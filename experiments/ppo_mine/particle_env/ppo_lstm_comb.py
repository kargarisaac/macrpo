import torch
import torch.nn as nn
from torch import distributions
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import optim

import numpy as np
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import json
import logging
import datetime
import os
import argparse
import sys

sys.path.append('/home/isaac/codes/autonomous_driving/multi-agent/')
from make_env import make_env as particke_env

import math
from dataclasses import dataclass
import time
from experiments.ppo_mine.particle_env.utils import make_env, calc_discounted_return, \
                                            compute_advantages, save_parameters,\
                                            save_checkpoint, get_env_space, StopConditions, \
                                            GumbelSoftmax, VecPyTorch, DummyVecEnv

#============================================================================================
# --------------------------------- settings ------------------------------------------------
# ===========================================================================================
# Save metrics for viewing with tensorboard.
SAVE_METRICS_TENSORBOARD = True

# Save actor & critic parameters for viewing in tensorboard.
SAVE_PARAMETERS_TENSORBOARD = False

# Save training state frequency in PPO iterations.
CHECKPOINT_FREQUENCY = 10

# Step env asynchronously using multiprocess or synchronously.
ASYNCHRONOUS_ENVIRONMENT = True

# Force using CPU for gathering trajectories.
FORCE_CPU_GATHER = True

# selfplay or not
SELFPLAY = True

# capture video or not
CAPTURE_VIDEO = False

#============================================================================================
# --------------------------------- training settings ---------------------------------------
# ===========================================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO agent')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
  
    args = parser.parse_args()

DEBUG = False
TRAIN = True
RESUME = False
N_AGENTS = 2
N_EVAL_EPISODE = 200
NORMALIZE_ADV = True

RESUME_CHECKPOINT_PATH = '/home/isaac/codes/autonomous_driving/multi-agent/experiments/ppo_mine/particle_env/checkpoints/2020_07_07/ppo_lstm_c2b_17:37:09/'

algo_name = "ppo_lstm_c2b_"

RANDOM_SEED = args.seed 

# Set random seed for consistant runs.
torch.random.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# Set maximum threads for torch to avoid inefficient use of multiple cpu cores.
torch.set_num_threads(10)
TRAIN_DEVICE = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
GATHER_DEVICE = "cpu" #"cuda" if torch.cuda.is_available() and not FORCE_CPU_GATHER else "cpu"

#============================================================================================
# --------------------------------- set directories and configs -----------------------------
# ===========================================================================================

WORKSPACE_PATH = "/home/isaac/codes/autonomous_driving/multi-agent/experiments/ppo_mine/particle_env/"
experiment_name = algo_name + str(datetime.datetime.today()).split(' ')[1].split('.')[0]
yyyymmdd = datetime.datetime.today().strftime("%Y_%m_%d")
EXPERIMENT_NAME = os.path.join(yyyymmdd, experiment_name)
BASE_CHECKPOINT_PATH = f"{WORKSPACE_PATH}/checkpoints/{EXPERIMENT_NAME}/"
LOG_DIR = f"{WORKSPACE_PATH}/logs/{EXPERIMENT_NAME}/"
VIDEO_DIR = f"{WORKSPACE_PATH}/videos/{EXPERIMENT_NAME}/"


@dataclass_json
@dataclass
class HyperParameters():
    actor_hidden_size:      int = 64
    critic_hidden_size:     int = 64
    # batch_size:           int   = 128 #128>32, 128>=512, 128>=1024
    discount:             float = 0.99 # 0.99>0.997, 0.98>>0.99
    gae_lambda:           float = 0.95 #0.95>0.94, 0.95>0.97, 0.95?0.96
    ppo_clip:             float = 0.2 #0.3 > 0.2 - 
    ppo_epochs:           int   = 10 #10>4, 10>20
    max_grad_norm:        float = 0.5 #1>0.5, 1>2, 1>10
    entropy_factor:       float = 0.00 #0.0001>0.001, 
    actor_learning_rate:  float = 1e-3 #1e-3>2e-3, 1e-3>8e-4
    critic_learning_rate: float = 1e-3 #1e-3>1e-4
    recurrent_seq_len:    int = 4
    recurrent_layers:     int = 1
    if DEBUG:
        rollout_steps:        int = 200
        parallel_rollouts:    int = 4
    else:
        rollout_steps:        int = 2048
        parallel_rollouts:    int = 100
    patience:             int = 200
    # Apply to continous action spaces only
    trainable_std_dev:    bool = False
    init_log_std_dev:     float = 0.0
    n_minibatch:           int  = 32
    batch_size:           int   = int(rollout_steps * parallel_rollouts / n_minibatch)



#============================================================================================
# --------------------------------- trajectory stuff ----------------------------------------
# ===========================================================================================

@dataclass
class TrajectorBatch():
    """
    Dataclass for storing data batch.
    """
    states: torch.tensor
    actions: torch.tensor
    action_probabilities: torch.tensor
    advantages: torch.tensor
    discounted_returns: torch.tensor
    batch_size: torch.tensor
    actor_hidden_states: torch.tensor
    actor_cell_states: torch.tensor
    critic_hidden_states: torch.tensor
    critic_cell_states: torch.tensor


class TrajectoryDataset():
    """
    Fast dataset for producing training batches from trajectories.
    """
    def __init__(self, trajectories, batch_size, device, batch_len, hp):
        # Combine multiple trajectories into
        self.trajectories = {key: value.to(device) for key, value in trajectories.items()}
        self.batch_len = batch_len
        truncated_seq_len = torch.clamp(trajectories["seq_len"] - batch_len + 1, 0, hp.rollout_steps)
        self.cumsum_seq_len = np.cumsum(np.concatenate((np.array([0]), truncated_seq_len.numpy())))
        self.batch_size = batch_size

    def __iter__(self):
        self.valid_idx = np.arange(self.cumsum_seq_len[-1])
        self.batch_count = 0
        return self

    def __next__(self):
        if self.batch_count * self.batch_size >= math.ceil(self.cumsum_seq_len[-1] / self.batch_len):
            raise StopIteration
        else:
            actual_batch_size = min(len(self.valid_idx), self.batch_size)
            start_idx = np.random.choice(self.valid_idx, size=actual_batch_size, replace=False)
            self.valid_idx = np.setdiff1d(self.valid_idx, start_idx) #remove start_idx from valid_idx
            eps_idx = np.digitize(start_idx, bins=self.cumsum_seq_len, right=False) - 1
            seq_idx = start_idx - self.cumsum_seq_len[eps_idx]
            series_idx = np.linspace(seq_idx, seq_idx + self.batch_len - 1, num=self.batch_len, dtype=np.int64)
            self.batch_count += 1

            return TrajectorBatch(**{key: value[eps_idx, series_idx] for key, value
                                    in self.trajectories.items() if key in TrajectorBatch.__dataclass_fields__.keys()},
                                    batch_size=actual_batch_size)



def gather_trajectories(input_data, hp):
    """
    Gather policy trajectories from gym environment.
    """

    # Unpack inputs.
    env = input_data["env"] #vectorized envs
    actor = input_data["actor"]
    critic = input_data["critic"]

    # Initialise variables.
    obsv = env.reset() #obsv shape:[n_envs, n_agents*obs shape of each agent]
    
    #separate trajs for two agents
    trajectory_data = {"states": [],
                        "actions": [],
                        "action_probabilities": [],
                        "rewards": [],
                        "values": [],
                        "terminals": [],
                        "actor_hidden_states": [],
                        "actor_cell_states": [],
                        "critic_hidden_states": [],
                        "critic_cell_states": [],
                        # "collisions": [],
                        # "min_dists": [],
                        # "occupied_landmarks": []
                        }

    terminal = torch.ones(hp.parallel_rollouts, 1)
    
    with torch.no_grad():

        episode_rewards = []
        episode_reward = torch.zeros(hp.parallel_rollouts)

        actor.get_init_state(hp.parallel_rollouts, GATHER_DEVICE)
        critic.get_init_state(hp.parallel_rollouts, GATHER_DEVICE)

        for i in range(hp.rollout_steps):

            action_list = []

            for agent_idx in range(N_AGENTS):
                
                trajectory_data["actor_hidden_states"].append(actor.hidden_cell[0].squeeze(0).cpu())
                trajectory_data["actor_cell_states"].append(actor.hidden_cell[1].squeeze(0).cpu())
                trajectory_data["critic_hidden_states"].append(critic.hidden_cell[0].squeeze(0).cpu())
                trajectory_data["critic_cell_states"].append(critic.hidden_cell[1].squeeze(0).cpu())
                
                # get state for agent_index
                state = obsv[:, agent_idx*obsv.shape[1]//N_AGENTS: (agent_idx+1)*obsv.shape[1]//N_AGENTS].clone()

                trajectory_data["states"].append(state)

                value = critic(state.unsqueeze(0).to(GATHER_DEVICE), terminal[:, 0].to(GATHER_DEVICE))

                trajectory_data["values"].append(value.squeeze(1).cpu())
                
                action_dist = actor(state.unsqueeze(0).to(GATHER_DEVICE), terminal[:, 0].to(GATHER_DEVICE))
                action = action_dist.sample().reshape(hp.parallel_rollouts, -1)
                if not actor.continuous_action_space:
                    action = action.squeeze(1)

                trajectory_data["actions"].append(action.cpu())
                trajectory_data["action_probabilities"].append(action_dist.log_prob(action).cpu())

                # action_list.append(action.cpu().numpy().squeeze())
                action_list.append(action)

            # Step environment
            action_array = torch.stack(action_list, axis=1).squeeze(1)
            obsv, reward, done, info = env.step(action_array)

            for a in range(N_AGENTS):
                episode_reward += reward[:, a]

            if done[0]:
                episode_rewards.append(torch.mean(episode_reward))
                # print(f'train episode rewrd mean: {np.mean(episode_reward)}, 100 last rewards mean: {np.mean(episode_rewards[-100:])}')
                print(f'train episode rewrd mean: {torch.mean(episode_reward)}')
                episode_reward = torch.zeros(hp.parallel_rollouts)
 
            terminal = torch.tensor(done).unsqueeze(1).float()

            #TODO: when env is done, only use one? 
            for _ in range(N_AGENTS - 1):
                trajectory_data["terminals"].append(torch.zeros_like(terminal[:, 0]))

            t = terminal[:, 0]
            if N_AGENTS > 1:
                for a in range(1, N_AGENTS):
                    t += terminal[:, 0] 
            t[t > 0] = 1
            trajectory_data["terminals"].append(t)
            
            for a in range(N_AGENTS):
                trajectory_data["rewards"].append(reward[:, a].float())

            # info has about (rew, collisions, min_dists, occupied_landmarks) for three agents
            # reward from env.step() is shared_reward -> sum(r1+r2+r3) or r1*3 -> all agents will get the same reward which is sum of the reward of all agents
            # reward from info is for each agent separately
            # for a in range(N_AGENTS):
            #     r = []
            #     col = []
            #     min_dist = []
            #     occupied_landmark = []
            #     for info_i in info:
            #         r.append(info_i['n'][a][0])
            #         col.append(info_i['n'][a][1])
            #         min_dist.append(info_i['n'][a][2])
            #         occupied_landmark.append(info_i['n'][a][3])

            #     trajectory_data["collisions"].append(torch.tensor(col).float())
            #     trajectory_data["min_dists"].append(torch.tensor(min_dist).float())
            #     trajectory_data["occupied_landmarks"].append(torch.tensor(occupied_landmark).float())

                        
        #---- end of loop -----

        # Compute final value to allow for incomplete episodes.
        #TODO: this bootstrap value should be for one agent or all? traj is combined and each agent is like one step.
        # maybe mean of 3 values
        # if we set rollout_steps to sth like 2000 which is devidable by 25 (max_time_steps), I think this bv will be unusable
        for agent_idx in range(1):
            # get state for agent_index
            state = obsv[:, agent_idx*obsv.shape[1]//N_AGENTS: (agent_idx+1)*obsv.shape[1]//N_AGENTS].clone()

            value = critic(state.unsqueeze(0).to(GATHER_DEVICE), terminal[:, 0].to(GATHER_DEVICE))

            trajectory_data["values"].append(value.squeeze(1).cpu())
        

    trajectory_tensors = {key: torch.stack(value) for key, value in trajectory_data.items()}

    return trajectory_tensors


def split_trajectories_episodes(trajectory_tensors, hp):
    """
    Split trajectories by episode.
    """

    states_episodes, actions_episodes, action_probabilities_episodes = [], [], []
    rewards_episodes, terminal_rewards_episodes, terminals_episodes, values_episodes = [], [], [], []
    policy_hidden_episodes, policy_cell_episodes, critic_hidden_episodes, critic_cell_episodes = [], [], [], []
    len_episodes = []
    trajectory_episodes = {key: [] for key in trajectory_tensors.keys()}

    for i in range(hp.parallel_rollouts):
        terminals_tmp = trajectory_tensors["terminals"].clone()
        terminals_tmp[0, i] = 1
        terminals_tmp[-1, i] = 1
        split_points = (terminals_tmp[:, i] == 1).nonzero() + 1

        split_lens = split_points[1:] - split_points[:-1]
        split_lens[0] += 1

        len_episode = [split_len.item() for split_len in split_lens]
        len_episodes += len_episode
        for key, value in trajectory_tensors.items():
            # Value includes additional step.
            if key == "values":
                value_split = list(torch.split(value[:, i], len_episode[:-1] + [len_episode[-1] + 1]))
                # Append extra 0 to values to represent no future reward for all episodes except final one
                # because the final episode in this rollout has its own future reward and it is not necessarily done. All the
                # previous episodes in this rollout are done, so their bootstrap value should be zero.
                for j in range(len(value_split) - 1):
                    value_split[j] = torch.cat((value_split[j], torch.zeros(1)))
                trajectory_episodes[key] += value_split
            else:
                trajectory_episodes[key] += torch.split(value[:, i], len_episode)
    return trajectory_episodes, len_episodes


def pad_and_compute_returns(trajectory_episodes, len_episodes, hp):
    """
    Pad the trajectories up to hp.rollout_steps so they can be combined in a
    single tensor.
    Add advantages and discounted_returns to trajectories.
    """

    episode_count = len(len_episodes) #number of all episodes for all parallel envs
    advantages_episodes, discounted_returns_episodes = [], []
    padded_trajectories = {key: [] for key in trajectory_episodes.keys()}
    padded_trajectories["advantages"] = []
    padded_trajectories["discounted_returns"] = []

    for i in range(episode_count):
        
        # in separated mode, the roll-out steps are half for each agent
        single_padding = torch.zeros(hp.rollout_steps - len_episodes[i])
        for key, value in trajectory_episodes.items():
            if value[i].ndim > 1:
                padding = torch.zeros(hp.rollout_steps - len_episodes[i], value[0].shape[1], dtype=value[i].dtype)
            else:
                padding = torch.zeros(hp.rollout_steps - len_episodes[i], dtype=value[i].dtype)
            padded_trajectories[key].append(torch.cat((value[i], padding)))
        padded_trajectories["advantages"].append(
            torch.cat((compute_advantages(rewards=trajectory_episodes["rewards"][i],
                                          values=trajectory_episodes["values"][i],
                                          discount=hp.discount,
                                          gae_lambda=hp.gae_lambda), single_padding)))
        padded_trajectories["discounted_returns"].append(
            torch.cat((calc_discounted_return(rewards=trajectory_episodes["rewards"][i],
                                              discount=hp.discount,
                                              final_value=trajectory_episodes["values"][i][-1]), single_padding)))
    return_val = {k: torch.stack(v) for k, v in padded_trajectories.items()}
    return_val["seq_len"] = torch.tensor(len_episodes)

    return return_val #[number of episodes, hp.rollout_steps]


#============================================================================================
# --------------------------------- neural nets ---------------------------------------------
# ===========================================================================================

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, continuous_action_space, trainable_std_dev, hp, init_log_std_dev=None):
        super().__init__()
        self.hp = hp
        self.embed = nn.Linear(state_dim, self.hp.actor_hidden_size)
        self.lstm = nn.LSTM(self.hp.actor_hidden_size, self.hp.actor_hidden_size, num_layers=self.hp.recurrent_layers)
        self.layer_hidden = nn.Linear(self.hp.actor_hidden_size, self.hp.actor_hidden_size)
        self.layer_policy_logits = nn.Linear(self.hp.actor_hidden_size, action_dim)
        self.action_dim = action_dim
        self.continuous_action_space = continuous_action_space
        self.log_std_dev = nn.Parameter(init_log_std_dev * torch.ones((action_dim), dtype=torch.float), requires_grad=trainable_std_dev)
        self.covariance_eye = torch.eye(self.action_dim).unsqueeze(0)
        self.hidden_cell = None
        self.nonlinearity = torch.tanh

    def get_init_state(self, batch_size, device):
        self.hidden_cell = (torch.zeros(self.hp.recurrent_layers, batch_size, self.hp.actor_hidden_size).to(device),
                            torch.zeros(self.hp.recurrent_layers, batch_size, self.hp.actor_hidden_size).to(device))

    def forward(self, state, terminal=None):
        batch_size = state.shape[1]
        device = state.device
        if self.hidden_cell is None or batch_size != self.hidden_cell[0].shape[1]:
            self.get_init_state(batch_size, device)
        if terminal is not None:
            self.hidden_cell = [value * (1. - terminal).reshape(1, batch_size, 1) for value in self.hidden_cell]
        
        embedded = self.nonlinearity(self.embed(state))
        lstm_out, self.hidden_cell = self.lstm(embedded, self.hidden_cell)
        # hidden_out = F.relu(self.layer_hidden(lstm_out)) #many to one- just use the last step hidden state
        policy_logits_out = self.layer_policy_logits(lstm_out).squeeze()
        # print(policy_logits_out.shape)
        if self.continuous_action_space:
            cov_matrix = self.covariance_eye.to(device).expand(batch_size, self.action_dim, self.action_dim) * torch.exp(self.log_std_dev.to(device))
            # We define the distribution on the CPU since otherwise operations fail with CUDA illegal memory access error.
            policy_dist = distributions.multivariate_normal.MultivariateNormal(policy_logits_out.to("cpu"), cov_matrix.to("cpu"))
        else:
            try:
                policy_dist = distributions.Categorical(F.softmax(policy_logits_out, dim=1).to("cpu"))
            except:
                policy_dist = distributions.Categorical(F.softmax(policy_logits_out, dim=0).to("cpu"))
        return policy_dist


class Critic(nn.Module):
    def __init__(self, state_dim, hp):
        super().__init__()
        self.hp = hp
        self.embed = nn.Linear(state_dim, self.hp.critic_hidden_size)
        self.layer_lstm = nn.LSTM(self.hp.critic_hidden_size, self.hp.critic_hidden_size, num_layers=self.hp.recurrent_layers)
        # self.layer_hidden = nn.Linear(self.hp.hidden_size, self.hp.hidden_size)
        self.layer_value = nn.Linear(self.hp.critic_hidden_size, 1)
        self.hidden_cell = None
        self.nonlinearity = torch.tanh

    def get_init_state(self, batch_size, device):
        self.hidden_cell = (torch.zeros(self.hp.recurrent_layers, batch_size, self.hp.critic_hidden_size).to(device),
                            torch.zeros(self.hp.recurrent_layers, batch_size, self.hp.critic_hidden_size).to(device))

    def forward(self, state, terminal=None):
        batch_size = state.shape[1]
        device = state.device
        if self.hidden_cell is None or batch_size != self.hidden_cell[0].shape[1]:
            self.get_init_state(batch_size, device)
        if terminal is not None:
            self.hidden_cell = [value * (1. - terminal).reshape(1, batch_size, 1) for value in self.hidden_cell]
        
        embedded = self.nonlinearity(self.embed(state))
        lstm_out, self.hidden_cell = self.layer_lstm(embedded, self.hidden_cell)
        # hidden_out = F.relu(self.layer_hidden(lstm_out)) #many to one- just use the last step hidden state
        value_out = self.layer_value(lstm_out)
        return value_out.squeeze(0)


#============================================================================================
# --------------------------------- helper functions ----------------------------------------
# ===========================================================================================

def evaluate_env(actor, render=False):
    """
    Evaluate policy
    """

    # Unpack inputs.
    env = particke_env('simple_spread_mine', True)

    # Initialise variables.
    reward_list = []
    col = []
    min_dist = []
    occupied_landmark = []

    obs = env.reset()

    # terminal = torch.ones(3)
    with torch.no_grad():

        actor.get_init_state(1, GATHER_DEVICE)

        # Take 1 additional step in order to collect the state and value for the final state.
        episode_reward = 0
        episode_col = 0
        episode_min_dist = 0
        episode_occupied_landmark = 0
        episode_steps = 0

        while True:
            action_list = []
            # Choose next action
            for agent_idx in range(N_AGENTS):
                # state = torch.tensor(obs[agent_idx], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                state = torch.tensor(obs[agent_idx], dtype=torch.float32).unsqueeze(0)
                # print('state shape: ', state.shape)
                action_dist = actor(state.unsqueeze(0).to(GATHER_DEVICE))
                # print('action: ', action_dist.probs.shape)
                # action = action_dist.probs
                action = torch.argmax(action_dist.probs)
                # action = action_dist.sample()
                # print(action)
                if not actor.continuous_action_space:
                    action = action.squeeze(0)

                # Step environment
                action_list.append(action.cpu().numpy().squeeze())

            # action_np = np.stack(action_list, axis=0)
            # print(action_list[0].shape)
            obs, reward, done, info = env.step(action_list)

            for i in range(N_AGENTS):
                episode_reward += reward[i]                
                episode_col += info['n'][i][1]
                episode_min_dist += info['n'][i][2]
                episode_occupied_landmark += info['n'][i][3]

            # terminal = torch.tensor(done).float()

            episode_steps += 1

            if render:
                env.render()
                time.sleep(0.1)

            if episode_steps == 25:
                reward_list.append(episode_reward)
                col.append(episode_col)
                min_dist.append(episode_min_dist)
                occupied_landmark.append(episode_occupied_landmark)
                obs = env.reset()
                actor.get_init_state(1, GATHER_DEVICE)
                if render:
                    print('episode reward: ', episode_reward)
                episode_reward = 0
                episode_steps = 0
                if len(reward_list) > N_EVAL_EPISODE:
                    break

    env.close()
    return np.mean(reward_list), np.mean(col), np.mean(min_dist), np.mean(occupied_landmark)


def train_model(actor,
                critic,
                actor_optimizer,
                critic_optimizer,
                stop_conditions,
                hp                
                ):
    # Vector environment manages multiple instances of the environment.
    # A key difference between this and the standard gym environment is it automatically resets.
    # Therefore when the done flag is active in the done vector the corresponding state is the first new state.
    
    env = VecPyTorch(DummyVecEnv([make_env(RANDOM_SEED+i, i) for i in range(hp.parallel_rollouts)]), GATHER_DEVICE)

    writer = SummaryWriter(log_dir=LOG_DIR)

    iteration = 0

    if RESUME:
        actor.load_state_dict(torch.load(RESUME_CHECKPOINT_PATH + 'actor.pt'))
        critic.load_state_dict(torch.load(RESUME_CHECKPOINT_PATH + 'critic.pt'))
        actor_optimizer.load_state_dict(torch.load(RESUME_CHECKPOINT_PATH + 'actor_optimizer.pt'))
        critic_optimizer.load_state_dict(torch.load(RESUME_CHECKPOINT_PATH + 'critic_optimizer.pt'))

    while iteration < stop_conditions.max_iterations:

        actor = actor.to(GATHER_DEVICE)
        critic = critic.to(GATHER_DEVICE)
        start_gather_time = time.time()

        # Gather trajectories.
        input_data = {"env": env, "actor": actor, "critic": critic, "discount": hp.discount,
                      "gae_lambda": hp.gae_lambda}
        trajectory_tensors = gather_trajectories(input_data, hp)
        trajectory_episodes, len_episodes = split_trajectories_episodes(trajectory_tensors, hp)
        trajectories = pad_and_compute_returns(trajectory_episodes, len_episodes, hp)
        # 'trajectories' is a tensor with trajectories collected from all parallel envs-> [number of episodes, hp.rollout_steps]

        # Calculate mean reward.
        
        complete_episode_count = trajectories["terminals"].sum().item()
        # just consider episodes that are done (completed)
        terminal_episodes_rewards = (
                    trajectories["terminals"].sum(axis=1) * trajectories["rewards"].sum(axis=1)).sum()
        # terminal_episodes_collisions = (
        #             trajectories["terminals"].sum(axis=1) * trajectories["collisions"].sum(axis=1)).sum()
        # terminal_episodes_min_dists = (
        #             trajectories["terminals"].sum(axis=1) * trajectories["min_dists"].sum(axis=1)).sum()
        # terminal_episodes_occupied_landmarks = (
        #             trajectories["terminals"].sum(axis=1) * trajectories["occupied_landmarks"].sum(axis=1)).sum()

        # cuz we separated trajs, and want to calculate the total reward of whole traj, two agents, we need to
        # consider half of the number of done trajs
        train_mean_reward = terminal_episodes_rewards / (complete_episode_count)
        # train_mean_collisions = terminal_episodes_collisions / (complete_episode_count)
        # train_mean_min_dists = terminal_episodes_min_dists / (complete_episode_count)
        # train_mean_occupied_landmarks = terminal_episodes_occupied_landmarks / (complete_episode_count)
        
        # train_mean_reward = trajectory_tensors['rewards'].sum()

        # # # Check stop conditions.
        # if train_mean_reward > stop_conditions.best_reward:
        #     stop_conditions.best_reward = train_mean_reward
        #     stop_conditions.fail_to_improve_count = 0
        # else:
        #     stop_conditions.fail_to_improve_count += 1
        # if stop_conditions.fail_to_improve_count > hp.patience:
        #     print(f"Policy has not yielded higher reward for {hp.patience} iterations...  Stopping now.")
        #     break

        trajectory_dataset = TrajectoryDataset(trajectories,
                                               batch_size=hp.batch_size,
                                               device=TRAIN_DEVICE,
                                               batch_len=hp.recurrent_seq_len,
                                               hp=hp)
        end_gather_time = time.time()

        #======= end of data collection and starting of training ===========
        start_train_time = time.time()

        actor = actor.to(TRAIN_DEVICE)
        critic = critic.to(TRAIN_DEVICE)

        # Train actor and critic.
        for epoch_idx in range(hp.ppo_epochs):
            for batch in trajectory_dataset:
                # Get batch
                actor.hidden_cell = (batch.actor_hidden_states[:1], batch.actor_cell_states[:1])
                
                # Update actor
                actor_optimizer.zero_grad()
                action_dist = actor(batch.states)

                action_probabilities = action_dist.log_prob(batch.actions.to("cpu")).to(TRAIN_DEVICE)
                probabilities_ratio = torch.exp(action_probabilities - batch.action_probabilities)
                if NORMALIZE_ADV:
                    advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
                else:
                    advantages = batch.advantages
                surrogate_loss_0 = probabilities_ratio * batch.advantages
                surrogate_loss_1 = torch.clamp(probabilities_ratio, 1. - hp.ppo_clip,
                                                1. + hp.ppo_clip) * batch.advantages

                surrogate_loss_2 = action_dist.entropy().to(TRAIN_DEVICE)
                actor_loss = -torch.mean(torch.min(surrogate_loss_0, surrogate_loss_1)) - torch.mean(
                    hp.entropy_factor * surrogate_loss_2)
                actor_loss.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), hp.max_grad_norm)
                actor_optimizer.step()

                # Update critic
                critic_optimizer.zero_grad()

                critic.hidden_cell = (batch.critic_hidden_states[:1], batch.critic_cell_states[:1])

                values = critic(batch.states)
                critic_loss = F.mse_loss(batch.discounted_returns, values.squeeze(2))
                torch.nn.utils.clip_grad.clip_grad_norm_(critic.parameters(), hp.max_grad_norm)
                critic_loss.backward()
                critic_optimizer.step()

        end_train_time = time.time()

        start_eval_time = time.time()
        test_mean_reward, test_mean_col, test_mean_min_dist, test_mean_occupied_lansmark = evaluate_env(actor)
        end_eval_time = time.time()

        print(
            f"Iteration: {iteration},  Mean reward: {test_mean_reward:.2f}, Mean Entropy: {torch.mean(surrogate_loss_2):.2f}, " +
            # f"Mean collision: {test_mean_col:.2f}, Mean min dist: {test_mean_min_dist:.2f}, Mean occupied landmark: {test_mean_occupied_lansmark:.2f}, " +
            f"complete_episode_count: {complete_episode_count:.2f}, Gather time: {end_gather_time - start_gather_time:.2f}s, " +
            f"Train time: {end_train_time - start_train_time:.2f}s, Eval Time: {end_eval_time - start_eval_time:.2f}s")

        print('=====================================================')

        if SAVE_METRICS_TENSORBOARD:
            writer.add_scalar("complete_episode_count", complete_episode_count, iteration)
            # writer.add_scalar("total_reward/test", test_mean_reward, iteration)
            writer.add_scalar("mean_reward", test_mean_reward, iteration)
            # writer.add_scalar("train_mean_collisions", train_mean_collisions, iteration)
            # writer.add_scalar("train_mean_min_dists", train_mean_min_dists, iteration)
            # writer.add_scalar("train_mean_occupied_landmarks", train_mean_occupied_landmarks, iteration)
            writer.add_scalar("actor_loss", actor_loss, iteration)
            writer.add_scalar("critic_loss", critic_loss, iteration)
            writer.add_scalar("policy_entropy", torch.mean(surrogate_loss_2), iteration)
        if SAVE_PARAMETERS_TENSORBOARD:
            save_parameters(writer, "actor", actor, iteration)
            save_parameters(writer, "value", critic, iteration)
        if iteration % CHECKPOINT_FREQUENCY == 0:
            save_checkpoint(actor,
                            critic,
                            actor_optimizer,
                            critic_optimizer,
                            iteration,
                            stop_conditions,
                            env_id='simple_spread_mine',
                            hp=hp,
                            base_checkpoint_path=BASE_CHECKPOINT_PATH)
        iteration += 1

    env.close()
    return stop_conditions.best_reward


#============================================================================================
# --------------------------------- training ------------------------------------------------
# ===========================================================================================


hp = HyperParameters()

cfg = dict(algo_config=hp.to_dict())
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
with open(os.path.join(LOG_DIR, 'params.json'), 'w') as fp:
    json.dump(cfg, fp)

batch_count = hp.parallel_rollouts * hp.rollout_steps / hp.recurrent_seq_len / hp.batch_size
print(f"batch_count: {batch_count}")
# if not DEBUG:
#     assert batch_count >= 1., "Less than 1 batch per trajectory.  Are you sure that's what you want?"

obsv_dim, action_dim, continuous_action_space = get_env_space()

actor = Actor(obsv_dim,
                action_dim,
                continuous_action_space=continuous_action_space,
                trainable_std_dev=hp.trainable_std_dev,
                hp=hp,
                init_log_std_dev=hp.init_log_std_dev)
critic = Critic(obsv_dim, hp=hp)

actor_optimizer = optim.Adam(actor.parameters(), lr=hp.actor_learning_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr=hp.critic_learning_rate)

stop_conditions = StopConditions()

if TRAIN:
    score = train_model(actor,
                        critic,
                        actor_optimizer,
                        critic_optimizer,
                        stop_conditions,
                        hp
                        )

## for evaluating the trained model
else:
    actor.load_state_dict(torch.load(
            RESUME_CHECKPOINT_PATH + 'actor.pt'
            ))
    evaluate_env(actor, True)