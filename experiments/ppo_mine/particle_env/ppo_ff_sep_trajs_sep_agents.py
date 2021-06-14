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
from dotmap import DotMap
import pathlib
import pickle

sys.path.append('/home/isaac/codes/autonomous_driving/multi-agent/')
from make_env import make_env as particke_env

import math
from dataclasses import dataclass
import time
from experiments.ppo_mine.particle_env.utils import make_env, calc_discounted_return, \
                                            compute_advantages, save_parameters,\
                                            save_checkpoint, get_env_space, StopConditions, \
                                            GumbelSoftmax, VecPyTorch, DummyVecEnv, one_hot_embedding
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
TRAIN = False
RESUME = False
N_AGENTS = 3
N_EVAL_EPISODE = 200
NORMALIZE_ADV = True

RESUME_CHECKPOINT_PATH = '/home/isaac/codes/autonomous_driving/multi-agent/experiments/ppo_mine/particle_env/checkpoints/2020_07_06/ppo_ff_c1aa_23:20:12/'

algo_name = "ppo_ff_c1aa_"

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
    discount:             float = 0.99
    gae_lambda:           float = 0.95
    ppo_clip:             float = 0.2
    ppo_epochs:           int   = 10
    max_grad_norm:        float = 0.5
    entropy_factor:       float = 0.0
    actor_learning_rate:  float = 1e-3
    critic_learning_rate: float = 1e-3
    recurrent_seq_len:    int = 20
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
    critic_input: torch.tensor
    actions: torch.tensor
    action_probabilities: torch.tensor
    advantages: torch.tensor
    discounted_returns: torch.tensor
    batch_size: torch.tensor


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

            # do these for ff
            np.random.shuffle(series_idx)  # shuffle on rows
            dataset_dict = {}
            for key, value in self.trajectories.items():
                if key in TrajectorBatch.__dataclass_fields__.keys():
                    if value.ndim == 2:
                        dataset_dict[key]= value[eps_idx, series_idx].view(-1, )
                    elif value.ndim == 3:
                        dataset_dict[key] = value[eps_idx, series_idx].view(-1, value.shape[-1])
            return TrajectorBatch(**dataset_dict, batch_size=actual_batch_size)


def gather_trajectories(input_data, hp):
    """
    Gather policy trajectories from gym environment.
    """

    # Unpack inputs.
    env = input_data["env"] #vectorized envs
    actors = input_data["actors"]
    critics = input_data["critics"]

    # Initialise variables.
    obsv = env.reset() #obsv shape:[n_envs, n_agents*obs shape of each agent]
    
    #separate trajs for two agents
    trajectory_data = {"states": {i:[] for i in range(N_AGENTS)},
                        "critic_input": {i:[] for i in range(N_AGENTS)},
                        "actions": {i:[] for i in range(N_AGENTS)},
                        "action_probabilities": {i:[] for i in range(N_AGENTS)},
                        "rewards": {i:[] for i in range(N_AGENTS)},
                        "values": {i:[] for i in range(N_AGENTS)},
                        "terminals": {i:[] for i in range(N_AGENTS)},
                        # "collisions": {i:[] for i in range(N_AGENTS)},
                        # "min_dists": {i:[] for i in range(N_AGENTS)},
                        # "occupied_landmarks": {i:[] for i in range(N_AGENTS)}
                        }

    terminal = torch.ones(hp.parallel_rollouts, 1)

    with torch.no_grad():
        
        episode_rewards = []
        episode_reward = torch.zeros(hp.parallel_rollouts)

        # prev_action = torch.zeros([hp.parallel_rollouts, N_AGENTS])

        for i in range(hp.rollout_steps):
            
            action_list = []

            for agent_idx in range(N_AGENTS):

                # get state for agent_index
                state = obsv[:, agent_idx*obsv.shape[1]//N_AGENTS: (agent_idx+1)*obsv.shape[1]//N_AGENTS].clone()

                trajectory_data["states"][agent_idx].append(state)
                
                action_dist = actors[agent_idx](state.unsqueeze(0).to(GATHER_DEVICE), terminal[:, 0].to(GATHER_DEVICE))
                action = action_dist.sample().reshape(hp.parallel_rollouts, -1)
                if not actors[agent_idx].continuous_action_space:
                    action = action.squeeze(1)
                trajectory_data["actions"][agent_idx].append(action.cpu())
                trajectory_data["action_probabilities"][agent_idx].append(action_dist.log_prob(action).cpu())
                
                # action_list.append(action.cpu().numpy().squeeze())
                action_list.append(action)

            #TODO: action of all agents should be considered for critic input. use
            action_1hot = torch.cat([one_hot_embedding(action_list[a], 5) for a in range(N_AGENTS)], dim=1)
            critic_input = torch.cat([obsv.clone(), action_1hot], dim=1)
            # critic_input = obsv.clone()

            for agent_idx in range(N_AGENTS):
                trajectory_data["critic_input"][agent_idx].append(critic_input)
                value = critics[agent_idx](critic_input.unsqueeze(0).to(GATHER_DEVICE), terminal[:, 0].to(GATHER_DEVICE))
                trajectory_data["values"][agent_idx].append(value.squeeze(1).cpu())

            # Step environment
            action_array = torch.stack(action_list, axis=1).squeeze(1)
            # prev_action = action_array
            obsv, reward, done, info = env.step(action_array)
 
            for a in range(N_AGENTS):
                episode_reward += reward[:, a]

            if done[0]:
                episode_rewards.append(torch.mean(episode_reward))
                # print(f'train episode rewrd mean: {np.mean(episode_reward)}, 100 last rewards mean: {np.mean(episode_rewards[-100:])}')
                print(f'train episode rewrd mean: {torch.mean(episode_reward)}')
                episode_reward = torch.zeros(hp.parallel_rollouts)

            terminal = torch.tensor(done).unsqueeze(1).float()
            
            for a in range(N_AGENTS):
                trajectory_data["terminals"][a].append(terminal[:, 0])

            for a in range(N_AGENTS):
                trajectory_data["rewards"][a].append(reward[:, a].float())


            # info has about (rew, collisions, min_dists, occupied_landmarks) for three agents
            # reward from env.step() is shared_reward -> sum(r1+r2+r3) or r1*3 -> all agents will get the same reward which is sum of the reward of all agents
            # reward from info is for each agent separately
            # for a in range():
            #     r = []
            #     col = []
            #     min_dist = []
            #     occupied_landmark = []
            #     for info_i in info:
            #         r.append(info_i['n'][a][0])
            #         col.append(info_i['n'][a][1])
            #         min_dist.append(info_i['n'][a][2])
            #         occupied_landmark.append(info_i['n'][a][3])

            #     trajectory_data["rewards"][a].append(torch.tensor(r).float())
            #     trajectory_data["collisions"][a].append(torch.tensor(col).float())
            #     trajectory_data["min_dists"][a].append(torch.tensor(min_dist).float())
            #     trajectory_data["occupied_landmarks"][a].append(torch.tensor(occupied_landmark).float())

                        
        #---- end of loop -----

        # Compute final value to allow for incomplete episodes.
        for agent_idx in range(N_AGENTS):
            # get state for agent_index
            # state = torch.tensor(obsv[:, agent_idx*obsv.shape[1]//N_AGENTS: (agent_idx+1)*obsv.shape[1]//N_AGENTS], dtype=torch.float32)
            action_1hot = torch.cat([one_hot_embedding(action_list[a], 5) for a in range(N_AGENTS)], dim=1)
            critic_input = torch.cat([obsv.clone(), action_1hot], dim=1)
            # critic_input = obsv.clone()
            value = critics[agent_idx](critic_input.unsqueeze(0).to(GATHER_DEVICE), terminal[:, 0].to(GATHER_DEVICE))

            trajectory_data["values"][agent_idx].append(value.squeeze(1).cpu())
        
        # state = torch.tensor(obsv, dtype=torch.float32)

        # value = critic(state.unsqueeze(0).to(GATHER_DEVICE), terminal.to(GATHER_DEVICE))
        # # Future value for terminal episodes is 0.
        # trajectory_data["values"][agent_idx].append(value.squeeze(1).cpu() * (1 - terminal))

    # trajectory_tensors = {key: torch.cat([torch.stack(value[0]), torch.stack(value[1]), torch.stack(value[2])], dim=1) for key, value in trajectory_data.items()}
    trajectory_tensors = []
    for a in range(N_AGENTS):
        trajectory_tensors.append({key: torch.stack(value[a]) for key, value in trajectory_data.items()})

    return trajectory_tensors #list of data for agents -> for n agents: list on n tensors


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

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # torch.nn.init.orthogonal_(layer.weight, std)
    # torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, continuous_action_space, trainable_std_dev, hp, init_log_std_dev=None):
        super().__init__()
        self.hp = hp
        self.embed = layer_init(nn.Linear(state_dim, self.hp.actor_hidden_size))
        self.layer_hidden = layer_init(nn.Linear(self.hp.actor_hidden_size, self.hp.actor_hidden_size))
        self.layer_policy_logits = layer_init(nn.Linear(self.hp.actor_hidden_size, action_dim), std=0.01)
        self.action_dim = action_dim
        self.continuous_action_space = continuous_action_space
        self.log_std_dev = nn.Parameter(init_log_std_dev * torch.ones((action_dim), dtype=torch.float),
                                        requires_grad=trainable_std_dev)
        self.covariance_eye = torch.eye(self.action_dim).unsqueeze(0)
        self.nonlinearity = torch.tanh

    def forward(self, state, terminal=None):
        state = state.squeeze(0)
        batch_size = state.shape[0]
        device = state.device
        embed = self.nonlinearity(self.embed(state))
        hidden_out = self.nonlinearity(self.layer_hidden(embed))
        policy_logits_out = self.layer_policy_logits(hidden_out)
        if self.continuous_action_space:
            cov_matrix = self.covariance_eye.to(device).expand(batch_size, self.action_dim,
                                                               self.action_dim) * torch.exp(self.log_std_dev.to(device))
            # We define the distribution on the CPU since otherwise operations fail with CUDA illegal memory access error.
            policy_dist = distributions.multivariate_normal.MultivariateNormal(policy_logits_out.to("cpu"),
                                                                               cov_matrix.to("cpu"))
        else:
            policy_dist = distributions.Categorical(F.softmax(policy_logits_out, dim=1).to("cpu"))
            # policy_dist = distributions.Categorical(F.gumbel_softmax(policy_logits_out, tau=1, hard=False).to("cpu"))
        return policy_dist


class Critic(nn.Module):
    def __init__(self, state_dim, hp):
        super().__init__()
        self.hp = hp
        self.embed = layer_init(nn.Linear(state_dim, self.hp.critic_hidden_size))
        self.layer_hidden = layer_init(nn.Linear(self.hp.critic_hidden_size, self.hp.critic_hidden_size))
        self.layer_value = layer_init(nn.Linear(self.hp.critic_hidden_size, 1), std=1.)
        self.nonlinearity = torch.tanh
        
    def forward(self, state, terminal=None):
        state = state.squeeze(0)
        device = state.device
        embed = self.nonlinearity(self.embed(state))
        hidden_out = self.nonlinearity(self.layer_hidden(embed))
        value_out = self.layer_value(hidden_out)
        return value_out



#============================================================================================
# --------------------------------- helper functions ----------------------------------------
# ===========================================================================================


def evaluate_env(actors, render=False):
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

    for i in range(N_AGENTS):
        actors[i] = actors[i].to(GATHER_DEVICE)

    # terminal = torch.ones(3)
    with torch.no_grad():
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
                state = torch.tensor(obs[agent_idx], dtype=torch.float32).unsqueeze(0)
                action_dist = actors[agent_idx](state.unsqueeze(0).to(GATHER_DEVICE))
                # print('action: ', action_dist.probs.shape)
                action = action_dist.probs
                action = torch.argmax(action_dist.probs)
                # action = action_dist.sample()
                # print(action)
                if not actors[agent_idx].continuous_action_space:
                    action = action.squeeze(0)

                # Step environment
                action_list.append(action.cpu().numpy().squeeze())

            # action_np = np.stack(action_list, axis=1)
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
                if render:
                    print('episode reward: ', episode_reward)
                episode_reward = 0
                episode_steps = 0
                if len(reward_list) > N_EVAL_EPISODE:
                    break

    env.close()
    return np.mean(reward_list), np.mean(col), np.mean(min_dist), np.mean(occupied_landmark)


def train_model(actors,
                critics,
                actor_optimizers,
                critic_optimizers,
                # actor_scheduler,
                # critic_scheduler,
                stop_conditions,
                hp                
                ):
    # Vector environment manages multiple instances of the environment.
    # A key difference between this and the standard gym environment is it automatically resets.
    # Therefore when the done flag is active in the done vector the corresponding state is the first new state.
    #     env = gym.vector.make(ENV, hp.parallel_rollouts, asynchronous=ASYNCHRONOUS_ENVIRONMENT, kwargs=env_config)
    # env = make_env(hp.parallel_rollouts, asynchronous=ASYNCHRONOUS_ENVIRONMENT)
    env = VecPyTorch(DummyVecEnv([make_env(RANDOM_SEED+i, i) for i in range(hp.parallel_rollouts)]), GATHER_DEVICE)

    # eval_env = env_cls(is_intersection_map = env_config['is_intersection_map'])  # for self-play to have 2 learning agents
    # eval_env.configure_env(env_config)

    writer = SummaryWriter(log_dir=LOG_DIR)

    iteration = 0

    if RESUME:
        for i in range(N_AGENTS):
            actors[i].load_state_dict(torch.load(RESUME_CHECKPOINT_PATH + 'actor_' + str(i+1) + '.pt'))
            critics[i].load_state_dict(torch.load(RESUME_CHECKPOINT_PATH + 'critic_' + str(i+1) + '.pt'))
            actor_optimizers[i].load_state_dict(torch.load(RESUME_CHECKPOINT_PATH + 'actor_optimizer_' + str(i+1) + '.pt'))
            critic_optimizers[i].load_state_dict(torch.load(RESUME_CHECKPOINT_PATH + 'critic_optimizer_' + str(i+1) + '.pt'))

    while iteration < stop_conditions.max_iterations:
        for i in range(len(actors)):
            actors[i] = actors[i].to(GATHER_DEVICE)
            critics[i] = critics[i].to(GATHER_DEVICE)

        start_gather_time = time.time()

        # Gather trajectories.
        input_data = {"env": env, "actors": actors, "critics": critics, "discount": hp.discount,
                      "gae_lambda": hp.gae_lambda}

        trajectory_tensors_list = gather_trajectories(input_data, hp)

        trajectories_list = []
        for traj_tens in trajectory_tensors_list:
            a, b = split_trajectories_episodes(traj_tens, hp)
            trajectories_list.append(pad_and_compute_returns(a, b, hp))
            
        # Calculate mean reward.
        # complete_episode_count1 = trajectories1["terminals"].sum().item()
        # complete_episode_count2 = trajectories2["terminals"].sum().item()

        # just consider episodes that are done (completed)
        # terminal_episodes_rewards = (
        #             trajectories["terminals"].sum(axis=1) * trajectories["rewards"].sum(axis=1)).sum()
        # terminal_episodes_collisions = (
        #             trajectories["terminals"].sum(axis=1) * trajectories["collisions"].sum(axis=1)).sum()
        # terminal_episodes_min_dists = (
        #             trajectories["terminals"].sum(axis=1) * trajectories["min_dists"].sum(axis=1)).sum()
        # terminal_episodes_occupied_landmarks = (
        #             trajectories["terminals"].sum(axis=1) * trajectories["occupied_landmarks"].sum(axis=1)).sum()

        # cuz we separated trajs, and want to calculate the total reward of whole traj, two agents, we need to
        # consider half of the number of done trajs
        # train_mean_reward = 3 * terminal_episodes_rewards / (complete_episode_count)
        # train_mean_collisions = 3 * terminal_episodes_collisions / (complete_episode_count)
        # train_mean_min_dists = 3 * terminal_episodes_min_dists / (complete_episode_count)
        # train_mean_occupied_landmarks = 3 * terminal_episodes_occupied_landmarks / (complete_episode_count)
        
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

        trajectory_dataset_list = []
        for traj in trajectories_list:
            trajectory_dataset_list.append(TrajectoryDataset(traj,
                                                    batch_size=hp.batch_size,
                                                    device=TRAIN_DEVICE,
                                                    batch_len=hp.recurrent_seq_len,
                                                    hp=hp))
        
        end_gather_time = time.time()

        #======= end of data collection and starting of training ===========
        start_train_time = time.time()

        for i in range(N_AGENTS):
            actors[i] = actors[i].to(TRAIN_DEVICE)
            critics[i] = critics[i].to(TRAIN_DEVICE)

        # Train actor and critic.
        for epoch_idx in range(hp.ppo_epochs):
            for agent_idx, trajectory_dataset in enumerate(trajectory_dataset_list):
                for batch in trajectory_dataset:
                    
                    # Update actor
                    actor_optimizers[agent_idx].zero_grad()
                    action_dist = actors[agent_idx](batch.states)

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
                    torch.nn.utils.clip_grad.clip_grad_norm_(actors[agent_idx].parameters(), hp.max_grad_norm)
                    actor_optimizers[agent_idx].step()

                    # Update critic
                    critic_optimizers[agent_idx].zero_grad()
                    values = critics[agent_idx](batch.critic_input)
                    critic_loss = F.mse_loss(batch.discounted_returns, values.squeeze(1))
                    torch.nn.utils.clip_grad.clip_grad_norm_(critics[agent_idx].parameters(), hp.max_grad_norm)
                    critic_loss.backward()
                    critic_optimizers[agent_idx].step()
            
        end_train_time = time.time()

        test_mean_reward, test_mean_col, test_mean_min_dist, test_mean_occupied_lansmark = evaluate_env(actors)

        print(
            f"Iteration: {iteration},  Mean reward: {test_mean_reward:.2f}, " + #Mean Entropy: {torch.mean(surrogate_loss_2):.2f}, " +
            # f"Mean collision: {test_mean_col:.2f}, Mean min dist: {test_mean_min_dist:.2f}, Mean occupied landmark: {test_mean_occupied_lansmark:.2f}, " +
            # f"complete_episode_count: {complete_episode_count:.2f}," +
            f"Gather time: {end_gather_time - start_gather_time:.2f}s, " +
            f"Train time: {end_train_time - start_train_time:.2f}s")
        
        print('=====================================================')

        if SAVE_METRICS_TENSORBOARD:
            # writer.add_scalar("complete_episode_count", complete_episode_count, iteration)
            # writer.add_scalar("total_reward/test", test_mean_reward, iteration)
            writer.add_scalar("mean_reward", test_mean_reward, iteration)
            # writer.add_scalar("train_mean_collisions", train_mean_collisions, iteration)
            # writer.add_scalar("train_mean_min_dists", train_mean_min_dists, iteration)
            # writer.add_scalar("train_mean_occupied_landmarks", train_mean_occupied_landmarks, iteration)
            # writer.add_scalar("actor_loss", actor_loss, iteration)
            # writer.add_scalar("critic_loss", critic_loss, iteration)
            # writer.add_scalar("policy_entropy", torch.mean(surrogate_loss_2), iteration)
            # writer.add_scalar("actor_lr", actor_optimizer.param_groups[0]['lr'], iteration)
            # writer.add_scalar("critic_lr", critic_optimizer.param_groups[0]['lr'], iteration)

        # if SAVE_PARAMETERS_TENSORBOARD:
        #     save_parameters(writer, "actor", actor, iteration)
        #     save_parameters(writer, "value", critic, iteration)
        if iteration % CHECKPOINT_FREQUENCY == 0:
            save_checkpoint(actors,
                            critics,
                            actor_optimizers,
                            critic_optimizers,
                            iteration,
                            stop_conditions,
                            env_id='simple_spread_mine',
                            hp=hp,
                            base_checkpoint_path=BASE_CHECKPOINT_PATH)
        iteration += 1
        # actor_scheduler.step()
        # critic_scheduler.step()

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
if not DEBUG:
    assert batch_count >= 1., "Less than 1 batch per trajectory.  Are you sure that's what you want?"

obsv_dim, action_dim, continuous_action_space = get_env_space()
obsv_dim_actor = obsv_dim
obsv_dim_critic = (obsv_dim + action_dim) * N_AGENTS
# obsv_dim_critic = (obsv_dim) * N_AGENTS

actors = []
critics = []
actor_optimizers = []
critic_optimizers = []
for i in range(N_AGENTS):
    actors.append(
        Actor(obsv_dim_actor,
            action_dim,
            continuous_action_space=continuous_action_space,
            trainable_std_dev=hp.trainable_std_dev,
            hp=hp,
            init_log_std_dev=hp.init_log_std_dev)
    )
    critics.append(Critic(obsv_dim_critic, hp=hp))

    actor_optimizers.append(optim.Adam(actors[i].parameters(), lr=hp.actor_learning_rate))
    critic_optimizers.append(optim.Adam(critics[i].parameters() , lr=hp.critic_learning_rate))

    
# actor_scheduler = optim.lr_scheduler.StepLR(actor_optimizer, step_size=1000, gamma=0.1)
# critic_scheduler = optim.lr_scheduler.StepLR(critic_optimizer, step_size=1000, gamma=0.1)

stop_conditions = StopConditions()

if TRAIN:
    score = train_model(actors,
                        critics,
                        actor_optimizers,
                        critic_optimizers,
                        # actor_scheduler,
                        # critic_scheduler,
                        stop_conditions,
                        hp
                        )

## for evaluating the trained model
else:
    for i in range(N_AGENTS):
            actors[i].load_state_dict(torch.load(RESUME_CHECKPOINT_PATH + 'actor_' + str(i+1) + '.pt'))
    evaluate_env(actors, True)

