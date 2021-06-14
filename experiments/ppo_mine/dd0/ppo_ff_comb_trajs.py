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
# sys.path.append('/home/isaac/codes/autonomous_driving/deepdrive-zero/')
# sys.path.append('/home/isaac/codes/autonomous_driving/multi-agent/')
sys.path.append('/scratch/work/kargare1/codes/deepdrive-zero/')
sys.path.append('/scratch/work/kargare1/codes/multi-agent2/')
from deepdrive_zero.envs.env import Deepdrive2DEnv

import math
from dataclasses import dataclass
import time
from experiments.ppo_mine.dd0.utils import make_env, calc_discounted_return, \
                                        compute_advantages, save_parameters,\
                                        save_checkpoint, get_env_space, StopConditions

import wandb

#============================================================================================
# --------------------------------- settings ------------------------------------------------
# ===========================================================================================
# Save metrics for viewing with tensorboard.
SAVE_METRICS_TENSORBOARD = True

# Save actor & critic parameters for viewing in tensorboard.
SAVE_PARAMETERS_TENSORBOARD = False

# Save training state frequency in PPO iterations.
CHECKPOINT_FREQUENCY = 1000

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
    parser.add_argument('--beta', type=float, default=1.0,
                        help='beta of A and R')
  
    args = parser.parse_args()


DEBUG = False
N_AGENTS = 2
algo_name = "ppo_ff_c1b_"

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

# WORKSPACE_PATH = "/home/isaac/codes/autonomous_driving/multi-agent/experiments/ppo_mine/dd0/"
WORKSPACE_PATH = "/scratch/work/kargare1/codes/multi-agent2/experiments/ppo_mine/dd0/"
# experiment_name = algo_name + "beta_" + str(args.beta) + "_" + str(datetime.datetime.today()).split(' ')[1].split('.')[0]
experiment_name = algo_name + "beta_" + str(args.beta)
yyyymmdd = datetime.datetime.today().strftime("%Y_%m_%d")
EXPERIMENT_NAME = os.path.join(yyyymmdd, experiment_name)
BASE_CHECKPOINT_PATH = f"{WORKSPACE_PATH}/checkpoints/{EXPERIMENT_NAME}/"
LOG_DIR = f"{WORKSPACE_PATH}/logs/{EXPERIMENT_NAME}/"
VIDEO_DIR = f"{WORKSPACE_PATH}/videos/{EXPERIMENT_NAME}/"


env_config = dict(
        env_id="deepdrive-2d-intersection-w-gs-allow-decel-v0",
        is_intersection_map=True,
        is_one_waypoint_map=False,
        expect_normalized_actions=True,
        expect_normalized_action_deltas=False,
        jerk_penalty_coeff=3.3e-6 * 0,
        gforce_penalty_coeff=0.000006 * 0,
        lane_penalty_coeff=0.001,  # 0.3,
        collision_penalty_coeff=4,
        speed_reward_coeff=0.50,
        steer_change_coeff=0.0001,  # 0.3,
        accel_change_coeff=0.0001,  # 0.05,
        pass_action_boundary_coeff=0.001,  # 0.05,
        gforce_threshold=None,
        end_on_harmful_gs=False,
        incent_win=True,  # reward for reaching the target
        incent_yield_to_oncoming_traffic=True,
        constrain_controls=False,
        physics_steps_per_observation=6,
        contain_prev_actions_in_obs=True,
        # dummy_accel_agent_indices=[1],  # for opponent
        # dummy_random_scenario=None,  # select randomly between 3 scenarios for dummy agent
        discrete_actions=None,
        # end_on_lane_violation=True
    )

@dataclass_json
@dataclass
class HyperParameters():
    actor_hidden_size:      int = 64
    critic_hidden_size:     int = 128
    batch_size:           int   = 512
    discount:             float = 0.99
    gae_lambda:           float = 0.94
    ppo_clip:             float = 0.15
    ppo_epochs:           int   = 4
    max_grad_norm:        float = 1.0
    entropy_factor:       float = 0.001
    actor_learning_rate:  float = 2e-4
    critic_learning_rate: float = 2e-4
    recurrent_seq_len:    int = 20
    recurrent_layers:     int = 1
    if DEBUG:
        rollout_steps:        int = 512
        parallel_rollouts:    int = 25
    else:
        rollout_steps:        int = 3010
        parallel_rollouts:    int = 100
    patience:             int = 200
    # Apply to continous action spaces only
    trainable_std_dev:    bool = True
    init_log_std_dev:     float = 0.0
    beta:                 float = args.beta


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
    actor = input_data["actor"]
    critic = input_data["critic"]

    # Initialise variables.
    obsv = env.reset()
    
    #separate trajs for two agents
    trajectory_data = {"states": [],
                        "actions": [],
                        "action_probabilities": [],
                        "rewards": [],
                        "values": [],
                        "terminals": []
                        }

    terminal = torch.ones(hp.parallel_rollouts)
    
    with torch.no_grad():

        for i in range(hp.rollout_steps):
            done_list = []
            for agent_idx in range(N_AGENTS):

                # Choose next action
                state = torch.tensor(obsv, dtype=torch.float32)

                trajectory_data["states"].append(state)

                value = critic(state.unsqueeze(0).to(GATHER_DEVICE), terminal.to(GATHER_DEVICE))

                trajectory_data["values"].append(value.squeeze(1).cpu())
                
                action_dist = actor(state.unsqueeze(0).to(GATHER_DEVICE), terminal.to(GATHER_DEVICE))
                action = action_dist.sample().reshape(hp.parallel_rollouts, -1)
                if not actor.continuous_action_space:
                    action = action.squeeze(1)

                trajectory_data["actions"].append(action.cpu())
                trajectory_data["action_probabilities"].append(action_dist.log_prob(action).cpu())

                # Step environment
                action_np = action.cpu().numpy()
                obsv, reward, done, info = env.step(action_np)

                terminal = torch.tensor(done).float()
                done_list.append(terminal)
                if agent_idx == 0:
                    trajectory_data["terminals"].append(torch.zeros_like(terminal))
                else:
                    #check if agent 0 was done
                    # if agent 0 was done -> set terminal to True, 
                    # if not -> set terminal to agent 1 terminal state
                    trajectory_data["terminals"].append(torch.clamp(done_list[0] + terminal, 0, 1))

                trajectory_data["rewards"].append(torch.tensor(reward).float())
                # trajectory_data["terminals"].append(terminal)


        #---- end of loop -----

        # Compute final value to allow for incomplete episodes.
        state = torch.tensor(obsv, dtype=torch.float32)
        value = critic(state.unsqueeze(0).to(GATHER_DEVICE), terminal.to(GATHER_DEVICE))
        # Future value for terminal episodes is 0.
        trajectory_data["values"].append(value.squeeze(1).cpu() * (1 - terminal))

    # Combine step lists into tensors.
    # 'values' has one more data step because of the bootstrap value after for loop
    trajectory_tensors = {key: torch.stack(value) for key, value in trajectory_data.items()}
    
    return trajectory_tensors


# def split_trajectories_episodes(trajectory_tensors, hp):
#     """
#     Split trajectories by episode.
#     """

#     states_episodes, actions_episodes, action_probabilities_episodes = [], [], []
#     rewards_episodes, terminal_rewards_episodes, terminals_episodes, values_episodes = [], [], [], []
#     policy_hidden_episodes, policy_cell_episodes, critic_hidden_episodes, critic_cell_episodes = [], [], [], []
#     len_episodes = []
#     trajectory_episodes = {key: [] for key in trajectory_tensors.keys()}

#     for i in range(hp.parallel_rollouts):
#         terminals_tmp = trajectory_tensors["terminals"].clone()
#         terminals_tmp[0, i] = 1
#         terminals_tmp[-1, i] = 1
#         split_points = (terminals_tmp[:, i] == 1).nonzero() + 1

#         split_lens = split_points[1:] - split_points[:-1]
#         split_lens[0] += 1

#         len_episode = [split_len.item() for split_len in split_lens]
#         len_episodes += len_episode
#         for key, value in trajectory_tensors.items():
#             # Value includes additional step.
#             if key == "values":
#                 value_split = list(torch.split(value[:, i], len_episode[:-1] + [len_episode[-1] + 1]))
#                 # Append extra 0 to values to represent no future reward for all episodes except final one
#                 # because the final episode in this rollout has its own future reward and it is not necessarily done. All the
#                 # previous episodes in this rollout are done, so their bootstrap value should be zero.
#                 for j in range(len(value_split) - 1):
#                     value_split[j] = torch.cat((value_split[j], torch.zeros(1)))
#                 trajectory_episodes[key] += value_split
#             else:
#                 trajectory_episodes[key] += torch.split(value[:, i], len_episode)
#     return trajectory_episodes, len_episodes



def split_trajectories_episodes(trajectory_tensors, hp):
    """
    Split trajectories by episode.
    """
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
        # len_episodes += len_episode
        len_episodes += len_episode[:-1]
        for key, value in trajectory_tensors.items():
            # Value includes additional step.
            if key == "values":
                # value_split = list(torch.split(value[:, i], len_episode[:-1] + [len_episode[-1] + 1]))
                value_split = list(torch.split(value[:, i], len_episode[:-1] + [len_episode[-1] + 1]))
                # Append extra 0 to values to represent no future reward for all episodes except final one
                # because the final episode in this rollout has its own future reward and it is not necessarily done. All the
                # previous episodes in this rollout are done, so their bootstrap value should be zero.
                for j in range(len(value_split) - 1):
                    value_split[j] = torch.cat((value_split[j], torch.zeros(1)))
                # trajectory_episodes[key] += value_split
                trajectory_episodes[key] += value_split[:-1]
            else:
                # trajectory_episodes[key] += torch.split(value[:, i], len_episode)
                trajectory_episodes[key] += torch.split(value[:, i], len_episode)[:-1]
    return trajectory_episodes, len_episodes


def calc_discounted_return_nagents(rewards, discount, final_value, alpha=1, beta=0):
    """
    Calculate discounted returns based on rewards and discount factor.
    """

    # if len(rewards) % 2 == 0:
    #     final_value_list = [final_value, torch.tensor(0.)]
    # else:
    #     final_value_list = [torch.tensor(0.), final_value]
    # alpha = 1
    # beta = 0.8

    final_value_list = [torch.tensor([0.]), torch.tensor([0.])]
    
    agent_specific_r = []
    discounted_returns = []
    seq_len = []

    for n in range(N_AGENTS):
        agent_specific_r.append(rewards[n::N_AGENTS])

    # switch rewards
    r0 = agent_specific_r[0]
    agent_specific_r[0] = agent_specific_r[1]
    agent_specific_r[1] = r0

    for n in range(N_AGENTS):
        seq_len.append(len(agent_specific_r[n]))
        discounted_returns.append(torch.zeros(seq_len[n]))

    for n in range(N_AGENTS):
        if seq_len[n] > 0:
            others_curr_reward = [
                beta * agent_specific_r[(n+j) % N_AGENTS][-1] for j in range(N_AGENTS)]
            agents_curr_reward_mean = torch.Tensor(
                others_curr_reward +
                [alpha * agent_specific_r[n][-1]]
            ).mean()
            discounted_returns[n][-1] = agents_curr_reward_mean + \
                discount * final_value_list[n]
    
            # discounted_returns[n][-1] = agent_specific_r[n][-1] + \
            #     discount * final_value_list[n]
    
    for n in range(N_AGENTS):
        for i in range(seq_len[n] - 2, -1, -1):
            # agents_next_discounted_returns_mean = discounted_returns[n][i + 1]
            others_curr_reward = [
                beta * agent_specific_r[(n+j) % N_AGENTS][i] for j in range(N_AGENTS)]
            agents_curr_reward_mean = torch.Tensor(
                others_curr_reward +
                [alpha * agent_specific_r[n][i]]
            ).mean()
            discounted_returns[n][i] = agents_curr_reward_mean + \
                discount * discounted_returns[n][i + 1]

            # agents_curr_reward_mean = torch.Tensor(
            #     [alpha * agent_specific_r[n][i]]
            # ).sum()
            # discounted_returns[n][i] = agent_specific_r[n][i] + \
                # discount * discounted_returns[n][i + 1]

    discounted_returns_final = torch.zeros(seq_len[0] + seq_len[1])
    for n in range(N_AGENTS):
        discounted_returns_final[n::N_AGENTS] = discounted_returns[n]

    return discounted_returns_final


def compute_advantages_nagents(rewards, values, discount, gae_lambda, alpha=1, beta=0):
    """
    Compute General Advantage.
    """
    # if len(rewards) % 2 == 0:
    #     final_value_list = [torch.tensor([values[-1]]), torch.tensor([0.])]
    # else:
    #     final_value_list = [torch.tensor([0.]), torch.tensor([values[-1]])]
    # alpha = 1
    # beta = 0.8

    final_value_list = [torch.tensor([0.]), torch.tensor([0.])]
    
    agent_specific_r = []
    agent_specific_v = []
    agent_specific_adv = []
    seq_len = []

    for n in range(N_AGENTS):
        agent_specific_r.append(rewards[n::N_AGENTS])
        agent_specific_v.append(
            torch.cat([values[n:-1:N_AGENTS], final_value_list[n]]))
        
    # switch rewards
    r0 = agent_specific_r[0]
    agent_specific_r[0] = agent_specific_r[1]
    agent_specific_r[1] = r0

    for n in range(N_AGENTS):
        seq_len.append(len(agent_specific_r[n]))
        agent_specific_adv.append(torch.zeros(seq_len[n] + 1))
        
    deltas = []
    for n in range(N_AGENTS):
        others_v = [beta * agent_specific_v[(n + i) % N_AGENTS]
                    for i in range(1, N_AGENTS)]
        agents_v_mean = torch.mean(torch.stack(
            others_v + [alpha * agent_specific_v[n]]
        ), dim=0)

        others_r = [beta * agent_specific_r[(n + i) % N_AGENTS]
                    for i in range(1, N_AGENTS)]
        agents_r_mean = torch.mean(torch.stack(
            others_r + [alpha * agent_specific_r[n]]
        ), dim=0)

        # deltas.append(agents_r_mean + discount * agent_specific_v[n][1:] - agent_specific_v[n][:-1])
        deltas.append(agents_r_mean + discount * agents_v_mean[1:] - agents_v_mean[:-1])

        # deltas.append(agent_specific_r[n] + discount * agent_specific_v[n][1:] - agent_specific_v[n][:-1])

    multiplier = discount * gae_lambda
    for n in range(N_AGENTS):
        for i in range(seq_len[n] - 1, -1, -1):
            agent_specific_adv[n][i] = deltas[n][i] + \
                multiplier * agent_specific_adv[n][i+1]

    advs = torch.zeros(seq_len[0] + seq_len[1])
    for n in range(N_AGENTS):
        advs[n::N_AGENTS] = agent_specific_adv[n][:-1]
    return advs


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
        
        single_padding = torch.zeros(hp.rollout_steps - len_episodes[i])
        
        for key, value in trajectory_episodes.items():
            if value[i].ndim > 1:
                padding = torch.zeros(hp.rollout_steps - len_episodes[i], value[0].shape[1], dtype=value[i].dtype)
            else:
                padding = torch.zeros(hp.rollout_steps - len_episodes[i], dtype=value[i].dtype)
            padded_trajectories[key].append(torch.cat((value[i], padding)))
        padded_trajectories["advantages"].append(
            torch.cat((compute_advantages_nagents(rewards=trajectory_episodes["rewards"][i],
                                          values=trajectory_episodes["values"][i],
                                          discount=hp.discount,
                                          gae_lambda=hp.gae_lambda,
                                          beta=hp.beta), single_padding)))
        padded_trajectories["discounted_returns"].append(
            torch.cat((calc_discounted_return_nagents(rewards=trajectory_episodes["rewards"][i],
                                              discount=hp.discount,
                                              final_value=trajectory_episodes["values"][i][-1],
                                              beta=hp.beta), single_padding)))
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
        self.layer_hidden = nn.Linear(self.hp.actor_hidden_size, self.hp.actor_hidden_size)
        self.layer_policy_logits = nn.Linear(self.hp.actor_hidden_size, action_dim)
        self.action_dim = action_dim
        self.continuous_action_space = continuous_action_space
        self.log_std_dev = nn.Parameter(init_log_std_dev * torch.ones((action_dim), dtype=torch.float),
                                        requires_grad=trainable_std_dev)
        self.covariance_eye = torch.eye(self.action_dim).unsqueeze(0)
        self.nonlinearity = F.tanh
        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)

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
        return policy_dist


class Critic(nn.Module):
    def __init__(self, state_dim, hp):
        super().__init__()
        self.hp = hp
        self.embed = nn.Linear(state_dim, self.hp.critic_hidden_size)
        self.layer_hidden = nn.Linear(self.hp.critic_hidden_size, self.hp.critic_hidden_size)
        self.layer_value = nn.Linear(self.hp.critic_hidden_size, 1)
        self.nonlinearity = F.tanh
        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)
        
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

def evaluate_env(env_cls, actor):
    """
    Evaluate policy
    """

    # Unpack inputs.
    env = env_cls(is_intersection_map=SELFPLAY)  # for self-play to have 2 learning agents
    env.configure_env(env_config)

    # Initialise variables.
    reward_list = []
    for _ in range(10):
        obs = env.reset()

        terminal = torch.ones(1)
        with torch.no_grad():
            # Take 1 additional step in order to collect the state and value for the final state.
            episode_reward = 0
            while True:
                # Choose next action
                state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action_dist = actor(state.unsqueeze(0).to(GATHER_DEVICE), terminal.to(GATHER_DEVICE))
                action = action_dist.loc[0]
                # action = action_dist.mean()
                if not actor.continuous_action_space:
                    action = action.squeeze(1)

                # Step environment
                action_np = action.cpu().numpy()
                obs, reward, done, info = env.step(action_np)
                episode_reward += reward
                terminal = torch.tensor(done).float()

                if done:
                    reward_list.append(episode_reward)
                    break

    env.close()
    return np.mean(reward_list)


def evaluate_env_parallel(env_cls, env_config, actor):
    """
    Evaluate policy
    """

    n_envs = 5
    env = make_env(env_cls=env_cls, num_envs=n_envs, asynchronous=True, env_config=env_config)

    # Initialise variables.
    score = np.zeros(n_envs)
    reward_list = []

    obs = env.reset()
    terminal = torch.ones(n_envs)
    with torch.no_grad():
        while len(reward_list) < 100:
            # Choose next action
            state = torch.tensor(obs, dtype=torch.float32)
            action_dist = actor(state.unsqueeze(0).to(GATHER_DEVICE), terminal.to(GATHER_DEVICE))
            action = action_dist.loc.reshape(n_envs, -1)
            if not actor.continuous_action_space:
                action = action.squeeze(1)

            # Step environment
            action_np = action.cpu().numpy()
            obs, reward, done, _ = env.step(action_np)

            score += reward
            
            terminal = torch.tensor(done).float()

            for n in range(n_envs):
                if done[n]:
                    reward_list.append(score[n])
                    score[n] = 0

    env.close()
    return np.mean(reward_list)


def train_model(actor,
                critic,
                actor_optimizer,
                critic_optimizer,
                stop_conditions,
                hp,
                env_cls                
                ):
    # Vector environment manages multiple instances of the environment.
    # A key difference between this and the standard gym environment is it automatically resets.
    # Therefore when the done flag is active in the done vector the corresponding state is the first new state.
    #     env = gym.vector.make(ENV, hp.parallel_rollouts, asynchronous=ASYNCHRONOUS_ENVIRONMENT, kwargs=env_config)
    env = make_env(env_cls, hp.parallel_rollouts, asynchronous=ASYNCHRONOUS_ENVIRONMENT, env_config=env_config)

    # eval_env = env_cls(is_intersection_map = env_config['is_intersection_map'])  # for self-play to have 2 learning agents
    # eval_env.configure_env(env_config)

    writer = SummaryWriter(log_dir=LOG_DIR)

    iteration = 0

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

        
        # cuz we separated trajs, and want to calculate the total reward of whole traj, two agents, we need to
        # consider half of the number of done trajs
        train_mean_reward = terminal_episodes_rewards / (complete_episode_count)
    
        # # # Check stop conditions.
        if train_mean_reward > stop_conditions.best_reward:
            stop_conditions.best_reward = train_mean_reward
            stop_conditions.fail_to_improve_count = 0
        else:
            stop_conditions.fail_to_improve_count += 1
        if stop_conditions.fail_to_improve_count > hp.patience:
            print(f"Policy has not yielded higher reward for {hp.patience} iterations...  Stopping now.")
            break

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
                
                # Update actor
                actor_optimizer.zero_grad()
                action_dist = actor(batch.states)

                action_probabilities = action_dist.log_prob(batch.actions.to("cpu")).to(TRAIN_DEVICE)
                probabilities_ratio = torch.exp(action_probabilities - batch.action_probabilities)
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
                values = critic(batch.states)
                critic_loss = F.mse_loss(batch.discounted_returns, values.squeeze(1))
                torch.nn.utils.clip_grad.clip_grad_norm_(critic.parameters(), hp.max_grad_norm)
                critic_loss.backward()
                critic_optimizer.step()

        end_train_time = time.time()

        #evaluate policy
        test_mean_reward = evaluate_env_parallel(env_cls, env_config, actor)

        print(
            f"Iteration: {iteration},  Mean reward: {train_mean_reward}, Mean Entropy: {torch.mean(surrogate_loss_2)}, " +
            f"complete_episode_count: {complete_episode_count}, Gather time: {end_gather_time - start_gather_time:.2f}s, " +
            f"Train time: {end_train_time - start_train_time:.2f}s")

        if SAVE_METRICS_TENSORBOARD:
            writer.add_scalar("complete_episode_count", complete_episode_count, iteration)
            writer.add_scalar("total_reward/test", test_mean_reward, iteration)
            writer.add_scalar("total_reward/train", train_mean_reward, iteration)
            writer.add_scalar("actor_loss", actor_loss, iteration)
            writer.add_scalar("critic_loss", critic_loss, iteration)
            writer.add_scalar("policy_entropy", torch.mean(surrogate_loss_2), iteration)
        if SAVE_PARAMETERS_TENSORBOARD:
            save_parameters(writer, "actor", actor, iteration)
            save_parameters(writer, "value", critic, iteration)
        if iteration > 0 and iteration % CHECKPOINT_FREQUENCY == 0:
            save_checkpoint(actor,
                            critic,
                            actor_optimizer,
                            critic_optimizer,
                            iteration,
                            stop_conditions,
                            env_id=env_config['env_id'],
                            hp=hp,
                            base_checkpoint_path=BASE_CHECKPOINT_PATH)
        iteration += 1
    
    env.close()
    return stop_conditions.best_reward


#============================================================================================
# --------------------------------- training ------------------------------------------------
# ===========================================================================================

hp = HyperParameters()

cfg = dict(env_config=env_config, algo_config=hp.to_dict())

if not DEBUG:
    wandb.init(
        project="dd0", 
        sync_tensorboard=True, 
        config=cfg,
        name=experiment_name, 
        # monitor_gym=True, 
        save_code=True,
        group=algo_name
        )
        
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
with open(os.path.join(LOG_DIR, 'params.json'), 'w') as fp:
    json.dump(cfg, fp)

batch_count = hp.parallel_rollouts * hp.rollout_steps / hp.recurrent_seq_len / hp.batch_size
print(f"batch_count: {batch_count}")
assert batch_count >= 1., "Less than 1 batch per trajectory.  Are you sure that's what you want?"

env_cls = Deepdrive2DEnv

obsv_dim, action_dim, continuous_action_space = get_env_space(env_cls, env_config)

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

score = train_model(actor,
                    critic,
                    actor_optimizer,
                    critic_optimizer,
                    stop_conditions,
                    hp,
                    env_cls
                    )


del actor
del critic
del actor_optimizer
del critic_optimizer