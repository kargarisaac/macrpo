import sys
sys.path.append('/scratch/work/kargare1/codes/multi-agent/')

from experiments.ppo_mine.particle_env.utils import make_env2, \
    save_parameters,\
    save_checkpoint, get_env_space, StopConditions, compute_advantages, calc_discounted_return
import time
import math
from make_env import make_env as particke_env
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

import wandb

# ============================================================================================
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

# ============================================================================================
# --------------------------------- training settings ---------------------------------------
# ===========================================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO agent')
    parser.add_argument('--seed', type=int, default=4,
                        help='seed of the experiment')
    parser.add_argument('--n_envs', type=int, default=20,
                        help='number of parallel environments')
    parser.add_argument('--ep_steps', type=int, default=2500,
                        help='steps in one training epoch')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='beta of A and R')
    args = parser.parse_args()


DEBUG = False
TRAIN = True
RESUME = False
N_AGENTS = 3
ENV_NAME = 'simple_spread_mine'
N_EVAL_EPISODE = 20
NORMALIZE_ADV = True

RESUME_CHECKPOINT_PATH = '/scratch/work/kargare1/codes/multi-agent/experiments/ppo_mine/particle_env/logs/2020_09_17/c2c_coslr_0.005_1.0_21:48:21/best/'

algo_name = f"c2c_coslr_{args.lr}_{args.beta}_"

# algo_name = "ep_steps_" + str(args.ep_steps) + "_"

RANDOM_SEED = args.seed

# Set random seed for consistant runs.
torch.random.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# Set maximum threads for torch to avoid inefficient use of multiple cpu cores.
torch.set_num_threads(10)
TRAIN_DEVICE = "cuda"  # "cuda" if torch.cuda.is_available() else "cpu"
# "cuda" if torch.cuda.is_available() and not FORCE_CPU_GATHER else "cpu"
GATHER_DEVICE = "cpu"
EVAL_DEVICE = "cpu"
# ============================================================================================
# --------------------------------- set directories and configs -----------------------------
# ===========================================================================================

# WORKSPACE_PATH = "/home/isaac/codes/autonomous_driving/multi-agent/experiments/ppo_mine/particle_env/"
WORKSPACE_PATH = "/scratch/work/kargare1/codes/multi-agent/experiments/ppo_mine/particle_env/"
experiment_name = algo_name + \
    str(datetime.datetime.today()).split(' ')[1].split('.')[0]
yyyymmdd = datetime.datetime.today().strftime("%Y_%m_%d")
EXPERIMENT_NAME = os.path.join(yyyymmdd, experiment_name)
BASE_CHECKPOINT_PATH = f"{WORKSPACE_PATH}/logs/{EXPERIMENT_NAME}/"
LOG_DIR = f"{WORKSPACE_PATH}/logs/{EXPERIMENT_NAME}/"
VIDEO_DIR = f"{WORKSPACE_PATH}/videos/{EXPERIMENT_NAME}/"


@dataclass_json
@dataclass
class HyperParameters():
    actor_hidden_size:      int = 128
    critic_hidden_size:     int = 128
    # batch_size:           int   = 128 #128>32, 128>=512, 128>=1024
    discount:             float = 0.99  # 0.95
    gae_lambda:           float = 0.95  # 0.95>0.94, 0.95>0.97, 0.95?0.96
    ppo_clip:             float = 0.2  # 0.2 > 0.15 -
    ppo_epochs:           int = 10  # 10>4, 10>20
    max_grad_norm:        float = 1.0  # 1>0.5, 1>2, 1>10
    entropy_factor:       float = 0.01  # 0.0001>0.001,
    actor_learning_rate:  float = args.lr  # 6e-4
    critic_learning_rate: float = args.lr  # 6e-4
    recurrent_seq_len:    int = N_AGENTS
    recurrent_layers:     int = 1
    if DEBUG:
        rollout_steps:        int = 200
        parallel_rollouts:    int = 4
    else:
        rollout_steps:        int = args.ep_steps
        parallel_rollouts:    int = args.n_envs
    patience:             int = 200
    # Apply to continous action spaces only
    trainable_std_dev:    bool = True
    init_log_std_dev:     float = 0.01
    n_minibatch:           int = 32
    batch_size:           int = int(
        rollout_steps * parallel_rollouts / n_minibatch)
    alpha:                float = 1.0  # 0
    beta:                 float = args.beta


# ============================================================================================
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
        self.trajectories = {key: value for key, value in trajectories.items()}
        self.device = device
        self.batch_len = batch_len
        truncated_seq_len = torch.clamp(
            trajectories["seq_len"] - batch_len + 1, 0, hp.rollout_steps)
        self.cumsum_seq_len = np.cumsum(np.concatenate(
            (np.array([0]), truncated_seq_len.numpy())))
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
            start_idx = np.random.choice(
                self.valid_idx, size=actual_batch_size, replace=False)
            # remove start_idx from valid_idx
            self.valid_idx = np.setdiff1d(self.valid_idx, start_idx)
            eps_idx = np.digitize(
                start_idx, bins=self.cumsum_seq_len, right=False) - 1
            seq_idx = start_idx - self.cumsum_seq_len[eps_idx]
            series_idx = np.linspace(
                seq_idx, seq_idx + self.batch_len - 1, num=self.batch_len, dtype=np.int64)
            self.batch_count += 1

            return TrajectorBatch(**{key: value[eps_idx, series_idx].to(self.device) for key, value
                                     in self.trajectories.items() if key in TrajectorBatch.__dataclass_fields__.keys()},
                                  batch_size=actual_batch_size)


def gather_trajectories(input_data, hp):
    """
    Gather policy trajectories from gym environment.
    """

    # Unpack inputs.
    env = input_data["env"]  # vectorized envs
    actors = input_data["actors"]
    critic = input_data["critic"]

    # Initialise variables.
    obsv = env.reset()  # obsv shape:[n_envs, n_agents*obs shape of each agent]

    # separate trajs for two agents
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

        for a in range(N_AGENTS):
            actors[a].get_init_state(hp.parallel_rollouts, GATHER_DEVICE)
        critic.get_init_state(hp.parallel_rollouts, GATHER_DEVICE)

        # have a list to store actor and critic rnn_state and swith them in each time step
        # actor_hidden_cell_alternating_list = [actor.hidden_cell] * 2 #the initial values are zero
        # critic_hidden_cell_alternating_list = [critic.hidden_cell] * 2

        for i in range(hp.rollout_steps):
            action_list = []
            for agent_idx in range(N_AGENTS):

                trajectory_data["actor_hidden_states"].append(
                    actors[agent_idx].hidden_cell[0].squeeze(0).cpu())
                trajectory_data["actor_cell_states"].append(
                    actors[agent_idx].hidden_cell[1].squeeze(0).cpu())
                trajectory_data["critic_hidden_states"].append(
                    critic.hidden_cell[0].squeeze(0).cpu())
                trajectory_data["critic_cell_states"].append(
                    critic.hidden_cell[1].squeeze(0).cpu())

                # get state for agent_index
                state = torch.tensor(
                    obsv[:, agent_idx*obsv.shape[1]//N_AGENTS: (agent_idx+1)*obsv.shape[1]//N_AGENTS])

                trajectory_data["states"].append(state)

                value = critic(state.unsqueeze(0).to(
                    GATHER_DEVICE), terminal[:, 0].to(GATHER_DEVICE))

                trajectory_data["values"].append(value.squeeze(1).cpu())

                # switch rnn_state
                # actor.hidden_cell = actor_hidden_cell_alternating_list[agent_idx]

                action_dist = actors[agent_idx](state.unsqueeze(0).to(
                    GATHER_DEVICE), terminal[:, 0].to(GATHER_DEVICE))
                action = action_dist.sample().reshape(hp.parallel_rollouts, -1)
                if not actors[agent_idx].continuous_action_space:
                    action = action.squeeze(1)

                trajectory_data["actions"].append(action.cpu())
                trajectory_data["action_probabilities"].append(
                    action_dist.log_prob(action).cpu())

                # action_list.append(action.cpu().numpy().squeeze())
                action_list.append(action)

                # actor_hidden_cell_alternating_list[agent_idx] = actor.hidden_cell

            # Step environment
            action_array = torch.stack(
                action_list, axis=1).squeeze(1).cpu().numpy()
            obsv, reward, done, info = env.step(action_array)

            for a in range(N_AGENTS):
                if ENV_NAME == 'simple_spread':
                    episode_reward += reward[:, a]
                else:
                    episode_reward += reward[:, a]

            if done[0]:
                episode_rewards.append(torch.mean(episode_reward))
                # print(f'train episode rewrd mean: {np.mean(episode_reward)}, 100 last rewards mean: {np.mean(episode_rewards[-100:])}')
                # print(f'train episode rewrd mean: {torch.mean(episode_reward)}')
                episode_reward = torch.zeros(hp.parallel_rollouts)

                # for a in range(N_AGENTS):
                #     actors[a].get_init_state(hp.parallel_rollouts, GATHER_DEVICE)
                # critic.get_init_state(hp.parallel_rollouts, GATHER_DEVICE)

            terminal = torch.tensor(done).unsqueeze(1).float()

            # TODO: when env is done, only use one?
            for _ in range(N_AGENTS - 1):
                trajectory_data["terminals"].append(
                    torch.zeros_like(terminal[:, 0]))

            t = terminal[:, 0]
            if N_AGENTS > 1:
                for a in range(1, N_AGENTS):
                    t += terminal[:, 0]
            t[t > 0] = 1
            trajectory_data["terminals"].append(t)

            for a in range(N_AGENTS):
                trajectory_data["rewards"].append(torch.tensor(reward[:, a]))

        # ---- end of loop -----

        # Compute final value to allow for incomplete episodes.
        # TODO: this bootstrap value should be for one agent or all? traj is combined and each agent is like one step.
        # maybe mean of 3 values
        # if we set rollout_steps to sth like 2000 which is devidable by 25 (max_time_steps), I think this bv will be unusable

        # value_mean = 0
        # for agent_idx in range(N_AGENTS):
        #     # get state for agent_index
        #     state = obsv[:, agent_idx*obsv.shape[1]//N_AGENTS: (agent_idx+1)*obsv.shape[1]//N_AGENTS].copy()
        #     value_mean += critic(state.unsqueeze(0).to(GATHER_DEVICE), terminal[:, 0].to(GATHER_DEVICE)).squeeze(1).cpu()
        # trajectory_data["values"].append(value_mean/N_AGENTS)

        for agent_idx in range(1):
            # get state for agent_index
            state = torch.tensor(
                obsv[:, agent_idx*obsv.shape[1]//N_AGENTS: (agent_idx+1)*obsv.shape[1]//N_AGENTS])
            value = critic(state.unsqueeze(0).to(GATHER_DEVICE),
                           terminal[:, 0].to(GATHER_DEVICE))
            trajectory_data["values"].append(value.squeeze(1).cpu())

    trajectory_tensors = {key: torch.stack(
        value) for key, value in trajectory_data.items()}
    
    return trajectory_tensors


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
        len_episodes += len_episode
        for key, value in trajectory_tensors.items():
            # Value includes additional step.
            if key == "values":
                value_split = list(torch.split(
                    value[:, i], len_episode[:-1] + [len_episode[-1] + 1]))
                # Append extra 0 to values to represent no future reward for all episodes except final one
                # because the final episode in this rollout has its own future reward and it is not necessarily done. All the
                # previous episodes in this rollout are done, so their bootstrap value should be zero.
                for j in range(len(value_split) - 1):
                    value_split[j] = torch.cat(
                        (value_split[j], torch.zeros(1)))
                trajectory_episodes[key] += value_split
            else:
                trajectory_episodes[key] += torch.split(
                    value[:, i], len_episode)
    return trajectory_episodes, len_episodes


# def calc_discounted_return_nagents(rewards, discount, final_value_mean, alpha=1, beta=1):
#     """
#     Calculate discounted returns based on rewards and discount factor.
#     """

#     seq_len = len(rewards)

#     agent_specific_r = []
#     discounted_returns = []
#     for n in range(N_AGENTS):
#         agent_specific_r.append(rewards[n::N_AGENTS])
#         discounted_returns.append(torch.zeros(int(seq_len/N_AGENTS)))

#     for n in range(N_AGENTS):
#         # final_value will be zero always because episode always is done
#         discounted_returns[n][-1] = agent_specific_r[n][-1] + \
#             discount * final_value_mean

#     for i in range(int(seq_len/N_AGENTS) - 2, -1, -1):
#         for n in range(N_AGENTS):

#             # others_next_discounted_returns = [
#             #     beta * discounted_returns[(n+j) % N_AGENTS][i + 1] for j in range(1, N_AGENTS)]
#             # agents_next_discounted_returns_mean = torch.tensor(
#             #     others_next_discounted_returns +
#             #     [alpha * discounted_returns[n][i + 1]]
#             # ).sum()
#             agents_next_discounted_returns_mean = discounted_returns[n][i + 1]
#             others_curr_reward = [
#                 beta * agent_specific_r[(n+j) % N_AGENTS][i] for j in range(N_AGENTS)]
#             agents_curr_reward_mean = torch.Tensor(
#                 others_curr_reward +
#                 [alpha * agent_specific_r[n][i]]
#             ).mean()
#             discounted_returns[n][i] = agents_curr_reward_mean + \
#                 discount * agents_next_discounted_returns_mean

#     discounted_returns_final = torch.zeros(seq_len)
#     for n in range(N_AGENTS):
#         discounted_returns_final[n::N_AGENTS] = discounted_returns[n]

#     return discounted_returns_final


# def compute_advantages_nagents(rewards, values, discount, gae_lambda, alpha=1, beta=1):
#     """
#     Compute General Advantage.
#     """
#     seq_len = len(rewards)

#     final_value = torch.tensor([values[-1]])

#     agent_specific_r = []
#     agent_specific_v = []
#     agent_specific_adv = []
#     for n in range(N_AGENTS):
#         agent_specific_r.append(rewards[n::N_AGENTS])
#         # the last value for particle env is zero because we sample for 25 steps each time which env will be done then.
#         agent_specific_v.append(
#             torch.cat([values[:-1][n::N_AGENTS], final_value]))
#         agent_specific_adv.append(torch.zeros(int(seq_len/N_AGENTS) + 1))

#     deltas = []
#     for n in range(N_AGENTS):
#         # 1st method
#         # others_v = [beta * agent_specific_v[(n + i) % N_AGENTS]
#         #             for i in range(1, N_AGENTS)]
#         # agents_v_mean = torch.mean(torch.stack(
#         #     others_v + [alpha * agent_specific_v[n]]
#         # ), dim=0)

#         others_r = [beta * agent_specific_r[(n + i) % N_AGENTS]
#                     for i in range(1, N_AGENTS)]
#         agents_r_mean = torch.mean(torch.stack(
#             others_r + [alpha * agent_specific_r[n]]
#         ), dim=0)

#         deltas.append(agents_r_mean + discount *
#                       agent_specific_v[n][1:] - agent_specific_v[n][:-1])

#     multiplier = discount * gae_lambda
#     for i in range(int(seq_len/N_AGENTS) - 1, -1, -1):
#         for n in range(N_AGENTS):
#             agent_specific_adv[n][i] = deltas[n][i] + \
#                 multiplier * agent_specific_adv[n][i+1]

#     advs = torch.zeros(seq_len)
#     for n in range(N_AGENTS):
#         advs[n::N_AGENTS] = agent_specific_adv[n][:-1]
#     return advs



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

    final_value_list = [torch.tensor([0.])] * N_AGENTS
    
    agent_specific_r = []
    discounted_returns = []
    seq_len = []

    for n in range(N_AGENTS):
        agent_specific_r.append(rewards[n::N_AGENTS])

    # switch rewards
    # r0 = agent_specific_r[0]
    # agent_specific_r[0] = agent_specific_r[1]
    # agent_specific_r[1] = r0

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

    discounted_returns_final = torch.zeros(sum(seq_len))
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

    final_value_list = [torch.tensor([0.])] * N_AGENTS
    
    agent_specific_r = []
    agent_specific_v = []
    agent_specific_adv = []
    seq_len = []

    for n in range(N_AGENTS):
        agent_specific_r.append(rewards[n::N_AGENTS])
        agent_specific_v.append(
            torch.cat([values[n:-1:N_AGENTS], final_value_list[n]]))
        
    # switch rewards
    # r0 = agent_specific_r[0]
    # agent_specific_r[0] = agent_specific_r[1]
    # agent_specific_r[1] = r0

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

    advs = torch.zeros(sum(seq_len))
    for n in range(N_AGENTS):
        advs[n::N_AGENTS] = agent_specific_adv[n][:-1]
    return advs



def pad_and_compute_returns(trajectory_episodes, len_episodes, hp):
    """
    Pad the trajectories up to hp.rollout_steps so they can be combined in a
    single tensor.
    Add advantages and discounted_returns to trajectories.
    """

    # number of all episodes for all parallel envs
    episode_count = len(len_episodes)
    padded_trajectories = {key: [] for key in trajectory_episodes.keys()}
    padded_trajectories["advantages"] = []
    padded_trajectories["discounted_returns"] = []

    for i in range(episode_count):

        # in separated mode, the roll-out steps are half for each agent
        single_padding = torch.zeros(hp.rollout_steps - len_episodes[i])
        for key, value in trajectory_episodes.items():
            if value[i].ndim > 1:
                padding = torch.zeros(
                    hp.rollout_steps - len_episodes[i], value[0].shape[1], dtype=value[i].dtype)
            else:
                padding = torch.zeros(
                    hp.rollout_steps - len_episodes[i], dtype=value[i].dtype)
            padded_trajectories[key].append(torch.cat((value[i], padding)))

        # padded_trajectories["advantages"].append(
        #     torch.cat((compute_advantages(rewards=trajectory_episodes["rewards"][i],
        #                                   values=trajectory_episodes["values"][i],
        #                                   discount=hp.discount,
        #                                   gae_lambda=hp.gae_lambda), single_padding)))
        # padded_trajectories["discounted_returns"].append(
        #     torch.cat((calc_discounted_return(rewards=trajectory_episodes["rewards"][i],
        #                                       discount=hp.discount,
        #                                       final_value=trajectory_episodes["values"][i][-1]), single_padding)))
    

        padded_trajectories["advantages"].append(
            torch.cat((compute_advantages_nagents(rewards=trajectory_episodes["rewards"][i],
                                                  values=trajectory_episodes["values"][i],
                                                  discount=hp.discount,
                                                  gae_lambda=hp.gae_lambda,
                                                  alpha=hp.alpha,
                                                  beta=hp.beta), single_padding)))
        padded_trajectories["discounted_returns"].append(
            torch.cat((calc_discounted_return_nagents(rewards=trajectory_episodes["rewards"][i],
                                                      discount=hp.discount,
                                                      final_value=trajectory_episodes["values"][i][-1],
                                                      alpha=hp.alpha,
                                                      beta=hp.beta), single_padding)))
    
    return_val = {k: torch.stack(v) for k, v in padded_trajectories.items()}
    return_val["seq_len"] = torch.tensor(len_episodes)

    return return_val  # [number of episodes, hp.rollout_steps]


# ============================================================================================
# --------------------------------- neural nets ---------------------------------------------
# ===========================================================================================

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, continuous_action_space, trainable_std_dev, hp, init_log_std_dev=None):
        super().__init__()
        self.hp = hp
        self.embed = nn.Linear(state_dim, self.hp.actor_hidden_size)
        self.lstm = nn.LSTM(self.hp.actor_hidden_size,
                            self.hp.actor_hidden_size, num_layers=self.hp.recurrent_layers)
        self.layer_hidden = nn.Linear(
            self.hp.actor_hidden_size, self.hp.actor_hidden_size)
        self.layer_policy_logits = nn.Linear(
            self.hp.actor_hidden_size, action_dim)
        self.action_dim = action_dim
        self.continuous_action_space = continuous_action_space
        self.log_std_dev = nn.Parameter(init_log_std_dev * torch.ones(
            (action_dim), dtype=torch.float), requires_grad=trainable_std_dev)
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
            self.hidden_cell = [
                value * (1. - terminal).reshape(1, batch_size, 1) for value in self.hidden_cell]

        embedded = self.nonlinearity(self.embed(state))
        lstm_out, self.hidden_cell = self.lstm(embedded, self.hidden_cell)
        # hidden_out = F.relu(self.layer_hidden(lstm_out)) #many to one- just use the last step hidden state
        policy_logits_out = self.layer_policy_logits(lstm_out).squeeze()
        # print(policy_logits_out.shape)
        if self.continuous_action_space:
            cov_matrix = self.covariance_eye.to(device).expand(
                batch_size, self.action_dim, self.action_dim) * torch.exp(self.log_std_dev.to(device))
            # We define the distribution on the CPU since otherwise operations fail with CUDA illegal memory access error.
            policy_dist = distributions.multivariate_normal.MultivariateNormal(
                policy_logits_out.to("cpu"), cov_matrix.to("cpu"))
        else:
            try:
                policy_dist = distributions.Categorical(
                    F.softmax(policy_logits_out, dim=1).to("cpu"))
            except:
                policy_dist = distributions.Categorical(
                    F.softmax(policy_logits_out, dim=0).to("cpu"))
        return policy_dist


class Critic(nn.Module):
    def __init__(self, state_dim, hp):
        super().__init__()
        self.hp = hp
        self.embed = nn.Linear(state_dim, self.hp.critic_hidden_size)
        self.layer_lstm = nn.LSTM(
            self.hp.critic_hidden_size, self.hp.critic_hidden_size, num_layers=self.hp.recurrent_layers)
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
            self.hidden_cell = [
                value * (1. - terminal).reshape(1, batch_size, 1) for value in self.hidden_cell]

        embedded = self.nonlinearity(self.embed(state))
        lstm_out, self.hidden_cell = self.layer_lstm(
            embedded, self.hidden_cell)
        # hidden_out = F.relu(self.layer_hidden(lstm_out)) #many to one- just use the last step hidden state
        value_out = self.layer_value(lstm_out)
        return value_out.squeeze(0)


# ============================================================================================
# --------------------------------- helper functions ----------------------------------------
# ===========================================================================================

def evaluate_env(actors, render=False):
    """
    Evaluate policy
    """

    from utils import ActionWrapper, ObsWrapper, TimeLimit
    from gym.wrappers import Monitor

    # Unpack inputs.
    env = particke_env(ENV_NAME, True)
    env = ObsWrapper(env)
    env = ActionWrapper(env)
    env = TimeLimit(env)
    env = Monitor(env, VIDEO_DIR, video_callable=lambda episode_id: True, force=True)

    # Initialise variables.
    reward_list = []
    col = []
    min_dist = []
    occupied_landmark = []

    for a in range(N_AGENTS):
        actors[a] = actors[a].to(EVAL_DEVICE)

    obs = env.reset()

    # terminal = torch.ones(3)
    with torch.no_grad():

        for a in range(N_AGENTS):
            actors[a].get_init_state(1, EVAL_DEVICE)
        # actor_hidden_cell_alternating_list = [actor.hidden_cell] * 2 #the initial values are zero

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
                state = torch.tensor(
                    obs[agent_idx], dtype=torch.float32).unsqueeze(0)
                # print('state shape: ', state.shape)

                # switch rnn_state
                # actor.hidden_cell = actor_hidden_cell_alternating_list[agent_idx]

                action_dist = actors[agent_idx](
                    state.unsqueeze(0).to(EVAL_DEVICE))
                # print('action: ', action_dist.probs.shape)
                # action = action_dist.probs
                action = torch.argmax(action_dist.probs)
                # action = action_dist.sample()
                # print(action)
                if not actors[agent_idx].continuous_action_space:
                    action = action.squeeze(0)

                # Step environment
                action_list.append(action.cpu().numpy().squeeze())

                # actor_hidden_cell_alternating_list[agent_idx] = actor.hidden_cell

            # action_np = np.stack(action_list, axis=0)
            # print(action_list[0].shape)
            obs, reward, done, info = env.step(action_list)

            for i in range(N_AGENTS):
                if ENV_NAME == 'simple_spread':
                    episode_reward += reward[i] / N_AGENTS
                else:
                    episode_reward += reward[i]

                episode_col += info['n'][i][1]
                episode_min_dist += info['n'][i][2]
                episode_occupied_landmark += info['n'][i][3]

            # terminal = torch.tensor(done).float()

            episode_steps += 1

            if render:
                env.render()
                time.sleep(0.1)

            # if episode_steps == 25:
            if done:
                reward_list.append(episode_reward)
                col.append(episode_col)
                min_dist.append(episode_min_dist)
                occupied_landmark.append(episode_occupied_landmark)
                obs = env.reset()

                if render:
                    print('episode reward: ', episode_reward)
                episode_reward = 0
                episode_steps = 0

                for a in range(N_AGENTS):
                    actors[a].get_init_state(1, EVAL_DEVICE)
                # actor_hidden_cell_alternating_list = [actor.hidden_cell] * 2 #the initial values are zero

                if len(reward_list) > N_EVAL_EPISODE:
                    break

    env.close()
    return np.mean(reward_list), np.mean(col), np.mean(min_dist), np.mean(occupied_landmark)


def evaluate_env_parallel(actors, render=False):
    """
    Evaluate policy
    """
    
    eval_env = make_env2(10, asynchronous=False, benchmark=True, env_name=ENV_NAME)

    # Initialise variables.
    reward_list = []
    col = []
    min_dist = []
    occupied_landmark = []

    episode_reward = np.zeros(eval_env.num_envs)
    episode_col = np.zeros(eval_env.num_envs)
    episode_min_dist = np.zeros(eval_env.num_envs)
    episode_occupied_landmark = np.zeros(eval_env.num_envs)
    episode_steps = 0
    total_step = 0

    for a in range(N_AGENTS):
        actors[a].get_init_state(eval_env.num_envs, GATHER_DEVICE)
        actors[a] = actors[a].to(EVAL_DEVICE)

    # actor_hidden_cell_alternating_list = [actor.hidden_cell] * 2 #the initial values are zero

    obsv = eval_env.reset()
    terminal = torch.ones(eval_env.num_envs, 1)

    with torch.no_grad():

        # have a list to store actor and critic rnn_state and swith them in each time step
        # actor_hidden_cell_alternating_list = [actor.hidden_cell] * 2 #the initial values are zero
        # critic_hidden_cell_alternating_list = [critic.hidden_cell] * 2

        while True:

            action_list = []

            for agent_idx in range(N_AGENTS):

                # get state for agent_index
                state = torch.tensor(
                    obsv[:, agent_idx*obsv.shape[1]//N_AGENTS: (agent_idx+1)*obsv.shape[1]//N_AGENTS])

                # switch rnn_state
                # actor.hidden_cell = actor_hidden_cell_alternating_list[agent_idx]

                action_dist = actors[agent_idx](state.unsqueeze(0).to(
                    GATHER_DEVICE), terminal[:, 0].to(GATHER_DEVICE))
                # action = action_dist.sample().reshape(hp.parallel_rollouts, -1)
                action = torch.argmax(action_dist.probs, dim=1)
                # if not actors[agent_idx].continuous_action_space:
                #     action = action.squeeze(1)

                # action_list.append(action.cpu().numpy().squeeze())
                action_list.append(action)

                # actor_hidden_cell_alternating_list[agent_idx] = actor.hidden_cell

            # Step environment
            action_array = torch.stack(
                action_list, axis=1).squeeze(1).cpu().numpy()
            obsv, reward, done, info = eval_env.step(action_array)

            for a in range(N_AGENTS):    
                episode_reward += reward[:, a]

            for n in range(reward.shape[0]):
                for a in range(N_AGENTS):    
                    episode_col[n] += info[n]['n'][a][1]
                    episode_min_dist[n] += info[n]['n'][a][2]
                    episode_occupied_landmark[n] += info[n]['n'][a][3]
                

            terminal = torch.tensor(done).unsqueeze(1).float()

            episode_steps += 1
            total_step += 1

            # if episode_steps == 25:
            if done[0]:
                reward_list += list(episode_reward)
                col += list(episode_col)
                min_dist += list(episode_min_dist)
                occupied_landmark += list(episode_occupied_landmark)
                
                # col.append(episode_col)
                # min_dist.append(episode_min_dist)
                # occupied_landmark.append(episode_occupied_landmark)
                obsv = eval_env.reset()

                # episode_reward = 0
                # episode_col = 0
                # episode_min_dist = 0
                # episode_occupied_landmark = 0

                episode_reward = torch.zeros(eval_env.num_envs)
                episode_col = torch.zeros(eval_env.num_envs)
                episode_min_dist = torch.zeros(eval_env.num_envs)
                episode_occupied_landmark = torch.zeros(eval_env.num_envs)

                episode_steps = 0

                for a in range(N_AGENTS):
                    actors[a].get_init_state(eval_env.num_envs, GATHER_DEVICE)
                # actor_hidden_cell_alternating_list = [actor.hidden_cell] * 2 #the initial values are zero

                if len(reward_list) >= N_EVAL_EPISODE:
                    break

    eval_env.close()
    return torch.tensor(np.mean(reward_list)), torch.tensor(np.mean(col)), torch.tensor(np.mean(min_dist)), torch.tensor(np.mean(occupied_landmark))



def train_model(actors,
                critic,
                actor_optimizers,
                critic_optimizer,
                actor_lr_schedulers,
                critic_lr_scheduler,
                stop_conditions,
                hp
                ):
    # Vector environment manages multiple instances of the environment.
    # A key difference between this and the standard gym environment is it automatically resets.
    # Therefore when the done flag is active in the done vector the corresponding state is the first new state.

    env = make_env2(hp.parallel_rollouts, asynchronous=False, env_name=ENV_NAME)

    if not DEBUG:
        writer = SummaryWriter(log_dir=LOG_DIR)

    iteration = 0

    if RESUME:
        for i in range(N_AGENTS):
            actors[i].load_state_dict(torch.load(
                RESUME_CHECKPOINT_PATH + 'actor_' + str(i+1) + '.pt'))
            actor_optimizers[i].load_state_dict(torch.load(
                RESUME_CHECKPOINT_PATH + 'actor_optimizer_' + str(i+1) + '.pt'))

        critic.load_state_dict(torch.load(
            RESUME_CHECKPOINT_PATH + 'critic.pt'))
        critic_optimizer.load_state_dict(torch.load(
            RESUME_CHECKPOINT_PATH + 'critic_optimizer.pt'))

    while iteration < stop_conditions.max_iterations:

        for i in range(N_AGENTS):
            actors[i] = actors[i].to(GATHER_DEVICE)

        critic = critic.to(GATHER_DEVICE)
        start_gather_time = time.time()

        # Gather trajectories.
        input_data = {"env": env, "actors": actors, "critic": critic, "discount": hp.discount,
                      "gae_lambda": hp.gae_lambda}

        trajectory_tensors = gather_trajectories(input_data, hp)
        trajectory_episodes, len_episodes = split_trajectories_episodes(
            trajectory_tensors, hp)
        trajectories = pad_and_compute_returns(
            trajectory_episodes, len_episodes, hp)
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
        train_mean_reward = terminal_episodes_rewards / \
            (complete_episode_count)
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

        # ======= end of data collection and starting of training ===========
        start_train_time = time.time()

        for a in range(N_AGENTS):
            actors[a] = actors[a].to(TRAIN_DEVICE)
        critic = critic.to(TRAIN_DEVICE)

        # Train actor and critic.
        for epoch_idx in range(hp.ppo_epochs):
            for batch in trajectory_dataset:

                for a in range(N_AGENTS):
                    actor_optimizers[a].zero_grad()

                probs = torch.zeros([hp.recurrent_seq_len, hp.batch_size, 5])
                action_probabilities = torch.zeros_like(
                    batch.action_probabilities)
                for i in range(N_AGENTS):
                    # select actor hidden_cell data for each agent
                    actors[i].hidden_cell = (
                        batch.actor_hidden_states[i:i+1], batch.actor_cell_states[i:i+1])
                    # Update actor - select data of each agent
                    # action_dist_temp = actors[i](batch.states[i::N_AGENTS])
                    action_dist_temp = actors[i](batch.states[i::N_AGENTS])

                    probs[i::N_AGENTS] = action_dist_temp.probs

                    action_probabilities[i::N_AGENTS] = action_dist_temp.log_prob(
                        batch.actions[i::N_AGENTS].to("cpu")).to(TRAIN_DEVICE)

                action_dist = distributions.Categorical(probs=probs)

                del action_dist_temp

                probabilities_ratio = torch.exp(
                    action_probabilities - batch.action_probabilities)
                if NORMALIZE_ADV:
                    advantages = (
                        batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
                else:
                    advantages = batch.advantages
                surrogate_loss_0 = probabilities_ratio * batch.advantages
                surrogate_loss_1 = torch.clamp(probabilities_ratio, 1. - hp.ppo_clip,
                                               1. + hp.ppo_clip) * batch.advantages

                surrogate_loss_2 = action_dist.entropy().to(TRAIN_DEVICE)
                actor_loss = -torch.mean(torch.min(surrogate_loss_0, surrogate_loss_1)) - torch.mean(
                    hp.entropy_factor * surrogate_loss_2)
                actor_loss.backward()
                for a in range(N_AGENTS):
                    torch.nn.utils.clip_grad.clip_grad_norm_(
                        actors[a].parameters(), hp.max_grad_norm)
                    actor_optimizers[a].step()

                # Update critic
                critic_optimizer.zero_grad()

                critic.hidden_cell = (
                    batch.critic_hidden_states[:1], batch.critic_cell_states[:1])

                # values = critic(batch.states)
                values = critic(batch.states)

                critic_loss = F.mse_loss(
                    batch.discounted_returns, values.squeeze(2))
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    critic.parameters(), hp.max_grad_norm)
                critic_loss.backward()
                critic_optimizer.step()

        for a in range(N_AGENTS):
            actor_lr_schedulers[a].step()
        critic_lr_scheduler.step()

        end_train_time = time.time()

        start_eval_time = time.time()
        # test_mean_reward, test_mean_col, test_mean_min_dist, test_mean_occupied_lansmark = evaluate_env(actors)
        test_mean_reward, test_mean_col, test_mean_min_dist, test_mean_occupied_landmark = evaluate_env_parallel(actors)
        end_eval_time = time.time()

        print(
            f"Iteration: {iteration},  Mean reward: {test_mean_reward:.2f}, Mean Entropy: {torch.mean(surrogate_loss_2):.2f}, " +
            f"Mean collision: {test_mean_col:.2f}, Mean min dist: {test_mean_min_dist:.2f}, Mean occupied landmark: {test_mean_occupied_landmark:.2f}, " +
            f"complete_episode_count: {complete_episode_count:.2f}, Gather time: {end_gather_time - start_gather_time:.2f}s, " +
            f"Train time: {end_train_time - start_train_time:.2f}s, Eval Time: {end_eval_time - start_eval_time:.2f}s")

        print('=====================================================')

        if SAVE_METRICS_TENSORBOARD and not DEBUG:
            writer.add_scalar("complete_episode_count",
                              complete_episode_count, iteration)
            writer.add_scalar("mean_reward_train",
                              train_mean_reward, iteration)
            writer.add_scalar("mean_reward", test_mean_reward, iteration)
            writer.add_scalar("mean_collisions", test_mean_col, iteration)
            writer.add_scalar("mean_min_dists", test_mean_min_dist, iteration)
            writer.add_scalar("mean_occupied_landmarks", test_mean_occupied_landmark, iteration)
            writer.add_scalar("actor_loss", actor_loss, iteration)
            writer.add_scalar("critic_loss", critic_loss, iteration)
            writer.add_scalar("policy_entropy", torch.mean(
                surrogate_loss_2), iteration)
            writer.add_scalar(
                "critic_lr", critic_optimizer.param_groups[0]['lr'], iteration)

        if SAVE_PARAMETERS_TENSORBOARD and not DEBUG:
            save_parameters(writer, "actor", actors, iteration)
            save_parameters(writer, "value", critic, iteration)

        if iteration % CHECKPOINT_FREQUENCY == 0:
            stop_conditions.best_reward = test_mean_reward
            save_checkpoint(actors,
                            critic,
                            actor_optimizers,
                            critic_optimizer,
                            iteration,
                            stop_conditions,
                            env_id=ENV_NAME,
                            hp=hp,
                            base_checkpoint_path=BASE_CHECKPOINT_PATH)

        if test_mean_reward >= stop_conditions.best_reward:
            stop_conditions.best_reward = test_mean_reward
            save_checkpoint(actors,
                            critic,
                            actor_optimizers,
                            critic_optimizer,
                            iteration,
                            stop_conditions,
                            env_id=ENV_NAME,
                            hp=hp,
                            base_checkpoint_path=BASE_CHECKPOINT_PATH + 'best/')

        iteration += 1

    env.close()
    return stop_conditions.best_reward


# ============================================================================================
# --------------------------------- training ------------------------------------------------
# ===========================================================================================

hp = HyperParameters()

cfg = dict(algo_config=hp.to_dict())

if not DEBUG:
    wandb.init(
        project="particle", 
        sync_tensorboard=True, 
        config=cfg,
        name=experiment_name, 
        monitor_gym=True, 
        save_code=True,
        group=algo_name
        )


if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
with open(os.path.join(LOG_DIR, 'params.json'), 'w') as fp:
    json.dump(cfg, fp)

batch_count = hp.parallel_rollouts * hp.rollout_steps / \
    hp.recurrent_seq_len / hp.batch_size
print(f"batch_count: {batch_count}")
# if not DEBUG:
#     assert batch_count >= 1., "Less than 1 batch per trajectory.  Are you sure that's what you want?"

stop_conditions = StopConditions()

obsv_dim, action_dim, continuous_action_space = get_env_space(ENV_NAME)

actors = []
actor_optimizers = []
actor_lr_schedulers = []
milestones = [30, 60, 90, 120]

for i in range(N_AGENTS):
    actors.append(Actor(obsv_dim,
                        action_dim,
                        continuous_action_space=continuous_action_space,
                        trainable_std_dev=hp.trainable_std_dev,
                        hp=hp,
                        init_log_std_dev=hp.init_log_std_dev)
                  )
    actor_optimizers.append(optim.Adam(
        actors[i].parameters(), lr=hp.actor_learning_rate))
    # actor_lr_schedulers.append(torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer=actor_optimizers[i], milestones=milestones, gamma=0.5))
    actor_lr_schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=actor_optimizers[i], T_max=stop_conditions.max_iterations, eta_min=1e-5))

critic = Critic(obsv_dim, hp=hp)
critic_optimizer = optim.Adam(critic.parameters(), lr=hp.critic_learning_rate)
# critic_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
#     optimizer=critic_optimizer, milestones=milestones, gamma=0.5)
critic_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=critic_optimizer, T_max=stop_conditions.max_iterations, eta_min=1e-5)

if TRAIN:
    score = train_model(actors,
                        critic,
                        actor_optimizers,
                        critic_optimizer,
                        actor_lr_schedulers,
                        critic_lr_scheduler,
                        stop_conditions,
                        hp
                        )

# for evaluating the trained model
else:
    for a in range(N_AGENTS):
        actors[a].load_state_dict(torch.load(
            RESUME_CHECKPOINT_PATH + "actor_" + str(a+1) + ".pt"
        ))
    evaluate_env(actors, False)
