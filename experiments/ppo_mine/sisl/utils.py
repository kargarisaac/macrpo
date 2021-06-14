try:
    from collections.abc import Iterable
except ImportError:
    Iterable = (tuple, list)

from gym.vector.async_vector_env import AsyncVectorEnv
from gym.vector.vector_env import VectorEnv
import gym
from gym import logger
from gym.vector.utils import concatenate, create_empty_array

import numpy as np
from copy import deepcopy

import torch
from torch import distributions
import torch.nn.functional as F
import numpy as np
import re
import os
from dataclasses import dataclass
from dotmap import DotMap
import pathlib
import time
import pickle
from torch import optim

from collections import OrderedDict
from copy import deepcopy
from typing import Sequence

import numpy as np

import multiprocessing
import random

from pettingzoo.sisl import multiwalker_v0

N_AGENTS = 3

def calc_discounted_return(rewards, discount, final_value):
    """
    Calculate discounted returns based on rewards and discount factor.
    """
    seq_len = len(rewards)
    discounted_returns = torch.zeros(seq_len)
    discounted_returns[-1] = rewards[-1] + discount * final_value
    for i in range(seq_len - 2, -1, -1):
        discounted_returns[i] = rewards[i] + \
            discount * discounted_returns[i + 1]
    return discounted_returns


def compute_advantages(rewards, values, discount, gae_lambda):
    """
    Compute General Advantage.
    """
    deltas = rewards + discount * values[1:] - values[:-1]
    seq_len = len(rewards)
    advs = torch.zeros(seq_len + 1)
    multiplier = discount * gae_lambda
    for i in range(seq_len - 1, -1, -1):
        advs[i] = advs[i + 1] * multiplier + deltas[i]
    return advs[:-1]


def save_parameters(writer, tag, model, batch_idx):
    """
    Save model parameters for tensorboard.
    """
    _INVALID_TAG_CHARACTERS = re.compile(r"[^-/\w\.]")
    for k, v in model.state_dict().items():
        shape = v.shape
        # Fix shape definition for tensorboard.
        shape_formatted = _INVALID_TAG_CHARACTERS.sub("_", str(shape))
        # Don't do this for single weights or biases
        if np.any(np.array(shape) > 1):
            mean = torch.mean(v)
            std_dev = torch.std(v)
            maximum = torch.max(v)
            minimum = torch.min(v)
            writer.add_scalars(
                "{}_weights/{}{}".format(tag, k, shape_formatted),
                {"mean": mean, "std_dev": std_dev, "max": maximum, "min": minimum},
                batch_idx,
            )
        else:
            writer.add_scalar("{}_{}{}".format(
                tag, k, shape_formatted), v.data, batch_idx)


def get_last_checkpoint_iteration(base_checkpoint_path):
    """
    Determine latest checkpoint iteration.
    """
    if os.path.isdir(base_checkpoint_path):
        max_checkpoint_iteration = max(
            [int(dirname) for dirname in os.listdir(base_checkpoint_path)])
    else:
        max_checkpoint_iteration = 0
    return max_checkpoint_iteration


def save_checkpoint(actor,
                    critic,
                    actor_optimizer,
                    critic_optimizer,
                    iteration,
                    stop_conditions,
                    hp,
                    base_checkpoint_path):
    """
    Save training checkpoint.
    """
    checkpoint = DotMap()
    # checkpoint.env = env_id
    checkpoint.iteration = iteration
    checkpoint.stop_conditions = stop_conditions
    checkpoint.hp = hp
    CHECKPOINT_PATH = base_checkpoint_path  # + f"{iteration}/"
    pathlib.Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH + "parameters.pt", "wb") as f:
        pickle.dump(checkpoint, f)
    # with open(CHECKPOINT_PATH + "actor_class.pt", "wb") as f:
    #     pickle.dump(Actor, f)
    # with open(CHECKPOINT_PATH + "critic_class.pt", "wb") as f:
    #     pickle.dump(Critic, f)
    if isinstance(actor, list):
        for i, (a, a_optimizer) in enumerate(zip(actor, actor_optimizer)):
            torch.save(a.state_dict(), CHECKPOINT_PATH +
                       "actor_" + str(i+1) + ".pt")
            # torch.save(c.state_dict(), CHECKPOINT_PATH + "critic_" + str(i+1) + ".pt")
            torch.save(a_optimizer.state_dict(), CHECKPOINT_PATH +
                       "actor_optimizer_" + str(i+1) + ".pt")
            # torch.save(c_optimizer.state_dict(), CHECKPOINT_PATH + "critic_optimizer_" + str(i+1) + ".pt")
        if isinstance(critic, list):
            for i, (c, c_optimizer) in enumerate(zip(critic, critic_optimizer)):
                # torch.save(a.state_dict(), CHECKPOINT_PATH + "actor_" + str(i+1) + ".pt")
                torch.save(c.state_dict(), CHECKPOINT_PATH +
                           "critic_" + str(i+1) + ".pt")
                # torch.save(a_optimizer.state_dict(), CHECKPOINT_PATH + "actor_optimizer_" + str(i+1) + ".pt")
                torch.save(c_optimizer.state_dict(), CHECKPOINT_PATH +
                           "critic_optimizer_" + str(i+1) + ".pt")
        else:
            # torch.save(actor.state_dict(), CHECKPOINT_PATH + "actor.pt")
            torch.save(critic.state_dict(), CHECKPOINT_PATH + "critic.pt")
            # torch.save(actor_optimizer.state_dict(), CHECKPOINT_PATH + "actor_optimizer.pt")
            torch.save(critic_optimizer.state_dict(),
                       CHECKPOINT_PATH + "critic_optimizer.pt")

    else:
        torch.save(actor.state_dict(), CHECKPOINT_PATH + "actor.pt")
        torch.save(critic.state_dict(), CHECKPOINT_PATH + "critic.pt")
        torch.save(actor_optimizer.state_dict(),
                   CHECKPOINT_PATH + "actor_optimizer.pt")
        torch.save(critic_optimizer.state_dict(),
                   CHECKPOINT_PATH + "critic_optimizer.pt")


@dataclass
class StopConditions():
    """
    Store parameters and variables used to stop training.
    """
    best_reward: float = -np.inf
    fail_to_improve_count: int = 0
    max_iterations: int = 501


def load_checkpoint(iteration, hp, base_checkpoint_path, env_id, train_device):
    """
    Load from training checkpoint.
    """
    CHECKPOINT_PATH = base_checkpoint_path + f"{iteration}/"
    with open(CHECKPOINT_PATH + "parameters.pt", "rb") as f:
        checkpoint = pickle.load(f)

    assert env_id == checkpoint.env, "To resume training environment must match current settings."
    # assert ENV_MASK_VELOCITY == checkpoint.env_mask_velocity, "To resume training model architecture must match current settings."
    assert hp == checkpoint.hp, "To resume training hyperparameters must match current settings."

    actor_state_dict = torch.load(
        CHECKPOINT_PATH + "actor.pt", map_location=torch.device(train_device))
    critic_state_dict = torch.load(
        CHECKPOINT_PATH + "critic.pt", map_location=torch.device(train_device))
    actor_optimizer_state_dict = torch.load(CHECKPOINT_PATH + "actor_optimizer.pt",
                                            map_location=torch.device(train_device))
    critic_optimizer_state_dict = torch.load(CHECKPOINT_PATH + "critic_optimizer.pt",
                                             map_location=torch.device(train_device))

    return (actor_state_dict, critic_state_dict,
            actor_optimizer_state_dict, critic_optimizer_state_dict,
            checkpoint.stop_conditions)


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
    labels: (LongTensor) class labels, sized [N,].
    num_classes: (int) number of classes.

    Returns:
    (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


def make_env(num_envs=1, asynchronous=True, wrappers=None, env_cfg=None, **kwargs):
    def _make_env(seed):
        def _make():
            env = multiwalker_v0.env(**env_cfg)
            env = EnvWrapper(env)
            # env = ObsWrapper(env)
            # env = ActionWrapper(env)
            # env = TimeLimit(env)
            return env
        return _make

    env_fns = [_make_env(i) for i in range(num_envs)]
    return AsyncVectorEnv(env_fns) if asynchronous else SyncVectorEnv(env_fns)


def get_env_space():
    """
    Return obsvervation dimensions, action dimensions and whether or not action space is continuous.
    """
    #     env = gym.make(ENV)
    env = multiwalker_v0.env()
    env = EnvWrapper(env)

    continuous_action_space = type(env.action_space) is gym.spaces.box.Box
    if continuous_action_space:
        action_dim = env.action_space.shape[0]
    # else:
    #     if type(env.action_space) == list:
    #         action_dim = env.action_spaces[0].n
    #     else:
    #         action_dim = env.action_space.n
    if type(env.observation_spaces) == list:
        obsv_dim = env.observation_space.shape[0]
    else:
        obsv_dim = env.observation_space.shape[0]
    return obsv_dim, action_dim, continuous_action_space


class EnvWrapper(gym.Wrapper):
    def __init__(self, env):
        # super().__init__(env)
        self.env = env
        self.observation_space = env.observation_spaces['walker_0']
        self.action_space = env.action_spaces['walker_0']
        # self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata

    def step(self, action):
        action = np.clip(action, -1, 1)
        reward, done, info = self.env.last()
        next_state = self.env.step(action)
        return next_state, reward, done, info
