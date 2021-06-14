try:
    from collections.abc import Iterable
except ImportError:
    Iterable = (tuple, list)

from gym.vector.async_vector_env import AsyncVectorEnv
from gym.vector.sync_vector_env import SyncVectorEnv
from gym.vector.vector_env import VectorEnv
import gym

import torch
import numpy as np
import re
import os
from dataclasses import dataclass
from dotmap import DotMap
import pathlib
import time
import pickle
from torch import optim

N_AGENTS = 2


def calc_discounted_return(rewards, discount, final_value):
    """
    Calculate discounted returns based on rewards and discount factor.
    """
    seq_len = len(rewards)
    discounted_returns = torch.zeros(seq_len)
    discounted_returns[-1] = rewards[-1] + discount * final_value
    for i in range(seq_len - 2, -1 , -1):
        discounted_returns[i] = rewards[i] + discount * discounted_returns[i + 1]
        # discounted_returns[i] = rewards[i] + discount * discounted_returns[i + 1] + 0.95 * rewards[i + 1] #for multi-agent
    return discounted_returns


# def calc_discounted_return(rewards, discount, final_value):
#     """
#     Calculate discounted returns based on rewards and discount factor.
#     """
#     seq_len = len(rewards)
#     discounted_returns = torch.zeros(seq_len)
#     discounted_returns[-1] = rewards[-1] + discount * final_value
#     if len(rewards) > 1:
#         discounted_returns[-2] = rewards[-2] + discount * final_value
#     for i in range(seq_len - 3, -1 , -1):
#         discounted_returns[i] = rewards[i] + discount * discounted_returns[i + 2]
#         # discounted_returns[i] = rewards[i] + discount * discounted_returns[i + 1] + 0.95 * rewards[i + 1] #for multi-agent
#     return discounted_returns


def compute_advantages(rewards, values, discount, gae_lambda):
    """
    Compute General Advantage.
    """    
    deltas = rewards + discount * values[1:] - values[:-1]
    seq_len = len(rewards)
    advs = torch.zeros(seq_len + 1)
    multiplier = discount * gae_lambda
    for i in range(seq_len - 1, -1 , -1):
        advs[i] = advs[i + 1] * multiplier + deltas[i]
    return advs[:-1]


# def compute_advantages(rewards, values, discount, gae_lambda):
#     """
#     Compute General Advantage.
#     """
#     deltas = torch.zeros_like(rewards)
#     # TODO:add final value twice. later fix it with final value from other agent
#     values = torch.cat([values, torch.tensor([values[-1]])])
#     deltas[::N_AGENTS] = rewards[::N_AGENTS] + discount * values[2::N_AGENTS] - values[:-2:N_AGENTS]
#     deltas[1::N_AGENTS] = rewards[1::N_AGENTS] + discount * values[3::N_AGENTS] - values[1:-3:N_AGENTS]
#     seq_len = len(rewards)
#     advs = torch.zeros(seq_len + 1)
#     multiplier = discount * gae_lambda
#     for i in range(seq_len - 2, -1 , -1):
#         advs[i] = advs[i + 2] * multiplier + deltas[i]
#     return advs[:-1]


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
            writer.add_scalar("{}_{}{}".format(tag, k, shape_formatted), v.data, batch_idx)


def get_env_space(env_cls, env_config):
    """
    Return obsvervation dimensions, action dimensions and whether or not action space is continuous.
    """
    #     env = gym.make(ENV)

    env = env_cls(is_intersection_map=True)  # for self-play to have 2 learning agents
    env.configure_env(env_config)

    continuous_action_space = type(env.action_space) is gym.spaces.box.Box
    if continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    obsv_dim = env.observation_space.shape[0]
    return obsv_dim, action_dim, continuous_action_space


def get_last_checkpoint_iteration(base_checkpoint_path):
    """
    Determine latest checkpoint iteration.
    """
    if os.path.isdir(base_checkpoint_path):
        max_checkpoint_iteration = max([int(dirname) for dirname in os.listdir(base_checkpoint_path)])
    else:
        max_checkpoint_iteration = 0
    return max_checkpoint_iteration


def save_checkpoint(actor,
                    critic,
                    actor_optimizer,
                    critic_optimizer,
                    iteration,
                    stop_conditions,
                    env_id,
                    hp,
                    base_checkpoint_path):
    """
    Save training checkpoint.
    """
    checkpoint = DotMap()
    checkpoint.env = env_id
    checkpoint.iteration = iteration
    checkpoint.stop_conditions = stop_conditions
    checkpoint.hp = hp
    CHECKPOINT_PATH = base_checkpoint_path + f"{iteration}/"
    pathlib.Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH + "parameters.pt", "wb") as f:
        pickle.dump(checkpoint, f)
    # with open(CHECKPOINT_PATH + "actor_class.pt", "wb") as f:
    #     pickle.dump(Actor, f)
    # with open(CHECKPOINT_PATH + "critic_class.pt", "wb") as f:
    #     pickle.dump(Critic, f)
    torch.save(actor.state_dict(), CHECKPOINT_PATH + "actor.pt")
    torch.save(critic.state_dict(), CHECKPOINT_PATH + "critic.pt")
    torch.save(actor_optimizer.state_dict(), CHECKPOINT_PATH + "actor_optimizer.pt")
    torch.save(critic_optimizer.state_dict(), CHECKPOINT_PATH + "critic_optimizer.pt")


@dataclass
class StopConditions():
    """
    Store parameters and variables used to stop training.
    """
    best_reward: float = -1e6
    fail_to_improve_count: int = 0
    max_iterations: int = 151


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

    actor_state_dict = torch.load(CHECKPOINT_PATH + "actor.pt", map_location=torch.device(train_device))
    critic_state_dict = torch.load(CHECKPOINT_PATH + "critic.pt", map_location=torch.device(train_device))
    actor_optimizer_state_dict = torch.load(CHECKPOINT_PATH + "actor_optimizer.pt",
                                            map_location=torch.device(train_device))
    critic_optimizer_state_dict = torch.load(CHECKPOINT_PATH + "critic_optimizer.pt",
                                             map_location=torch.device(train_device))

    return (actor_state_dict, critic_state_dict,
            actor_optimizer_state_dict, critic_optimizer_state_dict,
            checkpoint.stop_conditions)


def make_env(env_cls, num_envs=1, asynchronous=True, wrappers=None, env_config=None, **kwargs):
    """Create a vectorized environment from multiple copies of an environment,
    from its id
    Parameters
    ----------
    id : str
        The environment ID. This must be a valid ID from the registry.
    num_envs : int
        Number of copies of the environment.
    asynchronous : bool (default: `True`)
        If `True`, wraps the environments in an `AsyncVectorEnv` (which uses
        `multiprocessing` to run the environments in parallel). If `False`,
        wraps the environments in a `SyncVectorEnv`.

    wrappers : Callable or Iterable of Callables (default: `None`)
        If not `None`, then apply the wrappers to each internal
        environment during creation.
    Returns
    -------
    env : `gym.vector.VectorEnv` instance
        The vectorized environment.
    Example
    -------
    # >>> import gym
    # >>> env = gym.vector.make('CartPole-v1', 3)
    # >>> env.reset()
    array([[-0.04456399,  0.04653909,  0.01326909, -0.02099827],
           [ 0.03073904,  0.00145001, -0.03088818, -0.03131252],
           [ 0.03468829,  0.01500225,  0.01230312,  0.01825218]],
          dtype=float32)
    """
    from gym.envs import make as make_
    def _make_env():
        #         env = make_(id, **kwargs)
        env = env_cls(is_intersection_map = True)  # for self-play to have 2 learning agents
        env.configure_env(env_config)

        if wrappers is not None:
            if callable(wrappers):
                env = wrappers(env)
            elif isinstance(wrappers, Iterable) and all([callable(w) for w in wrappers]):
                for wrapper in wrappers:
                    env = wrapper(env)
            else:
                raise NotImplementedError

        return env

    env_fns = [_make_env for _ in range(num_envs)]
    return AsyncVectorEnv(env_fns) if asynchronous else SyncVectorEnv(env_fns)

