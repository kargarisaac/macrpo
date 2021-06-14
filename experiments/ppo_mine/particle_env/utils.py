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

from make_env import make_env as particke_env
# from stable_baselines3.common.vec_env import VecEnvWrapper
# from stable_baselines3.common.vec_env.subproc_vec_env import _worker

from collections import OrderedDict
from copy import deepcopy
from typing import Sequence

import numpy as np

# from stable_baselines3.common.vec_env.base_vec_env import VecEnv, CloudpickleWrapper
# from stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info
import multiprocessing
import random

N_AGENTS = 3
# ENV_NAME = 'simple_spread'


# class DummyVecEnv(VecEnv):
#     """
#     Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
#     Python process. This is useful for computationally simple environment such as ``cartpole-v1``,
#     as the overhead of multiprocess or multithread outweighs the environment computation time.
#     This can also be used for RL methods that
#     require a vectorized environment, but that you want a single environments to train with.

#     :param env_fns: ([Gym Environment]) the list of environments to vectorize
#     """

#     def __init__(self, env_fns):
#         self.envs = [fn() for fn in env_fns]
#         env = self.envs[0]
#         VecEnv.__init__(self, len(env_fns),
#                         env.observation_space, env.action_space)
#         obs_space = env.observation_space
#         self.keys, shapes, dtypes = obs_space_info(obs_space)

#         self.buf_obs = OrderedDict([
#             (k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]))
#             for k in self.keys])
#         self.buf_dones = np.zeros((self.num_envs, ), dtype=np.bool)
#         self.buf_rews = np.zeros((self.num_envs, N_AGENTS), dtype=np.float32)
#         self.buf_infos = [{} for _ in range(self.num_envs)]
#         self.actions = None
#         self.metadata = env.metadata

#     def step_async(self, actions):
#         self.actions = actions

#     def step_wait(self):
#         for env_idx in range(self.num_envs):
#             obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] =\
#                 self.envs[env_idx].step(self.actions[env_idx])
#             if self.buf_dones[env_idx]:
#                 # save final observation where user can get it, then reset
#                 self.buf_infos[env_idx]['terminal_observation'] = obs
#                 obs = self.envs[env_idx].reset()
#             self._save_obs(env_idx, obs)
#         return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
#                 deepcopy(self.buf_infos))

#     def seed(self, seed=None):
#         seeds = list()
#         for idx, env in enumerate(self.envs):
#             seeds.append(env.seed(seed + idx))
#         return seeds

#     def reset(self):
#         for env_idx in range(self.num_envs):
#             obs = self.envs[env_idx].reset()
#             self._save_obs(env_idx, obs)
#         return self._obs_from_buf()

#     def close(self):
#         for env in self.envs:
#             env.close()

#     def get_images(self) -> Sequence[np.ndarray]:
#         return [env.render(mode='rgb_array') for env in self.envs]

#     def render(self, mode: str = 'human'):
#         """
#         Gym environment rendering. If there are multiple environments then
#         they are tiled together in one image via ``BaseVecEnv.render()``.
#         Otherwise (if ``self.num_envs == 1``), we pass the render call directly to the
#         underlying environment.

#         Therefore, some arguments such as ``mode`` will have values that are valid
#         only when ``num_envs == 1``.

#         :param mode: The rendering type.
#         """
#         if self.num_envs == 1:
#             return self.envs[0].render(mode=mode)
#         else:
#             return super().render(mode=mode)

#     def _save_obs(self, env_idx, obs):
#         for key in self.keys:
#             if key is None:
#                 self.buf_obs[key][env_idx] = obs
#             else:
#                 self.buf_obs[key][env_idx] = obs[key]

#     def _obs_from_buf(self):
#         return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

#     def get_attr(self, attr_name, indices=None):
#         """Return attribute from vectorized environment (see base class)."""
#         target_envs = self._get_target_envs(indices)
#         return [getattr(env_i, attr_name) for env_i in target_envs]

#     def set_attr(self, attr_name, value, indices=None):
#         """Set attribute inside vectorized environments (see base class)."""
#         target_envs = self._get_target_envs(indices)
#         for env_i in target_envs:
#             setattr(env_i, attr_name, value)

#     def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
#         """Call instance methods of vectorized environments."""
#         target_envs = self._get_target_envs(indices)
#         return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

#     def _get_target_envs(self, indices):
#         indices = self._get_indices(indices)
#         return [self.envs[i] for i in indices]


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
    if N_AGENTS == 1:
        max_iterations: int = 41
    elif N_AGENTS == 2:
        max_iterations: int = 101
    elif N_AGENTS == 3:
        max_iterations: int = 601
    else:
        max_iterations: int = 201


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


# def make_env(seed, idx):
#     def thunk():
#         # env = gym.make(gym_id)
#         env = particke_env(ENV_NAME, False)
#         env = ObsWrapper(env)
#         env = ActionWrapper(env)
#         env = TimeLimit(env)

#         # if isinstance(env.action_space, Box):
#         #     env = ClipActionsWrapper(env)
#         # env = gym.wrappers.RecordEpisodeStatistics(env)
#         # if args.capture_video:
#         #     if idx == 0:
#         #         env = Monifrom stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrappertor(env, f'videos/{experiment_name}')
#         # env = NormalizedEnv(env)
#         # env.seed(seed)
#         # env.action_space.seed(seed)
#         # env.observation_space.seed(seed)
#         return env
#     return thunk


def make_env2(num_envs=1, asynchronous=True, wrappers=None, benchmark=False, env_name='simple_spread_mine', **kwargs):
    def _make_env(seed):
        def _make():
            env = particke_env(env_name, benchmark)
            env = ObsWrapper(env)
            env = ActionWrapper(env)
            env = TimeLimit(env)
            return env
        return _make

    env_fns = [_make_env(i) for i in range(num_envs)]
    return AsyncVectorEnv(env_fns) if asynchronous else SyncVectorEnv(env_fns)


def get_env_space(env_name="simple_spread_mine"):
    """
    Return obsvervation dimensions, action dimensions and whether or not action space is continuous.
    """
    #     env = gym.make(ENV)

    env = particke_env(env_name, False)

    continuous_action_space = type(env.action_space) is gym.spaces.box.Box
    if continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        if type(env.action_space) == list:
            action_dim = env.action_space[0].n
        else:
            action_dim = env.action_space.n
    if type(env.observation_space) == list:
        obsv_dim = env.observation_space[0].shape[0]
    else:
        obsv_dim = env.observation_space.shape[0]
    return obsv_dim, action_dim, continuous_action_space


# class VecPyTorch(VecEnvWrapper):
#     def __init__(self, venv, device):
#         super(VecPyTorch, self).__init__(venv)
#         self.device = device

#     def reset(self):
#         obs = self.venv.reset()
#         obs = torch.from_numpy(obs).float().to(self.device)
#         return obs

#     def step_async(self, actions):
#         # actions = actions.cpu().numpy()
#         self.venv.step_async(actions)

#     def step_wait(self):
#         obs, reward, done, info = self.venv.step_wait()
#         obs = torch.from_numpy(obs).float().to(self.device)
#         # reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
#         reward = torch.from_numpy(reward).float()
#         return obs, reward, done, info


class ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space[0].low[0],
            env.observation_space[0].high[0],
            (env.observation_space[0].shape[0] * len(env.observation_space), ))

    def observation(self, obs):
        # obs is a list of obs vectors for all agents in one env -> concatenate them
        new_obs = np.concatenate(obs)
        return new_obs


class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(N_AGENTS * 5)

    def action(self, a):
        # convert array of three action indexes into onehot vector for each one
        # b = np.zeros((a.size, 5))
        # b[np.arange(a.size),a] = 1
        # return list(b)
        if N_AGENTS == 1:
            return [a]
        else:
            return list(a)


class TimeLimit(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.episode_steps = 0

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.episode_steps += 1
        if self.episode_steps == 25:
            done = [True] * N_AGENTS
            self.episode_steps = 0
        # done is similar for all agents
        return next_state, np.array(reward), done[0], info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean([x], axis=0)
        batch_var = np.var([x], axis=0)
        batch_count = 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizedEnv(gym.core.Wrapper):
    def __init__(self, env, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        super(NormalizedEnv, self).__init__(env)
        self.ob_rms = RunningMeanStd(
            shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=(1,)) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(())
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, dones, infos = self.env.step(action)
        infos['real_reward'] = rews
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(np.array([self.ret].copy()))
            rews = np.clip(rews / np.sqrt(self.ret_rms.var +
                                          self.epsilon), -self.cliprew, self.cliprew)
        self.ret = self.ret * (1-float(dones))
        return obs, rews, dones, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var +
                                                             self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(())
        obs = self.env.reset()
        return self._obfilt(obs)


class GumbelSoftmax(distributions.RelaxedOneHotCategorical):
    '''
    A differentiable Categorical distribution using reparametrization trick with Gumbel-Softmax
    Explanation http://amid.fish/assets/gumbel.html
    NOTE: use this in place PyTorch's RelaxedOneHotCategorical distribution since its log_prob is not working right (returns positive values)
    Papers:
    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables (Maddison et al, 2017)
    [2] Categorical Reparametrization with Gumbel-Softmax (Jang et al, 2017)
    '''

    def sample(self, sample_shape=torch.Size()):
        '''Gumbel-softmax sampling. Note rsample is inherited from RelaxedOneHotCategorical'''
        u = torch.empty(self.logits.size(), device=self.logits.device,
                        dtype=self.logits.dtype).uniform_(0, 1)
        noisy_logits = self.logits - torch.log(-torch.log(u))
        return torch.argmax(noisy_logits, dim=-1)

    def rsample(self, sample_shape=torch.Size()):
        '''
        Gumbel-softmax resampling using the Straight-Through trick.
        Credit to Ian Temple for bringing this to our attention. To see standalone code of how this works, refer to https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
        '''
        rout = super().rsample(sample_shape)  # differentiable
        out = F.one_hot(torch.argmax(rout, dim=-1),
                        self.logits.shape[-1]).float()
        return (out - rout).detach() + rout

    def log_prob(self, value):
        '''value is one-hot or relaxed'''
        if value.shape != self.logits.shape:
            value = F.one_hot(value.long(), self.logits.shape[-1]).float()
            assert value.shape == self.logits.shape
        return - torch.sum(- value * F.log_softmax(self.logits, -1), -1)


# for gym vec in multi-agent envs
class SyncVectorEnv(VectorEnv):
    """Vectorized environment that serially runs multiple environments.

    Parameters
    ----------
    env_fns : iterable of callable
        Functions that create the environments.

    observation_space : `gym.spaces.Space` instance, optional
        Observation space of a single environment. If `None`, then the
        observation space of the first environment is taken.

    action_space : `gym.spaces.Space` instance, optional
        Action space of a single environment. If `None`, then the action space
        of the first environment is taken.

    copy : bool (default: `True`)
        If `True`, then the `reset` and `step` methods return a copy of the
        observations.
    """

    def __init__(self, env_fns, observation_space=None, action_space=None,
                 copy=True):
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        self.copy = copy

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or self.envs[0].observation_space
            action_space = action_space or self.envs[0].action_space
        super(SyncVectorEnv, self).__init__(num_envs=len(env_fns),
                                            observation_space=observation_space, action_space=action_space)

        self._check_observation_spaces()
        self.observations = create_empty_array(self.single_observation_space,
                                               n=self.num_envs, fn=np.zeros)
        self._rewards = np.zeros((self.num_envs, N_AGENTS), dtype=np.float64)
        self._dones = np.zeros((self.num_envs,), dtype=np.bool_)
        self._actions = None

    def seed(self, seeds=None):
        if seeds is None:
            seeds = [None for _ in range(self.num_envs)]
        if isinstance(seeds, int):
            seeds = [seeds + i for i in range(self.num_envs)]
        assert len(seeds) == self.num_envs

        for env, seed in zip(self.envs, seeds):
            env.seed(seed)

    def reset_wait(self):
        self._dones[:] = False
        observations = []
        for env in self.envs:
            observation = env.reset()
            observations.append(observation)
        concatenate(observations, self.observations,
                    self.single_observation_space)

        return np.copy(self.observations) if self.copy else self.observations

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        observations, infos = [], []
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            observation, self._rewards[i], self._dones[i], info = env.step(
                action)
            if self._dones[i]:
                observation = env.reset()
            observations.append(observation)
            infos.append(info)
        concatenate(observations, self.observations,
                    self.single_observation_space)

        return (deepcopy(self.observations) if self.copy else self.observations,
                np.copy(self._rewards), np.copy(self._dones), infos)

    def close_extras(self, **kwargs):
        [env.close() for env in self.envs]

    def _check_observation_spaces(self):
        for env in self.envs:
            if not (env.observation_space == self.single_observation_space):
                break
        else:
            return True
        raise RuntimeError('Some environments have an observation space '
                           'different from `{0}`. In order to batch observations, the '
                           'observation spaces from all environments must be '
                           'equal.'.format(self.single_observation_space))
