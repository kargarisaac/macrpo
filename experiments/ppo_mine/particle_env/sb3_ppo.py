
import gym
import numpy as np

from stable_baselines3 import PPO

from make_env import make_env as particke_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from typing import Callable
import time

class ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space[0].low[0], 
            env.observation_space[0].high[0], 
            (env.observation_space[0].shape[0] * len(env.observation_space), ))
    
    def observation(self, obs):
        return obs[0]

class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(5)
    
    def action(self, a):
        # a = np.array(a)
        # b = np.zeros((a.size, 5))
        # b[np.arange(a.size),a] = 1
        # return list(b)
        return [a]


class ReturnsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.episode_steps = 0

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.episode_steps += 1
        if self.episode_steps == 25:
            done = [True] * 1
            self.episode_steps = 0
        return next_state, reward[0], done[0], info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

# make a single agent env from particle-env
def make_single_env():
    env = particke_env('simple_spread_mine', False)
    env = ObsWrapper(env)
    env = ActionWrapper(env)
    env = ReturnsWrapper(env)
    # env = Monitor(env)
    return env


def make_env() -> Callable:
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = make_single_env()
        return env
    return _init


if __name__ == '__main__':
    train = True

    if train:
        # num_cpu = 27  # Number of processes to use
        # env = SubprocVecEnv([make_env() for i in range(num_cpu)])
        env = make_single_env()
        # eval_env = make_single_env()

        # Use deterministic actions for evaluation
        # eval_callback = EvalCallback(eval_env, best_model_save_path='./logs_sb3/',
        #                             log_path='./logs_sb3/', eval_freq=500,
        #                             deterministic=True, render=False)

        model = PPO(
            'MlpPolicy', 
            env, 
            verbose=1, 
            tensorboard_log="./sb3_ppo_particle_tensorboard/",
            learning_rate=1e-3,
            ent_coef=0.001, 
            batch_size=32, 
            clip_range=0.2
            )
        model.learn(total_timesteps=500000)
        model.save("sb3_ppo_particle_3")
        # del model 
        print('training finished ...')
    else:
        print('evaluate agent ...')
        env = make_single_env()
        model = PPO.load("sb3_ppo_particle_2")
        obs = env.reset()
        t = 0
        total_reward = 0
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            total_reward += rewards
            time.sleep(0.1)
            env.render()
            t += 1
            if t >25:
                obs = env.reset()
                t = 0
                print('episode reward: ', total_reward)
                total_reward = 0
