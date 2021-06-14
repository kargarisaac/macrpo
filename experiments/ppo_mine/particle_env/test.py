from gym.vector.async_vector_env import AsyncVectorEnv
from make_env import make_env as particke_env
import gym
import numpy as np

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
        return next_state, np.array(reward), done[0], info #done is similar for all agents

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


def _make_env(seed):
    def _make():
        env = particke_env('simple_spread', False)
        env = ObsWrapper(env)
        env = ActionWrapper(env)
        env = TimeLimit(env)
        return env
    return _make

env_fns = [_make_env(i) for i in range(4)]
envs = AsyncVectorEnv(env_fns)

print(envs.reset())