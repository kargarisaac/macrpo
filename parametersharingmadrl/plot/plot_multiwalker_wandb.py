import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy
from scipy import signal
import wandb

# algo_name = "ppo_lstm_c3c_"


data_path = "plot/data/multi_walker/"

experiment_name = 'random'

# data = np.genfromtxt(os.path.join(data_path, 'gupta_ddpg.csv'), delimiter=',')
# data = np.genfromtxt(os.path.join(data_path, 'gupta_trpo.csv'), delimiter=',')

# df = pd.read_csv(os.path.join(data_path, 'ppo1.csv'))
# df = df[['episodes_total', "episode_reward_mean"]]
# data = df.to_numpy()

# df = pd.read_csv(os.path.join(data_path,'impala.csv'))
# df = df[['episodes_total', "episode_reward_mean"]]
# data = df.to_numpy()

# df = pd.read_csv(os.path.join(data_path,'a2c.csv'))
# df = df[['episodes_total', "episode_reward_mean"]]
# data = df.to_numpy()

# df = pd.read_csv(os.path.join(data_path,'apex_ddpg.csv'))
# df = df[['episodes_total', "episode_reward_mean"]]
# data = df.to_numpy()

# df = pd.read_csv(os.path.join(data_path, 'sac.csv'))
# df = df[['episodes_total', "episode_reward_mean"]]
# data = df.to_numpy()

# df = pd.read_csv(os.path.join(data_path,'td3.csv'))
# df = df[['episodes_total', "episode_reward_mean"]]
# data = df.to_numpy()

# df = pd.read_csv(os.path.join(data_path,'ddpg.csv'))
# df = df[['episodes_total', "episode_reward_mean"]]
# data = df.to_numpy()

# df = pd.read_csv(os.path.join(data_path, 'maddpg.csv'))
# df = df[['episode', "reward"]]
# data = df.to_numpy()

# df = pd.read_csv(os.path.join('plot/qmix_results/multiwalker', 'cout.txt.csv'))
# df = df[['episode', "return_mean"]]
# data = df.to_numpy()


wandb.init(
    project="multiwalker", 
    name=experiment_name, 
    # group=algo_name
    )

for ep in range(data.shape[0]):
    wandb.log({'prev_paper/episode_reward': data[ep, 1], 'episode': int(data[ep, 0])})

# plt.plot(np.array([0,60000]),np.array([-1e5,-1e5]), label='Rainbow DQN', linewidth=0.6, color='tab:blue', linestyle=(0, (3, 3)))
# plt.plot(np.array([0,60000]),np.array([-1e5,-1e5]), label='ApeX DQN',    linewidth=0.6, color='tab:brown', linestyle=(0, (1, 1)))
# plt.plot(np.array([0,60000]),np.array([-1e5,-1e5]), label='DQN', linewidth=0.6, color='tab:cyan', linestyle=(0, (3, 3)))
# plt.plot(np.array([0,60000]),np.array([-102.05,-102.05]), label='Random', linewidth=0.6, color='red', linestyle=(0, (1, 1)))

# wandb.log({'prev_paper/episode_reward': -102.05, 'episode': 0})
# wandb.log({'prev_paper/episode_reward': -102.05, 'episode': 60000})