import sys
import gym
import random
import numpy as np
import argparse
from gym.spaces import Tuple, MultiDiscrete, Dict, Discrete

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2 as Model


from ray.tune.registry import register_env
from ray.rllib.utils import try_import_tf
from ray.tune.integration.wandb import WandbLogger
from ray.tune.logger import DEFAULT_LOGGERS
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiEnvDict

from make_env import make_env as particle_env

import wandb

# tf = try_import_tf()
import tensorflow as tf


class MLPModel(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        last_layer = tf.layers.dense(
                input_dict["obs"], 400, activation=tf.nn.relu, name="fc1")
        last_layer = tf.layers.dense(
            last_layer, 300, activation=tf.nn.relu, name="fc2")
        output = tf.layers.dense(
            last_layer, num_outputs, activation=None, name="fc_out")
        return output, last_layer


class MLPModelV2(Model):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name="my_model"):
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        # Simplified to one layer.
        # print(obs_space)
        input = tf.keras.layers.Input(obs_space.shape, dtype=obs_space.dtype)
        output = tf.keras.layers.Dense(num_outputs, activation=None)(input)
        self.base_model = tf.keras.Model(input, output)
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        return self.base_model(input_dict["obs"]), []


class MAWrapper(MultiAgentEnv):
    def __init__(self, env):
        # super().__init__(env)
        self.env = env
        self.observation_space = {i:env.observation_space[i] for i in range(3)}
        self.action_space = {i:env.action_space[i] for i in range(3)}
        # self.reward_range = [-np.inf, np.inf]

    def reset(self):
        # obs = np.expand_dims(self.env.reset(), 0)
        obs = self.env.reset()
        return {i:obs[i] for i in range(3)}
        
    def step(self, action_dict):
        # total_reward = 0.0
        action_list = [action_dict[i] for i in range(3)]
        
        # obs, reward, done, info = self.env.step(np.argmax(action_dict[i]))
        obs, reward, done, info = self.env.step(action_list)
        # print(obs)
        # print(info)
        obs = {i:obs[i] for i in range(3)}
        reward = {i:reward[i] for i in range(3)}
        done = {i:done for i in range(3)}
        info = {i:{} for i in range(3)}
        
        done['__all__'] = done[0]
        # return obses, {0:sum(rewards), 1:sum(rewards)}, dones, infos
        return obs, reward, done, info


class TimeLimit(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.episode_steps = 0

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.episode_steps += 1
        if self.episode_steps == 25:
            done = [True] * 3 #number of agents
            self.episode_steps = 0
        # done is similar for all agents
        return next_state, np.array(reward), done[0], info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

        
# DQN and Apex-DQN do not work with continuous actions
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='RLlib')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--episodes-total', type=int, default=250000,
                        help='total number of episodes')
    parser.add_argument('--method', type=str, default="A2C",
                        help='name of RL method')

    args = parser.parse_args()

    # assert len(sys.argv) == 2, "Input the learning method as the second argument"
    method = args.method #sys.argv[1]
    # assert method in methods, "Method should be one of {}".format(methods)
    seed = args.seed
    num_gpus = 0
    num_workers = 8
    num_envs_per_worker = 8

    def env_creator(args):
        env = particle_env('simple_spread_mine', False)
        env = TimeLimit(env)
        env = MAWrapper(env)
        return env

    env = env_creator(1)
    if method == "QMIX":
        grouping = {
            "group_1": [0, 1, 2],
        }
        obs_space = Tuple([
            env.observation_space[0],
            env.observation_space[1],
            env.observation_space[2],
        ])
        act_space = Tuple([
            env.action_space[0],
            env.action_space[1],
            env.action_space[2],
        ])
        register_env(
            "particle",
            lambda config: env_creator(1).with_agent_groups(
                grouping, obs_space=obs_space, act_space=act_space))

    else:
        register_env("particle", env_creator)
        obs_space = env.observation_space[0]
        act_space = env.action_space[0]
      
    # ray.init()
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN

    if method in ["ADQN", "RDQN", "DQN"]:
        ModelCatalog.register_custom_model("MLPModelV2", MLPModelV2)
        def gen_policyV2(i):
            config = {
                "model": {
                    "custom_model": "MLPModelV2",
                },
                "gamma": 0.99,
            }
            return (None, obs_space, act_space, config)
        policies = {"policy_0": gen_policyV2(0)}

    elif method == "QMIX":
        def gen_policy(i):
            config = {
                "gamma": 0.99,
            }
            return (None, obs_space, act_space, config)
        policies = {"policy_0": gen_policy(0)}
        
    else:
        # ModelCatalog.register_custom_model("MLPModel", MLPModel)
        def gen_policy(i):
            config = {
                # "model": {
                #     "custom_model": "MLPModel",
                # },
                "gamma": 0.99,
            }
            return (None, obs_space, act_space, config)
        policies = {"policy_0": gen_policy(0)}
    
    policy_ids = list(policies.keys())

    if method == "ADQN":
        # APEX-DQN
        tune.run(
            "APEX",
            local_dir='./ray_results/particle/',
            stop={"episodes_total": args.episodes_total},
            checkpoint_freq=10000000,
            config={
        
                # Enviroment specific
                "env": "particle",
        
                # General
                "log_level": "INFO",
                "num_gpus": num_gpus,
                "num_workers": num_workers,
                "num_envs_per_worker": num_envs_per_worker,
                "learning_starts": 1000,
                "buffer_size": int(1e5),
                "compress_observations": False,
                # "sample_batch_size": 20,
                "train_batch_size": 5000,
                "gamma": .99,
                "lr": 0.01,
        
                # Method specific
        
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
                # wandb configuration
                "logger_config" : {
                    "wandb": {
                        "project": "particle",
                        # "api_key_file": "/path/to/file",
                        "log_config": True
                    }
                }
            },
            loggers=(WandbLogger, )
        )

    elif method == "DQN":
        # plain DQN
        tune.run(
            "DQN", 
            local_dir='./ray_results/particle/',
            stop={"episodes_total": args.episodes_total},
            checkpoint_freq=10000000,
            config={
                # Enviroment specific
                "env": "particle",
                # General
                "log_level": "ERROR",
                "num_gpus": num_gpus,
                "num_workers": num_workers,
                "num_envs_per_worker": num_envs_per_worker,
                "learning_starts": 1000,
                "buffer_size": int(1e5),
                "compress_observations": False,
                # "sample_batch_size": 20,
                "train_batch_size": 5000,
                "gamma": .99,
                # Method specific
                "dueling": True,
                "double_q": True,
                "lr":0.001,
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
                # wandb configuration
                "logger_config" : {
                    "wandb": {
                        "project": "particle",
                        # "api_key_file": "/path/to/file",
                        "log_config": True
                    }
                }
            },
            loggers=(WandbLogger, )
        )

    # psuedo-rainbow DQN
    elif method == "RDQN":
        tune.run(
            "DQN",
            local_dir='./ray_results/particle/',
            stop={"episodes_total": args.episodes_total},
            checkpoint_freq=10000000,
            config={
                # Enviroment specific
                "env": "particle",
        
                # General
                "log_level": "ERROR",
                "num_gpus": num_gpus,
                "num_workers": num_workers,
                "num_envs_per_worker": num_envs_per_worker,
                "learning_starts": 1000,
                "buffer_size": int(1e5),
                "compress_observations": False,
                # "sample_batch_size": 20,
                "train_batch_size": 1000,
                "gamma": .99,

                # Method specific
                "num_atoms": 51,
                "dueling": True,
                "double_q": True,
                "n_step": 2,
                "batch_mode": "complete_episodes",
                "prioritized_replay": True,
                "lr": 0.001,

                # # alternative 1
                # "noisy": True,
                # alternative 2
                # "parameter_noise": True,

                # based on expected return
                "v_min": 0,
                "v_max": 1500,
        
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
                # wandb configuration
                "logger_config" : {
                    "wandb": {
                        "project": "particle",
                        # "api_key_file": "/path/to/file",
                        "log_config": True
                    }
                }
            },
            loggers=(WandbLogger, )
        )

    elif method == "QMIX":
        tune.run(
            "QMIX",
            local_dir='./ray_results/particle/',
            stop={"episodes_total": args.episodes_total},
            checkpoint_freq=10000000,
            config={

                # Enviroment specific
                "env": "particle",
        
                # General
                "log_level": "ERROR",
                "num_gpus": num_gpus,
                "num_workers": num_workers,
                "num_envs_per_worker": num_envs_per_worker,
                "learning_starts": 1000,
                "buffer_size": int(1e5),
                "compress_observations": False,
                # "sample_batch_size": 20,
                "train_batch_size": 5000,
                "gamma": .99,
        
                # === QMix ===
                # Mixing network. Either "qmix", "vdn", or None
                "mixer": "qmix",
                # Size of the mixing network embedding
                "mixing_embed_dim": 32,
                # Whether to use Double_Q learning
                "double_q": True,
                # Optimize over complete episodes by default.
                "batch_mode": "complete_episodes",

                # === Exploration Settings ===
                "exploration_config": {
                    # The Exploration class to use.
                    "type": "EpsilonGreedy",
                    # Config for the Exploration class' constructor:
                    "initial_epsilon": 1.0,
                    "final_epsilon": 0.02,
                    "epsilon_timesteps": 1000,  # Timesteps over which to anneal epsilon.

                    # For soft_q, use:
                    # "exploration_config" = {
                    #   "type": "SoftQ"
                    #   "temperature": [float, e.g. 1.0]
                    # }
                },

                # === Evaluation ===
                # Evaluate with epsilon=0 every `evaluation_interval` training iterations.
                # The evaluation stats will be reported under the "evaluation" metric key.
                # Note that evaluation is currently not parallelized, and that for Ape-X
                # metrics are already only reported for the lowest epsilon workers.
                "evaluation_interval": None,
                # Number of episodes to run per evaluation period.
                "evaluation_num_episodes": 10,
                # Switch to greedy actions in evaluation workers.
                "evaluation_config": {
                    "explore": True,
                },

                # Number of env steps to optimize for before returning
                "timesteps_per_iteration": 2500,
                # Update the target network every `target_network_update_freq` steps.
                "target_network_update_freq": 500,

                # === Optimization ===
                # Learning rate for RMSProp optimizer
                "lr": 0.001,
                # RMSProp alpha
                "optim_alpha": 0.99,
                # RMSProp epsilon
                "optim_eps": 0.00001,
                # If not None, clip gradients during optimization at this value
                "grad_norm_clipping": 10,
                # Update the replay buffer with this many samples at once. Note that
                # this setting applies per-worker if num_workers > 1.
                "rollout_fragment_length": 4,
                # Size of a batched sampled from replay buffer for training. Note that
                # if async_updates is set, then each worker returns gradients for a
                # batch of this size.
                # "train_batch_size": 32,

                # === Parallelism ===
                # Number of workers for collecting samples with. This only makes sense
                # to increase if your environment is particularly slow to sample, or if
                # you"re using the Async or Ape-X optimizers.
                # "num_workers": 0,
                # Whether to use a distribution of epsilons across workers for exploration.
                "per_worker_exploration": False,
                # Whether to compute priorities on workers.
                "worker_side_prioritization": False,
                # Prevent iterations from going lower than this time span
                "min_iter_time_s": 1,

                # === Model ===
                # "model": {
                #     "lstm_cell_size": 64,
                #     "max_seq_len": 999999,
                # },

                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
                # wandb configuration
                "logger_config" : {
                    "wandb": {
                        "project": "particle",
                        # "api_key_file": "/path/to/file",
                        "log_config": True
                    }
                }
            },
            loggers=(WandbLogger, )
        )

    elif method == "A2C":
        tune.run(
            "A2C",
            local_dir='./ray_results/particle/',
            stop={"episodes_total": args.episodes_total},
            checkpoint_freq=10000000,
            config={
        
                # Enviroment specific
                "env": "particle",
        
                # General
                "log_level": "ERROR",
                "num_gpus": num_gpus,
                "num_workers": num_workers,
                "num_envs_per_worker": num_envs_per_worker,
                "compress_observations": False,
                # "sample_batch_size": 20,
                "train_batch_size": 5000,
                "gamma": .99,
        
                "lr_schedule": [[0, 0.0007],[20000000, 0.000000000001]],
        
                # Method specific
        
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
                # wandb configuration
                "logger_config" : {
                    "wandb": {
                        "project": "particle",
                        # "api_key_file": "/path/to/file",
                        "log_config": True
                    }
                }
            },
            loggers=(WandbLogger, )
        )

    elif method == "IMPALA":
        tune.run(
            "IMPALA",
            local_dir='./ray_results/particle/',
            stop={"episodes_total": args.episodes_total},
            checkpoint_freq=10000000,
            config={
        
                # Enviroment specific
                "env": "particle",
        
                # General
                "log_level": "ERROR",
                "num_gpus": num_gpus,
                "num_workers": num_workers,
                "num_envs_per_worker": num_envs_per_worker,
                "compress_observations": True,
                # "sample_batch_size": 20,
                "train_batch_size": 500,
                "gamma": .99,
        
                "clip_rewards": False,
                # "lr_schedule": [[0, 0.0005],[20000000, 0.000000000001]],
                "lr": 0.01,
        
                # Method specific
        
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
                # wandb configuration
                "logger_config" : {
                    "wandb": {
                        "project": "particle",
                        # "api_key_file": "/path/to/file",
                        "log_config": True
                    }
                }
            },
            loggers=(WandbLogger, )
        )

    elif method == "PPO":
        tune.run(
            "PPO",
            local_dir='./ray_results/particle/',
            stop={"episodes_total": args.episodes_total},
            checkpoint_freq=10000000,
            config={
                # Enviroment specific
                "env": "particle",
        
                # General
                "log_level": "ERROR",
                "num_gpus": num_gpus,
                "num_workers": num_workers,
                "num_envs_per_worker": num_envs_per_worker,
                "compress_observations": True,
                "gamma": .99,
        
                "lambda": 0.95,
                "kl_coeff": 0.5,
                "clip_rewards": False,
                "clip_param": 0.5,
                "vf_clip_param": 1.0,
                "entropy_coeff": 0.001,
                "train_batch_size": 5000,
                # "sample_batch_size": 100,
                "sgd_minibatch_size": 1000,
                "num_sgd_iter": 50,
                "batch_mode": 'truncate_episodes',
                "vf_share_layers": True,
                "lr": 1e-2,
                # Method specific
        
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
                # wandb configuration
                "logger_config" : {
                    "wandb": {
                        "project": "particle",
                        # "api_key_file": "/path/to/file",
                        "log_config": True
                    }
                }
            },
            loggers=(WandbLogger, )
        )
    
    elif method == "SAC":
        tune.run(
            "SAC",
            local_dir='./ray_results/particle/',
            stop={"episodes_total": args.episodes_total},
            checkpoint_freq=10000000,
            config={
                "seed":seed,
                
                # Enviroment specific
                "env": "particle",
        
                # General
                "log_level": "ERROR",
                "num_gpus": num_gpus,
                "num_workers": num_workers,
                "num_envs_per_worker": num_envs_per_worker,
                "learning_starts": 1000,
                "buffer_size": int(1e5),
                "compress_observations": True,
                # "train_batch_size": 20,
                "train_batch_size": 5000,
                "gamma": .99,
        
                "horizon": 200,
                "soft_horizon": False,
                "Q_model": {
                  "fcnet_activation": "relu",
                  "fcnet_hiddens": [256, 256]
                  },
                "tau": 0.005,
                "target_entropy": "auto",
                "no_done_at_end": True,
                "n_step": 5,
                "prioritized_replay": True,
                "target_network_update_freq": 1000,
                "timesteps_per_iteration": 1000,
                # "exploration_enabled": True,
                "optimization": {
                  "actor_learning_rate": 0.003,
                  "critic_learning_rate": 0.003,
                  "entropy_learning_rate": 0.003,
                  },
                "clip_actions": False,
                #TODO -- True
                "normalize_actions": False,
                "evaluation_interval": 1,
                "metrics_smoothing_episodes": 5,
        
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
            
                # wandb configuration
                "logger_config" : {
                    "wandb": {
                        "project": "particle",
                        # "api_key_file": "/path/to/file",
                        "log_config": True
                    }
                }
                },
            loggers=(WandbLogger, )
        )