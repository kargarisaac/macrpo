import sys
import gym
import random
import numpy as np
import argparse

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2 as Model

from ray.tune.registry import register_env
from ray.rllib.utils import try_import_tf
from ray.tune.integration.wandb import WandbLogger
from ray.tune.logger import DEFAULT_LOGGERS

from deepdrive_zero.envs.env import Deepdrive2DEnv
import wandb

tf = try_import_tf()

class MLPModel(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        last_layer = tf.layers.dense(
                input_dict["obs"], 400, activation=tf.nn.relu, name="fc1")
        last_layer = tf.layers.dense(
            last_layer, 300, activation=tf.nn.relu, name="fc2")
        output = tf.layers.dense(
            last_layer, num_outputs, activation=None, name="fc_out")
        return output, last_layer

        
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

# DQN and Apex-DQN do not work with continuous actions
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='RLlib')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--episodes-total', type=int, default=120000,
                        help='total number of episodes')
    parser.add_argument('--method', type=str, default="PPO",
                        help='name of RL method')

    args = parser.parse_args()

    methods = ["A2C", "APEX_DDPG", "DDPG", "IMPALA", "PPO", "SAC", "TD3"]
    
    # assert len(sys.argv) == 2, "Input the learning method as the second argument"
    method = args.method #sys.argv[1]
    # assert method in methods, "Method should be one of {}".format(methods)
    seed = args.seed
    num_gpus = 0
    num_workers = 5
    num_envs_per_worker = 5
    
    # ray.init()
    
    ModelCatalog.register_custom_model("MLPModel", MLPModel)
    
    # dd0
    def env_creator(args):
        # return dd0.env()
        env = Deepdrive2DEnv(is_intersection_map=True)  # for self-play to have 2 learning agents
        env.configure_env(env_config)
        return env

    env = env_creator(1)
    register_env("dd0", env_creator)
    
    obs_space = env.observation_space  # gym.spaces.Box(low=0, high=1, shape=(148,), dtype=np.float32)
    act_space = env.action_space  # gym.spaces.Discrete(5)
    
    def gen_policy(i):
        config = {
            "model": {
                "custom_model": "MLPModel",
            },
            "gamma": 0.99,
        }
        return (None, obs_space, act_space, config)
    
    policies = {"policy_0": gen_policy(0)}
    policy_ids = list(policies.keys())

    if method == "A2C":
        tune.run(
            "A2C",
            local_dir='./ray_results/dd0/',
            stop={"episodes_total": args.episodes_total},
            checkpoint_freq=10000000,
            config={
                "seed":seed,
                # Enviroment specific
                "env": "dd0",
        
                # General
                "log_level": "ERROR",
                "num_gpus": num_gpus,
                "num_workers": num_workers,
                "num_envs_per_worker": num_envs_per_worker,
                "compress_observations": False,
                # "train_batch_size": 20,
                "train_batch_size": 512,
                "gamma": .99,
        
                "lr_schedule": [[0, 0.0007],[20000000, 0.000000000001]],
        
                # Method specific
        
                # "multiagent": {
                #     "policies": policies,
                #     "policy_mapping_fn": (
                #         lambda agent_id: policy_ids[0]),
                # },
                # wandb configuration
                "logger_config" : {
                    "wandb": {
                        "project": "dd0",
                        # "api_key_file": "/path/to/file",
                        "log_config": True
                    }
                }
            },
            loggers=(WandbLogger, )
        )

    elif method == "APEX_DDPG":
        tune.run(
            "APEX_DDPG",
            local_dir='./ray_results/dd0/',
            stop={"episodes_total": args.episodes_total},
            checkpoint_freq=10000000,
            config={
                "seed":seed,
                
                # Enviroment specific
                "env": "dd0",
        
                # General
                "log_level": "ERROR",
                "num_gpus": num_gpus,
                "num_workers": num_workers,
                "num_envs_per_worker": num_envs_per_worker,
                "learning_starts": 1000,
                "buffer_size": int(1e5),
                "compress_observations": False, #True
                # "train_batch_size": 20,
                "train_batch_size": 512,
                "gamma": .99,
        
                "n_step": 3,
                "lr": .0001,
                "prioritized_replay_alpha": 0.5,
                # "beta_annealing_fraction": 1.0,
                "final_prioritized_replay_beta": 1.0,
                "target_network_update_freq": 5000,
                "timesteps_per_iteration": 2500,
                # "exploration_config": {"type": "PerWorkerEpsilonGreedy"},
                # Method specific
        
                # "multiagent": {
                #     "policies": policies,
                #     "policy_mapping_fn": (
                #         lambda agent_id: policy_ids[0]),
                # },
                # wandb configuration
                "logger_config" : {
                    "wandb": {
                        "project": "dd0",
                        # "api_key_file": "/path/to/file",
                        "log_config": True
                    }
                }
            },
            loggers=(WandbLogger, )
        )

    # plain DDPG
    elif method == "DDPG":
        tune.run(
            "DDPG",
            local_dir='./ray_results/dd0/',
            stop={"episodes_total": args.episodes_total},
            checkpoint_freq=10000000,
            config={
                "seed":seed,

                # Enviroment specific
                "env": "dd0",
                # General
                "log_level": "ERROR",
                "num_gpus": num_gpus,
                "num_workers": num_workers,
                "num_envs_per_worker": num_envs_per_worker,
                "learning_starts": 5000,
                "buffer_size": int(1e5),
                "compress_observations": True,
                # "train_batch_size": 20,
                "train_batch_size": 512,
                "gamma": .99,
                "critic_hiddens": [256, 256],
                # Method specific
                # "multiagent": {
                #     "policies": policies,
                #     "policy_mapping_fn": (
                #         lambda agent_id: policy_ids[0]),
                # },
                # wandb configuration
                "logger_config" : {
                    "wandb": {
                        "project": "dd0",
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
            local_dir='./ray_results/dd0/',
            stop={"episodes_total": args.episodes_total},
            checkpoint_freq=10000000,
            config={
                "seed":seed,
                
                # Enviroment specific
                "env": "dd0",
        
                # General
                "log_level": "ERROR",
                "num_gpus": num_gpus,
                "num_workers": num_workers,
                "num_envs_per_worker": num_envs_per_worker,
                "compress_observations": False,
                # "train_batch_size": 20,
                "train_batch_size": 5000,
                "gamma": .99,
        
                "clip_rewards": False,
                # "lr_schedule": [[0, 0.0005],[20000000, 0.000000000001]],
                "lr": 2e-4,
                # Method specific
        
                # "multiagent": {
                #     "policies": policies,
                #     "policy_mapping_fn": (
                #         lambda agent_id: policy_ids[0]),
                # },
                # wandb configuration
                "logger_config" : {
                    "wandb": {
                        "project": "dd0",
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
            local_dir='./ray_results/dd0/',
            stop={"episodes_total": args.episodes_total},
            checkpoint_freq=10000000,
            config={
                "seed":seed,
                
                # Enviroment specific
                "env": "dd0",
        
                # General
                "log_level": "ERROR",
                "num_gpus": num_gpus,
                "num_workers": num_workers,
                "num_envs_per_worker": num_envs_per_worker,
                "compress_observations": False,
                "gamma": .99,
        
                "lambda": 0.94,
                "kl_coeff": 0.5,
                "clip_rewards": False,
                "clip_param": 0.1,
                "vf_clip_param": 5.0,
                "entropy_coeff": 0.001,
                "train_batch_size": 5000,
                # "train_batch_size": 100,
                "sgd_minibatch_size": 500,
                "num_sgd_iter": 10,
                "batch_mode": 'truncate_episodes',
                "vf_share_layers": False,
                "lr": 1e-4,
        
                # Method specific
                # "multiagent": {
                #     "policies": policies,
                #     "policy_mapping_fn": (
                #         lambda agent_id: policy_ids[0]),
                # },
                # wandb configuration
                "logger_config" : {
                    "wandb": {
                        "project": "dd0",
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
            local_dir='./ray_results/dd0/',
            stop={"episodes_total": args.episodes_total},
            checkpoint_freq=10000000,
            config={
                "seed":seed,
                
                # Enviroment specific
                "env": "dd0",
        
                # General
                "log_level": "ERROR",
                "num_gpus": num_gpus,
                "num_workers": num_workers,
                "num_envs_per_worker": num_envs_per_worker,
                "learning_starts": 1000,
                "buffer_size": int(1e5),
                "compress_observations": True,
                # "train_batch_size": 20,
                "train_batch_size": 512,
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
                "n_step": 1,
                "prioritized_replay": False,
                "target_network_update_freq": 1,
                "timesteps_per_iteration": 1000,
                # "exploration_enabled": True,
                "optimization": {
                  "actor_learning_rate": 0.0003,
                  "critic_learning_rate": 0.0003,
                  "entropy_learning_rate": 0.0003,
                  },
                "clip_actions": False,
                #TODO -- True
                "normalize_actions": False,
                "evaluation_interval": 1,
                "metrics_smoothing_episodes": 5,
        
                # "multiagent": {
                #     "policies": policies,
                #     "policy_mapping_fn": (
                #         lambda agent_id: policy_ids[0]),
                # },
            
                # wandb configuration
                "logger_config" : {
                    "wandb": {
                        "project": "dd0",
                        # "api_key_file": "/path/to/file",
                        "log_config": True
                    }
                }
                },
            loggers=(WandbLogger, )
        )

    elif method == "TD3":
        tune.run(
            "TD3",
            local_dir='./ray_results/dd0/',
            stop={"episodes_total": args.episodes_total},
            checkpoint_freq=10000000,
            config={
                "seed":seed,
                
                # Enviroment specific
                "env": "dd0",
        
                # General
                "log_level": "ERROR",
                "num_gpus": num_gpus,
                "num_workers": num_workers,
                "num_envs_per_worker": num_envs_per_worker,
                "learning_starts": 5000,
                "buffer_size": int(1e5),
                "compress_observations": True,
                # "train_batch_size": 20,
                "train_batch_size": 512,
                "gamma": .99,
        
                "critic_hiddens": [256, 256],
                # "pure_exploration_steps": 5000,
        
                # Method specific
        
                # "multiagent": {
                #     "policies": policies,
                #     "policy_mapping_fn": (
                #         lambda agent_id: policy_ids[0]),
                # },
                # wandb configuration
                "logger_config" : {
                    "wandb": {
                        "project": "dd0",
                        # "api_key_file": "/path/to/file",
                        "log_config": True
                    }
                }
            },
            loggers=(WandbLogger, )
        )