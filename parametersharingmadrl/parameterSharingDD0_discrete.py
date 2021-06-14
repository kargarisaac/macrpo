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

from deepdrive_zero.envs.env import Deepdrive2DEnv
from deepdrive_zero.discrete.comfortable_actions import COMFORTABLE_ACTIONS
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

class MLPModelV2(Model):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name="my_model"):
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        # Simplified to one layer.
        input = tf.keras.layers.Input(obs_space.shape, dtype=obs_space.dtype)
        output = tf.keras.layers.Dense(num_outputs, activation=None)
        self.base_model = tf.keras.models.Sequential([input, output])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        return self.base_model(input_dict["obs"]), []

class MAWrapper(MultiAgentEnv):
    def __init__(self, env):
        # super().__init__(env)
        self.env = env
        self.observation_space = {0:env.observation_space, 1:env.observation_space}
        self.action_space = {0:env.action_space, 1:env.action_space}
        self.agent_ids = [0, 1]
        self.num_agents = len(self.agent_ids)

    def reset(self):
        # obs = np.expand_dims(self.env.reset(), 0)
        obs = self.env.reset()
        return {0:obs, 1:obs}
        
    def step(self, action_dict):
        # total_reward = 0.0
        obses = dict()
        dones = dict()
        rewards = dict()
        infos = dict()

        for i in range(2):
            obs, reward, done, info = self.env.step(action_dict[i])
            obses[i] = obs #np.expand_dims(obs, 0)
            dones[i] = done
            rewards[i] = reward
            infos[i] = info
            # total_reward += reward
            # if done:
            #     break
        dones['__all__'] = dones[0] or dones[1]

        # print('_'*20)
        # print(obses)
        # print('_'*20)

        # return obses, {0:sum(rewards), 1:sum(rewards)}, dones, infos
        return obses, rewards, dones, infos


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
        discrete_actions=COMFORTABLE_ACTIONS,
        # end_on_lane_violation=True
    )

# DQN and Apex-DQN do not work with continuous actions
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='RLlib')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--episodes-total', type=int, default=250000,
                        help='total number of episodes')
    parser.add_argument('--method', type=str, default="QMIX",
                        help='name of RL method')

    args = parser.parse_args()

    # assert len(sys.argv) == 2, "Input the learning method as the second argument"
    method = args.method #sys.argv[1]
    # assert method in methods, "Method should be one of {}".format(methods)
    seed = args.seed
    num_gpus = 0
    num_workers = 8
    num_envs_per_worker = 8
    
    # dd0
    def env_creator(args):
        # return dd0.env()
        env = Deepdrive2DEnv(is_intersection_map=True)  # for self-play to have 2 learning agents
        env.configure_env(env_config)
        if method == "QMIX":
            env = MAWrapper(env)
        return env

    env = env_creator(1)
    if method == "QMIX":
        grouping = {
            "group_1": [0, 1],
        }
        obs_space = Tuple([
            env.observation_space[0],
            env.observation_space[0],
        ])
        act_space = Tuple([
            env.action_space[0],
            env.action_space[0],
        ])
        register_env(
            "dd0",
            lambda config: env_creator(1).with_agent_groups(
                grouping, obs_space=obs_space, act_space=act_space))

    else:
        register_env("dd0", env_creator)
        obs_space = env.observation_space 
        act_space = env.action_space  
      
    # ray.init()
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN

    if method in ["ADQN", "RDQN"]:
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
        ModelCatalog.register_custom_model("MLPModel", MLPModel)
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

    if method == "ADQN":
        # APEX-DQN
        tune.run(
            "APEX",
            local_dir='./ray_results/dd0/',
            stop={"episodes_total": args.episodes_total},
            checkpoint_freq=10000000,
            config={
        
                # Enviroment specific
                "env": "dd0",
        
                # General
                "log_level": "INFO",
                "num_gpus": num_gpus,
                "num_workers": num_workers,
                "num_envs_per_worker": num_envs_per_worker,
                "learning_starts": 1000,
                "buffer_size": int(1e5),
                "compress_observations": False,
                # "sample_batch_size": 20,
                "train_batch_size": 32,
                "gamma": .99,
        
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

    elif method == "DQN":
        # plain DQN
        tune.run(
            "DQN", 
            local_dir='./ray_results/dd0/',
            stop={"episodes_total": args.episodes_total},
            checkpoint_freq=10000000,
            config={
                # Enviroment specific
                "env": "dd0",
                # General
                "log_level": "ERROR",
                "num_gpus": num_gpus,
                "num_workers": num_workers,
                "num_envs_per_worker": num_envs_per_worker,
                "learning_starts": 1000,
                "buffer_size": int(1e5),
                "compress_observations": False,
                # "sample_batch_size": 20,
                "train_batch_size": 32,
                "gamma": .99,
                # Method specific
                "dueling": False,
                "double_q": False,
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

    # psuedo-rainbow DQN
    elif method == "RDQN":
        tune.run(
            "DQN",
            local_dir='./ray_results/dd0/',
            stop={"episodes_total": args.episodes_total},
            checkpoint_freq=10000000,
            config={
        
                # Enviroment specific
                "env": "dd0",
        
                # General
                "log_level": "ERROR",
                "num_gpus": num_gpus,
                "num_workers": num_workers,
                "num_envs_per_worker": num_envs_per_worker,
                "learning_starts": 1000,
                "buffer_size": int(1e5),
                "compress_observations": False,
                # "sample_batch_size": 20,
                "train_batch_size": 32,
                "gamma": .99,
        
                # Method specific
                "num_atoms": 51,
                "dueling": True,
                "double_q": True,
                "n_step": 2,
                "batch_mode": "complete_episodes",
                "prioritized_replay": True,

                # # alternative 1
                # "noisy": True,
                # alternative 2
                # "parameter_noise": True,

                # based on expected return
                "v_min": 0,
                "v_max": 1500,
        
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

    elif method == "QMIX":
        tune.run(
            "QMIX",
            local_dir='./ray_results/dd0/',
            stop={"episodes_total": args.episodes_total},
            checkpoint_freq=10000000,
            config={
        
                # Enviroment specific
                "env": "dd0",
        
                # General
                "log_level": "ERROR",
                "num_gpus": num_gpus,
                "num_workers": num_workers,
                "num_envs_per_worker": num_envs_per_worker,
                "learning_starts": 1000,
                "buffer_size": int(10e3),
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
                    "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.

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
                "evaluation_num_episodes": 1,
                # Switch to greedy actions in evaluation workers.
                "evaluation_config": {
                    "explore": False,
                },

                # Number of env steps to optimize for before returning
                "timesteps_per_iteration": 2000,
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
                        "project": "dd0",
                        # "api_key_file": "/path/to/file",
                        "log_config": True
                    }
                }
            },
            loggers=(WandbLogger, )
        )