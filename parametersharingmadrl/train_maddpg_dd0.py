import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import gym
import os

import sys
sys.path.append('maddpg/')

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

# import sys
# sys.path.append('..')

from sisl_games.pursuit import pursuit
from sisl_games.waterworld import waterworld
from sisl_games.multiwalker import multiwalker

from deepdrive_zero.envs.env import Deepdrive2DEnv
from deepdrive_zero.discrete.comfortable_actions import COMFORTABLE_ACTIONS
import wandb

from vector_env import VectorEnv, ParallelVectorEnv, init_parallel_env


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

wandb.init(
    project='dd0', 
    # entity=args.wandb_entity, 
    config=env_config, 
    group="MADDPG", 
    name="MADDPG", 
    # monitor_gym=True, 
    # save_code=True
)

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="multiwalker", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=5000, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=250000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    parser.add_argument("--seed", type=int, default=3)

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-envs", type=int, default=1, help="number of parallel environments")
    parser.add_argument("--num-cpus", type=int, default=1, help="number of parallel environments")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="main_experiment", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


class MAWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space_dict = {0: env.observation_space, 1:env.observation_space}
        self.action_space_dict = {0:env.action_space, 1:env.action_space}
        self.agent_ids = [0, 1]
        self.num_agents = len(self.agent_ids)

    def reset(self):
        obs = np.expand_dims(self.env.reset(), 0)
        return [obs]*2
        
    def step(self, action_dict):
        # total_reward = 0.0
        obses = []
        dones = []
        rewards = []
        infos = []
        for i in range(2):
            obs, reward, done, info = self.env.step(np.argmax(action_dict[i]))
            obses.append(np.expand_dims(obs, 0))
            dones.append(done)
            rewards.append(reward)
            infos.append(info)
            # total_reward += reward
            # if done:
            #     break

        return obses, [sum(rewards)]*2, dones, infos


def make_env(scenario_name, arglist):
    #env = scenario.env()
    env_config['seed_value'] = arglist.seed
    env = Deepdrive2DEnv(is_intersection_map=True)  # for self-play to have 2 learning agents
    env.configure_env(env_config)
    env = MAWrapper(env)
    # env = ParallelVectorEnv(Deepdrive2DEnv, arglist.num_envs, arglist.num_cpus)
    # env = VectorEnv(scenario.env, arglist.num_envs)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    action_spaces = [env.action_space_dict[p] for p in env.agent_ids]
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, action_spaces, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.num_agents):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, action_spaces, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers

def train(arglist):
    init_parallel_env()
    env = make_env(arglist.scenario, arglist)
    with U.single_threaded_session():
        # Create environment
        # Create agent trainers
        obs_shape_n = [env.observation_space_dict[i].shape for i in env.agent_ids]
        num_adversaries = min(env.num_agents, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.num_agents)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        print('Starting iterations...')
        while True:
            # get action
            action_vec = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # environment step
            new_obs_ns, reward_ns, dones, infos_ns = env.step(action_vec)
            episode_step += 1
            all_done = any(dones)
            terminal = (episode_step >= arglist.max_episode_len) or all_done

            # collect experience
            # for j in range(env.num_envs):
            done = all_done
            for i, agent in enumerate(trainers):
                rew = reward_ns[i]
                #print(rew,done)
                if not done or rew != 0:
                    agent.experience(obs_n[i], action_vec[i], rew, new_obs_ns[i], dones[i], terminal)

                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            # increment global step counter
            train_step += 1

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            #for e in range(env.num_envs):
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                    wandb.log({'episode_reward_mean': np.mean(episode_rewards[-arglist.save_rate:]), 'episodes_total': len(episode_rewards)})
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards)*arglist.num_envs > arglist.num_episodes:
                os.makedirs(arglist.plots_dir,exist_ok=True)
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

            obs_n = new_obs_ns
            if all_done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)

    env.close()

if __name__ == '__main__':
    
    arglist = parse_args()
    train(arglist)
