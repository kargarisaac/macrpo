import os
import sys

from deepdrive_zero.constants import FPS
from deepdrive_zero.experiments import utils
from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch
import torch

experiment_name = os.path.basename(__file__)[:-3]
notes = """Testing garbage-in-garbage-out (some things are worth forgetting) 
 idea with single waypoint"""


env_config = dict(
    env_name='deepdrive-2d-one-waypoint-v0',
    is_intersection_map=True,
    expect_normalized_action_deltas=False,
    jerk_penalty_coeff=3.3e-5,
    gforce_penalty_coeff=0.006 * 5,
    collision_penalty_coeff=4,
    lane_penalty_coeff=0.02,
    speed_reward_coeff=0.50,
    gforce_threshold=None,
    incent_win=True,
    constrain_controls=False,
    incent_yield_to_oncoming_traffic=True,
    physics_steps_per_observation=12,
)

net_config = dict(
    hidden_units=(64, 64),
    activation=torch.nn.Tanh
)

eg = ExperimentGrid(name=experiment_name)
eg.add('env_name', env_config['env_name'], '', False)
# eg.add('seed', 0)
# eg.add('resume', '/home/c2/src/tmp/spinningup/data/intersection_2_agents_fine_tune_add_left_yield2/intersection_2_agents_fine_tune_add_left_yield2_s0_2020_03-23_22-40.11')
# eg.add('reinitialize_optimizer_on_resume', True)
# eg.add('num_inputs_to_add', 0)
# eg.add('pi_lr', 3e-6)
# eg.add('vf_lr', 1e-5)
# eg.add('boost_explore', 5)
pso = env_config['physics_steps_per_observation']
effective_horizon_seconds = 10
eg.add('gamma', 0)
# eg.add('episode_cull_ratio', 0)
eg.add('epochs', 20000)
eg.add('steps_per_epoch', 500)
eg.add('ac_kwargs:hidden_sizes', net_config['hidden_units'], 'hid')
eg.add('ac_kwargs:activation', net_config['activation'], '')
eg.add('notes', notes, '')
eg.add('run_filename', os.path.realpath(__file__), '')
eg.add('env_config', env_config, '')

def train():
    eg.run(ppo_pytorch)


if __name__ == '__main__':
    utils.run(train_fn=train, env_config=env_config, net_config=net_config)