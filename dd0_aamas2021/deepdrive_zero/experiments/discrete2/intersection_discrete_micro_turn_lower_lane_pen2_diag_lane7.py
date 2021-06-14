import os
import sys

from deepdrive_zero.discrete.comfortable_actions2 import COMFORTABLE_ACTIONS2
from deepdrive_zero.experiments import utils
from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch
import torch

experiment_name = os.path.basename(__file__)[:-3]
notes = """Reduce lane margin to zero, too difficult in intersection, and
speed needs to be too high to overcome negative reward when sitting perfectly
in lane at low speeds"""

env_config = dict(
    env_name='deepdrive-2d-intersection-w-gs-allow-decel-v0',
    is_intersection_map=True,
    expect_normalized_action_deltas=False,
    jerk_penalty_coeff=3.3e-4,
    gforce_penalty_coeff=0.006 * 5,
    collision_penalty_coeff=4,
    lane_penalty_coeff=0.02,
    speed_reward_coeff=0.50,
    gforce_threshold=1.0,
    end_on_lane_violation=False,

    # https://iopscience.iop.org/article/10.1088/0143-0807/37/6/065008/pdf
    # Importantly they depict the threshold
    # for admissible acceleration onset or jerk as j = 15g/s or ~150m/s^3.
    jerk_threshold=150.0,  # 15g/s
    incent_win=True,
    constrain_controls=False,
    incent_yield_to_oncoming_traffic=True,
    physics_steps_per_observation=12,
    discrete_actions=COMFORTABLE_ACTIONS2,
)

net_config = dict(
    hidden_units=(256, 256),
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
eg.add('epochs', 20000)
eg.add('steps_per_epoch', 4000)
eg.add('ac_kwargs:hidden_sizes', net_config['hidden_units'], 'hid')
eg.add('ac_kwargs:activation', net_config['activation'], '')
eg.add('notes', notes, '')
eg.add('run_filename', os.path.realpath(__file__), '')
eg.add('env_config', env_config, '')

def train():
    eg.run(ppo_pytorch)


if __name__ == '__main__':
    utils.run(train_fn=train, env_config=env_config, net_config=net_config)