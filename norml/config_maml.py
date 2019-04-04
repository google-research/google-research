# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configurations for MAML training (Reinforcement Learning).

See maml_rl.py for usage examples.
An easy task to get started with is: RL_MINITAUR_POINT_CONFIG_CIRCLE.
"""

# b/128310658.
# pytype: skip-file

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import random
import numpy as np
import tensorflow as tf

from norml import networks
from norml import policies
from norml.envs import cartpole_sensor_bias_env
from norml.envs import halfcheetah_motor_env
from norml.envs import move_point_env


def _early_termination_avg(rewards, num_steps, avg_reward):
  """Early termination based on average reward."""
  flat_reward = np.array(rewards).ravel()
  len_ok = len(flat_reward) >= num_steps
  val_ok = np.mean(flat_reward[-num_steps:]) >= avg_reward
  return len_ok and val_ok

MOVE_POINT_ROTATE_MAML = dict(
    random_seed=random.randint(0, 1000000),
    num_outer_iterations=1000,
    task_generator=functools.partial(
        move_point_env.MovePointEnv,
        start_pos=(0, 0),
        end_pos=(1, 0),
        goal_reached_distance=-1,
        trial_length=10),
    task_env_modifiers=[{
        '_action_rotation': i
    } for i in np.linspace(-np.pi, np.pi, 5000)],
    network_generator=networks.FullyConnectedNetworkGenerator(
        dim_input=2,
        dim_output=2,
        layer_sizes=(
            50,
            50,
        ),
        activation_fn=tf.nn.tanh),
    input_dims=2,
    pol_log_std_init=-3.,
    output_dims=2,
    reward_disc=0.9,
    learn_offset=False,
    policy=policies.GaussianPolicy,
    tasks_batch_size=10,
    num_inner_rollouts=25,
    outer_optimizer_algo=tf.train.AdamOptimizer,
    advantage_function='returns-values',
    whiten_values=False,
    always_full_rollouts=False,
    inner_lr_init=0.02,
    outer_lr_init=7e-3,
    outer_lr_decay=True,
    first_order=False,
    learn_inner_lr=True,
    learn_inner_lr_tensor=True,
    fixed_tasks=False,
    ppo=True,
    ppo_clip_value=0.2,
    max_num_batch_env=1000,
    max_rollout_len=10,
    log_every=10,
)

MOVE_POINT_ROTATE_MAML_OFFSET = MOVE_POINT_ROTATE_MAML.copy()
MOVE_POINT_ROTATE_MAML_OFFSET.update(
    learn_offset=True,
    inner_lr_init=0.1,
    outer_lr_init=3e-3,
    pol_log_std_init=-3.)

MOVE_POINT_ROTATE_MAML_LAF = MOVE_POINT_ROTATE_MAML.copy()
MOVE_POINT_ROTATE_MAML_LAF.update(
    learn_inner_lr=False,
    learn_inner_lr_tensor=False,
    learn_advantage_function_inner=True,
    advantage_generator=networks.FullyConnectedNetworkGenerator(
        dim_input=2 * 2 + 2,
        dim_output=1,
        layer_sizes=(
            50,
            50,
        ),
        activation_fn=tf.nn.tanh),
    inner_lr_init=0.7,
    outer_lr_init=6e-4,
    pol_log_std_init=-3.25)

MOVE_POINT_ROTATE_NORML = MOVE_POINT_ROTATE_MAML_LAF.copy()
MOVE_POINT_ROTATE_NORML.update(
    learn_offset=True,
    inner_lr_init=10.,
    outer_lr_init=6e-3,
    pol_log_std_init=-0.75)

MOVE_POINT_ROTATE_SPARSE_MAML = MOVE_POINT_ROTATE_MAML.copy()
MOVE_POINT_ROTATE_SPARSE_MAML.update(
    max_rollout_len=100,
    task_generator=functools.partial(
        move_point_env.MovePointEnv,
        start_pos=(0, 0),
        end_pos=(1, 0),
        goal_reached_distance=0.1,
        trial_length=100,
        sparse_reward=True),
    inner_lr_init=1e-4,
    outer_lr_init=2e-3,
    pol_log_std_init=-1.25)

MOVE_POINT_ROTATE_SPARSE_MAML_OFFSET = MOVE_POINT_ROTATE_MAML_OFFSET.copy()
MOVE_POINT_ROTATE_SPARSE_MAML_OFFSET.update(
    max_rollout_len=100,
    task_generator=functools.partial(
        move_point_env.MovePointEnv,
        start_pos=(0, 0),
        end_pos=(1, 0),
        goal_reached_distance=0.1,
        trial_length=100,
        sparse_reward=True),
    inner_lr_init=7.,
    outer_lr_init=2e-3,
    pol_log_std_init=-0.5)

MOVE_POINT_ROTATE_SPARSE_MAML_LAF = MOVE_POINT_ROTATE_MAML_LAF.copy()
MOVE_POINT_ROTATE_SPARSE_MAML_LAF.update(
    max_rollout_len=100,
    task_generator=functools.partial(
        move_point_env.MovePointEnv,
        start_pos=(0, 0),
        end_pos=(1, 0),
        goal_reached_distance=0.1,
        trial_length=100,
        sparse_reward=True),
    inner_lr_init=2e-5,
    outer_lr_init=1e-3,
    pol_log_std_init=-1.)

MOVE_POINT_ROTATE_SPARSE_NORML = MOVE_POINT_ROTATE_NORML.copy()
MOVE_POINT_ROTATE_SPARSE_NORML.update(
    max_rollout_len=100,
    task_generator=functools.partial(
        move_point_env.MovePointEnv,
        start_pos=(0, 0),
        end_pos=(1, 0),
        goal_reached_distance=0.1,
        trial_length=100,
        sparse_reward=True),
    inner_lr_init=9.,
    outer_lr_init=2.6e-3,
    pol_log_std_init=-0.6)

CARTPOLE_SENSOR_DR = dict(
    random_seed=random.randint(0, 1000000),
    num_outer_iterations=1000,
    task_generator=functools.partial(
        cartpole_sensor_bias_env.CartpoleSensorBiasEnv),
    task_env_modifiers=[{
        '_angle_observation_bias': theta
    } for theta in np.linspace(-np.pi / 18, np.pi / 18, 5000)],
    network_generator=networks.FullyConnectedNetworkGenerator(
        dim_input=4,
        dim_output=1,
        layer_sizes=(
            50,
            50,
        ),
        activation_fn=tf.nn.tanh),
    input_dims=4,
    pol_log_std_init=-4.0,
    output_dims=1,
    reward_disc=0.97,
    learn_offset=False,
    policy=policies.GaussianPolicy,
    tasks_batch_size=10,
    num_inner_rollouts=25,
    outer_optimizer_algo=tf.train.AdamOptimizer,
    advantage_function='returns-values',
    whiten_values=False,
    always_full_rollouts=False,
    inner_lr_init=0.,
    outer_lr_init=2e-4,
    outer_lr_decay=True,
    first_order=False,
    learn_inner_lr=False,
    learn_inner_lr_tensor=False,
    fixed_tasks=False,
    ppo=True,
    ppo_clip_value=0.2,
    max_num_batch_env=1000,
    max_rollout_len=500,
    log_every=10,
)

CARTPOLE_SENSOR_MAML = CARTPOLE_SENSOR_DR.copy()
CARTPOLE_SENSOR_MAML.update(
    learn_inner_lr=True,
    learn_inner_lr_tensor=True,
    inner_lr_init=1e-2,
    outer_lr_init=1e-2,
    pol_log_std_init=-0.5)

CARTPOLE_SENSOR_MAML_OFFSET = CARTPOLE_SENSOR_MAML.copy()
CARTPOLE_SENSOR_MAML_OFFSET.update(
    learn_offset=True,
    inner_lr_init=1e-1,
    outer_lr_init=4e-4,
    pol_log_std_init=-3.5)

CARTPOLE_SENSOR_MAML_LAF = CARTPOLE_SENSOR_MAML.copy()
CARTPOLE_SENSOR_MAML_LAF.update(
    learn_advantage_function_inner=True,
    advantage_generator=networks.FullyConnectedNetworkGenerator(
        dim_input=2 * 4 + 1,
        dim_output=1,
        layer_sizes=(
            50,
            50,
        ),
        activation_fn=tf.nn.tanh),
    inner_lr_init=1e-5,
    outer_lr_init=3e-3,
    pol_log_std_init=-0.5)

CARTPOLE_SENSOR_NORML = CARTPOLE_SENSOR_MAML_LAF.copy()
CARTPOLE_SENSOR_NORML.update(
    learn_offset=True,
    inner_lr_init=7e-4,
    outer_lr_init=3e-4,
    pol_log_std_init=-3.5)

HALFCHEETAH_MOTOR_DR = dict(
    random_seed=random.randint(0, 1000000),
    num_outer_iterations=1000,
    task_generator=functools.partial(halfcheetah_motor_env.HalfcheetahMotorEnv),
    task_env_modifiers=[{
        '_swap_action': True
    }, {
        '_swap_action': False
    }] * 2,
    network_generator=networks.FullyConnectedNetworkGenerator(
        dim_input=14,
        dim_output=6,
        layer_sizes=(100,),
        activation_fn=tf.identity),
    pol_log_std_init=-1.61,
    input_dims=14,
    output_dims=6,
    reward_disc=0.99,
    learn_offset=False,
    policy=policies.GaussianPolicy,
    tasks_batch_size=4,
    num_inner_rollouts=50,
    outer_optimizer_algo=tf.train.AdamOptimizer,
    advantage_function='returns-values',
    whiten_values=True,
    always_full_rollouts=False,  # also learn from failures?
    inner_lr_init=0.,
    outer_lr_init=0.0012,
    outer_lr_decay=True,
    first_order=False,
    learn_inner_lr=False,
    learn_inner_lr_tensor=False,
    fixed_tasks=False,
    log_every=10,
    ppo=True,
    ppo_clip_value=0.2,
    max_rollout_len=1000,
    max_num_batch_env=300,
)

HALFCHEETAH_MOTOR_MAML = HALFCHEETAH_MOTOR_DR.copy()
HALFCHEETAH_MOTOR_MAML.update(
    learn_inner_lr=True,
    learn_inner_lr_tensor=True,
    learn_offset=False,
    inner_lr_init=4.5e-4,
    outer_lr_init=8.7e-5,
    pol_log_std_init=-1.17)

HALFCHEETAH_MOTOR_MAML_OFFSET = HALFCHEETAH_MOTOR_MAML.copy()
HALFCHEETAH_MOTOR_MAML_OFFSET.update(
    learn_offset=True,
    inner_lr_init=5e-5,
    outer_lr_init=1.5e-4,
    pol_log_std_init=-0.8)

HALFCHEETAH_MOTOR_MAML_LAF = HALFCHEETAH_MOTOR_MAML.copy()
HALFCHEETAH_MOTOR_MAML_LAF.update(
    learn_advantage_function_inner=True,
    advantage_generator=networks.FullyConnectedNetworkGenerator(
        dim_input=14 * 2 + 6,
        dim_output=1,
        layer_sizes=(50,),
        activation_fn=tf.nn.relu),
    learn_inner_lr=False,
    learn_inner_lr_tensor=False,
    inner_lr_init=3e-5,
    outer_lr_init=5e-4,
    pol_log_std_init=-1.5)

HALFCHEETAH_MOTOR_NORML = HALFCHEETAH_MOTOR_MAML_LAF.copy()
HALFCHEETAH_MOTOR_NORML.update(
    learn_offset=True,
    inner_lr_init=3e-5,
    outer_lr_init=5e-4,
    pol_log_std_init=-1.9)
