# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Helper functions for PWIL training script."""

import os
import pickle

from acme import wrappers
from acme.tf import networks
from acme.tf import utils as tf2_utils
import dm_env
import gym
from gym.wrappers.time_limit import TimeLimit
import numpy as np
import sonnet as snt
import tensorflow as tf



def make_d4pg_networks(
    action_spec,
    policy_layer_sizes=(256, 256, 256),
    critic_layer_sizes=(512, 512, 256),
    vmin=-150.,
    vmax=150.,
    num_atoms=201):
  """Creates networks used by the d4pg agent."""

  num_dimensions = np.prod(action_spec.shape, dtype=int)
  policy_layer_sizes = list(policy_layer_sizes) + [int(num_dimensions)]

  policy_network = snt.Sequential([
      networks.LayerNormMLP(policy_layer_sizes),
      networks.TanhToSpec(action_spec)
  ])

  # The multiplexer concatenates the (maybe transformed) observations/actions.
  critic_network = snt.Sequential([
      networks.CriticMultiplexer(
          critic_network=networks.LayerNormMLP(
              critic_layer_sizes, activate_final=True)),
      networks.DiscreteValuedHead(vmin, vmax, num_atoms)
  ])

  return {
      'policy': policy_network,
      'critic': critic_network,
      'observation': tf2_utils.batch_concat,
  }


def load_demonstrations(demo_dir, env_name):
  """Load expert demonstrations.

  Outputs come with the following format:
    [
      [{observation: o_1, action: a_1}, ...], # episode 1
      [{observation: o'_1, action: a'_1}, ...], # episode 2
      ...
    ]

  Args:
    demo_dir: directory path of expert demonstrations
    env_name: name of the environment

  Returns:
    demonstrations: list of expert demonstrations
  """
  demonstrations_filename = os.path.join(demo_dir, '{}.pkl'.format(env_name))
  demonstrations_file = tf.io.gfile.GFile(demonstrations_filename, 'rb')
  demonstrations = pickle.load(demonstrations_file)
  return demonstrations


def load_environment(env_name):
  """Outputs a wrapped gym environment."""
  environment = gym.make(env_name)
  environment = TimeLimit(environment, max_episode_steps=1000)
  environment = wrappers.gym_wrapper.GymWrapper(environment)
  environment = wrappers.SinglePrecisionWrapper(environment)
  return environment


def prefill_rb_with_demonstrations(agent, demonstrations,
                                   num_transitions_rb, reward):
  """Fill the agent's replay buffer with expert transitions."""
  num_demonstrations = len(demonstrations)
  for _ in range(num_transitions_rb // num_demonstrations):
    for i in range(num_demonstrations):
      transition = demonstrations[i]
      observation = transition['observation']
      step_type = transition['step_type']
      discount = 1.0

      ts = dm_env.TimeStep(step_type, reward, discount, observation)
      ts = wrappers.single_precision._convert_value(ts)  # pylint:disable=protected-access

      if step_type == dm_env.StepType.FIRST:
        agent.observe_first(ts)
      else:
        action = demonstrations[i-1]['action']
        # We take the previous action to comply with acme's api.
        agent.observe(action, ts)
