# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Load D4RL data and envs
"""

import typing
from typing import Mapping, Sequence
import collections

import tensorflow.data as tf_data
import numpy as np

import d4rl
import gym
import dm_env
from acme import wrappers


Inputs = collections.namedtuple('inputs', ['data'])


def create_d4rl_env_and_dataset(
    task_name,
    batch_size
):
  """Create gym environment and dataset for d4rl.

  Args:
    task_name: Name of d4rl task.
    batch_size: Mini batch size.
  Returns:
    Gym env and dataset.
  """
  env = gym.make(task_name)
  env = wrappers.GymWrapper(env)
  dataset = d4rl.qlearning_dataset(env)

  states = np.array(dataset['observations'], dtype=np.float32)
  actions = np.array(dataset['actions'], dtype=np.float32)
  rewards = np.array(dataset['rewards'], dtype=np.float32)
  discounts = np.array(np.logical_not(dataset['terminals']), dtype=np.float32)
  next_states = np.array(dataset['next_observations'], dtype=np.float32)

  dataset = tf_data.Dataset.from_tensor_slices(
      Inputs(data=(states, actions, rewards, discounts, next_states))
      ).cache().shuffle(
          states.shape[0], reshuffle_each_iteration=True
          ).repeat().batch(
            batch_size, drop_remainder=True
          ).prefetch(tf_data.experimental.AUTOTUNE)
  return env, dataset
