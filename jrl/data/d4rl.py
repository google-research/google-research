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

"""Create the data iterators for D4RL."""

import collections
import typing

import gym
import numpy as np
import tensorflow.data as tf_data

from jrl.data import d4rl_get_dataset


Inputs = collections.namedtuple('inputs', ['data'])


def create_d4rl_data_iter(
    task_name,
    batch_size
):
  """Create gym environment and dataset for d4rl.

  Args:
    task_name: Name of d4rl task.
    batch_size: Mini batch size.
  Returns:
    dataset iterator.
  """
  # INCLUDE_NEXT_ACTS = False
  INCLUDE_NEXT_ACTS = True

  env = gym.make(task_name)
  if not INCLUDE_NEXT_ACTS:
    dataset = d4rl_get_dataset.qlearning_dataset(env)
  else:
    dataset = d4rl_get_dataset.qlearning_dataset(env, include_next_actions=True)
  del env

  states = np.array(dataset['observations'], dtype=np.float32)
  actions = np.array(dataset['actions'], dtype=np.float32)
  rewards = np.array(dataset['rewards'], dtype=np.float32)

  # For antmaze CQL etc. modify the rewards like this
  if 'antmaze' in task_name:
    # rewards = 2. * rewards - 1.
    rewards = 4. * (rewards - 0.5) # same as cql github repo

  discounts = np.array(np.logical_not(dataset['terminals']), dtype=np.float32)
  next_states = np.array(dataset['next_observations'], dtype=np.float32)
  if INCLUDE_NEXT_ACTS:
    next_actions = np.array(dataset['next_actions'], dtype=np.float32)

  if not INCLUDE_NEXT_ACTS:
    tensors = Inputs(data=(states, actions, rewards, discounts, next_states))
  else:
    tensors = Inputs(data=(states, actions, rewards, discounts, next_states, next_actions))

  dataset = tf_data.Dataset.from_tensor_slices(
            tensors
      ).cache().shuffle(
          states.shape[0], reshuffle_each_iteration=True
          ).repeat().batch(
            batch_size, drop_remainder=True
          ).prefetch(tf_data.experimental.AUTOTUNE)
  return dataset.as_numpy_iterator()
