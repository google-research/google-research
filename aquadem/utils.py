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

"""AQuaDem utils."""

from typing import Callable, Iterator

import acme
from acme import wrappers
from acme.datasets import tfds as acme_tfds
from acme.datasets.tfds import _episode_steps_to_transition
import d4rl  # pylint:disable=unused-import
import dm_env
import gym
import jax
import rlds
from rlds import rlds_types
import tensorflow as tf
import tensorflow_datasets as tfds

from aquadem import wrappers as env_wrappers


# Sparse rewards thresholds indicates whether the goals of Adroit tasks are
# achieved e.g. if r > threshold than goal_achieved = 1. else 0.
SPARSE_REWARD_THRESHOLDS = {'door': 15, 'hammer': 50, 'pen': 30, 'relocate': 5}


def make_environment(task, evaluation = False):
  """Creates an OpenAI Gym environment."""

  # Load the gym environment.
  environment = gym.make(task)

  environment = env_wrappers.AdroitSparseRewardWrapper(environment)

  # Make sure the environment obeys the dm_env.Environment interface.
  environment = wrappers.GymWrapper(environment)
  # Clip the action returned by the agent to the environment spec.
  environment = wrappers.CanonicalSpecWrapper(environment, clip=True)
  environment = wrappers.SinglePrecisionWrapper(environment)

  if evaluation:
    environment = env_wrappers.SuccessRewardWrapper(environment,
                                                    success_threshold=1.)

  return environment


def get_make_demonstrations_fn(
    env_name, num_demonstrations, seed = 0
):
  """Returns the demonstrations to be passed to the builder."""
  tfds_dataset_id = _d4rl_dataset_name(env_name)

  episodes = tfds.load(tfds_dataset_id)['train']
  if num_demonstrations:
    episodes = episodes.take(num_demonstrations)

  task_id = env_name.split('-')[0]  # e.g. task_id := 'door'
  assert task_id in SPARSE_REWARD_THRESHOLDS
  task_success_threshold = SPARSE_REWARD_THRESHOLDS[task_id]

  def sparsify_reward(step):
    reward = step[rlds_types.REWARD]
    sparse_reward = tf.cast(reward > task_success_threshold, tf.float32)
    step[rlds_types.REWARD] = sparse_reward
    return step

  episodes = rlds.transformations.map_nested_steps(episodes, sparsify_reward)
  transitions_iterator = episodes.flat_map(_episode_steps_to_transition)

  def make_demonstrations(batch_size):
    return acme_tfds.JaxInMemoryRandomSampleIterator(
        transitions_iterator, jax.random.PRNGKey(seed), batch_size)
  return make_demonstrations


def _d4rl_dataset_name(env_name):
  """Obtains the TFDS D4RL name."""
  split_name = env_name.split('-')
  task = split_name[0]
  version = split_name[-1]
  level = '-'.join(split_name[1:-1])
  return f'd4rl_adroit_{task}/{version}-{level}'
