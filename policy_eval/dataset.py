# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Utilities for loading data."""
import pickle
import typing

import numpy as np
import tensorflow as tf


def augment_data(dataset,
                 noise_scale):
  """Augments the data.

  Args:
    dataset: Dictionary with data.
    noise_scale: Scale of noise to apply.

  Returns:
    Augmented data.
  """
  noise_std = np.std(np.concatenate(dataset['rewards'], 0))
  for k, v in dataset.items():
    dataset[k] = np.repeat(v, 3, 0)

  dataset['rewards'][1::3] += noise_std * noise_scale
  dataset['rewards'][2::3] -= noise_std * noise_scale

  return dataset


def weighted_moments(x, weights):
  mean = np.sum(x * weights, 0) / np.sum(weights)
  sqr_diff = np.sum((x - mean)**2 * weights, 0)
  std = np.sqrt(sqr_diff / (weights.sum() - 1))
  return mean, std


class Dataset(object):
  """Dataset class for policy evaluation."""

  def __init__(self,
               data_file_name,
               num_trajectories,
               normalize_states = False,
               normalize_rewards = False,
               eps = 1e-5,
               noise_scale = 0.0,
               bootstrap = True):
    """Loads data from a file.

    Args:
      data_file_name: filename with data.
      num_trajectories: number of trajectories to select from the dataset.
      normalize_states: whether to normalize the states.
      normalize_rewards: whether to normalize the rewards.
      eps: Epsilon used for normalization.
      noise_scale: Data augmentation noise scale.
      bootstrap: Whether to generated bootstrapped weights.
    """
    with tf.io.gfile.GFile(data_file_name, 'rb') as f:
      dataset = pickle.load(f)

    for k, v in dataset['trajectories'].items():
      dataset['trajectories'][k] = v[:num_trajectories]

    if noise_scale > 0.0:
      dataset['trajectories'] = augment_data(dataset['trajectories'],
                                             noise_scale)

    dataset['trajectories']['steps'] = [
        np.arange(len(state_trajectory))
        for state_trajectory in dataset['trajectories']['states']
    ]

    dataset['initial_states'] = np.stack([
        state_trajectory[0]
        for state_trajectory in dataset['trajectories']['states']
    ])

    num_trajectories = len(dataset['trajectories']['states'])
    if bootstrap:
      dataset['initial_weights'] = np.random.multinomial(
          num_trajectories, [1.0 / num_trajectories] * num_trajectories,
          1).astype(np.float32)[0]
    else:
      dataset['initial_weights'] = np.ones(num_trajectories, dtype=np.float32)

    dataset['trajectories']['weights'] = []
    for i in range(len(dataset['trajectories']['masks'])):
      dataset['trajectories']['weights'].append(
          np.ones_like(dataset['trajectories']['masks'][i]) *
          dataset['initial_weights'][i])

    dataset['initial_weights'] = tf.convert_to_tensor(
        dataset['initial_weights'])
    dataset['initial_states'] = tf.convert_to_tensor(dataset['initial_states'])
    for k, v in dataset['trajectories'].items():
      if 'initial' not in k:
        dataset[k] = tf.convert_to_tensor(
            np.concatenate(dataset['trajectories'][k], axis=0))

    self.states = dataset['states']
    self.actions = dataset['actions']
    self.next_states = dataset['next_states']
    self.masks = dataset['masks']
    self.weights = dataset['weights']
    self.rewards = dataset['rewards']
    self.steps = dataset['steps']

    self.initial_states = dataset['initial_states']
    self.initial_weights = dataset['initial_weights']

    self.eps = eps
    self.model_filename = dataset['model_filename']

    if normalize_states:
      self.state_mean = tf.reduce_mean(self.states, 0)
      self.state_std = tf.math.reduce_std(self.states, 0)

      self.initial_states = self.normalize_states(self.initial_states)
      self.states = self.normalize_states(self.states)
      self.next_states = self.normalize_states(self.next_states)
    else:
      self.state_mean = 0.0
      self.state_std = 1.0

    if normalize_rewards:
      self.reward_mean = tf.reduce_mean(self.rewards)
      if tf.reduce_min(self.masks) == 0.0:
        self.reward_mean = tf.zeros_like(self.reward_mean)
      self.reward_std = tf.math.reduce_std(self.rewards)

      self.rewards = self.normalize_rewards(self.rewards)
    else:
      self.reward_mean = 0.0
      self.reward_std = 1.0

  def normalize_states(self, states):
    dtype = tf.convert_to_tensor(states).dtype
    return ((states - self.state_mean) /
            tf.maximum(tf.cast(self.eps, dtype), self.state_std))

  def unnormalize_states(self, states):
    dtype = tf.convert_to_tensor(states).dtype
    return (states * tf.maximum(tf.cast(self.eps, dtype), self.state_std)
            + self.state_mean)

  def normalize_rewards(self, rewards):
    return (rewards - self.reward_mean) / tf.maximum(self.reward_std, self.eps)

  def unnormalize_rewards(self, rewards):
    return rewards * tf.maximum(self.reward_std, self.eps) + self.reward_mean

  def with_uniform_sampling(self, sample_batch_size):
    return tf.data.Dataset.from_tensor_slices(
        (self.states, self.actions, self.next_states, self.rewards, self.masks,
         self.weights, self.steps)).repeat().shuffle(
             self.states.shape[0], reshuffle_each_iteration=True).batch(
                 sample_batch_size, drop_remainder=True).prefetch(100)

  def with_geometric_sampling(self, sample_batch_size,
                              discount):
    """Creates tf dataset with geometric sampling.

    Args:
      sample_batch_size: Batch size for sampling.
      discount: MDP discount.

    Returns:
      TensorFlow dataset.
    """

    sample_weights = discount**tf.cast(self.steps, tf.float32)
    weight_sum = tf.math.cumsum(sample_weights)

    def sample_batch(_):
      values = tf.random.uniform((sample_batch_size,), 0.0,
                                 weight_sum[-1])
      ind = tf.searchsorted(weight_sum, values)
      return (tf.gather(self.states, ind,
                        0), tf.gather(self.actions, ind, 0),
              tf.gather(self.next_states, ind,
                        0), tf.gather(self.rewards, ind, 0),
              tf.gather(self.masks, ind,
                        0), tf.gather(self.weights, ind, 0),
              tf.gather(self.steps, ind, 0))

    return tf.data.experimental.Counter().map(sample_batch).prefetch(100)


class D4rlDataset(Dataset):
  """Dataset class for policy evaluation."""

  # pylint: disable=super-init-not-called
  def __init__(self,
               d4rl_env,
               normalize_states = False,
               normalize_rewards = False,
               eps = 1e-5,
               noise_scale = 0.0,
               bootstrap = True):
    """Processes data from D4RL environment.

    Args:
      d4rl_env: gym.Env corresponding to D4RL environment.
      normalize_states: whether to normalize the states.
      normalize_rewards: whether to normalize the rewards.
      eps: Epsilon used for normalization.
      noise_scale: Data augmentation noise scale.
      bootstrap: Whether to generated bootstrapped weights.
    """
    dataset = dict(
        trajectories=dict(
            states=[],
            actions=[],
            next_states=[],
            rewards=[],
            masks=[]))
    d4rl_dataset = d4rl_env.get_dataset()
    dataset_length = len(d4rl_dataset['actions'])
    new_trajectory = True
    for idx in range(dataset_length):
      if new_trajectory:
        trajectory = dict(
            states=[], actions=[], next_states=[], rewards=[], masks=[])

      trajectory['states'].append(d4rl_dataset['observations'][idx])
      trajectory['actions'].append(d4rl_dataset['actions'][idx])
      trajectory['rewards'].append(d4rl_dataset['rewards'][idx])
      trajectory['masks'].append(1.0 - d4rl_dataset['terminals'][idx])
      if not new_trajectory:
        trajectory['next_states'].append(d4rl_dataset['observations'][idx])

      end_trajectory = (d4rl_dataset['terminals'][idx] or
                        d4rl_dataset['timeouts'][idx])
      if end_trajectory:
        trajectory['next_states'].append(d4rl_dataset['observations'][idx])
        if d4rl_dataset['timeouts'][idx] and not d4rl_dataset['terminals'][idx]:
          for key in trajectory:
            del trajectory[key][-1]
        if trajectory['actions']:
          for k, v in trajectory.items():
            assert len(v) == len(trajectory['actions'])
            dataset['trajectories'][k].append(np.array(v, dtype=np.float32))
          print('Added trajectory %d with length %d.' % (
              len(dataset['trajectories']['actions']),
              len(trajectory['actions'])))

      new_trajectory = end_trajectory

    if noise_scale > 0.0:
      dataset['trajectories'] = augment_data(dataset['trajectories'],
                                             noise_scale)

    dataset['trajectories']['steps'] = [
        np.arange(len(state_trajectory))
        for state_trajectory in dataset['trajectories']['states']
    ]

    dataset['initial_states'] = np.stack([
        state_trajectory[0]
        for state_trajectory in dataset['trajectories']['states']
    ])

    num_trajectories = len(dataset['trajectories']['states'])
    if bootstrap:
      dataset['initial_weights'] = np.random.multinomial(
          num_trajectories, [1.0 / num_trajectories] * num_trajectories,
          1).astype(np.float32)[0]
    else:
      dataset['initial_weights'] = np.ones(num_trajectories, dtype=np.float32)

    dataset['trajectories']['weights'] = []
    for i in range(len(dataset['trajectories']['masks'])):
      dataset['trajectories']['weights'].append(
          np.ones_like(dataset['trajectories']['masks'][i]) *
          dataset['initial_weights'][i])

    dataset['initial_weights'] = tf.convert_to_tensor(
        dataset['initial_weights'])
    dataset['initial_states'] = tf.convert_to_tensor(dataset['initial_states'])
    for k, v in dataset['trajectories'].items():
      if 'initial' not in k:
        dataset[k] = tf.convert_to_tensor(
            np.concatenate(dataset['trajectories'][k], axis=0))

    self.states = dataset['states']
    self.actions = dataset['actions']
    self.next_states = dataset['next_states']
    self.masks = dataset['masks']
    self.weights = dataset['weights']
    self.rewards = dataset['rewards']
    self.steps = dataset['steps']

    self.initial_states = dataset['initial_states']
    self.initial_weights = dataset['initial_weights']

    self.eps = eps
    self.model_filename = None

    if normalize_states:
      self.state_mean = tf.reduce_mean(self.states, 0)
      self.state_std = tf.math.reduce_std(self.states, 0)

      self.initial_states = self.normalize_states(self.initial_states)
      self.states = self.normalize_states(self.states)
      self.next_states = self.normalize_states(self.next_states)
    else:
      self.state_mean = 0.0
      self.state_std = 1.0

    if normalize_rewards:
      self.reward_mean = tf.reduce_mean(self.rewards)
      if tf.reduce_min(self.masks) == 0.0:
        self.reward_mean = tf.zeros_like(self.reward_mean)
      self.reward_std = tf.math.reduce_std(self.rewards)

      self.rewards = self.normalize_rewards(self.rewards)
    else:
      self.reward_mean = 0.0
      self.reward_std = 1.0
  # pylint: enable=super-init-not-called
