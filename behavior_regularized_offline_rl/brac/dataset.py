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

"""Dataset for offline RL (or replay buffer for online RL)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow.compat.v1 as tf


Transition = collections.namedtuple(
    'Transition', 's1, s2, a1, a2, discount, reward')


class DatasetView(object):
  """Interface for reading from dataset."""

  def __init__(self, dataset, indices):
    self._dataset = dataset
    self._indices = indices

  def get_batch(self, indices):
    real_indices = self._indices[indices]
    return self._dataset.get_batch(real_indices)

  @property
  def size(self):
    return self._indices.shape[0]


def save_copy(data, ckpt_name):
  """Creates a copy of the current data and save as a checkpoint."""
  new_data = Dataset(
      observation_spec=data.config['observation_spec'],
      action_spec=data.config['action_spec'],
      size=data.size,
      circular=False)
  full_batch = data.get_batch(np.arange(data.size))
  new_data.add_transitions(full_batch)
  data_ckpt = tf.train.Checkpoint(data=new_data)
  data_ckpt.write(ckpt_name)


class Dataset(tf.Module):
  """Tensorflow module of dataset of transitions."""

  def __init__(
      self,
      observation_spec,
      action_spec,
      size,
      circular=True,
      ):
    super(Dataset, self).__init__()
    self._size = size
    self._circular = circular
    obs_shape = list(observation_spec.shape)
    obs_type = observation_spec.dtype
    action_shape = list(action_spec.shape)
    action_type = action_spec.dtype
    self._s1 = self._zeros([size] + obs_shape, obs_type)
    self._s2 = self._zeros([size] + obs_shape, obs_type)
    self._a1 = self._zeros([size] + action_shape, action_type)
    self._a2 = self._zeros([size] + action_shape, action_type)
    self._discount = self._zeros([size], tf.float32)
    self._reward = self._zeros([size], tf.float32)
    self._data = Transition(
        s1=self._s1, s2=self._s2, a1=self._a1, a2=self._a2,
        discount=self._discount, reward=self._reward)
    self._current_size = tf.Variable(0)
    self._current_idx = tf.Variable(0)
    self._capacity = tf.Variable(self._size)
    self._config = collections.OrderedDict(
        observation_spec=observation_spec,
        action_spec=action_spec,
        size=size,
        circular=circular)

  @property
  def config(self):
    return self._config

  def create_view(self, indices):
    return DatasetView(self, indices)

  def get_batch(self, indices):
    indices = tf.constant(indices)
    def get_batch_(data_):
      return tf.gather(data_, indices)
    transition_batch = tf.nest.map_structure(get_batch_, self._data)
    return transition_batch

  @property
  def data(self):
    return self._data

  @property
  def capacity(self):
    return self._size

  @property
  def size(self):
    return self._current_size.numpy()

  def _zeros(self, shape, dtype):
    """Create a variable initialized with zeros."""
    return tf.Variable(tf.zeros(shape, dtype))

  @tf.function
  def add_transitions(self, transitions):
    assert isinstance(transitions, Transition)
    batch_size = transitions.s1.shape[0]
    effective_batch_size = tf.minimum(
        batch_size, self._size - self._current_idx)
    indices = self._current_idx + tf.range(effective_batch_size)
    for key in transitions._asdict().keys():
      data = getattr(self._data, key)
      batch = getattr(transitions, key)
      tf.scatter_update(data, indices, batch[:effective_batch_size])
    # Update size and index.
    if tf.less(self._current_size, self._size):
      self._current_size.assign_add(effective_batch_size)
    self._current_idx.assign_add(effective_batch_size)
    if self._circular:
      if tf.greater_equal(self._current_idx, self._size):
        self._current_idx.assign(0)
