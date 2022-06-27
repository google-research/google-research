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

"""Useful for instantiating different distance metrics to the initial state distribution."""

import numpy as np
import tensorflow as tf


class L2Distance(tf.Module):

  def __init__(self, initial_state_shape, name=None):
    super(L2Distance, self).__init__(name=name)
    self._initial_state_shape = initial_state_shape
    self._reset_values()

  def _reset_values(self):
    self.initial_state = tf.Variable(
        initial_value=np.zeros(self._initial_state_shape, dtype=np.float32),
        shape=self._initial_state_shape,
        trainable=True,
        name='initial_state')
    self.num_states = tf.Variable(
        initial_value=0.,
        trainable=True,
        name='initial_state_count',
        dtype=self.initial_state.dtype)

  def update(self, initial_states, update_type='incremental'):
    tf.debugging.assert_equal(self.initial_state.shape,
                              initial_states.shape[1:])
    if update_type == 'complete':
      self._reset_values()

    new_mean = self.initial_state * self.num_states + tf.reduce_sum(
        initial_states, axis=0)
    self.num_states.assign_add(
        tf.cast(initial_states.shape[0], dtype=self.initial_state.dtype))
    self.initial_state.assign(new_mean / self.num_states)

  # only works for batch of 1-D states
  def distance(self, states):
    tf.debugging.assert_equal(self.initial_state.shape, states.shape[1:])
    return tf.norm(states - self.initial_state, axis=1)


class StateDiscriminator(tf.Module):

  def __init__(self, name=None):
    super(StateDiscriminator, self).__init__(name=name)
