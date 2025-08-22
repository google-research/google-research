# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Orthogonal Matching Pursuit.

Adapted to neural networks in Sequential Attention for Feature Selection

https://arxiv.org/abs/2209.14881
"""

import tensorflow as tf


class OrthogonalMatchingPursuit(tf.Module):
  """A greedy selection strategy for combinatorial optimization."""

  def __init__(
      self,
      num_inputs,
      num_inputs_to_select,
      num_inputs_to_select_per_step=1,
      name='orthogonal_matching_pursuit',
      **kwargs,
  ):
    super().__init__(name=name, **kwargs)

    self._num_inputs = num_inputs
    self._num_inputs_to_select = num_inputs_to_select
    self._num_inputs_to_select_per_step = num_inputs_to_select_per_step
    self.selected_features = tf.Variable(tf.zeros(num_inputs), trainable=False)
    self._num_selected_features = tf.Variable(0, trainable=False)
    self.gradients = tf.Variable(tf.zeros(num_inputs), trainable=False)

    self.selected_features_history = tf.Variable(
        initial_value=tf.zeros(
            shape=self._num_inputs_to_select, dtype=tf.int32
        ),
        trainable=False,
        dtype=tf.int32,
        name='selected_features_history',
    )

    self.need_gradient = tf.Variable(False, trainable=False)
    self.set_gradient = tf.Variable(False, trainable=False)

  def __call__(self, training_percentage):
    curr_phase = tf.cast(
        tf.math.ceil(training_percentage * self._num_inputs_to_select), tf.int32
    )

    should_select_feature = tf.greater(
        curr_phase, self._num_selected_features, name='should_select_feature'
    )

    unselected_features = tf.ones(self._num_inputs) - self.selected_features
    _, indices = tf.math.top_k(
        tf.multiply(self.gradients, unselected_features),
        k=self._num_inputs_to_select_per_step,
    )
    new_features = tf.math.reduce_sum(tf.one_hot(indices, self._num_inputs), 0)
    new_features = tf.cond(
        should_select_feature,
        lambda: new_features,
        lambda: tf.zeros(self._num_inputs),
    )
    selected_features = self.selected_features.assign_add(new_features)
    unselected_features = tf.ones(self._num_inputs) - selected_features

    # update history
    support = tf.one_hot(
        tf.range(
            self._num_selected_features,
            self._num_selected_features + self._num_inputs_to_select_per_step,
        ),
        self._num_inputs_to_select,
        dtype=tf.int32,
    )
    indices = tf.reshape(
        indices, shape=[self._num_inputs_to_select_per_step, 1]
    )
    history = tf.cond(
        should_select_feature,
        lambda: tf.reduce_sum(support * indices, 0),
        lambda: tf.zeros(self._num_inputs_to_select, dtype=tf.int32),
    )
    history = self.selected_features_history.assign_add(history)

    # update num_selected_features
    with tf.control_dependencies([history]):  # force update
      num_new_features = tf.cond(
          should_select_feature,
          lambda: self._num_inputs_to_select_per_step,
          lambda: 0,
      )
      num_selected_features = self._num_selected_features.assign_add(
          num_new_features
      )
    with tf.control_dependencies([num_selected_features]):  # force update
      return self.selected_features
