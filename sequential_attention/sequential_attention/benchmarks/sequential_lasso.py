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

"""Sequential Lasso.

Shan Luo; Zehua Chen

https://arxiv.org/abs/1107.2734

Adapted to neural networks in Sequential Attention for Feature Selection

https://arxiv.org/abs/2209.14881
"""

import tensorflow as tf


class SequentialLassoRegularizer(tf.Module):
  """Sequential Lasso for feature selection."""

  def __init__(
      self,
      num_inputs,
      num_inputs_to_select,
      num_inputs_to_select_per_step=1,
      preselected_features=None,
      group_lasso_scale=0.01,
      name='sequential_lasso_regularizer',
      **kwargs,
  ):
    super(SequentialLassoRegularizer, self).__init__(name=name, **kwargs)

    assert num_inputs_to_select % num_inputs_to_select_per_step == 0

    self.group_lasso_scale = group_lasso_scale

    with self.name_scope:
      self._num_inputs = num_inputs
      self._num_inputs_to_select = num_inputs_to_select
      self._num_inputs_to_select_per_step = num_inputs_to_select_per_step
      if preselected_features is not None:
        num_preselected_features = len(preselected_features)
        preselected_features = tf.math.reduce_sum(
            tf.one_hot(preselected_features, num_inputs, dtype=tf.float32), 0
        )
        self.selected_features = tf.Variable(
            preselected_features, trainable=False, name='selected_features'
        )
        self._num_inputs_to_select -= num_preselected_features
      else:
        self.selected_features = tf.Variable(
            tf.zeros(num_inputs), trainable=False, name='selected_features'
        )
      self._num_selected_features = tf.Variable(
          0, trainable=False, name='num_selected_features'
      )

      self.selected_features_history = tf.Variable(
          initial_value=tf.zeros(
              shape=self._num_inputs_to_select, dtype=tf.int32
          ),
          trainable=False,
          dtype=tf.int32,
          name='selected_features_history',
      )

  @tf.Module.with_name_scope
  def __call__(self, x, training_percentage):
    start_percentage = 0.1
    end_percentage = 0.999
    effective_percentage = (training_percentage - start_percentage) / (
        end_percentage - start_percentage
    )

    curr_phase = tf.cast(
        tf.math.ceil(
            effective_percentage
            * (
                self._num_inputs_to_select
                // self._num_inputs_to_select_per_step
            )
        ),
        tf.int32,
    )

    new_phase = tf.greater(
        curr_phase,
        self._num_selected_features // self._num_inputs_to_select_per_step,
        name='new_phase',
    )
    done_phase = tf.greater(
        curr_phase,
        self._num_inputs_to_select // self._num_inputs_to_select_per_step,
        name='done_phase',
    )
    should_select_feature = tf.math.logical_and(
        new_phase, tf.math.logical_not(done_phase)
    )

    unselected_features = tf.ones(self._num_inputs) - self.selected_features
    _, indices = tf.math.top_k(
        tf.multiply(x, unselected_features),
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
      return self.group_lasso_scale * tf.reduce_sum(
          tf.multiply(x, unselected_features)
      )
