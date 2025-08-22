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

"""Sequential Attention for Feature Selection.

https://arxiv.org/abs/2209.14881
"""

import tensorflow as tf


class SequentialAttention(tf.Module):
  """SequentialAttention module."""

  def __init__(
      self,
      num_candidates,
      num_candidates_to_select,
      num_candidates_to_select_per_step=1,
      start_percentage=0.1,
      stop_percentage=1.0,
      name='sequential_attention',
      reset_weights=True,
      **kwargs,
  ):
    """Creates a new SequentialAttention module."""

    super(SequentialAttention, self).__init__(name=name, **kwargs)

    assert num_candidates_to_select % num_candidates_to_select_per_step == 0, (
        'num_candidates_to_select must be a multiple of '
        'num_candidates_to_select_per_step.'
    )

    with self.name_scope:
      self._num_candidates = num_candidates
      self._num_candidates_to_select_per_step = (
          num_candidates_to_select_per_step
      )
      self._num_steps = (
          num_candidates_to_select // num_candidates_to_select_per_step
      )
      self._start_percentage = start_percentage
      self._stop_percentage = stop_percentage
      self._reset_weights = reset_weights

      init_attention_weights = tf.random.normal(
          shape=[num_candidates], stddev=0.00001, dtype=tf.float32
      )
      self._attention_weights = tf.Variable(
          initial_value=lambda: init_attention_weights,
          dtype=tf.float32,
          name='attention_weights',
      )

      self.selected_features = tf.Variable(
          tf.zeros(shape=[num_candidates], dtype=tf.float32),
          trainable=False,
          name='selected_features',
      )

  @tf.Module.with_name_scope
  def __call__(self, training_percentage):
    """Calculates attention weights for all candidates.

    Args:
      training_percentage: Percentage of training process that has been done.
        This input argument should be between 0 and 1 and should be montonically
        increasing.

    Returns:
      A vector of attention weights of size self._num_candidates. All the
      weights
      are between 0 and 1 and sum to 1.
    """
    percentage = (training_percentage - self._start_percentage) / (
        self._stop_percentage - self._start_percentage
    )
    curr_index = tf.cast(
        tf.math.floor(percentage * self._num_steps), dtype=tf.float32
    )
    curr_index = tf.math.minimum(curr_index, self._num_steps - 1.0)

    should_train = tf.less(curr_index, 0.0)

    num_selected = tf.math.reduce_sum(self.selected_features)
    should_select = tf.greater_equal(curr_index, num_selected)
    _, new_indices = tf.math.top_k(
        self._softmax_with_mask(
            self._attention_weights, 1.0 - self.selected_features
        ),
        k=self._num_candidates_to_select_per_step,
    )
    new_indices = self._k_hot_mask(new_indices, self._num_candidates)
    new_indices = tf.cond(
        should_select,
        lambda: new_indices,
        lambda: tf.zeros(self._num_candidates),
    )
    select_op = self.selected_features.assign_add(new_indices)
    init_attention_weights = tf.random.normal(
        shape=[self._num_candidates], stddev=0.00001, dtype=tf.float32
    )
    should_reset = tf.logical_and(should_select, self._reset_weights)
    new_weights = tf.cond(
        should_reset,
        lambda: init_attention_weights,
        lambda: self._attention_weights,
    )
    reset_op = self._attention_weights.assign(new_weights)

    with tf.control_dependencies([select_op, reset_op]):
      candidates = 1.0 - self.selected_features
      softmax = self._softmax_with_mask(self._attention_weights, candidates)
      return tf.cond(
          should_train,
          lambda: tf.ones(self._num_candidates),
          lambda: softmax + self.selected_features,
      )

  @tf.Module.with_name_scope
  def _k_hot_mask(self, indices, depth, dtype=tf.float32):
    return tf.math.reduce_sum(tf.one_hot(indices, depth, dtype=dtype), 0)

  @tf.Module.with_name_scope
  def _softmax_with_mask(self, logits, mask):
    shifted_logits = logits - tf.math.reduce_max(logits)
    exp_shifted_logits = tf.math.exp(shifted_logits)
    masked_exp_shifted_logits = tf.multiply(exp_shifted_logits, mask)
    return tf.math.divide_no_nan(
        masked_exp_shifted_logits, tf.math.reduce_sum(masked_exp_shifted_logits)
    )
