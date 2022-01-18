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

"""Implements helper functions to compute alignment measure."""

import tensorflow as tf


def plot_class_alignment(batch,
                         labels,
                         num_labels,
                         step,
                         tf_summary_key='alignment'):
  """Plots class level alignment as a summary scalar."""
  for label in range(num_labels):
    indices = tf.squeeze(
        tf.where(tf.equal(labels, tf.cast(label, tf.int64))), axis=1)
    class_batch = tf.gather(batch, indices)
    class_list = tf.unstack(class_batch)
    if len(class_list) > 1:
      alignment = compute_alignment(class_list)
      tf.summary.scalar(
          '%s/%s' % (tf_summary_key, str(label)), alignment, step=step)


def compute_alignment(input_list):
  """Computes alignment measure given a list of vectors in O(n) time."""
  if len(input_list) < 2:
    return None

  # Compute mean norm.
  norms = [tf.norm(v) for v in input_list]
  norms_mean = tf.math.reduce_mean(norms)

  # Normalize each vector by mean norm.
  normalized_input_list = [
      v / (tf.where(tf.math.greater(norms_mean, 0), norms_mean, 1.0))
      for v in input_list
  ]
  n = len(normalized_input_list)

  # O(n) implementation of alignment.
  sum_norm_square = tf.math.square(
      tf.norm(tf.math.reduce_sum(normalized_input_list, axis=0)))
  norm_squares = [tf.math.square(tf.norm(v)) for v in normalized_input_list]
  norm_squares_sum = tf.math.reduce_sum(norm_squares)
  alignment = (sum_norm_square - norm_squares_sum) / (n * (n - 1))
  return alignment
