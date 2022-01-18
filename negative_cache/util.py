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

"""Utilities for negative cache training."""

import tensorflow.compat.v2 as tf


def approximate_top_k_with_indices(negative_scores, k):
  """Approximately mines the top k highest scoreing negatives with indices.

  This function groups the negative scores into num_negatives / k groupings and
  returns the highest scoring element from each group. It also returns the index
  where the selected elements were found in the score matrix.

  Args:
    negative_scores: A matrix with the scores of the negative elements.
    k: The number of negatives to mine.

  Returns:
    The tuple (top_k_scores, top_k_indices), where top_k_indices describes the
    index of the mined elements in the given score matrix.
  """
  bs = tf.shape(negative_scores)[0]
  num_elem = tf.shape(negative_scores)[1]
  batch_indices = tf.range(num_elem)
  indices = tf.tile(tf.expand_dims(batch_indices, axis=0), multiples=[bs, 1])
  grouped_negative_scores = tf.reshape(negative_scores, [bs * k, -1])
  grouped_batch_indices = tf.range(tf.shape(grouped_negative_scores)[0])
  grouped_top_k_scores, grouped_top_k_indices = tf.math.top_k(
      grouped_negative_scores)
  grouped_top_k_indices = tf.squeeze(grouped_top_k_indices, axis=1)
  gather_indices = tf.stack([grouped_batch_indices, grouped_top_k_indices],
                            axis=1)
  grouped_indices = tf.reshape(indices, [bs * k, -1])
  grouped_top_k_indices = tf.gather_nd(grouped_indices, gather_indices)
  top_k_indices = tf.reshape(grouped_top_k_indices, [bs, k])
  top_k_scores = tf.reshape(grouped_top_k_scores, [bs, k])
  return top_k_scores, top_k_indices


def cross_replica_concat(tensor, axis):
  replica_context = tf.distribute.get_replica_context()
  return replica_context.all_gather(tensor, axis=axis)
