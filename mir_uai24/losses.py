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

"""Loss utilities for wtd-Assign MIR."""

import tensorflow as tf

tfk = tf.keras

poisson_loss = tfk.losses.Poisson()
bce_loss = tfk.losses.BinaryCrossentropy()
mae_loss = tfk.losses.MeanAbsoluteError()
mse_loss = tfk.losses.MeanSquaredError()


def posterior_sum_1(bag_ids, posterior):
  """Loss function to encourage that the primeness posterior of a bag sum to 1.

  Args:
    bag_ids: A 1D tensor containing the bag id of each instance.
    posterior: A 1D tensor containing the primeness posterior of each instance.

  Returns:
    A scalar tensor containing the MAE loss.
  """
  _, bag_ids = tf.unique(bag_ids)
  aggregation_matrix = tf.transpose(
      tf.one_hot(bag_ids, tf.reduce_max(bag_ids) + 1))
  posterior_mae_loss = mae_loss(
      tf.matmul(aggregation_matrix, posterior),
      tf.ones(tf.reduce_max(bag_ids) + 1)[: None])
  return posterior_mae_loss


def overlap_posterior_max_sum_1(overlap_posterior):
  """Loss function to encourage that an instance's overlap posterior sums to 1.

  Args:
    overlap_posterior: A 2D tensor containing the overlap posterior of each
    instance.

  Returns:
    A scalar tensor containing the MAE loss.
  """
  overlap_posterior_sum = tf.reduce_sum(
      overlap_posterior, axis=1, keepdims=True)
  return mae_loss(
      tf.maximum(overlap_posterior_sum, 1.0), tf.ones_like(overlap_posterior))

