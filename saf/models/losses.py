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

"""Misc loss functions."""

import tensorflow as tf


def nan_mean(terms):
  """Returns the average discarding the NaNs."""
  return tf.reduce_mean(tf.boolean_mask(terms, tf.math.is_finite(terms)))


def mae_per_batch(y_true, y_pred):
  """Returns the MAE computed on the batch of y_true and y_pred.

  Args:
    y_true: Ground truth values for the outputs, of shape: [batch_size,
      num_timesteps]
    y_pred: Predicted quantiles, of shape: [batch_size, num_timesteps]

  Returns:
    Average loss value.
  """

  return nan_mean(tf.math.abs(y_true - y_pred))


def mse_per_batch(y_true, y_pred):
  """Returns the MSE computed on the batch of y_true and y_pred.

  Args:
    y_true: Ground truth values for the outputs, of shape: [batch_size,
      num_timesteps]
    y_pred: Predicted quantiles, of shape: [batch_size, num_timesteps]

  Returns:
    Average loss value.
  """

  return nan_mean(tf.math.abs((y_true - y_pred)**2))


def mape_per_batch(y_true, y_pred, epsilon=0.0001):
  """Returns the MAPE computed on the batch of y_true and y_pred.

  Args:
    y_true: Ground truth values for the outputs, of shape: [batch_size,
      num_timesteps]
    y_pred: Predicted quantiles, of shape: [batch_size, num_timesteps]
    epsilon: Small number of 0 division.

  Returns:
    Average loss value.
  """

  return 100 * nan_mean(tf.math.abs(1 - y_pred / (epsilon + y_true)))


def q_score_denominator_per_batch(y_true):
  """Get denominator for q-Risk computation."""
  return nan_mean(tf.math.abs(y_true))


def quantile_loss(y_true, y_pred, probabilities):
  """Returns quantile loss.

  Args:
    y_true: Ground truth values for the outputs, of shape: [batch_size,
      num_timesteps]
    y_pred: Predicted quantiles, of shape: [batch_size, num_timesteps,
      num_quantiles]
    probabilities: Probability values corresponding to quantiles, of shape:
      [num_quantiles]

  Returns:
    Average quantile loss value.
  """
  y_diff = tf.expand_dims(y_true, 2) - y_pred
  quantile_loss_probabilities = (
      probabilities * tf.nn.relu(y_diff) +
      (1.0 - probabilities) * tf.nn.relu(-y_diff))

  return tf.reduce_mean(quantile_loss_probabilities)
