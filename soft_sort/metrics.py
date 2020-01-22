# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Metrics related to soft sorting."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import gin
import tensorflow.compat.v2 as tf


def quantile_error(y_true, y_pred, quantile, power=1.0):
  """In regression, computes the quantile of errors between y_true and y_pred.

  Args:
   y_true: Tensor<float>[batch]
   y_pred: Tensor<float>[batch, 1] or Tensor<float>[batch].
   quantile: (float) the quantile to look at in [0, 1].
   power: (float) an optional power on the absolute difference.

  Returns:
    A Tensor<float>[1]: the quantile of the errors on the batch.
  """
  error = tf.pow(tf.abs(tf.squeeze(y_pred) - y_true), power)
  sorted_error = tf.sort(error, direction='ASCENDING', axis=0)
  idx = tf.cast(tf.shape(y_pred)[0], dtype=tf.float32) * quantile
  idx = tf.cast(idx, dtype=tf.int32)
  return sorted_error[idx]


@gin.configurable
class QuantileErrorMetric(tf.metrics.Mean):
  """A tf.Metric for the regression quantile error."""

  def __init__(self, quantile=0.5, power=1.0):
    self._quantile = quantile
    self._power = power
    super(QuantileErrorMetric, self).__init__(name='quantile_error')

  def update_state(self, y_true, y_pred, sample_weight=None):
    super(QuantileErrorMetric, self).update_state(quantile_error(
        y_true, y_pred, quantile=self._quantile, power=self._power))


def trimmed_error(
    y_true, y_pred, start_quantile, end_quantile, power=1.0):
  """In regression, computes the mean of trimmed errors.

  Args:
   y_true: Tensor<float>[batch]
   y_pred: Tensor<float>[batch, 1] or Tensor<float>[batch].
   start_quantile: (float) errors below this quantile (as computed on all
    [batch] examples) are discarded. value in [0, 1].
   end_quantile: (float) errors above this quantile (as computed on all
    [batch] examples) are discarded. value in [0, 1].
   power: (float) an optional power on the absolute difference between y_pred
    and y_true.

  Returns:
    A Tensor<float>[1]: the mean error on the selected range of quantiles.
  """
  error = tf.pow(tf.abs(tf.squeeze(y_pred) - y_true), power)
  sorted_error = tf.sort(error, direction='ASCENDING', axis=0)
  n = tf.cast(tf.shape(y_true)[0], tf.float32)
  n_start = tf.cast(tf.math.floor(n * start_quantile), dtype=tf.int32)
  n_end = tf.cast(tf.math.floor(n * end_quantile), dtype=tf.int32)
  return tf.reduce_mean(sorted_error[n_start:n_end], axis=0)


@gin.configurable
class TrimmedErrorMetric(tf.metrics.Mean):
  """A tf.Metric for the regression trimmed error."""

  def __init__(self, start_quantile=0.0, end_quantile=1.0, power=1.0):
    self._start_quantile = start_quantile
    self._end_quantile = end_quantile
    self._power = power
    super(TrimmedErrorMetric, self).__init__(name='trimmed_error')

  def update_state(self, y_true, y_pred, sample_weight=None):
    super(TrimmedErrorMetric, self).update_state(trimmed_error(
        y_true, y_pred, start_quantile=self._start_quantile,
        end_quantile=self._end_quantile, power=self._power))
