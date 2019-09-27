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

"""Soft sort based losses for different learning tasks.

The main paradigm shift here, is to design losses that penalizes the ranks
instead of the values of the predictions.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import gin
import tensorflow.compat.v2 as tf

from soft_sort import metrics
from soft_sort import ops


@gin.configurable
class SoftErrorLoss(tf.losses.Loss):
  """Implementation of the soft error loss for classification."""

  def __init__(self, topk=1, power=1.0, **kwargs):
    self._topk = topk
    self._power = power
    self._kwargs = kwargs
    super(SoftErrorLoss, self).__init__(name='soft_error')

  def _soft_topk_accuracy(self, y_true, y_pred):
    """Computes the soft topk accuracy of the prediction w.r.t the true values.

    Args:
     y_true: Tensor<float>[batch]: the true labels in [0, n-1].
     y_pred: Tensor<float>[batch, n]: n activation values for each input.

    Returns:
     A Tensor<float>[batch] of accuracy per batch.
    """
    num_activations = tf.shape(y_pred)[-1]
    topk = tf.cast(self._topk, dtype=y_pred.dtype)
    ranks = ops.softranks(
        y_pred, direction='ASCENDING', axis=-1, zero_based=True, **self._kwargs)
    # If the ranks are above topk then the accuracy is 1. Below that threshold
    # the accuracy decreases to 0.
    accuracies = tf.math.minimum(
        1.0, ranks / (tf.cast(num_activations, dtype=y_pred.dtype) - topk))
    # Multiply with the one hot encoding of the label to select only the soft
    # topk accuracy of the true labels.
    true_labels = tf.one_hot(tf.cast(y_true, dtype=tf.int32),
                             depth=num_activations,
                             dtype=y_pred.dtype)
    return tf.reduce_sum(accuracies * true_labels, axis=-1)

  def call(self, y_true, y_pred):
    acc = self._soft_topk_accuracy(y_true, y_pred)
    return tf.pow(1.0 - acc, self._power)


@gin.configurable
class LeastQuantileRegressionLoss(tf.losses.Loss):
  """A loss for least quantile regression based on plain sort operations."""

  def __init__(self, quantile=0.5, power=1.0, **kwargs):
    self._quantile = quantile
    self._power = power
    self._kwargs = kwargs
    super(LeastQuantileRegressionLoss, self).__init__(name='lqr')

  def call(self, y_true, y_pred):
    return tf.reduce_mean(metrics.quantile_error(
        y_true, y_pred, quantile=self._quantile, power=self._power))


@gin.configurable
class SoftLeastQuantileRegressionLoss(tf.losses.Loss):
  """A loss for least quantile regression based on soft sort operators."""

  def __init__(self, quantile=0.5, power=1.0, **kwargs):
    self._quantile = quantile
    self._power = power
    self._kwargs = kwargs
    super(SoftLeastQuantileRegressionLoss, self).__init__(name='soft_lqr')

  def call(self, y_true, y_pred):
    error = tf.pow(tf.abs(tf.squeeze(y_pred) - y_true), self._power)
    return ops.softquantiles(error, self._quantile, axis=0, **self._kwargs)


@gin.configurable
class TrimmedRegressionLoss(tf.losses.Loss):
  """A loss for trimmed quantile regression."""

  def __init__(self, start_quantile, end_quantile, power=1.0, **kwargs):
    self._start_quantile = start_quantile
    self._end_quantile = end_quantile
    self._power = power
    self._kwargs = kwargs
    super(TrimmedRegressionLoss, self).__init__(name='trimmed')

  def call(self, y_true, y_pred):
    return tf.reduce_mean(metrics.trimmed_error(
        y_true, y_pred,
        self._start_quantile, self._end_quantile, power=self._power))


@gin.configurable
class SoftTrimmedRegressionLoss(tf.losses.Loss):
  """A loss for least quantile regression based on soft sort operators.

  In trimmed regression, we want to minimize the (usually quadratic) error
  between the prediction and the true value, discarding both the best errors and
  the worst ones and focusing on the median-ish ones, more precisely the ones
  between two given quantiles.

  In terms of soft sorting operator, this is obtained by transporting the
  predictions errors onto 3 sorted targets with weights based on those quantile
  values.

  To do this, we use the softquantiles operator with the proper target quantile
  and target quantile width.
  """

  def __init__(self, start_quantile, end_quantile, power=1.0, **kwargs):
    if end_quantile < start_quantile:
      raise ValueError(
          'Start quantile {:.3f} should be lower than end {:.3f}'.format(
              start_quantile, end_quantile))

    self._start_quantile = start_quantile
    self._end_quantile = end_quantile
    self._power = power
    self._kwargs = kwargs
    super(SoftTrimmedRegressionLoss, self).__init__(name='soft_trimmed')

  def call(self, y_true, y_pred):
    error = tf.pow(tf.abs(tf.squeeze(y_pred) - y_true), self._power)
    width = self._end_quantile - self._start_quantile
    quantile = 0.5 * (self._end_quantile + self._start_quantile)
    return ops.softquantiles(
        error, quantile, quantile_width=width, axis=0, **self._kwargs)
