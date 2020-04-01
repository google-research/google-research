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
class TopKErrorLoss(tf.losses.Loss):
  """Based on a ranks function, compute the topk loss."""

  def __init__(self, topk, rescale_ranks=True, name=None):
    name = name if name is not None else 'topk_error'
    super(TopKErrorLoss, self).__init__(name=name)
    self._topk = topk
    self._rescale_ranks = rescale_ranks

  def get_ranks(self, y):
    """Compute the (hard) ranks of each entry in y."""
    diff = y[:, :, tf.newaxis] - y[:, tf.newaxis, :]
    diff -= tf.eye(y.shape[-1])[tf.newaxis, :, :] * 1e9
    s = tf.cast(diff > 0.0, dtype=y.dtype)
    return tf.math.reduce_sum(s, axis=-1)

  def rescale_ranks(self, ranks):
    mu = tf.math.reduce_mean(ranks, axis=-1)
    std = tf.math.reduce_std(ranks, axis=-1)
    n = tf.cast(tf.shape(ranks)[-1], ranks.dtype)
    tgt_mu = (n - 1) / 2
    tgt_std = tf.sqrt((n - 1) * (n + 1) / 12)
    return (ranks - mu[:, tf.newaxis]) / std[:, tf.newaxis] * tgt_std + tgt_mu

  def accuracy(self, y_true, y_pred):
    """Computes the soft topk accuracy of the prediction w.r.t the true values.

    Args:
     y_true: Tensor<float>[batch]: the true labels in [0, n-1].
     y_pred: Tensor<float>[batch, n]: n activation values for each input.

    Returns:
     A Tensor<float>[batch] of accuracy per batch.
    """
    ranks = self.get_ranks(y_pred)
    if self._rescale_ranks:
      ranks = self.rescale_ranks(ranks)

    # If the ranks are above topk then the accuracy is 1. Below that threshold
    # the accuracy decreases to 0.
    topk = tf.cast(self._topk, dtype=y_pred.dtype)
    num_activations = tf.shape(y_pred)[-1]
    accuracies = tf.math.minimum(
        tf.cast(1.0, dtype=y_pred.dtype),
        ranks / (tf.cast(num_activations, dtype=y_pred.dtype) - topk))
    # Multiply with the one hot encoding of the label to select only the soft
    # topk accuracy of the true labels.
    true_labels = tf.one_hot(tf.cast(y_true, dtype=tf.int32),
                             depth=num_activations,
                             dtype=y_pred.dtype)
    return tf.reduce_sum(accuracies * true_labels, axis=-1)

  def call(self, y_true, y_pred):
    return 1.0 - self.accuracy(y_true, y_pred)


@gin.configurable
class SoftHeavisideErrorLoss(TopKErrorLoss):
  """Implements an error loss based on soft heavysides."""

  def __init__(self, topk=1, rescale_ranks=True, epsilon=1e-3):
    super(SoftHeavisideErrorLoss, self).__init__(
        topk=topk, rescale_ranks=rescale_ranks, name='soft_heaviside_error')
    self._epsilon = epsilon

  def get_ranks(self, y):

    @tf.custom_gradient
    def _ranks(y):
      diff = y[:, :, tf.newaxis] - y[:, tf.newaxis, :]
      diff -= tf.eye(tf.shape(y)[-1], dtype=y.dtype)[tf.newaxis, :, :] * 1e6
      s = 1.0 / (1 + tf.math.exp(-diff / self._epsilon))

      def grad(dy):
        return dy * tf.math.reduce_sum(s * (1.0 - s), axis=1) / self._epsilon

      return tf.math.reduce_sum(s, axis=-1), grad

    return _ranks(y)


@gin.configurable
class SoftErrorLoss(TopKErrorLoss):
  """Implementation of the soft error loss for classification."""

  def __init__(self, topk=1, rescale_ranks=True, **kwargs):
    super(SoftErrorLoss, self).__init__(
        topk=topk, rescale_ranks=rescale_ranks, name='soft_error')
    self._kwargs = kwargs

  @tf.function
  def get_ranks(self, y):
    return ops.softranks(
        y, direction='ASCENDING', axis=-1, zero_based=True, **self._kwargs)


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
