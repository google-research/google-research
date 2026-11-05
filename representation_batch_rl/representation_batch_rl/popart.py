# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Implements PopArt reward normalization.

The implementation is based on the paper:
Learning values across many orders of magnitude.
https://arxiv.org/pdf/1602.07714.pdf

The key difference to the paper is that we do not correct the weights and biases
of the last neural network layer but maintain separate, trainable compensation
variables that define an affine transformation which should be applied to model
output.
"""

import gin
import tensorflow.compat.v2 as tf

from representation_batch_rl.representation_batch_rl import logging_module
from representation_batch_rl.representation_batch_rl import running_statistics


@gin.configurable
class PopArt(tf.Module, logging_module.LoggingModule):
  """TensorFlow module for PopArt reward normalization.

  PopArt reward normalization works by normalizing value targets to zero mean
  and unit variance by rescaling with the exponential moving average of the
  mean m and standard deviation s, i.e.,
      normalized_target = (target - m)/s

  Whenever m and s are updated to new values m' and s', compensation variables
  a and b are updated to a' and b' to guarantee that for all values x:
      s*(x*a + b) + m = s'*(x*a' + b') + m'

  This guarantees that the update of the PopArt statistics does not change
  implicit value predictions.

  The module should be used as follows:
  - Initially, `init_normalization_stats()` should be called to initialize the
    variables.
  - To compute bootstrap values from the value function (to compute targets and
    advantages), first `correct_prediction(...)` and then
    `unnormalize_prediction(...)` should be applied to the model output.
  - To compute the baseline loss, `correct_prediction(...)` should be applied to
    model predictions and `normalize_target(...)` to the computed baseline
    targets.
  - To update the policy, computed advantages should be normalized with
    `normalize_advantage(...)`.
  - When using an optimizer, gradient updates can be applied to the trainable
    parameters of this module (i.e., to the compensation parameters).
  - Finally, the computed target values should be used to update the module
    parameters via `update_normalization_statistics(...)`.
  """

  def __init__(self, mean_std_tracker, compensate=True):
    """Creates a PopArt.

    Args:
      mean_std_tracker: Instance of running_statistics.MeanStd used for tracking
        the mean and the standard deviation.
      compensate: Whether to introduce a trainbale linear transform which
        compensates for normalization. If False, correct_prediction is
        an identity mapping.
    """
    if not isinstance(mean_std_tracker, running_statistics.MeanStd):
      raise ValueError('`mean_std_tracker` needs to be an instance of MeanStd.')
    self.mean_std_tracker = mean_std_tracker
    self.compensate = compensate
    self._compensation_mean = None
    self._compensation_std = None
    self.initialized = False

  def init(self):
    """Initializes normalization variables.

    This is done explicitly in a manual step since variables need to be created
    under the correct distribution strategy scope.
    """
    if self.initialized:
      return
    self.mean_std_tracker.init(1)
    if self.compensate:
      self._compensation_mean = tf.Variable(
          name='running_mean',
          shape=[],
          trainable=True,
          dtype=tf.float32,
          initial_value=tf.zeros(shape=[]),
          aggregation=tf.VariableAggregation.MEAN)
      self._compensation_std = tf.Variable(
          name='running_std',
          shape=[],
          trainable=True,
          dtype=tf.float32,
          initial_value=tf.ones(shape=[]),
          aggregation=tf.VariableAggregation.MEAN)
    self.initialized = True

  @property
  def compensation_mean(self):
    if self._compensation_mean is None:
      raise ValueError('`init` was not called.')
    return self._compensation_mean

  @property
  def compensation_std(self):
    if self._compensation_std is None:
      raise ValueError('`init` was not called.')
    return self._compensation_std

  def normalize_target(self, x):
    """Normalizes target values x using past target statistics.

    Args:
      x: <float32> tensor.

    Returns:
      <float32> normalized tensor.
    """
    vector = self.mean_std_tracker.normalize(tf.expand_dims(x, -1))
    return tf.squeeze(vector, -1)

  def normalize_advantage(self, x):
    """Normalizes advantage values x using past target statistics.

    Args:
      x: <float32> tensor.

    Returns:
      <float32> normalized tensor.
    """
    # Advantage values are differences, thus we only need to divide by the
    # standard deviation.
    _, std = self.mean_std_tracker.get_mean_std()
    return x / std

  def correct_prediction(self, x):
    """Corrects a prediction x using compensation parameters.

    Args:
      x: <float32> tensor.

    Returns:
      <float32> corrected tensor.
    """
    if self.compensate:
      return self.compensation_std * x + self.compensation_mean
    else:
      return x

  def unnormalize_prediction(self, x):
    """Unnormalizes a corrected prediction x using past target statistics.

    Args:
      x: <float32>[batch,] tensor.

    Returns:
      <float32>[batch] unnormalized tensor.
    """
    vector = self.mean_std_tracker.unnormalize(tf.expand_dims(x, -1))
    return tf.squeeze(vector, -1)

  def update_normalization_statistics(self, data):
    """Updates running statistics and compensation statistics.

    Args:
      data: <float32>[time, batch_size].
    """
    # Update the running statistics.
    mean1, std1 = self.mean_std_tracker.get_mean_std()
    self.mean_std_tracker.update(tf.expand_dims(data, -1))
    mean2, std2 = self.mean_std_tracker.get_mean_std()

    self.log('PopArt/mean', mean2)
    self.log('PopArt/std', std2)

    if self.compensate:
      # Update the compensation parameters.
      new_compensation_std = std1 / std2 * self.compensation_std
      new_compensation_mean = (mean1 - mean2 +
                               std1 * self.compensation_mean) / std2
      self.compensation_mean.assign(tf.squeeze(new_compensation_mean, -1))
      self.compensation_std.assign(tf.squeeze(new_compensation_std, -1))
