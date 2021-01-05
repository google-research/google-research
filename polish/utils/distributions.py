# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Implements multivariate normal distribution on R^k."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow.compat.v1 as tf


class MultiVariateNormalDiag(object):
  """The multivariate normal distribution on R^k.

  The covariance matrix is a diagonal matrix.
  wiki: https://en.wikipedia.org/wiki/Multivariate_normal_distribution

  Attributes:
    mean: Mean of the k-dimensional distribution.
    logstd: Log of standard deviation of the k-dimensional distribution.
    std: Standard deviation the k-dimensional distribution.
  """
  # TODO(ayazdan): Replace the class with the standard library
  #   tfp.distributions.MultivariateNormalDiag

  def __init__(self, mean, logstd):
    self.mean = mean
    self.logstd = logstd
    self.std = tf.exp(self.logstd)

  def sample(self):
    """Draws a k-dimensional sample from the distribution."""
    return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

  def negative_log_prob(self, sample):
    """Negative log of likelihood of a drawn sample from the distribution.

    wiki: https://en.wikipedia.org/wiki/Multivariate_normal_distribution

    Args:
      sample: A drawn k-dimensional sample from the distribution.

    Returns:
      Computed negative likelihood of a sample.
    """
    return 0.5 * tf.reduce_sum(
        tf.square((sample - self.mean) / self.std),
        axis=-1) + 0.5 * np.log(2.0 * np.pi) * tf.cast(
            tf.shape(sample)[-1], tf.float32) + tf.reduce_sum(
                self.logstd, axis=-1)

  def kl_divergence(self, other):
    """Computes the Kullback-Leibler divergence.

    Both distributions must be instantiated from the same distribution.

    Args:
      other: Another k-dimensional MultiVariateNormalDiag distribution.

    Returns:
      The Kullback-Leibler divergence between `self` and `other` distributions.
    """
    assert isinstance(other, MultiVariateNormalDiag)

    return tf.reduce_sum(
        other.logstd - self.logstd +
        (tf.square(self.std) + tf.square(self.mean - other.mean)) /
        (2.0 * tf.square(other.std)) - 0.5,
        axis=-1)

  def entropy(self):
    """Computes entropy of the current distribution.

    Returns:
      The Shannon entropy in nats.
    """
    return tf.reduce_sum(.5 * self.logstd + .5 * np.log(2.0 * np.pi * np.e),
                         axis=-1)
