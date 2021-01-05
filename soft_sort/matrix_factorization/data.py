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

# Lint as: python3
"""Different data for quantile matrix factorization."""

import gin
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp


def hard_quantile_normalization(inputs, quantiles):
  """Applies the quantile function `quantiles` to the inputs."""
  n_rows = inputs.shape[0]
  rows = tf.range(n_rows)[:, tf.newaxis] * tf.ones_like(inputs, dtype=tf.int32)
  indices = tf.stack(
      [rows, tf.argsort(tf.argsort(inputs, axis=1), axis=1)], axis=-1)
  ordered_quantiles = tf.gather_nd(quantiles, tf.reshape(indices, (-1, 2)))
  return tf.reshape(ordered_quantiles, inputs.shape)


@gin.configurable
class SyntheticData:
  """A class to synthesize data for matrix factorization."""

  def __init__(self,
               num_features,
               num_individuals,
               low_rank,
               poisson_lam = 2.0,
               log_quantile_gap_std = 2.0,
               normalize = True,
               noise = 0.0):
    self.num_individuals = num_individuals
    self.num_features = num_features
    self.rank = low_rank
    self.shape = (self.num_features, self.num_individuals)
    self._noise = noise
    self._normalize = normalize

    # Sample some matrices.
    self._quantiles = tf.math.cumsum(
        tf.math.exp(log_quantile_gap_std * tf.random.normal(self.shape)),
        axis=1)
    self._u = tf.random.poisson((self.num_features, self.rank), lam=poisson_lam)
    self._v = tf.transpose(tfp.distributions.Dirichlet(
        0.5 * tf.ones(self.rank)).sample(self.num_individuals))
    self.uv = tf.matmul(self._u, self._v)
    self._additive_noise = tf.zeros(self.uv.shape)

  def make(self, normalize = None, noise = None):
    """Builds a synthentic input tensor to be fed to a factorization."""
    normalize = self._normalize if normalize is None else normalize
    noise = self._noise if noise is None else noise
    result = self.uv
    if noise > 0.0:
      self._noise = noise * tf.random.normal(result.shape)
      result = result + self._additive_noise
    if normalize:
      result = hard_quantile_normalization(result, self._quantiles)
    return result

