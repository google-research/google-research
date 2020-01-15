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

# Lint as: python3
"""A jax implementation of the SoftQuantilizer.

Computes all sort of statistical objects related to (soft) sorting, such as the
ranks, the sorted values, the cumulative distribution function or apply quantile
normalization, among other things.

For practical purposes, we encourage the use of the operators defined in ops.py
instead of a direct use of the SoftQuantilizer defined in this module.

It is based on:
"Differentiable Sorting using Optimal Transport: The Sinkhorn CDF and Quantile
Operator" by Cuturi M., Teboul O., Vert JP.
(see https://arxiv.org/pdf/1905.11885.pdf)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import jax
import jax.numpy as np
from soft_sort.jax import sinkhorn


@gin.configurable
def squash(x, scale=1.0, min_std=1e-10):
  """Applies a sigmoid function on a whitten version of the input."""
  mu = np.mean(x, axis=1)
  s = scale * np.sqrt(3.0) / np.pi * np.maximum(np.std(x, axis=1), min_std)
  return jax.scipy.special.expit((x - mu[:, np.newaxis]) / s[:, np.newaxis])


@gin.configurable
class SoftQuantilizer(object):
  """Jax implementation of the SoftQuantilizer.

  Attributes:
   x: np.ndarray<float>[batch, n] : the input 1D point clouds.
   weights: np.ndarray<float>[batch, n] : the input weights
   y: np.ndarray<float>[batch, m] : the sorted target 1D point clouds.
   target_weights: np.ndarray<float>[batch, m] : the target weights.
   transport: np.ndarray<float>[batch, n, m] the transport map.
   dtype: the type of the input data.
   softcdf: the soft cdf (normalized rank) of each point within its point cloud.
   softsort: the soft sorted values within each point cloud.
  """

  def __init__(
      self, x=None, weights=None, num_targets=None, target_weights=None, y=None,
      descending=False, scale_input_fn=squash, **kwargs):
    self._scale_input_fn = scale_input_fn
    self._descending = descending
    self._sinkhorn = sinkhorn.Sinkhorn1D(**kwargs)
    self.reset(x, y, weights, target_weights, num_targets)

  def reset(
      self, x, y=None, weights=None, target_weights=None, num_targets=None):
    """Resets the Sinkhorn matrix for new inputs x, y, a, b."""
    self.x = np.array(x)
    self.dtype = self.x.dtype
    self._batch = self.x.shape[0]
    self._set_input(self.x, weights)
    self._set_target(y, num_targets, target_weights)
    # We run sinkhorn on the rescaled input values x_s.
    self.transport = self._sinkhorn(
        self._x_s, self.y, self.weights, self.target_weights)

  @property
  def softcdf(self):
    return 1.0 / self.weights * jax.vmap(np.matmul)(
        self.transport, np.cumsum(self.target_weights, axis=1))

  @property
  def softsort(self):
    pt = np.transpose(self.transport, (0, 2, 1))
    return 1.0 / self.target_weights * jax.vmap(np.matmul)(pt, self.x)

  def _may_repeat(self, z):
    """Enforces rank 2 on z, repeating itself if needed to match the batch."""
    z = np.array(z)
    if len(z.shape) < len(self.x.shape):
      z = np.reshape(np.tile(z, [self._batch]), (self._batch, -1))
    return z

  def _set_input(self, x, weights):
    """Sets the input vector and the input weigths."""
    self._n = x.shape[1]
    self._x_s = self._scale_input_fn(x) if self._scale_input_fn else self.x

    if weights is None:
      weights = np.ones(self.x.shape, dtype=self.dtype) / self._n
    self.weights = self._may_repeat(weights)

  def _set_target(self, y, num_targets, target_weights):
    """Sets the target defined by at least one of y, b, and m."""
    if y is None and num_targets is None and target_weights is None:
      num_targets = self._n

    # First we set the number of targets
    self._num_targets = num_targets
    if self._num_targets is None:
      reference = y if y is not None else target_weights
      self._num_targets = np.array(reference).shape[-1]

    # Then we set the target vector itself. It must be sorted.
    if y is None:
      m = self._num_targets
      y = np.arange(0, m, dtype=self.dtype) / np.maximum(1.0, m - 1.0)
    self.y = self._may_repeat(y)
    if self._descending:
      self.y = np.flip(self.y, axis=-1)

    # Last we set target_weights
    if target_weights is None:
      target_weights = np.ones(
          self.y.shape, dtype=self.dtype) / self._num_targets
    self.target_weights = self._may_repeat(target_weights)
