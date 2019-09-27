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

"""Soft Sorting core library.

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

import math
import gin
import tensorflow.compat.v2 as tf
from soft_sort import sinkhorn


@gin.configurable
def group_rescale(x, scale=1.0, min_std=1e-10, is_logistic=True):
  """Applies a sigmoid map on standardized centered inputs.

  This logistic map, when applied on the output of a neural network,
  redistributes the activations into [0,1] in a smooth adaptative way, helping
  the numerical stability of the Sinkhorn algorithm while maintaining a
  well behaved back propagation.

  In case of logistic sigmoid, this map is exactly the CDF of a logistic
  distribution. See https://en.wikipedia.org/wiki/Logistic_distribution for
  details, in particular the variance of the distribution. In case of an atan
  sigmoid (is_logistic == False), it is somewhat related to the CDF of a Cauchy
  distribution and presents the advantage to have better behaved gradients.

  Args:
   x: Tensor<float>[batch, n]
   scale: (float) a scale to be applied after standardizing and centering the
    inputs.
   min_std: (float) minimum standard deviation to consider to avoid degenerated
    values when centering and rescaling the input x.
   is_logistic: uses either a logistic sigmoid or an arctan.

  Returns:
   A Tensor<float>[batch, n] after application of the sigmoid map.
  """
  mean = tf.math.reduce_mean(x, axis=1)
  std = tf.maximum(tf.math.reduce_std(x, axis=1), min_std)
  if is_logistic:
    scale *= math.sqrt(3.0) / math.pi
  s = tf.cast(scale * std, dtype=x.dtype)
  squashing_fn = tf.math.sigmoid if is_logistic else tf.math.atan
  return squashing_fn((x - mean[:, tf.newaxis]) / s[:, tf.newaxis])


@gin.configurable
class SoftQuantilizer(object):
  """Computes soft-ranks, soft-sorts and soft-quantile normalizations.

  It is based on:
  "Differentiable Sorting using Optimal Transport: The Sinkhorn CDF and Quantile
  Operator" by Cuturi M., Teboul O., Vert JP.
  (see https://arxiv.org/pdf/1905.11885.pdf)

  This class:
   - prepares the data to build the sorted target tensor and its weights.
   - runs the Sinkhorn algorithm to obtain the transport matrix.
   - recomposes the cdf and the sorted values from the input data and the
     transport matrix as described in section 3 of the paper.

  Attributes:
   x: Tensor<float>[batch, n], the input tensor being soft sorted.
   dtype: The type of the input.
   y: Tensor<float>[batch, m], the target for the OT problem.
   weights: Tensor<float>[batch_size, n], the weights of the inputs.
   target_weights: Tensor<float>[batch, n], the weights of the target.
   transport: Tensor<float>[batch, n, m], the transport matrix obtain via
    Sinkhorn algorithm.
   num_iterations: (int) the number of Sinkhorn updates.
   sinkhorn_error: (float) the error in the Sinkhorn algorithm, due to the fact
    that we stop at a given iteration.
   softcdf: Tensor<float>[batch, n]: the obtained cdf of the input x.
   softsort: Tensor<float>[batch, n]: the obtained soft sorted values of x.
  """

  def __init__(
      self, x=None, weights=None, num_targets=None, target_weights=None, y=None,
      epsilon=1e-3, p=2.0, sinkhorn_threshold=1e-3, stabilized=True,
      descending=False, scale_input_fn=group_rescale, **kwargs):
    """Initializes the internal state of the SoftSorter.

    Args:
     x: the Tensor<float>[batch, n] to be soft-sorted.
     weights: Tensor<float>[n] or None. May be given with a batch dimension.
      When these weights are uniform one recovers usual sorting behaviour.
      If left to None, we use uniform weights.
     num_targets: used when y is not assigned to set a uniform target vector on
      [0,1]. num_targets is equal to n by default. If smaller than n, it will
      lead to some quantization of the input vector.
     target_weights: vector of weights for each element in y. Uniform by
      default.
     y: Tensor<float>[m] or None. May be given with a batch dimension.
      We encourage the default use of None for most of the cases. If provided,
      the list, array or tensor must be sorted in increasing order in order to
      perform a soft sort. If left to None, it will be set to num_targets values
      [0,1/(num_targets-1),...,1] copied N times.
     epsilon: scale of the entropic relaxation (see sinkhorn.py).
     p: (float) power of the distance function (see sinkhorn.py).
     sinkhorn_threshold: (float) treshold (see sinkhorn.py).
     stabilized : log-space computations when true. Slower but numerically more
      stable.
     descending: (bool), if True, targets will be reversed so as to produce a
      decending sorting.
     scale_input_fn: function used to scale input entries so that they fit into
      the [0, 1] segment [0,1]. This is not only useful to stabilize
      computations but also to ensure that the regularization parameter epsilon
      is valid throughout gradient iterations, regardless of the variations of
      the input values'range.
     **kwargs: extra parameters to the Sinkhorn algorithm.
    """
    self._stabilized = stabilized
    self._scale_input_fn = scale_input_fn
    self._kwargs = kwargs  # The sinkhorn params.
    self.num_iterations = 0
    self._descending = descending
    self._epsilon = epsilon
    self._p = p
    self._sinkhorn_threshold = sinkhorn_threshold
    self._kwargs = kwargs
    self.reset(x, y, weights, target_weights, num_targets)

  def reset(
      self, x, y=None, weights=None, target_weights=None, num_targets=None):
    """Resets the Sinkhorn matrix for new inputs x, y, a, b."""
    self.x = x
    if x is None:
      return

    self.dtype = x.dtype
    self._batch = tf.shape(x)[0]
    self._set_input(x, weights)
    self._set_target(y, num_targets, target_weights)

    sinkhorn_fn = sinkhorn.log_sinkhorn if self._stabilized else sinkhorn.sinkhorn
    # We run sinkhorn on the rescaled input values x_s.
    self.transport, self.sinkhorn_error, self.num_iterations = sinkhorn_fn(
        self._x_s, self.y, self.weights, self.target_weights,
        self._epsilon, self._p, self._sinkhorn_threshold,
        **self._kwargs)

  @property
  def softcdf(self):
    return 1 / self.weights * tf.linalg.matvec(
        self.transport, tf.cumsum(self.target_weights, axis=1))

  @property
  def softsort(self):
    pt = tf.transpose(self.transport, (0, 2, 1))
    return 1.0 / self.target_weights * tf.linalg.matvec(pt, self.x)

  def quantile_normalization(self, f):
    """Quantile normalization: returns the values of f in the order of x."""
    f = self._cast_may_repeat(f)
    return 1.0 / self.weights * tf.linalg.matvec(self.transport, f)

  def _cast_may_repeat(self, z):
    """Enforces rank 2 on z, repeating itself if needed to match the batch."""
    z = tf.cast(z, dtype=self.dtype)
    if z.shape.rank < self.x.shape.rank:
      return tf.reshape(
          tf.tile(z, [self._batch]), (self._batch, tf.shape(z)[0]))
    else:
      return z

  def _set_input(self, x, weights):
    """Sets the input vector and the input weigths."""
    self._n = tf.shape(x)[1]
    self._x_s = self._scale_input_fn(x) if self._scale_input_fn else self.x

    if weights is None:
      n = tf.cast(self._n, dtype=self.dtype)
      weights = tf.ones(tf.shape(self.x), dtype=self.dtype) / n
    self.weights = self._cast_may_repeat(weights)

  def _set_target(self, y, num_targets, target_weights):
    """Sets the target defined by at least one of y, b, and m."""
    if y is None and num_targets is None and target_weights is None:
      num_targets = tf.cast(self._n, dtype=self.dtype)

    # First we set the number of targets
    self._num_targets = num_targets
    if self._num_targets is None:
      if target_weights is not None:
        if isinstance(target_weights, tf.Tensor):
          self._num_targets = tf.shape(target_weights)[-1]
        else:
          self._num_targets = len(target_weights)
      elif y is not None:
        self._num_targets = tf.shape(y)[1]

    # Then we set the target vector itself. It must be sorted.
    if y is None:
      m = tf.cast(self._num_targets, dtype=self.dtype)
      y = tf.range(0, m, dtype=self.dtype) / tf.math.maximum(1.0, m - 1.0)
    self.y = self._cast_may_repeat(y)
    if self._descending:
      self.y = tf.reverse(self.y, (1,))

    # Last we set target_weights
    if target_weights is None:
      m = tf.cast(self._num_targets, dtype=self.dtype)
      target_weights = tf.ones(tf.shape(self.y), dtype=self.dtype) / m
    self.target_weights = self._cast_may_repeat(target_weights)
