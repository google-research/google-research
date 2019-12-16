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

"""A Sinkhorn implementation for 1D Optimal Transport.

Sinkhorn algorithm was introduced in 1967 by R. Sinkhorn in the article
"Diagonal equivalence to matrices with prescribed row and column sums." in
The American Mathematical Monthly. It is an iterative algorithm that turns an
input matrix into a bi-stochastic, alternating between normalizing rows and
columns.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import gin
import tensorflow.compat.v2 as tf


@gin.configurable
class Sinkhorn1D(object):
  """Runs the Sinkhorn algorithm for 1D inputs.

  This class implements the stabilized Sinkhorn algorithm in log domain with
  epsilon decay to speed up convergence.

  Attributes:
   cost: Tensor<float>[batch_size, n, m] the cost matrix of optimal transport.
   eps: (float) the current level of regularization. This changes over time due
    to the epsilon decay scheme.
   epsilon: (float) the level of entropic regularization wanted.
   epsilon_0: (float) the initial level of entropic regularization.
   epsilon_decay: (float) a multiplicative factor applied at each iteration
    until reaching the epsilon value.
   inner_num_iter: (int32) the Sinkhorn error is not computed at each iteration
    but every inner_num_iter instead to avoid computational overhead.
   iterations: (int32) the actual number of applied iterations.
   max_iterations: (int32) the maximum number of Sinkhorn iterations.
   power: (float) power of the p-norm used in the cost matrix.
   threshold: (float) the relative threshold on the Sinkhorn error to stop the
    Sinkhorn iterations.
  """

  def __init__(
      self, epsilon=1e-3, epsilon_0=1e-1, epsilon_decay=0.95, power=2.0,
      threshold=1e-2, inner_num_iter=5, max_iterations=2000):
    self.epsilon = epsilon
    self.epsilon_0 = epsilon_0
    self.epsilon_decay = epsilon_decay
    self.power = power
    self.threshold = threshold
    self.inner_num_iter = inner_num_iter
    self.max_iterations = max_iterations
    self._max_outer_iterations = max_iterations // inner_num_iter

  def center(self, f, g):
    """Centers the cost matrix relatively to dual variables f and g."""
    return self.cost - f[:, :, tf.newaxis] - g[:, tf.newaxis, :]

  def softmin(self, f, g, eps, axis):
    return -eps * tf.reduce_logsumexp(-self.center(f, g) / eps, axis=axis)

  def error(self, f, g, eps, b):
    """Computes the maximum relative sinkhorn error over the batch."""
    b_target = tf.math.reduce_sum(
        tf.math.exp(-self.center(f, g) / eps), axis=1)
    return tf.reduce_max(tf.abs(b_target - b) / b, axis=None)

  def __call__(self, x, y, a, b):
    """Runs the Sinkhorn algorithm on input (x, a) and target (y, b).

    Args:
     x: Tensor<float>[batch, n]: the input point clouds.
     y: Tensor<float>[batch, m]: the target point clouds.
     a: Tensor<float>[batch, n]: the weight of each input point.
     b: Tensor<float>[batch, m]: the weight of each target point.

    Returns:
     A Tensor<float>[batch, n, m] transport map. As a side effect, it also
     stores the cost matrix, the number of applied iterations and the obtained
     level of entropic regularization.
    """
    self._b = b
    loga = tf.math.log(a)
    logb = tf.math.log(b)
    self.cost = tf.pow(
        tf.math.abs(x[:, :, tf.newaxis] - y[:, tf.newaxis, :]), self.power)

    def body_fn(f, g, eps, num_iter):
      """A small loop of N Sinkhorn iterations."""
      for _ in range(self.inner_num_iter):
        g = eps * logb + self.softmin(f, g, eps, axis=1) + g
        f = eps * loga + self.softmin(f, g, eps, axis=2) + f
        eps = tf.math.maximum(eps * self.epsilon_decay, self.epsilon)
      return [f, g, eps, num_iter + self.inner_num_iter]

    def cond_fn(f, g, eps, num_iter):
      return tf.math.reduce_all([
          tf.math.less(num_iter, self.max_iterations),
          tf.math.reduce_any([
              tf.math.greater(eps, self.epsilon),
              tf.math.greater(self.error(f, g, eps, b), self.threshold)
          ])
      ])

    self._f, self._g, self.eps, self.iterations = tf.while_loop(
        cond_fn, body_fn, [
            tf.zeros(tf.shape(loga), dtype=x.dtype),
            tf.zeros(tf.shape(logb), dtype=x.dtype),
            tf.cast(self.epsilon_0, dtype=x.dtype),
            tf.constant(0, dtype=tf.int32)
        ],
        parallel_iterations=1,
        maximum_iterations=self._max_outer_iterations + 1)

    return tf.math.exp(-self.center(self._f, self._g) / self.eps)
