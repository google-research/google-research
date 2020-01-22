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

# Lint as: python3
"""A Jax version of Sinkhorn's algorithm."""

from jax import scipy
import jax.numpy as np


class Sinkhorn1D(object):
  """Jax implementation of Sinkhorn1D.

  This class implements the alternated Sinkhorn iterations that solves the 1D
  regularized transport problem.

  It implements an epsilon decay scheme to speed up convergence by starting from
  a large regularization and decreasing it at each iteration until the desired
  level of regularization.

  Attributes:
   power: (float) use to define the cost function.
   epsilon: (float) the strength of the entropic regularization.
   epsilon_0: (float) the initial strength of the entropic regularization.
   epsilon_decay: (float) the decay of the entropic regularization between two
    consecutive iteration.
   threshold: (float) the maximum accepted relative error.
   inner_iterations: (int) the number of consecutive iterations without
    computing the error.
   max_iterations: (int) the maximum number of Sinkhorn iterations.
   iterations: (int) the effective number of iterations done.
  """

  def __init__(
      self, power=2.0, epsilon=1e-3, epsilon_0=0.1, epsilon_decay=0.95,
      threshold=1e-3, inner_iterations=5, max_iterations=1000):
    self.power = power
    self.epsilon = epsilon
    self.epsilon_0 = epsilon_0
    self.epsilon_decay = epsilon_decay
    self.threshold = threshold
    self.max_iterations = max_iterations
    self.inner_iterations = inner_iterations
    self.iterations = 0

  def __call__(self, x, y, a, b):
    """Runs the Sinkhorn algorithm on two discrete measures (x, a) and (y, b).

    Args:
     x: np.ndarray<float>[batch, n] the 1D position of the input points.
     y: np.ndarray<float>[batch, m] the 1D position of the target points.
     a: np.ndarray<float>[batch, n] the weight of each input point.
     b: np.ndarray<float>[batch, m] the weight of each target point.

    Returns:
     np.ndarray<float>[batch, n, m]: the transport map between the two measures.
    """
    c = (x[:, :, np.newaxis] - y[:, np.newaxis, :]) ** self.power

    def center(f, g):
      return c - f[:, :, np.newaxis] - g[:, np.newaxis, :]

    def softmin(f, g, eps, axis):
      return -eps * scipy.special.logsumexp(-center(f, g) / eps, axis=axis)

    def error(f, g, eps):
      b_target = np.sum(np.exp(-center(f, g) / eps), axis=1)
      return np.max(np.abs(b_target - b) / b, axis=None)

    eps = self.epsilon_0
    loga = np.log(a)
    logb = np.log(b)
    f = np.zeros(np.shape(a), dtype=x.dtype)
    g = np.zeros(np.shape(b), dtype=x.dtype)
    err = self.threshold + 1.0
    self.iterations = 0
    while (self.iterations < self.max_iterations) and (
        err >= self.threshold or eps > self.epsilon):
      for _ in range(self.inner_iterations):
        self.iterations += 1
        g = eps * logb + softmin(f, g, eps, axis=1) + g
        f = eps * loga + softmin(f, g, eps, axis=2) + f
        eps = max(eps * self.epsilon_decay, self.epsilon)

      if eps <= self.epsilon:
        err = error(f, g, eps)

    return np.exp(-center(f, g) / eps)
