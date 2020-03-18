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

import gin
from jax import scipy
import jax.numpy as np


def center(cost, f, g):
  return cost - f[:, :, np.newaxis] - g[:, np.newaxis, :]


def softmin(cost, f, g, eps, axis):
  return -eps * scipy.special.logsumexp(-center(cost, f, g) / eps, axis=axis)


def error(cost, f, g, eps, b):
  b_target = np.sum(transport(cost, f, g, eps), axis=1)
  return np.max(np.abs(b_target - b) / b, axis=None)


def transport(cost, f, g, eps):
  return np.exp(-center(cost, f, g) / eps)


def cost_fn(x, y, power):
  """A transport cost in the form |x-y|^p and its derivative."""
  delta = x[:, :, np.newaxis] - y[:, np.newaxis, :]
  if power == 1.0:
    cost = np.abs(delta)
    derivative = np.sign(delta)
  elif power == 2.0:
    cost = delta ** 2.0
    derivative = 2.0 * delta
  else:
    abs_diff = np.abs(delta)
    cost = abs_diff ** power
    derivative = power * np.sign(delta) * abs_diff ** (power - 1.0)
  return cost, derivative


@gin.configurable
def sinkhorn_iterations(x,
                        y,
                        a,
                        b,
                        power = 2.0,
                        epsilon = 1e-2,
                        epsilon_0 = 0.1,
                        epsilon_decay = 0.95,
                        threshold = 1e-2,
                        inner_iterations = 10,
                        max_iterations = 2000):
  """Runs the Sinkhorn's algorithm from (x, a) to (y, b).

  Args:
   x: np.ndarray<float>[batch, n]: the input point clouds.
   y: np.ndarray<float>[batch, m]: the target point clouds.
   a: np.ndarray<float>[batch, n]: the weight of each input point. The sum of
    all elements of b must match that of a to converge.
   b: np.ndarray<float>[batch, m]: the weight of each target point. The sum of
    all elements of b must match that of a to converge.
   power: (float) the power of the distance for the cost function.
   epsilon: (float) the level of entropic regularization wanted.
   epsilon_0: (float) the initial level of entropic regularization.
   epsilon_decay: (float) a multiplicative factor applied at each iteration
    until reaching the epsilon value.
   threshold: (float) the relative threshold on the Sinkhorn error to stop the
    Sinkhorn iterations.
   inner_iterations: (int32) the Sinkhorn error is not recomputed at each
    iteration but every inner_num_iter instead to avoid computational overhead.
   max_iterations: (int32) the maximum number of Sinkhorn iterations.

  Returns:
   A 5-tuple containing: the values of the conjugate variables f and g, the
   final value of the entropic parameter epsilon, the cost matrix and the number
   of iterations.
  """
  loga = np.log(a)
  logb = np.log(b)
  cost, d_cost = cost_fn(x, y, power)
  f = np.zeros(np.shape(a), dtype=x.dtype)
  g = np.zeros(np.shape(b), dtype=x.dtype)
  err = threshold + 1.0
  iterations = 0
  eps = epsilon_0
  while (iterations < max_iterations) and (err >= threshold or eps > epsilon):
    for _ in range(inner_iterations):
      iterations += 1
      g = eps * logb + softmin(cost, f, g, eps, axis=1) + g
      f = eps * loga + softmin(cost, f, g, eps, axis=2) + f
      eps = max(eps * epsilon_decay, epsilon)

    if eps <= epsilon:
      err = error(cost, f, g, eps, b)

  return f, g, eps, cost, d_cost, iterations


def sinkhorn(x,
             y,
             a,
             b,
             **kwargs):
  """Computes the transport between (x, a) and (y, b) via Sinkhorn algorithm."""
  f, g, eps, cost, _, _ = sinkhorn_iterations(x, y, a, b, **kwargs)
  return transport(cost, f, g, eps)
