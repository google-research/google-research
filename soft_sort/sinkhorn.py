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

import tensorflow.compat.v2 as tf


def build_distances(x, y):
  """Computes the distance matrix in 1D."""
  xn = tf.tile(x[:, :, tf.newaxis], (1, 1, tf.shape(y)[1]))
  ym = tf.transpose(
      tf.tile(y[:, :, tf.newaxis], (1, 1, tf.shape(x)[1])), (0, 2, 1))
  return tf.math.abs(xn - ym)


@tf.function
def sinkhorn(
    x, y, a, b, eps, p, threshold, inner_num_iter=20, max_iterations=1000):
  """The Sinkhorn algorithm in 1D.

  Computes the Sinkhorn's algorithm to transport the points x with weights a
  onto the points y with weights b.

  Args:
   x: the input Tensor[batch, n].
   y: the target Tensor[batch, m].
   a: the weights tensor associated to x (same shape).
   b: the weights tensor associated to y (same shape).
   eps: the value of the entropic regularization parameter.
   p: the power to be applied the distance kernel.
   threshold: the value of sinkhorn error below which to stop iterating.
   inner_num_iter: the number of iteration before computing the error.
   max_iterations: the total number of iterations.

  Returns:
   a tuple: (transport matrix, sinkhorn error, num iterations done).
  """
  v = tf.ones(tf.shape(b), dtype=x.dtype)
  u = tf.ones(tf.shape(a), dtype=x.dtype)
  c = tf.pow(build_distances(x, y), p)
  k = tf.exp(-c / eps)
  kt = tf.transpose(k, (0, 2, 1))

  def body_fn(u, v, err, num_iter):
    """A small loop of Sinkhorn iterations."""
    del err  # Unused.
    for _ in range(inner_num_iter):
      u = a / tf.linalg.matvec(k, v)
      v = b / tf.linalg.matvec(kt, u)
    u = a / tf.linalg.matvec(k, v)
    return u, v, error(u, v), num_iter + inner_num_iter

  def error(u, v):
    """Computes the maximum relative Sinkhorn error."""
    b_target = v * tf.linalg.matvec(kt, u)
    return tf.reduce_max(tf.abs(b_target - b) / b, axis=None)

  def cond_fn(u, v, err, num_iter):
    """The condition to stop Sinkhorn's iterations."""
    del u, v, num_iter  # Unused.
    return err >= threshold

  num_iter = tf.constant(0, dtype=tf.int32)
  max_iterations //= inner_num_iter
  err = tf.constant(10.0, dtype=x.dtype)
  u, v, err, num_iter = tf.while_loop(
      cond_fn, body_fn, [u, v, err, num_iter],
      parallel_iterations=1, maximum_iterations=max_iterations)
  transport = u[:, :, tf.newaxis] * k * v[:, tf.newaxis, :]
  return transport, err, num_iter


@tf.function
def log_sinkhorn(
    x, y, a, b, eps, p, threshold, inner_num_iter=20, max_iterations=1000):
  """The stabilized Sinkhorn algorithm in log space.

  Computes the stabilized version of the Sinkhorn's algorithm to transport the
  points x with weights a onto the points y with weights b.

  Args:
   x: the input Tensor[batch, n].
   y: the target Tensor[batch, m].
   a: the weights tensor associated to x (same shape).
   b: the weights tensor associated to y (same shape).
   eps: the value of the entropic regularization parameter.
   p: the power to be applied the distance kernel.
   threshold: the value of sinkhorn error below which to stop iterating.
   inner_num_iter: the number of iteration before computing the error.
   max_iterations: the total number of iterations.

  Returns:
   a tuple: transport matrix, sinkhorn error, num iterations done.
  """
  c = tf.pow(build_distances(x, y), p)
  eps = tf.cast(eps, dtype=x.dtype)

  def center(f, g):
    return c - f[:, :, tf.newaxis] - g[:, tf.newaxis, :]

  def softmin(f, g, eps, axis):
    return -eps * tf.reduce_logsumexp(-center(f, g) / eps, axis=axis)

  def error(f, g):
    """Computes the maximum relative sinkhorn error."""
    b_target = tf.math.reduce_sum(tf.math.exp(-center(f, g) / eps), axis=1)
    return tf.reduce_max(tf.abs(b_target - b) / b, axis=None)

  loga = tf.math.log(a)
  logb = tf.math.log(b)
  f = tf.zeros(tf.shape(loga), dtype=x.dtype)
  g = tf.zeros(tf.shape(logb), dtype=x.dtype)
  num_iter = tf.constant(0, dtype=tf.int32)

  def body_fn(f, g, err, num_iter):
    """A small loop of N Sinkhorn iterations."""
    del err  # Unused.
    for _ in range(inner_num_iter):
      g = eps * logb + softmin(f, g, eps, axis=1) + g
      f = eps * loga + softmin(f, g, eps, axis=2) + f

    return [f, g, error(f, g), num_iter + inner_num_iter]

  def cond_fn(f, g, err, num_iter):
    """The condition to stop Sinkhorn's iterations."""
    del num_iter, f, g  # Unused.
    return err >= threshold

  max_iterations //= inner_num_iter
  err = tf.constant(10.0, dtype=x.dtype)
  f, g, err, num_iter = tf.while_loop(
      cond_fn, body_fn, [f, g, err, num_iter],
      parallel_iterations=1, maximum_iterations=max_iterations)

  return tf.math.exp(-center(f, g) / eps), err, num_iter
