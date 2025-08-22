# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Library of utils to smooth arrays / enforce monotonicity."""

import numpy as np
import scipy


# pylint: disable=invalid-name

cumulative_min = np.minimum.accumulate
none = lambda x: x


def rolling_mean(array, window = 10):
  return np.append(
      [np.nan] * (window - 1),
      np.convolve(array, np.ones(window), mode='valid') / window,
  )


def isotonic(array):
  """Enforces array to be monotonically decreasing."""
  return scipy.optimize.isotonic_regression(array, increasing=False).x


def _decreasing_sequence(n):
  """Matrix that encodes the constraint x_{i+1} <= x_i."""
  return scipy.sparse.diags(
      [1, -1], offsets=[0, -1], shape=(n, n), format='csr'
  )[1:]


def _decreasing_gaps(n):
  """Matrix that encodes the constraint x_i - x_{i+1} <= x_{i-1} - x_i."""
  return scipy.sparse.diags(
      [-1, 2, -1], offsets=[1, 0, -1], shape=(n, n), format='csr'
  )[1:-1]


def smooth_second_order(y):
  """Smooths an array so that the values and differences are non-increasing."""
  n = y.size
  A = scipy.sparse.vstack([_decreasing_sequence(n), _decreasing_gaps(n)])

  def loss_and_grad(x):
    diff = x - y
    return 0.5 * (diff @ diff), diff

  return scipy.optimize.minimize(
      loss_and_grad,
      x0=y.copy(),
      jac=True,
      constraints=scipy.optimize.LinearConstraint(A, ub=0),
  ).x


def smooth_second_order_2d(
    Y, decreasing_gaps_axes = (0,)
):
  """Smooths a 2D array to satisfy certain monotonicity properties.

  Specifically, the returned array has the same shape as the input, and the
  values are non-increasing along both axes, while the differences are
  non-increasing along the first axis.  This function solves the following
  linear program:

  minimize_{X} || X - Y ||_1
  s.t. X[i+1] <= X[i]  (values decreasing along first axis)
       X[:, i+1] <= X[:, i]  (values decreasing along second axis)
       # differences decreasing along first/second axis
       X[i] - X[i+1] <= X[i-1] - X[i] (if 0 in decreasing_gaps_axes)
       X[:, i] - X[:, i+1] <= X[:, i-1] - X[:, i] (if 1 in decreasing_gaps_axes)

  Second order smoothing may be desirable in some settings, to e.g., enforce
  that the decrease in Cross Entropy is larger from step 1000 to 1100 than from
  1100 to 1200 (i.e., impose the law of decreasing marginal returns).  Imposing
  this constraint makes the fits a bit more smooth, while still falling under
  the umbrella of a non-parameteric approach.

  Args:
    Y: The 2D array to smooth.
    decreasing_gaps_axes: The axes along which the differences should be
      monotonically decreasing.

  Returns:
    A smoothed version of the input array.
  """
  y = Y.flatten()
  n, m = Y.shape

  A1 = (
      scipy.sparse.vstack([_decreasing_sequence(n), _decreasing_gaps(n)])
      if 0 in decreasing_gaps_axes
      else _decreasing_sequence(n)
  )

  A2 = (
      scipy.sparse.vstack([_decreasing_sequence(m), _decreasing_gaps(m)])
      if 1 in decreasing_gaps_axes
      else _decreasing_sequence(m)
  )

  A = scipy.sparse.vstack([
      scipy.sparse.kron(A1, scipy.sparse.eye(m)),
      scipy.sparse.kron(scipy.sparse.eye(n), A2),
  ])

  # We introduce auxialary variables t representing abs(x - y)
  Z = scipy.sparse.csr_matrix((A.shape[0], n * m))
  I = scipy.sparse.eye(n * m)

  A_ub = scipy.sparse.vstack([
      scipy.sparse.hstack([A, Z]),
      scipy.sparse.hstack([I, -I]),
      scipy.sparse.hstack([-I, -I]),
  ])

  b_ub = np.concatenate([np.zeros(A.shape[0]), y, -y])

  c = np.append(np.zeros(n * m), np.ones(n * m))

  return scipy.optimize.linprog(c, A_ub, b_ub).x[: n * m].reshape(n, m)
