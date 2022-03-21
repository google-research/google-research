# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

# pylint: disable=invalid-name
"""Module to evaluate Chebyshev polynomials.

Whenever possible, we try to follow the same API as the functions
scipy.special.eval_cheby*
"""
from jax import numpy as jnp
from sobolev import utils


def rec_chebyt_classical():
  """Recurrence for the classical Chebyshev polynomial."""
  yield 0, 1, -1
  while True:
    yield 0, 2, -1


def rec_chebyt_residual(low, high):
  """Recurrence for the Residual Chebyshev polynomial."""
  # Reference: https://fa.bianp.net/blog/2020/polyopt/
  yield 1, - 2 / (low + high), 0
  rho = (high - low) / (high + low)
  wt = 2
  while True:
    wt = 1 / (1 - rho * rho * wt / 4)
    yield wt, -2 * wt / (low + high), 1 - wt


def eval_chebyt(n,
                x,
                low=-1.,
                high=1.,
                normalization="classical"):
  """Evaluate Chebyshev polynomial of the first kind at a point.

  Args:
    n : positive integer.
      Degree of the polynomial, must be an integer.
    x : array_like,
      Points at which to evaluate the Chebyshev polynomial.
    low: float.
      Lower bound for the interval of orthogonality, which is (low, high).
    high: float.
      Upper bound for the interval of orthogonality, which is (low, high).
    normalization : string, one of {'classical', 'residual'}.
      Type of normalization to use. 'classical' will use the classical
      definition of Chebyshev polynomials defined by the recurrence
      P_{t+1}(x) = 2 x P_t(x) - P_{t-1}(x), P_1(x) = x, P_0(x) = 1
      for polynomials with interval of orthogonality (-1, 1).
      'residual' normalization will normalize the polynomials so that P_t(0) = 1
      for all degrees t.

  Returns:
    T : ndarray
      Values of the Chebyshev polynomial

  Notes:
    This function follows the same API as scipy.special.eval_chebyt
  """
  if not isinstance(n, int):
    raise NotImplementedError(
        "This function is currently implemented only for " +
        "integer values of n")
  # TODO(pedregosa): what happens if n is negative?

  if normalization == "classical":
    if low != -1. and high != 1.:
      raise NotImplementedError(
          "Shifted Chebyshev polynomials are not implemented with the classical"
          + "normalization")
    coefs = rec_chebyt_classical()
  elif normalization == "residual":
    if low <= 0:
      raise ValueError(
          "For residual polynomials, low should be a positive number")
    coefs = rec_chebyt_residual(low, high)
  return utils.eval_three_term(n, coefs, x)
