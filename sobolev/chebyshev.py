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

# pylint: disable=invalid-name
"""Module to evaluate Chebyshev polynomials.

Whenever possible, we try to follow the same API as the functions
scipy.special.eval_cheby*
"""
import math
from jax import numpy as jnp
from sobolev import utils


def recurrence_chebyt(
    low = -1.,
    high = 1.,
    normalization = "classical"):
  """Recurrence for Chebyshev polynomials of the first kind.

  Returns coefficients (a_t, b_t, c_t) in the recurrence

      P_{t+1}(x) = (a_t + b_t x) P_t(x) + c_t P_{t-1}(x)

  Args:
    low: float Lower bound for the interval of orthogonality, which is
      (low, high).
    high : float. Upper bound for the interval of orthogonality, which is (low,
      high).
    normalization : string, one of {'classical', 'residual'}. Type of
      normalization to use. 'classical' will use the classical definition of
      Chebyshev polynomials defined by the recurrence P_{t+1}(x) = 2 x P_t(x) -
      P_{t-1}(x), P_1(x) = x, P_0(x) = 1 for polynomials with interval of
      orthogonality (-1, 1). 'residual' normalization will normalize the
      polynomials so that P_t(0) = 1 for all degrees t.

  Yields:
    Coefficients (at, bt, ct) in the recurrence.

  Notes:
    Reference for the residual normalization:
    https://fa.bianp.net/blog/2020/polyopt/
  """
  if normalization == "classical":
    if not (math.isclose(low, -1.) and math.isclose(high, 1.)):
      raise NotImplementedError(
          "Shifted Chebyshev polynomials are only implemented with the residual"
          + " normalization")
    yield 0., 1., -1.
    while True:
      yield 0., 2., -1.
  elif normalization == "residual":
    if low <= 0:
      raise ValueError(
          "For residual polynomials, low should be a positive number")
    yield 1., -2 / (low + high), 0.
    rho = (high - low) / (high + low)
    wt = 2.
    while True:
      wt = 1.0 / (1 - rho * rho * wt / 4)
      yield wt, -2 * wt / (low + high), 1 - wt
  else:
    raise NotImplementedError


def eval_chebyt(n,
                x,
                low = -1.,
                high = 1.,
                normalization = "classical"):
  """Evaluate Chebyshev polynomial of the first kind.

  Args:
    n: positive integer. Degree of the polynomial, must be an integer.
    x: array_like. Points at which to evaluate the Chebyshev polynomial.
    low: float. Lower bound for the interval of orthogonality, which is (low,
      high).
    high: float. Upper bound for the interval of orthogonality, which is (low,
      high).
    normalization: string, one of {'classical', 'residual'}. Type of
      normalization to use. 'classical' will use the classical definition of
      Chebyshev polynomials defined by the recurrence P_{t+1}(x) = 2 x P_t(x) -
      P_{t-1}(x), P_1(x) = x, P_0(x) = 1 for polynomials with interval of
      orthogonality (-1, 1). 'residual' normalization will normalize the
      polynomials so that P_t(0) = 1 for all degrees t.

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
  coefs = recurrence_chebyt(low, high, normalization)
  return utils.eval_three_term(n, coefs, x)
