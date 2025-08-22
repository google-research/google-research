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

# pylint: skip-file
from dataclasses import dataclass
import logging
import math
from typing import Sequence, Tuple

from factoring_sqif import fpylll_helpers, number_theory
from fpylll import GSO, IntegerMatrix, LLL
import numpy as np


def sample(
    N, n, c, rs
):
  """Returns a random sample of the CVP problem from https://arxiv.org/pdf/2212.12372.pdf p13.

  Args:
      N: The number to factor.
      n: lattice dimension.
      c: precision parameter.
      rs: RandomState.

  Returns:
      B: An (n+1)xn IntegerMatrix constructed as instructed in
      https://arxiv.org/pdf/2212.12372.pdf p13.
      t: A sparse vector with a single non-zero element (last entry).

  Raises:
      NotImplemented: if n is larger than 25.
  """
  f = rs.permutation([(i + 1) // 2 for i in range(1, n + 1)])
  last_row = np.round(10**c * np.log(number_theory.first_n_primes(n)))
  B = [[0] * n for _ in range(n)] + [[int(x) for x in last_row]]
  for i in range(n):
    B[i][i] = int(f[i])  # fill diagonal
  B = IntegerMatrix.from_matrix(B)

  t = [0] * (n + 1)
  t[-1] = int(round(10**c * math.log(N)))
  return B, tuple(t)


def cvp_solution_to_smooth_pair(prime_exponents):
  n, e = prime_exponents.shape
  # Need to use dtype=object to prevent integer overflow when computing prime powers.
  primes = np.array(number_theory.first_n_primes(e), dtype=object)
  u_exp = np.multiply(prime_exponents, (prime_exponents > 0).astype(int))
  v_exp = np.multiply(-prime_exponents, (prime_exponents < 0).astype(int))
  u = np.prod(np.power(primes, u_exp), axis=1)
  v = np.prod(np.power(primes, v_exp), axis=1)
  return np.stack((u, v), axis=-1)


@dataclass
class BabaiResult:
  prime_basis: np.ndarray
  lll_reduced_basis: np.ndarray
  weights: np.ndarray
  residual_vector: np.ndarray
  rounding_direction: Tuple[bool, Ellipsis]


def babai_algorithm(
    B, t, delta = 0.75
):
  """Returns an approximate solution to CVP.

      Reference
      https://github.com/fplll/fpylll/blob/master/docs/tutorial.rst#svp-and-cvp-tools
  Args:
      B: Integer matrix whose columns are the lattice vectors.
      t: the target vector.
      delta: the delta parameter for the LLL algorithm.

  Returns:
      - difference between an approximate solution and the target vector.
      - the weights for the approximate solution.
  """
  prime_basis = fpylll_helpers.integer_matrix_to_numpy(B)
  D = B.transpose()
  LLL.reduction(D, delta=delta)
  logging.debug('D =\n' + str(D))
  M = GSO.Mat(D, update=True)
  w = M.babai(t)

  A = IntegerMatrix(2 * D.nrows, D.ncols)
  for i in range(D.nrows):
    for j in range(D.ncols):
      A[i, j] = D[i, j]

  b = np.array(t)
  for i in reversed(range(D.nrows)):
    for j in range(D.ncols):
      A[i + D.nrows, j] = int(b[j])
    b -= w[i] * np.array(D[i])

  M = GSO.Mat(A, update=True)
  rounding_direction = []
  for i in range(D.nrows):
    mu = M.get_mu(i + D.nrows, i)
    logging.debug(f'\t{mu=} c={w[i]}')
    rounding_direction.append(w[i] > mu)
  return BabaiResult(
      prime_basis=prime_basis,
      residual_vector=np.array(t) - np.array(D.multiply_left(w)),
      weights=w,
      lll_reduced_basis=fpylll_helpers.integer_matrix_to_numpy(D.transpose()),
      rounding_direction=tuple(rounding_direction),
  )
