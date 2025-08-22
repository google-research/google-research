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
from typing import Iterator, List, Optional, Sequence, Tuple

import cupy
import numba
import number_theory
import numpy as np


def sr_pair_to_differences(
    u, v, N, n
):
  diff = u - v * N
  if diff == 0:  # trivial case
    return None
  Ep = [int(diff < 0)]
  E = [0]
  diff = abs(diff)
  for p in number_theory.first_n_primes(n):
    E.append(number_theory.exponent(u, p))
    Ep.append(number_theory.exponent(diff, p))
    diff //= p ** Ep[-1]
  if diff != 1:
    raise ValueError(
        f'u - vN = {u-v*N} contains a prime factor larger than the nth prime.'
    )
  return [ep - e for ep, e in zip(Ep, E)]


def exponents_to_candidates(exponents):
  A = -1 if (exponents[0] & 1) else 1
  B = 1
  for p, e in zip(
      number_theory.first_n_primes(len(exponents) - 1), exponents[1:]
  ):
    if e == 0:
      continue
    if e > 0:
      A *= p**e
    else:
      B *= p ** (-e)
  return abs(A - B), abs(A + B)


@numba.jit(nopython=True)
def _gaussian_elementation_Z2(A):
  i: int = 0
  for p in range(A.shape[1]):
    r = i
    while r < A.shape[0] and A[r, p] == 0:
      r += 1
    if r == A.shape[0]:
      continue
    if r != i:
      # swap rows i and r
      for c in range(p, A.shape[1]):
        A[r, c], A[i, c] = A[i, c], A[r, c]
    for r in range(A.shape[0]):
      if r == i or A[r, p] == 0:
        continue
      A[r, :] = (A[r, :] - A[i, :]) % 2
    i += 1
  return A


def _null_space_basis(A):
  is_free = [True] * A.shape[1]
  for i in range(A.shape[0]):
    cols = [j for j in range(A.shape[1]) if A[i, j]]
    if len(cols) != 0:
      is_free[cols[0]] = False
  free_variables = [i for i in range(len(is_free)) if is_free[i]]
  basis = []
  for f in free_variables:
    values = [-1] * A.shape[1]
    for j in free_variables:
      values[j] = 0
    values[f] = 1
    for r in range(A.shape[0] - 1, -1, -1):
      unknowns = []
      s = 0
      for j in range(A.shape[1]):
        if A[r, j] == 0:
          continue
        if values[j] == -1:
          unknowns.append(j)
        else:
          s += values[j]
      assert (
          len(unknowns) <= 1
      ), f'{A[r, :]} {values} {is_free} {free_variables} {unknowns}'
      s %= 2
      if len(unknowns) == 0:
        assert s == 0, f'{A[r, :]} {values} {is_free} {free_variables}'
      if len(unknowns):
        values[unknowns[0]] = s
    basis.append(cupy.array(values))
  return basis


def _null_space(
    dE, limit = None
):
  """Returns the null space of dE mod \mathbb{Z}_2

  Args:
      dE: the system of equations.
      limit: maximum number of solutions to return since the number of solutions
        is exponential. The number of solutions is equal to $2^{n - rank(dE)}$
        if dE is and m by n matrix.

  Returns:
      min(limit, 2^{n - rank(dE)}) exponents.
  """
  dE = _gaussian_elementation_Z2(dE % 2)
  bases = _null_space_basis(dE)  # number of bases is equal to n - rank(dE)
  if limit is None or limit.bit_length() >= len(bases):
    limit = 1 << len(bases)

  for msk in range(limit):
    s = cupy.zeros((dE.shape[1],), dtype=int)
    for e, b in zip(number_theory.iter_bits(msk, len(bases)), bases):
      s += e * b
    yield s


def differences_to_exponents(
    dE, limit = None
):
  vdE = cupy.array(dE)  # move to gpu
  for t in _null_space(dE % 2, limit=limit):
    yield ((vdE @ t) // 2).tolist()


if __name__ == '__main__':
  A = np.array([[0, 1, 0], [1, 1, 1]])
  A = _gaussian_elementation_Z2(A)
  print(_null_space_basis(A))
