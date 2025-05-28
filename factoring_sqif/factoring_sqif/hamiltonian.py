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
from collections import defaultdict
import functools
import logging
import time
from typing import Callable, DefaultDict, Optional, Sequence, Tuple, TypeVar

import cirq
import cupy
from factoring_sqif import closest_vector_problem as cvp
from factoring_sqif import number_theory
import numba
import numpy as np


@numba.jit(nopython=True)
def _apply_singles(
    single_bits,
    states,
    result,
):
  for bit, coef in single_bits:
    negate_coef = (states >> bit) & 1
    sgn = 1 - 2 * negate_coef  # (-1)**n == 1 - 2(n & 1)
    result += sgn * coef


@numba.jit(forceobj=True)
def _apply_pairs(
    pairs,
    states,
    result,
):
  for (a, b), coef in pairs:
    negate_coef = ((states >> a) & 1) != ((states >> b) & 1)
    sgn = 1 - 2 * negate_coef
    result += sgn * coef


_T = TypeVar('_T')


def _in_batches(
    I,
    states,
    f,
    result,
    dummy,
    batch_size=128,
    cuda = False,
):
  """Applies f in batches appending the last batch with the `dummy` so that the `f` doesn't get recompiled."""
  if cuda:
    states = cupy.array(states)
    res = cupy.zeros_like(result)
    for i in range(0, len(I), batch_size):
      batch = I[i : i + batch_size]
      if len(batch) < batch_size:
        batch = batch + (dummy,) * (batch_size - len(batch))
      f(batch, states, res)
    result += cupy.asnumpy(res)
    del res
    del states
    cupy._default_memory_pool.free_all_blocks()
  else:
    for i in range(0, len(I), batch_size):
      batch = I[i : i + batch_size]
      if len(batch) < batch_size:
        batch = batch + (dummy,) * (batch_size - len(batch))
      f(batch, states, result)


def energy_from_integer_states(
    hamiltonian, states
):
  qubits = hamiltonian.qubits
  single_bits: DefaultDict[int, float] = defaultdict(lambda: 0.0)
  pairs: DefaultDict[Tuple[int, int], float] = defaultdict(lambda: 0.0)

  offset = 0
  for ps in hamiltonian:
    idx = [(len(qubits) - qubits.index(q) - 1) for q in ps.qubits]
    if len(idx) == 1:
      single_bits[idx[0]] += ps.coefficient.real
    elif len(idx) == 2:
      a, b = idx
      if a > b:
        a, b = b, a
      pairs[(a, b)] += ps.coefficient.real
    else:
      offset += ps.coefficient.real
  ct = time.time_ns()
  result = np.zeros(states.shape, dtype=np.float32) + offset
  _in_batches(
      tuple(single_bits.items()),
      states,
      _apply_singles,
      result,
      (100, 0.0),
      len(qubits),
  )
  _in_batches(
      tuple(pairs.items()),
      states,
      _apply_pairs,
      result,
      ((100, 100), 0.0),
      len(qubits),
      cuda=True,
  )
  elapsed = time.time_ns() - ct
  logging.info(f'energy_from_integer_states took {elapsed*1e-6}ms')
  np.testing.assert_array_almost_equal(np.imag(result), 0)
  return np.real(result)


@functools.lru_cache()
def _get_integer_states(n):
  return np.array([*range(2**n)], dtype=int)


def brute_force_lowest_energy_states(
    hamiltonian, num_samples = None
):
  energies = energy_from_integer_states(
      hamiltonian, _get_integer_states(len(hamiltonian.qubits))
  )
  indexes = np.argsort(energies, kind='stable')
  return cupy.array(indexes[:num_samples] if num_samples else indexes)


def get_example_problem_hamiltonian(num_qubits):
  q = cirq.LineQubit.range(num_qubits)
  import os

  with open(
      os.path.dirname(__file__) + f'/test_data/{num_qubits}q/hamiltonian.txt'
  ) as f:
    coefficients = eval(f.readline())
  assert len(coefficients) == (num_qubits * (num_qubits + 1)) // 2 + 1
  H = coefficients[0] * cirq.I(q[0])
  idx = 0
  for i in range(num_qubits):
    for j in range(i + 1, num_qubits):
      idx = idx + 1
      H += cirq.Z(q[i]) * cirq.Z(q[j]) * coefficients[idx]
    idx = idx + 1
    H += cirq.Z(q[i]) * coefficients[idx]
  assert idx == len(coefficients) - 1
  return H


def _get_operators(
    qs, rounding_direction
):
  X = []
  for i, (up, q) in enumerate(zip(rounding_direction, qs)):
    x = (cirq.I(q) + -cirq.Z(q)) / 2
    if up:
      x = -x
    X.append(x)
  return X


def hamiltonian_from_babai_result(
    qs, babai_result
):
  X = _get_operators(qs, babai_result.rounding_direction)
  H = cirq.PauliSum()
  for j in range(babai_result.lll_reduced_basis.shape[0]):
    h = babai_result.residual_vector[j]
    for i in range(babai_result.lll_reduced_basis.shape[1]):
      h -= X[i] * babai_result.lll_reduced_basis[j, i]
    H += h**2
  return H


def ndarray_to_binary(d, m):
  if not isinstance(d, cupy.ndarray):
    d = cupy.asarray(d)
  return (((d[:, None] & (1 << cupy.arange(m)[::-1]))) > 0).astype(int)


def integer_states_to_lattice_vectors(
    states, babai_result
):
  add_if_set = (-1) ** cupy.array(babai_result.rounding_direction, dtype=int)
  weights = cupy.array(babai_result.weights)
  lll_reduced_basis = cupy.array(babai_result.lll_reduced_basis)
  states_bin = ndarray_to_binary(states, len(add_if_set))
  add_bits = cupy.multiply(states_bin, add_if_set)
  new_weights = weights + add_bits
  # For large arrays (states with 2 ** 23 rows), this runs into https://github.com/cupy/cupy/issues/4387
  lattice_vector = new_weights @ lll_reduced_basis.T
  return lattice_vector


def u_v_pairs_from_lattice_vectors(
    lattice_vectors, prime_basis
):
  pinv = cupy.linalg.pinv(cupy.asarray(prime_basis, dtype=int))
  prime_exponents = cupy.rint(lattice_vectors @ pinv.T).astype(int)
  ret = cvp.cvp_solution_to_smooth_pair(prime_exponents.get())
  return ret


def sr_pairs_from_uv_pairs(
    uv_pairs, N, smooth_bound
):
  return [
      tuple(uv)
      for uv in uv_pairs
      if number_theory.is_smooth(abs(int(uv[0]) - N * int(uv[1])), smooth_bound)
  ]
