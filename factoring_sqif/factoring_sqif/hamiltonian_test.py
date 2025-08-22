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
import os

import cirq
import cupy
from factoring_sqif import closest_vector_problem as cvp
from factoring_sqif import hamiltonian
import numpy as np
import pytest


@pytest.mark.parametrize('num_qubits', [3, 10])
def test_hamiltonian(num_qubits):
  with open(
      os.path.dirname(__file__)
      + f'/test_data/{num_qubits}q/low_energy_states.txt'
  ) as f:
    expected_energies, expected_states = eval(f.readline())

  h = hamiltonian.get_example_problem_hamiltonian(num_qubits)
  low_energy_states = hamiltonian.brute_force_lowest_energy_states(h)[
      : len(expected_states)
  ].get()
  energies = hamiltonian.energy_from_integer_states(h, low_energy_states)

  np.testing.assert_array_almost_equal(energies, expected_energies)
  for state, binary_repr in zip(low_energy_states, expected_states):
    assert f'{state:0{num_qubits}b}' == binary_repr


def test_hamiltonian_construction():
  D = cvp.IntegerMatrix.from_matrix(
      [[1, -4, -3], [-2, 1, 2], [2, 2, 0], [3, -2, 4]]
  )
  babai_result = cvp.babai_algorithm(D, [0, 0, 0, 240])
  qs = cirq.LineQid.range(3, dimension=2)
  H = hamiltonian.hamiltonian_from_babai_result(qs, babai_result)
  want = hamiltonian.get_example_problem_hamiltonian(3)
  assert H == want


def test_u_v_pairs_from_integer_states():
  D = [[1, -4, -3], [-2, 1, 2], [2, 2, 0], [3, -2, 4]]
  babai_result = cvp.babai_algorithm(
      cvp.IntegerMatrix.from_matrix(D), [0, 0, 0, 240]
  )
  np.testing.assert_array_almost_equal(babai_result.prime_basis, np.asarray(D))
  babai_result.prime_basis = np.asarray(
      [[1, 0, 0], [0, 1, 0], [0, 0, 2], [22, 35, 51]]
  )
  with open(
      os.path.dirname(__file__) + '/test_data/3q/low_energy_states.txt'
  ) as f:
    _, expected_states = eval(f.readline())
  expected_states = cupy.array([int(x, 2) for x in expected_states], dtype=int)
  lattice_vectors = hamiltonian.integer_states_to_lattice_vectors(
      expected_states, babai_result
  )
  u_v_pairs = hamiltonian.u_v_pairs_from_lattice_vectors(
      lattice_vectors, babai_result.prime_basis
  )
  assert [tuple(uv) for uv in u_v_pairs] == [
      (1800, 1),
      (1944, 1),
      (2025, 1),
      (3645, 2),
  ]


def test_sr_pairs_from_uv_pairs():
  uv_pairs = [(1800, 1), (1944, 1), (2025, 1), (3645, 2)]
  smooth_bound = 15
  N = 1961
  sr_pairs = [(1800, 1), (1944, 1), (2025, 1)]
  assert (
      hamiltonian.sr_pairs_from_uv_pairs(uv_pairs, N, smooth_bound) == sr_pairs
  )
