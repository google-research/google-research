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

"""Tests for stochastic_descent_lib."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from bangbang_qaoa import circuit_lib
from bangbang_qaoa import stochastic_descent_lib
from bangbang_qaoa.two_sat import dnf_circuit_lib
from bangbang_qaoa.two_sat import dnf_lib


class StochasticDescentTest(parameterized.TestCase):

  def test_get_all_protocols_single(self):
    self.assertCountEqual(
        list(stochastic_descent_lib.get_all_protocols(1)),
        [
            (
                circuit_lib.HamiltonianType.X,
            ),
            (
                circuit_lib.HamiltonianType.CONSTRAINT,
            )
        ])

  def test_get_all_protocols_multiple(self):
    self.assertCountEqual(
        list(stochastic_descent_lib.get_all_protocols(3)),
        [
            (
                circuit_lib.HamiltonianType.X,
                circuit_lib.HamiltonianType.X,
                circuit_lib.HamiltonianType.X
            ),
            (
                circuit_lib.HamiltonianType.X,
                circuit_lib.HamiltonianType.X,
                circuit_lib.HamiltonianType.CONSTRAINT
            ),
            (
                circuit_lib.HamiltonianType.X,
                circuit_lib.HamiltonianType.CONSTRAINT,
                circuit_lib.HamiltonianType.X
            ),
            (
                circuit_lib.HamiltonianType.X,
                circuit_lib.HamiltonianType.CONSTRAINT,
                circuit_lib.HamiltonianType.CONSTRAINT
            ),
            (
                circuit_lib.HamiltonianType.CONSTRAINT,
                circuit_lib.HamiltonianType.X,
                circuit_lib.HamiltonianType.X
            ),
            (
                circuit_lib.HamiltonianType.CONSTRAINT,
                circuit_lib.HamiltonianType.X,
                circuit_lib.HamiltonianType.CONSTRAINT
            ),
            (
                circuit_lib.HamiltonianType.CONSTRAINT,
                circuit_lib.HamiltonianType.CONSTRAINT,
                circuit_lib.HamiltonianType.X
            ),
            (
                circuit_lib.HamiltonianType.CONSTRAINT,
                circuit_lib.HamiltonianType.CONSTRAINT,
                circuit_lib.HamiltonianType.CONSTRAINT
            )
        ])

  def test_get_all_protocols_neg_num_chunks(self):
    with self.assertRaisesRegex(
        ValueError, 'num_chunks should be positive, not -3'):
      stochastic_descent_lib.get_all_protocols(-3)

  @parameterized.parameters(
      (0,
       [circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.CONSTRAINT]),
      (1,
       [circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.CONSTRAINT]),
      (2,
       [circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.X]),
      (3,
       [circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.X]),
      )
  def test_get_random_protocol(self, seed, expected_protocol):
    self.assertListEqual(
        stochastic_descent_lib.get_random_protocol(
            num_chunks=5, random_state=np.random.RandomState(seed)),
        expected_protocol)

  def test_get_random_protocol_neg_chunks(self):
    with self.assertRaisesRegex(
        ValueError, 'num_chunks should be positive, not -4'):
      stochastic_descent_lib.get_random_protocol(-4)

  @parameterized.parameters(
      (0,
       [circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.CONSTRAINT]),
      (1,
       [circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.CONSTRAINT]),
      (2,
       [circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.CONSTRAINT]),
      (3,
       [circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.CONSTRAINT]),
      )
  def test_get_random_adiabatic_protocol_ascending(
      self, seed, expected_protocol):
    self.assertListEqual(
        stochastic_descent_lib.get_random_adiabatic_protocol(
            num_chunks=5,
            ascending=True,
            random_state=np.random.RandomState(seed)),
        expected_protocol)

  @parameterized.parameters(
      (0,
       [circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.X]),
      (1,
       [circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.X]),
      (2,
       [circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.X]),
      (3,
       [circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.X]),
      )
  def test_get_random_adiabatic_protocol_descending(
      self, seed, expected_protocol):
    self.assertListEqual(
        stochastic_descent_lib.get_random_adiabatic_protocol(
            num_chunks=5,
            ascending=False,
            random_state=np.random.RandomState(seed)),
        expected_protocol)

  def test_get_random_adiabatic_protocol_neg_chunks(self):
    with self.assertRaisesRegex(
        ValueError, 'num_chunks should be positive, not -4'):
      stochastic_descent_lib.get_random_adiabatic_protocol(
          num_chunks=-4, random_state=None)

  @parameterized.parameters(
      (
          [
              circuit_lib.HamiltonianType.CONSTRAINT,
              circuit_lib.HamiltonianType.CONSTRAINT,
              circuit_lib.HamiltonianType.CONSTRAINT
          ],
          [],
          [
              circuit_lib.HamiltonianType.CONSTRAINT,
              circuit_lib.HamiltonianType.CONSTRAINT,
              circuit_lib.HamiltonianType.CONSTRAINT
          ]
      ),
      (
          [
              circuit_lib.HamiltonianType.X,
              circuit_lib.HamiltonianType.CONSTRAINT,
              circuit_lib.HamiltonianType.X
          ],
          [2],
          [
              circuit_lib.HamiltonianType.X,
              circuit_lib.HamiltonianType.CONSTRAINT,
              circuit_lib.HamiltonianType.CONSTRAINT
          ],
      ),
      (
          [
              circuit_lib.HamiltonianType.X,
              circuit_lib.HamiltonianType.CONSTRAINT,
              circuit_lib.HamiltonianType.CONSTRAINT
          ],
          [0, 1, 2],
          [
              circuit_lib.HamiltonianType.CONSTRAINT,
              circuit_lib.HamiltonianType.X,
              circuit_lib.HamiltonianType.X
          ],
      ),
  )
  def test_apply_changes_to_protocol(self, starting_protocol, changes,
                                     expected_protocol):
    self.assertEqual(
        stochastic_descent_lib._apply_changes_to_protocol(starting_protocol,
                                                          changes),
        expected_protocol)

  @parameterized.parameters(
      ([-1, 3], 'Each index should be between 0 and 2, not -1'),
      ([4], 'Each index should be between 0 and 2, not 4'),
  )
  def test_apply_changes_to_protocol_neg_max_num_flips(self,
                                                       changes,
                                                       expected_error):
    protocol = [
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.X
    ]

    with self.assertRaisesRegex(
        IndexError,
        expected_error
    ):
      stochastic_descent_lib._apply_changes_to_protocol(protocol, changes)

  def test_get_all_changes(self):
    self.assertEqual(
        list(stochastic_descent_lib._get_all_changes(5, 2)),
        [(0,), (1,), (2,), (3,), (4,), (0, 1,), (0, 2,), (0, 3,), (0, 4,),
         (1, 2,), (1, 3,), (1, 4,), (2, 3,), (2, 4,), (3, 4,)])

  def test_get_all_changes_neg_max_num_flips(self):
    with self.assertRaisesRegex(
        ValueError, 'max_num_flips should be positive, not -5'):
      stochastic_descent_lib._get_all_changes(10, -5)

  def test_get_all_changes_num_chunks_too_small(self):
    with self.assertRaisesRegex(
        ValueError, 'num_chunks should be at least max_num_flips'):
      stochastic_descent_lib._get_all_changes(5, 10)

  def test_get_all_new_protocols(self):
    protocol = [
        circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.X
    ]
    self.assertCountEqual(
        list(stochastic_descent_lib.get_all_new_protocols(protocol, 2)),
        [
            [
                circuit_lib.HamiltonianType.X,
                circuit_lib.HamiltonianType.CONSTRAINT,
                circuit_lib.HamiltonianType.CONSTRAINT
            ],
            [
                circuit_lib.HamiltonianType.X,
                circuit_lib.HamiltonianType.X,
                circuit_lib.HamiltonianType.X
            ],
            [
                circuit_lib.HamiltonianType.CONSTRAINT,
                circuit_lib.HamiltonianType.CONSTRAINT,
                circuit_lib.HamiltonianType.X
            ],
            [
                circuit_lib.HamiltonianType.X,
                circuit_lib.HamiltonianType.X,
                circuit_lib.HamiltonianType.CONSTRAINT
            ],
            [
                circuit_lib.HamiltonianType.CONSTRAINT,
                circuit_lib.HamiltonianType.CONSTRAINT,
                circuit_lib.HamiltonianType.CONSTRAINT
            ],
            [
                circuit_lib.HamiltonianType.CONSTRAINT,
                circuit_lib.HamiltonianType.X,
                circuit_lib.HamiltonianType.X
            ],
        ])

  @parameterized.parameters(
      (True, -3, -2, True),
      (True, -1, 3, True),
      (True, 0, 4, True),
      (True, -10, 0, True),
      (True, 2, 25, True),
      (True, 5, 0, False),
      (True, 209423, 23, False),
      (True, 32, -1, False),
      (True, 8, 8, False),
      (False, 5, 2, True),
      (False, -1, -10, True),
      (False, 0, -3, True),
      (False, 2, 5, False),
      (False, 0, 0, False),
      (False, -5, -3, False),
  )
  def test_more_optimal(self, minimize, new_val, old_val, expected_output):
    self.assertEqual(
        stochastic_descent_lib._more_optimal(minimize, new_val, old_val),
        expected_output)

  def test_stochastic_descent_epoch_minimize(self):
    # Every 2-SAT with just one clause will not be satisfied by 1/4 of all
    # possible literal assignments.
    dnf = dnf_lib.DNF(3, [dnf_lib.Clause(0, 1, False, True)])
    circuit = dnf_circuit_lib.BangBangProtocolCircuit(1, dnf)
    protocol, evaluation = stochastic_descent_lib._stochastic_descent_epoch(
        circuit=circuit,
        bangbang_protocol=[
            circuit_lib.HamiltonianType.X,
            circuit_lib.HamiltonianType.X,
            circuit_lib.HamiltonianType.CONSTRAINT
        ],
        max_num_flips=2,
        previous_eval=0.75,
        minimize=True
    )
    self.assertLen(protocol, 3)
    self.assertIsInstance(evaluation, float)
    self.assertLessEqual(evaluation, 0.75)

  def test_stochastic_descent_epoch_maximize(self):
    # Every 2-SAT with just one clause will not be satisfied by 1/4 of all
    # possible literal assignments.
    dnf = dnf_lib.DNF(
        5,
        [dnf_lib.Clause(0, 1, False, True), dnf_lib.Clause(2, 3, True, True)])
    circuit = dnf_circuit_lib.BangBangProtocolCircuit(1, dnf)
    protocol, evaluation = stochastic_descent_lib._stochastic_descent_epoch(
        circuit=circuit,
        bangbang_protocol=[
            circuit_lib.HamiltonianType.X,
            circuit_lib.HamiltonianType.X,
            circuit_lib.HamiltonianType.X
        ],
        max_num_flips=2,
        previous_eval=0.75,
        minimize=False
    )
    self.assertLen(protocol, 3)
    self.assertIsInstance(evaluation, float)
    self.assertGreaterEqual(evaluation, 0.75)

  def test_stochastic_descent_epoch_neg_max_num_flips(self):
    dnf = dnf_lib.DNF(3, [dnf_lib.Clause(0, 1, False, True)])
    circuit = dnf_circuit_lib.BangBangProtocolCircuit(1, dnf)
    with self.assertRaisesRegex(
        ValueError,
        'max_num_flips should be positive, not 0'
    ):
      stochastic_descent_lib._stochastic_descent_epoch(
          circuit=circuit,
          bangbang_protocol=[
              circuit_lib.HamiltonianType.X,
              circuit_lib.HamiltonianType.X,
              circuit_lib.HamiltonianType.CONSTRAINT
          ],
          max_num_flips=0,
          previous_eval=0.75,
          minimize=False
      )

  def test_stochastic_descent_minimize(self):
    # Every 2-SAT with just one clause will not be satisfied by 1/4 of all
    # possible literal assignments.
    # With only one bang, the only way to do better than random guessing is to
    # apply the DNF hamiltonian.
    dnf = dnf_lib.DNF(3, [dnf_lib.Clause(0, 1, False, True)])
    circuit = dnf_circuit_lib.BangBangProtocolCircuit(1, dnf)
    random_protocol = stochastic_descent_lib.get_random_protocol(5)
    random_eval = circuit.get_constraint_expectation(
        circuit.get_wavefunction(random_protocol))
    protocol, evaluation, num_epoch = stochastic_descent_lib.stochastic_descent(
        circuit=circuit,
        max_num_flips=2,
        initial_protocol=stochastic_descent_lib.get_random_protocol(5),
        minimize=True)
    self.assertLen(protocol, 5)
    self.assertIsInstance(evaluation, float)
    self.assertLessEqual(evaluation, random_eval)
    # Contain at least 1 epoch of stochastic descent.
    self.assertGreaterEqual(num_epoch, 1)

  def test_stochastic_descent_maximize(self):
    # Every 2-SAT with just one clause will not be satisfied by 1/4 of all
    # possible literal assignments.
    # With only one bang, the only way to do better than random guessing is to
    # apply the DNF hamiltonian.
    dnf = dnf_lib.DNF(2, [dnf_lib.Clause(0, 1, True, True)])
    circuit = dnf_circuit_lib.BangBangProtocolCircuit(1, dnf)
    random_protocol = stochastic_descent_lib.get_random_protocol(2)
    random_eval = circuit.get_constraint_expectation(
        circuit.get_wavefunction(random_protocol))
    protocol, evaluation, num_epoch = stochastic_descent_lib.stochastic_descent(
        circuit=circuit,
        max_num_flips=2,
        initial_protocol=stochastic_descent_lib.get_random_protocol(5),
        minimize=False)
    self.assertLen(protocol, 5)
    self.assertIsInstance(evaluation, float)
    self.assertGreaterEqual(evaluation, random_eval)
    # Contain at least 1 epoch of stochastic descent.
    self.assertGreaterEqual(num_epoch, 1)

  def test_stochastic_descent_neg_max_num_flips(self):
    # Every 2-SAT with just one clause will not be satisfied by 1/4 of all
    # possible literal assignments.
    # With only one bang, the only way to do better than random guessing is to
    # apply the DNF hamiltonian.
    dnf = dnf_lib.DNF(3, [dnf_lib.Clause(0, 1, False, True)])
    circuit = dnf_circuit_lib.BangBangProtocolCircuit(1, dnf)
    with self.assertRaisesRegex(
        ValueError, 'max_num_flips should be positive, not -10'):
      stochastic_descent_lib.stochastic_descent(
          circuit=circuit,
          max_num_flips=-10,
          initial_protocol=stochastic_descent_lib.get_random_protocol(5),
          minimize=False)

  def test_stochastic_descent_skip_search(self):
    dnf = dnf_lib.DNF(2, [dnf_lib.Clause(0, 1, True, True)])
    circuit = dnf_circuit_lib.BangBangProtocolCircuit(1, dnf)
    random_protocol = stochastic_descent_lib.get_random_protocol(2)
    random_eval = circuit.get_constraint_expectation(
        circuit.get_wavefunction(random_protocol))
    protocol, evaluation, num_epoch = stochastic_descent_lib.stochastic_descent(
        circuit=circuit,
        max_num_flips=1,
        initial_protocol=random_protocol,
        minimize=False,
        skip_search=True)
    self.assertListEqual(protocol, random_protocol)
    self.assertIsInstance(evaluation, float)
    self.assertAlmostEqual(evaluation, random_eval)
    # Zero epoch of stochastic descent.
    self.assertEqual(num_epoch, 0)


if __name__ == '__main__':
  absltest.main()
