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

"""Tests for two_sat.dnf_circuit_lib."""

import math

from absl.testing import absltest
from absl.testing import parameterized
import cirq
from cirq.circuits import insert_strategy
import numpy as np

from bangbang_qaoa import circuit_lib
from bangbang_qaoa.two_sat import dnf_circuit_lib
from bangbang_qaoa.two_sat import dnf_lib


class HamiltonianGeneratorTest(parameterized.TestCase):

  @parameterized.parameters(
      (True, True, 1),
      (True, False, -1),
      (False, True, -1),
      (False, False, 1),
  )
  def test_get_sign(self, is_negative, other_is_negative, expected):
    self.assertEqual(dnf_circuit_lib._get_sign(is_negative, other_is_negative),
                     expected)

  @parameterized.parameters(
      (
          dnf_lib.Clause(4, 2, True, False),
          1,
          [
              (
                  (cirq.LineQubit(2),),
                  cirq.ZPowGate(exponent=-2/math.pi, global_shift=-0.5)
              ),
              (
                  (cirq.LineQubit(4),),
                  cirq.ZPowGate(exponent=2/math.pi, global_shift=-0.5)
              ),
              (
                  (cirq.LineQubit(2), cirq.LineQubit(4)),
                  cirq.ZZPowGate(exponent=2/math.pi, global_shift=-0.5)
              ),
          ],
      ),
      (
          dnf_lib.Clause(1, 3, False, True),
          0,
          [
              (
                  (cirq.LineQubit(1),),
                  cirq.ZPowGate(exponent=0, global_shift=-0.5)
              ),
              (
                  (cirq.LineQubit(3),),
                  cirq.ZPowGate(exponent=0, global_shift=-0.5)
              ),
              (
                  (cirq.LineQubit(1), cirq.LineQubit(3)),
                  cirq.ZZPowGate(exponent=0, global_shift=-0.5)
              ),
          ],
      )
  )
  def test_generate_clause_hamiltonian_exponential(self, clause, time,
                                                   expected_answers):
    circuit = cirq.Circuit()
    circuit.append(
        dnf_circuit_lib.generate_clause_hamiltonian_exponential(clause, time),
        strategy=insert_strategy.InsertStrategy.NEW_THEN_INLINE)

    generator = circuit.all_operations()
    answer_generator = iter(expected_answers)

    operation = next(generator)
    expected_qubits, expected_gate = next(answer_generator)
    self.assertTupleEqual(operation.qubits, expected_qubits)
    self.assertEqual(operation.gate, expected_gate)

    operation = next(generator)
    expected_qubits, expected_gate = next(answer_generator)
    self.assertTupleEqual(operation.qubits, expected_qubits)
    self.assertEqual(operation.gate, expected_gate)

    operation = next(generator)
    expected_qubits, expected_gate = next(answer_generator)
    self.assertTupleEqual(operation.qubits, expected_qubits)
    self.assertEqual(operation.gate, expected_gate)

    with self.assertRaises(StopIteration):
      next(generator)
    with self.assertRaises(StopIteration):
      next(answer_generator)

  def test_generate_dnf_hamiltonian_order(self):
    circuit = cirq.Circuit()
    dnf = dnf_lib.DNF(10, [dnf_lib.Clause(5, 7, False, False)])
    circuit.append(
        dnf_circuit_lib.generate_dnf_hamiltonian_exponential(dnf, 0.5),
        strategy=insert_strategy.InsertStrategy.NEW_THEN_INLINE)

    generator = circuit.all_operations()

    operation = next(generator)
    self.assertEqual(
        operation,
        cirq.ZPowGate(exponent=-1.0 / math.pi,
                      global_shift=-0.5).on(cirq.LineQubit(5)))

    operation = next(generator)
    self.assertEqual(
        operation,
        cirq.ZPowGate(exponent=-1.0 / math.pi,
                      global_shift=-0.5).on(cirq.LineQubit(7)))

    operation = next(generator)
    self.assertEqual(
        operation,
        cirq.ZZPowGate(exponent=-1.0 / math.pi,
                       global_shift=-0.5).on(cirq.LineQubit(5),
                                             cirq.LineQubit(7)))

    with self.assertRaises(StopIteration):
      next(generator)

  def test_generate_dnf_hamiltonian_content(self):
    clauses = [
        dnf_lib.Clause(2, 4, True, False),
        dnf_lib.Clause(5, 4, True, True)
    ]
    dnf = dnf_lib.DNF(10, clauses)

    circuit = cirq.Circuit()
    circuit.append(
        dnf_circuit_lib.generate_dnf_hamiltonian_exponential(dnf, 0.25),
        strategy=insert_strategy.InsertStrategy.NEW_THEN_INLINE)

    operations = list(circuit.all_operations())
    self.assertCountEqual(
        operations,
        [
            cirq.ZPowGate(exponent=-0.5 / math.pi,
                          global_shift=-0.5).on(cirq.LineQubit(4)),
            cirq.ZPowGate(exponent=0.5 / math.pi,
                          global_shift=-0.5).on(cirq.LineQubit(2)),
            cirq.ZZPowGate(exponent=0.5 / math.pi,
                           global_shift=-0.5).on(cirq.LineQubit(2),
                                                 cirq.LineQubit(4)),
            cirq.ZPowGate(exponent=0.5 / math.pi,
                          global_shift=-0.5).on(cirq.LineQubit(4)),
            cirq.ZPowGate(exponent=0.5 / math.pi,
                          global_shift=-0.5).on(cirq.LineQubit(5)),
            cirq.ZZPowGate(exponent=-0.5 / math.pi,
                           global_shift=-0.5).on(cirq.LineQubit(4),
                                                 cirq.LineQubit(5)),
        ]
    )


class BangBangProtocolCircuitTest(parameterized.TestCase):

  def test_bangbang_protocol_circuit_init(self):
    dnf = dnf_lib.DNF(4, [dnf_lib.Clause(0, 3, True, True)])
    circuit = dnf_circuit_lib.BangBangProtocolCircuit(5.3, dnf)
    self.assertEqual(circuit.dnf, dnf)
    self.assertEqual(circuit.chunk_time, 5.3)
    self.assertEqual(circuit.num_qubits, 4)

  def test_get_hamiltonian_diagonal(self):
    dnf = dnf_lib.DNF(4, [dnf_lib.Clause(0, 3, True, True)])
    circuit = dnf_circuit_lib.BangBangProtocolCircuit(5.3, dnf)
    np.testing.assert_allclose(
        circuit.get_hamiltonian_diagonal(),
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0])

  def test_bangbang_protocol_circuit_init_neg_chunk_time(self):
    with self.assertRaisesRegex(
        ValueError,
        'chunk_time must be positive, not -1.2'
    ):
      dnf_circuit_lib.BangBangProtocolCircuit(-1.2, dnf_lib.DNF(22, []))

  def test_generate_qaoa_circuit(self):
    dnf = dnf_lib.DNF(5, [dnf_lib.Clause(1, 2, False, False)])
    circuit = dnf_circuit_lib.BangBangProtocolCircuit(0.2, dnf)
    qaoa_circuit = circuit.qaoa_circuit([
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.X,
        circuit_lib.HamiltonianType.X,
        ])
    # Should 21 contain gates
    # 5 gates from Hadamard Layer
    # 2 layers of X Hamiltonian, which has 5 gates each
    # 2 layers of DNF Hamiltonian, which has 3 gates each
    # 5 + 2*5 + 2*3  = 21
    self.assertLen(list(qaoa_circuit.all_operations()), 21)

  def test_get_wavefunction(self):
    dnf = dnf_lib.DNF(2, [dnf_lib.Clause(0, 1, True, False)])
    # time is chosen so that ZPowGate = sqrt(Z) and Rx = sqrt(X) up to phase.
    circuit = dnf_circuit_lib.BangBangProtocolCircuit(math.pi / 4, dnf)
    bangbang_protocol = [
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.X,
    ]
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit.get_wavefunction(bangbang_protocol),
        np.array([0, (1 + 1j) / 2, (-1 + 1j) / 2, 0]),
        atol=0.000001)

  def test_get_probabilities_wrong_shape(self):
    dnf = dnf_lib.DNF(2, [dnf_lib.Clause(0, 1, True, False)])
    # time is chosen so that ZPowGate = sqrt(Z) and Rx = sqrt(X) up to phase.
    circuit = dnf_circuit_lib.BangBangProtocolCircuit(math.pi / 4, dnf)
    with self.assertRaisesRegex(
        ValueError,
        r'The shape of wavefunction should be \(4\,\) but got \(3\,\)'):
      circuit.get_probabilities(wavefunction=np.array([1., 0., 0.]))

  def test_get_probabilities(self):
    dnf = dnf_lib.DNF(2, [dnf_lib.Clause(0, 1, True, False)])
    # time is chosen so that ZPowGate = sqrt(Z) and Rx = sqrt(X) up to phase.
    circuit = dnf_circuit_lib.BangBangProtocolCircuit(math.pi / 4, dnf)
    bangbang_protocol = [
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.X,
    ]
    probabilities = circuit.get_probabilities(
        circuit.get_wavefunction(bangbang_protocol))
    np.testing.assert_allclose(
        probabilities,
        [0, 0.5, 0.5, 0],
        atol=0.00001)

  @parameterized.parameters(
      ([False, False, False, False], 3 / 4),
      ([False, False, False, True], 1),
      ([False, False, True, False], 1),
      ([False, False, True, True], 1),
      ([False, True, False, False], 3 /4),
      ([False, True, False, True], 1),
      ([False, True, True, False], 1),
      ([False, True, True, True], 1),
      ([True, False, False, False], 3 / 4),
      ([True, False, False, True], 1),
      ([True, False, True, False], 1),
      ([True, False, True, True], 1),
      ([True, True, False, False], 3 / 4),
      ([True, True, False, True], 1),
      ([True, True, True, False], 1),
      ([True, True, True, True], 1),
  )
  def test_constraint_evaluation(self, measurement, expected_value):
    dnf = dnf_lib.DNF(4, [dnf_lib.Clause(0, 1, False, False),
                          dnf_lib.Clause(0, 1, True, True),
                          dnf_lib.Clause(0, 1, False, True),
                          dnf_lib.Clause(0, 1, True, False),
                          dnf_lib.Clause(2, 3, False, False)])
    self.assertEqual(dnf.optimal_num_satisfied, 4)
    circuit = dnf_circuit_lib.BangBangProtocolCircuit(1, dnf)
    self.assertEqual(circuit.constraint_evaluation(measurement),
                     expected_value)

  def test_get_constraint_expectation(self):
    dnf = dnf_lib.DNF(2, [dnf_lib.Clause(0, 1, True, False)])
    # time is chosen so that ZPowGate = sqrt(Z) and Rx = sqrt(X) up to phase.
    circuit = dnf_circuit_lib.BangBangProtocolCircuit(math.pi / 4, dnf)
    bangbang_protocol = [
        circuit_lib.HamiltonianType.CONSTRAINT,
        circuit_lib.HamiltonianType.X,
    ]
    self.assertAlmostEqual(
        circuit.get_constraint_expectation(
            circuit.get_wavefunction(bangbang_protocol)),
        0.5,
        places=5)

if __name__ == '__main__':
  absltest.main()
