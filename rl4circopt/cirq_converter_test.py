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

# Lint as: python3
"""Tests for cirq_converter."""

from absl.testing import absltest
from absl.testing import parameterized

import cirq
import numpy as np
from scipy import stats

from rl4circopt import circuit
from rl4circopt import cirq_converter


class TestExportAndImport(parameterized.TestCase):

  @parameterized.parameters([
      circuit.PhasedXGate(0.47, 0.11),
      circuit.RotZGate(0.42),
      circuit.ControlledZGate(),
      circuit.MatrixGate(stats.unitary_group.rvs(2)),  # random 1-qubit unitary
      circuit.MatrixGate(stats.unitary_group.rvs(4)),  # random 2-qubit unitary
      circuit.MatrixGate(stats.unitary_group.rvs(8)),  # random 3-qubit unitary
      circuit.MatrixGate(stats.unitary_group.rvs(16))  # random 4-qubit unitary
  ])
  def test_gates(self, gate_orig):
    # export the gate to Cirq
    gate_exported = cirq_converter.export_to_cirq(gate_orig)

    # check that the gates are equivalent
    self.assertIsInstance(gate_exported, cirq.Gate)
    np.testing.assert_allclose(
        gate_orig.get_pauli_transform(),
        circuit.compute_pauli_transform(cirq.unitary(gate_exported)),
        rtol=1e-5, atol=1e-8
    )

    # reimport the gate from Cirq
    gate_reimported = cirq_converter.import_from_cirq(gate_exported)

    # check that the original and the reimported gate are equivalent
    self.assertIs(type(gate_reimported), type(gate_orig))
    self.assertEqual(
        gate_reimported.get_num_qubits(),
        gate_orig.get_num_qubits()
    )
    np.testing.assert_allclose(
        gate_orig.get_operator(),
        gate_reimported.get_operator(),
        rtol=1e-5, atol=1e-8
    )

  @parameterized.parameters([1, 2])
  def test_operations(self, num_qubits):
    # construct the original operation
    op_orig = circuit.Operation(
        circuit.MatrixGate(stats.unitary_group.rvs(2 ** num_qubits)),
        np.random.permutation(10)[:num_qubits]
    )

    # export the operation to Cirq
    op_exported = cirq_converter.export_to_cirq(op_orig)

    # check that the operations are equivalent
    self.assertIsInstance(op_exported, cirq.GateOperation)
    np.testing.assert_allclose(
        op_orig.get_gate().get_operator(),
        cirq.unitary(op_exported.gate),
        rtol=1e-5, atol=1e-8
    )
    self.assertTupleEqual(
        op_orig.get_qubits(),
        tuple(qubit.x for qubit in op_exported.qubits)
    )

    # reimport the operation from Cirq
    op_reimported = cirq_converter.import_from_cirq(op_exported)

    # check that the original and the reimported operation are equivalent
    self.assertIs(type(op_reimported), circuit.Operation)
    self.assertEqual(op_reimported.get_num_qubits(), op_orig.get_num_qubits())
    np.testing.assert_allclose(
        op_orig.get_gate().get_operator(),
        op_reimported.get_gate().get_operator()
    )
    self.assertTupleEqual(op_orig.get_qubits(), op_reimported.get_qubits())

  def test_circuit(self):
    # construct the original circuit
    circ_orig = circuit.Circuit(2, [
        circuit.Operation(circuit.PhasedXGate(0.47, 0.11), [0]),
        circuit.Operation(circuit.ControlledZGate(), [0, 1]),
        circuit.Operation(circuit.RotZGate(0.42), [1]),
    ])

    # export the circuit to Cirq
    circ_exported = cirq_converter.export_to_cirq(circ_orig)

    # check the type of circ_exported
    self.assertIsInstance(circ_exported, cirq.Circuit)

    # TODO(tfoesel):
    # a direct comparison between circ_orig and circ_exported would be better

    # reimport the circuit from Cirq
    circ_reimported = cirq_converter.import_from_cirq(circ_exported)

    # check that the number of operations and the gate types are conserved
    self.assertEqual(len(circ_orig), len(circ_reimported))
    self.assertEqual(
        [type(operation.get_gate()) for operation in circ_orig],
        [type(operation.get_gate()) for operation in circ_orig]
    )

  def test_export_unknown_type_error(self):
    with self.assertRaisesRegex(TypeError, r'unknown type: range'):
      cirq_converter.export_to_cirq(range(42))

  def test_import_unknown_type_error(self):
    with self.assertRaisesRegex(TypeError, r'unknown type: range'):
      cirq_converter.import_from_cirq(range(42))

  def test_import_partial_cz_error(self):
    partial_cz = cirq.CZPowGate(exponent=0.37)

    with self.assertRaisesRegex(
        ValueError,
        r'partial ControlledZ gates are not supported'):
      cirq_converter.import_from_cirq(partial_cz)

  def test_import_non_line_qubits_error(self):
    qubit_a = cirq.GridQubit(0, 1)
    qubit_b = cirq.GridQubit(0, 2)

    circ = cirq.Circuit(
        cirq.PhasedXPowGate(exponent=0.47, phase_exponent=0.11).on(qubit_a),
        cirq.CZPowGate(exponent=1.0).on(qubit_a, qubit_b),
        cirq.ZPowGate(exponent=0.42).on(qubit_b)
    )

    with self.assertRaisesRegex(
        ValueError,
        r'import is supported for circuits on LineQubits only \[found qubit'
        r' type\(s\): GridQubit\]'):
      cirq_converter.import_from_cirq(circ)


if __name__ == '__main__':
  absltest.main()
