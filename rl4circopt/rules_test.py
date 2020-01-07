# coding=utf-8
# Copyright 2019 The Google Research Authors.
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
"""Tests for rules."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
from scipy import stats

from rl4circopt import architecture
from rl4circopt import circuit
from rl4circopt import rules
from rl4circopt import transform


# TODO(tfoesel):
# in the unit tests for the concrete Transformation rules (InvertCnot,
# CancelOperations, ExchangeCommutingOperations, ExchangePhasedXwithRotZ,
# ExchangePhasedXwithControlledZ, CompressLocalOperations), add an additional
# check that the operator for the output circuit matches that of the input
# circuit.


def _random_operation(*qubits):
  return circuit.Operation(
      circuit.MatrixGate(stats.unitary_group.rvs(2 ** len(qubits))),
      qubits
  )


def _elementwise_is(sequence_a, sequence_b):
  sequence_a = tuple(sequence_a)
  sequence_b = tuple(sequence_b)

  if len(sequence_a) == len(sequence_b):
    return all(
        elem_a is elem_b
        for elem_a, elem_b in zip(sequence_a, sequence_b)
    )
  else:
    return False


def _cmp_attention_circs(attention_circ_a, attention_circ_b):
  return (
      _elementwise_is(
          attention_circ_a.focus(),
          attention_circ_b.focus()
      )
      and _elementwise_is(
          attention_circ_a.context().before(),
          attention_circ_b.context().before()
      )
      and _elementwise_is(
          attention_circ_a.context().between(),
          attention_circ_b.context().between()
      )
      and _elementwise_is(
          attention_circ_a.context().after(),
          attention_circ_b.context().after()
      )
      and attention_circ_a.locations() == attention_circ_b.locations()
  )


class DummyArchitecture:

  def __init__(self, test_case, *gates):
    self.test_case = test_case
    self.gates = gates

  def decompose_single_qubit_gate(self, pauli_transform):
    self.test_case.assertIs(type(pauli_transform), np.ndarray)
    self.test_case.assertEqual(pauli_transform.dtype, float)
    self.test_case.assertTupleEqual(pauli_transform.shape, (3, 3))

    for gate in self.gates:
      if np.allclose(pauli_transform, gate.get_pauli_transform()):
        return [gate]

    assert False, 'pauli_transform does not match any of the given gates'


class UncallableArchitecture:

  def decompose_single_qubit_gate(self, pauli_transform):  # argument needed to match signature, so pylint: disable=unused-argument
    assert False, 'this method should never be called'


class TransformationRuleTest(parameterized.TestCase):

  def test_scan(self):
    # preparation work
    # (Note that ExchangeCommutingOperations.transformations_from_scanner() is
    # trusted from its own unit test.)
    rule = rules.ExchangeCommutingOperations()
    circ = circuit.Circuit(7, [
        circuit.Operation(circuit.ControlledZGate(), [0, 1]),
        circuit.Operation(circuit.RotZGate(0.42), [0]),
        circuit.Operation(circuit.ControlledZGate(), [1, 2]),
        circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [1]),
        circuit.Operation(circuit.ControlledZGate(), [0, 1]),
        circuit.Operation(circuit.ControlledZGate(), [1, 2])
    ])

    # call the function to be tested
    transformations = tuple(rule.scan(circ))

    # check the length of transformations
    self.assertLen(transformations, 4)
    self.assertTrue(all(
        isinstance(transformation, transform.Transformation)
        for transformation in transformations
    ))

    # sort the transformations to make their order canonical
    transformations = sorted(
        transformations,
        key=lambda transformation: transformation.locations()
    )

    # check the locations to make sure that the correct operations are selected
    self.assertTupleEqual(transformations[0].locations(), (0, 1))
    self.assertTupleEqual(transformations[1].locations(), (0, 2))
    self.assertTupleEqual(transformations[2].locations(), (1, 4))
    self.assertTupleEqual(transformations[3].locations(), (4, 5))

  def test_apply_first(self):
    # preparation work
    # (Note that CancelOperations.transformations_from_scanner() is trusted
    # from its own unit test.)
    rule = rules.CancelOperations()

    operation_a = circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [0])
    operation_b = circuit.Operation(circuit.ControlledZGate(), [1, 2])
    operation_c = circuit.Operation(circuit.ControlledZGate(), [0, 1])
    operation_d = circuit.Operation(circuit.RotZGate(0.137), [2])
    operation_e = circuit.Operation(circuit.RotZGate(0.42), [0])
    operation_f = circuit.Operation(circuit.RotZGate(-0.42), [0])
    operation_g = circuit.Operation(circuit.ControlledZGate(), [0, 1])
    operation_h = circuit.Operation(circuit.ControlledZGate(), [1, 2])

    circ_0 = circuit.Circuit(3, [
        operation_a,
        operation_b,
        operation_c,
        operation_d,
        operation_e,
        operation_f,
        operation_g,
        operation_h
    ])

    # call the function to be tested for the first time
    circ_1 = rule.apply_first(circ_0)

    # check type for circ_1
    self.assertTrue(type(circ_1), circuit.Circuit)

    # check value for circ_1
    self.assertLen(circ_1, 6)
    self.assertTrue(_elementwise_is(
        circ_1.get_operation_sequence(),
        [
            operation_a,
            operation_b,
            operation_c,
            operation_d,
            operation_g,
            operation_h
        ]
    ))

    # call the function to be tested for the second time
    circ_2 = rule.apply_first(circ_1)

    # check type for circ_2
    self.assertTrue(type(circ_2), circuit.Circuit)

    # check value for circ_2
    self.assertLen(circ_2, 4)
    self.assertTrue(_elementwise_is(
        circ_2.get_operation_sequence(),
        [
            operation_a,
            operation_b,
            operation_d,
            operation_h
        ]
    ))

    # call the function to be tested for the second time
    circ_3 = rule.apply_first(circ_2)
    self.assertIsNone(circ_3)  # there was no transformation applicable anymore

  def test_apply_greedily(self):
    # preparation work
    # (Note that CancelOperations.transformations_from_scanner() is trusted
    # from its own unit test.)
    rule = rules.CancelOperations()

    operation_a = circuit.Operation(circuit.RotZGate(0.1234), [0])
    operation_b = circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [0])
    operation_c = circuit.Operation(circuit.ControlledZGate(), [1, 2])
    operation_d = circuit.Operation(circuit.RotZGate(-0.42), [3])
    operation_e = circuit.Operation(circuit.ControlledZGate(), [2, 3])
    operation_f = circuit.Operation(circuit.RotZGate(0.42), [3])
    operation_g = circuit.Operation(circuit.PhasedXGate(-0.815, 0.4711), [0])
    operation_h = circuit.Operation(circuit.ControlledZGate(), [1, 2])
    operation_i = circuit.Operation(circuit.RotZGate(0.137), [1])
    operation_j = circuit.Operation(circuit.RotZGate(-0.137), [1])
    operation_k = circuit.Operation(circuit.ControlledZGate(), [0, 1])

    circ_in = circuit.Circuit(4, [
        operation_a,
        operation_b,
        operation_c,
        operation_d,
        operation_e,
        operation_f,
        operation_g,
        operation_h,
        operation_i,
        operation_j,
        operation_k
    ])

    # call the function to be tested
    circ_out = rule.apply_greedily(circ_in)

    # check type for circ_out
    self.assertTrue(type(circ_out), circuit.Circuit)

    # check value for circ_out
    self.assertLen(circ_out, 7)
    self.assertTrue(_elementwise_is(
        circ_out.get_operation_sequence(),
        [
            operation_a,
            operation_c,
            operation_d,
            operation_e,
            operation_f,
            operation_h,
            operation_k
        ]
    ))


class InvertCnotTest(parameterized.TestCase):

  @parameterized.parameters([
      [1, 5],
      [4, 3]
  ])
  def test_normal_cnot(self, src_qubit, tgt_qubit):
    # preparation work
    cnot_gate = circuit.MatrixGate(np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ]))
    operation_in = circuit.Operation(cnot_gate, [src_qubit, tgt_qubit])

    hadamard_gate = circuit.MatrixGate(np.sqrt(0.5) * np.array([
        [1.0, 1.0],
        [1.0, -1.0]
    ]))
    rule = rules.InvertCnot(DummyArchitecture(self, hadamard_gate))

    # check type and value for rule.accept(...)
    is_accepted = rule.accept(operation_in)
    self.assertIs(type(is_accepted), bool)
    self.assertTrue(is_accepted)

    # call rule.perform()
    operations_out = tuple(rule.perform(operation_in))

    # check the length of operations_out
    self.assertLen(operations_out, 5)
    self.assertTrue(all(
        type(operation) is circuit.Operation  # don't want possible subtypes, so pylint: disable=unidiomatic-typecheck
        for operation in operations_out
    ))

    # check that operations_out[0] is a Hadamard gate on src_qubit
    self.assertIs(operations_out[0].get_gate(), hadamard_gate)
    self.assertTupleEqual(operations_out[0].get_qubits(), (src_qubit,))

    # check that operations_out[1] is a Hadamard gate on tgt_qubit
    self.assertIs(operations_out[1].get_gate(), hadamard_gate)
    self.assertTupleEqual(operations_out[1].get_qubits(), (tgt_qubit,))

    # check that operations_out[2] is the CNOT gate on the inverted qubits
    self.assertIs(operations_out[2].get_gate(), cnot_gate)
    self.assertTupleEqual(operations_out[2].get_qubits(),
                          (tgt_qubit, src_qubit))

    # check that operations_out[3] is a Hadamard gate on src_qubit
    self.assertIs(operations_out[3].get_gate(), hadamard_gate)
    self.assertTupleEqual(operations_out[3].get_qubits(), (src_qubit,))

    # check that operations_out[4] is a Hadamard gate on tgt_qubit
    self.assertIs(operations_out[4].get_gate(), hadamard_gate)
    self.assertTupleEqual(operations_out[4].get_qubits(), (tgt_qubit,))

  @parameterized.parameters([
      [1, 5],
      [4, 3]
  ])
  def test_inverted_cnot(self, src_qubit, tgt_qubit):
    # preparation work
    inv_cnot_gate = circuit.MatrixGate(np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ]))
    operation_in = circuit.Operation(inv_cnot_gate, [src_qubit, tgt_qubit])

    hadamard_gate = circuit.MatrixGate(np.sqrt(0.5) * np.array([
        [1.0, 1.0],
        [1.0, -1.0]
    ]))
    rule = rules.InvertCnot(DummyArchitecture(self, hadamard_gate))

    # check type and value for rule.accept(...)
    is_accepted = rule.accept(operation_in)
    self.assertIs(type(is_accepted), bool)
    self.assertTrue(is_accepted)

    # call rule.perform()
    operations_out = tuple(rule.perform(operation_in))

    # check the length of operations_out
    self.assertLen(operations_out, 5)
    self.assertTrue(all(
        type(operation) is circuit.Operation  # don't want possible subtypes, so pylint: disable=unidiomatic-typecheck
        for operation in operations_out
    ))

    # check that operations_out[0] is a Hadamard gate on src_qubit
    self.assertIs(operations_out[0].get_gate(), hadamard_gate)
    self.assertTupleEqual(operations_out[0].get_qubits(), (src_qubit,))

    # check that operations_out[1] is a Hadamard gate on tgt_qubit
    self.assertIs(operations_out[1].get_gate(), hadamard_gate)
    self.assertTupleEqual(operations_out[1].get_qubits(), (tgt_qubit,))

    # check that operations_out[2] is the inv_cnot_gate on the inverted qubits
    self.assertIs(operations_out[2].get_gate(), inv_cnot_gate)
    self.assertTupleEqual(operations_out[2].get_qubits(),
                          (tgt_qubit, src_qubit))

    # check that operations_out[3] is a Hadamard gate on src_qubit
    self.assertIs(operations_out[3].get_gate(), hadamard_gate)
    self.assertTupleEqual(operations_out[3].get_qubits(), (src_qubit,))

    # check that operations_out[4] is a Hadamard gate on tgt_qubit
    self.assertIs(operations_out[4].get_gate(), hadamard_gate)
    self.assertTupleEqual(operations_out[4].get_qubits(), (tgt_qubit,))

  @parameterized.parameters([
      circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [1]),
      circuit.Operation(circuit.RotZGate(0.42), [3]),
      circuit.Operation(circuit.ControlledZGate(), [1, 2])
  ])
  def test_negative(self, operation):
    # preparation work
    rule = rules.InvertCnot(UncallableArchitecture())

    # check type and value for rule.accept(...)
    is_accepted = rule.accept(operation)
    self.assertIs(type(is_accepted), bool)
    self.assertFalse(is_accepted)

    # check that rule.perform(...) raises a RuleNotApplicableError
    with self.assertRaises(rules.RuleNotApplicableError):
      rule.perform(operation)

  def test_transformations_from_scanner(self):
    # preparation work
    rule = rules.InvertCnot(UncallableArchitecture())

    cnot_gate = circuit.MatrixGate(np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ]))
    inv_cnot_gate = circuit.MatrixGate(np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ]))

    scanner = rules.CircuitScanner(circuit.Circuit(7, [
        circuit.Operation(circuit.ControlledZGate(), [1, 2]),
        circuit.Operation(inv_cnot_gate, [2, 3]),
        circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [3]),
        circuit.Operation(circuit.ControlledZGate(), [3, 4]),
        circuit.Operation(cnot_gate, [4, 5]),
        circuit.Operation(circuit.RotZGate(0.42), [5]),
        circuit.Operation(circuit.ControlledZGate(), [5, 6])
    ]))

    # call the function to be tested
    transformations = tuple(rule.transformations_from_scanner(scanner))

    # check the length of transformations
    self.assertLen(transformations, 2)
    self.assertTrue(all(
        type(transformation) is transform.PointTransformation  # don't want possible subtypes, so pylint: disable=unidiomatic-typecheck
        for transformation in transformations
    ))

    # sort the transformations to make their order canonical
    transformations = sorted(
        transformations,
        key=lambda transformation: transformation.locations()
    )

    # check the locations to make sure that the correct operations are selected
    self.assertTupleEqual(transformations[0].locations(), (1,))
    self.assertTupleEqual(transformations[1].locations(), (4,))


class CancelOperationsTest(parameterized.TestCase):

  @parameterized.parameters([
      [
          circuit.PhasedXGate(0.815, 0.4711),
          circuit.PhasedXGate(-0.815, 0.4711), (3,)
      ],
      [
          circuit.RotZGate(0.815),
          circuit.RotZGate(-0.815), (3,)
      ],
      [
          circuit.ControlledZGate(),
          circuit.ControlledZGate(),
          (2, 5)
      ]
  ])
  def test_positive(self, gate_first, gate_second, qubits):
    # preparation work
    rule = rules.CancelOperations()
    operation_in_first = circuit.Operation(gate_first, qubits)
    operation_in_second = circuit.Operation(gate_second, qubits)

    # check type and value for rule.accept(...)
    is_accepted = rule.accept(operation_in_first, operation_in_second)
    self.assertIs(type(is_accepted), bool)
    self.assertTrue(is_accepted)

    # check rule.perform(...)
    operations_out_first, operations_out_second = \
        rule.perform(operation_in_first, operation_in_second)
    self.assertEmpty(tuple(operations_out_first))
    self.assertEmpty(tuple(operations_out_second))

  @parameterized.parameters([
      [
          circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [3]),
          circuit.Operation(circuit.PhasedXGate(0.42, 0.137), [3])
      ],
      [
          circuit.Operation(circuit.RotZGate(0.815), [2]),
          circuit.Operation(circuit.RotZGate(0.4711), [2])
      ],
      [
          circuit.Operation(circuit.ControlledZGate(), [1, 2]),
          circuit.Operation(circuit.ControlledZGate(), [2, 3])
      ],
      [
          circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [5]),
          circuit.Operation(circuit.RotZGate(0.42), [5])
      ],
      [
          circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [4]),
          circuit.Operation(circuit.ControlledZGate(), [3, 4])
      ],
      [
          circuit.Operation(circuit.RotZGate(0.42), [5]),
          circuit.Operation(circuit.ControlledZGate(), [4, 5])
      ]
  ])
  def test_negative(self, operation_first, operation_second):
    # preparation work
    rule = rules.CancelOperations()

    # check type and value for rule.accept(...)
    is_accepted = rule.accept(operation_first, operation_second)
    self.assertIs(type(is_accepted), bool)
    self.assertFalse(is_accepted)

    # check that rule.perform(...) raises a RuleNotApplicableError
    with self.assertRaises(rules.RuleNotApplicableError):
      rule.perform(operation_first, operation_second)

  def test_transformations_from_scanner(self):
    # preparation work
    rule = rules.CancelOperations()
    scanner = rules.CircuitScanner(circuit.Circuit(7, [
        circuit.Operation(circuit.ControlledZGate(), [0, 1]),
        circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [3]),
        circuit.Operation(circuit.ControlledZGate(), [1, 2]),
        circuit.Operation(circuit.PhasedXGate(-0.815, 0.4711), [3]),
        circuit.Operation(circuit.ControlledZGate(), [3, 4]),
        circuit.Operation(circuit.RotZGate(-0.42), [5]),
        circuit.Operation(circuit.RotZGate(0.42), [5]),
        circuit.Operation(circuit.ControlledZGate(), [4, 5])
    ]))

    # call the function to be tested
    transformations = tuple(rule.transformations_from_scanner(scanner))

    # check the length of transformations
    self.assertLen(transformations, 2)
    self.assertTrue(all(
        type(transformation) is transform.PairTransformation  # don't want possible subtypes, so pylint: disable=unidiomatic-typecheck
        for transformation in transformations
    ))

    # sort the transformations to make their order canonical
    transformations = sorted(
        transformations,
        key=lambda transformation: transformation.locations()
    )

    # check the locations to make sure that the correct operations are selected
    self.assertTupleEqual(transformations[0].locations(), (1, 3))
    self.assertTupleEqual(transformations[1].locations(), (5, 6))


class ExchangeCommutingOperationsTest(parameterized.TestCase):

  @parameterized.parameters([
      [
          circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [3]),
          circuit.Operation(circuit.PhasedXGate(0.42, 0.4711), [3])
      ],
      [
          circuit.Operation(circuit.RotZGate(0.815), [5]),
          circuit.Operation(circuit.RotZGate(0.42), [5])
      ],
      [
          circuit.Operation(circuit.ControlledZGate(), [0, 1]),
          circuit.Operation(circuit.ControlledZGate(), [0, 1])
      ],
      [
          circuit.Operation(circuit.ControlledZGate(), [2, 3]),
          circuit.Operation(circuit.ControlledZGate(), [3, 4])
      ],
      [
          circuit.Operation(circuit.RotZGate(0.815), [2]),
          circuit.Operation(circuit.ControlledZGate(), [1, 2])
      ]
  ])
  def test_positive(self, operation_in_first, operation_in_second):
    # preparation work
    rule = rules.ExchangeCommutingOperations()

    # check type and value for rule.accept(...)
    is_accepted = rule.accept(operation_in_first, operation_in_second)
    self.assertIs(type(is_accepted), bool)
    self.assertTrue(is_accepted)

    # check rule.perform(...)
    operations_out_first, operations_out_second = \
        rule.perform(operation_in_first, operation_in_second)
    self.assertTrue(_elementwise_is(
        operations_out_first,
        [operation_in_second]
    ))
    self.assertTrue(_elementwise_is(
        operations_out_second,
        [operation_in_first]
    ))

  @parameterized.parameters([
      [
          circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [3]),
          circuit.Operation(circuit.PhasedXGate(0.42, 0.137), [3])
      ],
      [
          circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [5]),
          circuit.Operation(circuit.RotZGate(0.42), [5])
      ],
      [
          circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [4]),
          circuit.Operation(circuit.ControlledZGate(), [3, 4])
      ]
  ])
  def test_negative(self, operation_first, operation_second):
    # preparation work
    rule = rules.ExchangeCommutingOperations()

    # check type and value for rule.accept(...)
    is_accepted = rule.accept(operation_first, operation_second)
    self.assertIs(type(is_accepted), bool)
    self.assertFalse(is_accepted)

    # check that rule.perform(...) raises a RuleNotApplicableError
    with self.assertRaises(rules.RuleNotApplicableError):
      rule.perform(operation_first, operation_second)

  def test_transformations_from_scanner(self):
    # preparation work
    rule = rules.ExchangeCommutingOperations()
    scanner = rules.CircuitScanner(circuit.Circuit(7, [
        circuit.Operation(circuit.ControlledZGate(), [0, 1]),
        circuit.Operation(circuit.RotZGate(0.42), [0]),
        circuit.Operation(circuit.ControlledZGate(), [1, 2]),
        circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [1]),
        circuit.Operation(circuit.ControlledZGate(), [0, 1]),
        circuit.Operation(circuit.ControlledZGate(), [1, 2])
    ]))

    # call the function to be tested
    transformations = tuple(rule.transformations_from_scanner(scanner))

    # check the length of transformations
    self.assertLen(transformations, 4)
    self.assertTrue(all(
        type(transformation) is transform.PairTransformation  # don't want possible subtypes, so pylint: disable=unidiomatic-typecheck
        for transformation in transformations
    ))

    # sort the transformations to make their order canonical
    transformations = sorted(
        transformations,
        key=lambda transformation: transformation.locations()
    )

    # check the locations to make sure that the correct operations are selected
    self.assertTupleEqual(transformations[0].locations(), (0, 1))
    self.assertTupleEqual(transformations[1].locations(), (0, 2))
    self.assertTupleEqual(transformations[2].locations(), (1, 4))
    self.assertTupleEqual(transformations[3].locations(), (4, 5))


class ExchangePhasedXwithRotZTest(parameterized.TestCase):

  @parameterized.parameters([
      [
          circuit.PhasedXGate(0.815, 0.42),
          circuit.RotZGate(0.4711),
          circuit.RotZGate(0.4711),
          circuit.PhasedXGate(0.815, 0.42+0.4711)
      ],
      [
          circuit.RotZGate(0.4711),
          circuit.PhasedXGate(0.815, 0.42),
          circuit.PhasedXGate(0.815, 0.42-0.4711),
          circuit.RotZGate(0.4711)
      ]
  ])
  def test_positive(self,
                    gate_in_first, gate_in_second,
                    gate_out_expected_first, gate_out_expected_second):
    # check consistency (note that the order is reversed)
    np.testing.assert_allclose(
        np.dot(gate_in_second.get_pauli_transform(),
               gate_in_first.get_pauli_transform()),
        np.dot(gate_out_expected_second.get_pauli_transform(),
               gate_out_expected_first.get_pauli_transform()),
        rtol=1e-5, atol=1e-8
    )

    # preparation work
    rule = rules.ExchangePhasedXwithRotZ()
    qubit = np.random.randint(10)
    operation_in_first = circuit.Operation(gate_in_first, [qubit])
    operation_in_second = circuit.Operation(gate_in_second, [qubit])

    # check type and value for rule.accept(...)
    is_accepted = rule.accept(operation_in_first, operation_in_second)
    self.assertIs(type(is_accepted), bool)
    self.assertTrue(is_accepted)

    # call rule.perform(...)
    operations_out_first, operations_out_second = \
        rule.perform(operation_in_first, operation_in_second)

    # check operations_out_first
    operations_out_first = tuple(operations_out_first)
    self.assertLen(operations_out_first, 1)

    operation_out_first, = operations_out_first
    self.assertIs(type(operation_out_first), circuit.Operation)

    gate_out_first = operation_out_first.get_gate()
    self.assertIs(type(gate_out_first), type(gate_out_expected_first))
    np.testing.assert_allclose(
        gate_out_first.get_operator(),
        gate_out_expected_first.get_operator(),
        rtol=1e-5, atol=1e-8
    )

    self.assertTupleEqual(operation_out_first.get_qubits(), (qubit,))

    # check operations_out_second
    operations_out_second = tuple(operations_out_second)
    self.assertLen(operations_out_second, 1)

    operation_out_second, = operations_out_second
    self.assertIs(type(operation_out_second), circuit.Operation)

    gate_out_second = operation_out_second.get_gate()
    self.assertIs(type(gate_out_second), type(gate_out_expected_second))
    np.testing.assert_allclose(
        gate_out_second.get_operator(),
        gate_out_expected_second.get_operator(),
        rtol=1e-5, atol=1e-8
    )

    self.assertTupleEqual(operation_out_second.get_qubits(), (qubit,))

  @parameterized.parameters([
      [
          circuit.Operation(circuit.PhasedXGate(0.4711, 0.137), [4]),
          circuit.Operation(circuit.PhasedXGate(0.815, 0.42), [4])
      ],
      [
          circuit.Operation(circuit.RotZGate(0.4711), [2]),
          circuit.Operation(circuit.RotZGate(0.815), [2])
      ],
      [
          circuit.Operation(circuit.PhasedXGate(0.4711, 0.137), [1]),
          circuit.Operation(circuit.ControlledZGate(), [0, 1])
      ],
      [
          circuit.Operation(circuit.ControlledZGate(), [2, 3]),
          circuit.Operation(circuit.PhasedXGate(0.4711, 0.137), [2])
      ],
      [
          circuit.Operation(circuit.RotZGate(0.815), [1]),
          circuit.Operation(circuit.ControlledZGate(), [0, 1])
      ],
      [
          circuit.Operation(circuit.ControlledZGate(), [2, 3]),
          circuit.Operation(circuit.RotZGate(0.815), [2])
      ]
  ])
  def test_negative(self, operation_first, operation_second):
    # preparation work
    rule = rules.ExchangePhasedXwithRotZ()

    # check type and value for rule.accept(...)
    is_accepted = rule.accept(operation_first, operation_second)
    self.assertIs(type(is_accepted), bool)
    self.assertFalse(is_accepted)

    # check that rule.perform(...) raises a RuleNotApplicableError
    with self.assertRaises(rules.RuleNotApplicableError):
      rule.perform(operation_first, operation_second)

  def test_transformations_from_scanner(self):
    # preparation work
    rule = rules.ExchangePhasedXwithRotZ()
    scanner = rules.CircuitScanner(circuit.Circuit(7, [
        circuit.Operation(circuit.RotZGate(0.42), [0]),
        circuit.Operation(circuit.PhasedXGate(0.73, 0.37), [2]),
        circuit.Operation(circuit.ControlledZGate(), [0, 1]),
        circuit.Operation(circuit.RotZGate(0.137), [2]),
        circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [2]),
        circuit.Operation(circuit.ControlledZGate(), [2, 3]),
        circuit.Operation(circuit.RotZGate(0.1234), [2]),
    ]))

    # call the function to be tested
    transformations = tuple(rule.transformations_from_scanner(scanner))

    # check the length of transformations
    self.assertLen(transformations, 2)
    self.assertTrue(all(
        type(transformation) is transform.PairTransformation  # don't want possible subtypes, so pylint: disable=unidiomatic-typecheck
        for transformation in transformations
    ))

    # sort the transformations to make their order canonical
    transformations = sorted(
        transformations,
        key=lambda transformation: transformation.locations()
    )

    # check the locations to make sure that the correct operations are selected
    self.assertTupleEqual(transformations[0].locations(), (1, 3))
    self.assertTupleEqual(transformations[1].locations(), (3, 4))


class ExchangePhasedXwithControlledZTest(parameterized.TestCase):

  @parameterized.parameters([
      [
          circuit.Operation(circuit.PhasedXGate(np.pi, 0.4711), [3]),
          circuit.Operation(circuit.ControlledZGate(), [3, 4]),
          circuit.Operation(circuit.RotZGate(np.pi), [4])
      ],
      [
          circuit.Operation(circuit.ControlledZGate(), [1, 2]),
          circuit.Operation(circuit.PhasedXGate(np.pi, 0.4711), [1]),
          circuit.Operation(circuit.RotZGate(np.pi), [2])
      ]
  ])
  def test_positive(self,
                    operation_in_first,
                    operation_in_second,
                    operation_new):
    # preparation work
    arch = DummyArchitecture(self, operation_new.get_gate())
    rule = rules.ExchangePhasedXwithControlledZ(arch)

    # check type and value for rule.accept(...)
    is_accepted = rule.accept(operation_in_first, operation_in_second)
    self.assertIs(type(is_accepted), bool)
    self.assertTrue(is_accepted)

    # call rule.perform(...)
    operations_out_first, operations_out_second = \
        rule.perform(operation_in_first, operation_in_second)

    # check operations_out_first
    self.assertTrue(_elementwise_is(
        operations_out_first,
        [operation_in_second]
    ))

    # check operations_out_second
    operations_out_second = tuple(operations_out_second)
    self.assertLen(operations_out_second, 2)

    # check operations_out_second[0]
    self.assertIs(type(operations_out_second[0]), circuit.Operation)
    self.assertIs(operations_out_second[0].get_gate(),
                  operation_new.get_gate())
    self.assertTupleEqual(operations_out_second[0].get_qubits(),
                          operation_new.get_qubits())

    # check operations_out_second[1]
    self.assertIs(operations_out_second[1], operation_in_first)

  @parameterized.parameters([
      [
          # rotation angle of PhasedX is not pi (or equivalent)
          circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [3]),
          circuit.Operation(circuit.ControlledZGate(), [3, 4])
      ],
      [
          # rotation angle of PhasedX is not pi (or equivalent)
          circuit.Operation(circuit.ControlledZGate(), [1, 2]),
          circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [1])
      ],
      [
          circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [0]),
          circuit.Operation(circuit.RotZGate(0.42), [0])
      ],
      [
          circuit.Operation(circuit.RotZGate(0.42), [2]),
          circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [2])
      ],
      [
          circuit.Operation(circuit.RotZGate(0.42), [4]),
          circuit.Operation(circuit.ControlledZGate(), [4, 5])
      ],
      [
          circuit.Operation(circuit.ControlledZGate(), [5, 6]),
          circuit.Operation(circuit.RotZGate(0.42), [6])
      ]
  ])
  def test_negative(self, operation_first, operation_second):
    # preparation work
    rule = rules.ExchangePhasedXwithControlledZ(UncallableArchitecture())

    # check type and value for rule.accept(...)
    is_accepted = rule.accept(operation_first, operation_second)
    self.assertIs(type(is_accepted), bool)
    self.assertFalse(is_accepted)

    # check that rule.perform(...) raises a RuleNotApplicableError
    with self.assertRaises(rules.RuleNotApplicableError):
      rule.perform(operation_first, operation_second)

  def test_transformations_from_scanner(self):
    # preparation work
    rule = rules.ExchangePhasedXwithControlledZ(UncallableArchitecture())
    scanner = rules.CircuitScanner(circuit.Circuit(3, [
        circuit.Operation(circuit.ControlledZGate(), [1, 2]),
        circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [1]),
        circuit.Operation(circuit.PhasedXGate(np.pi, 0.42), [2]),
        circuit.Operation(circuit.RotZGate(0.137), [0]),
        circuit.Operation(circuit.ControlledZGate(), [0, 1]),
        circuit.Operation(circuit.ControlledZGate(), [1, 2]),
    ]))

    # call the function to be tested
    transformations = tuple(rule.transformations_from_scanner(scanner))

    # check the length of transformations
    self.assertLen(transformations, 2)
    self.assertTrue(all(
        type(transformation) is transform.PairTransformation  # don't want possible subtypes, so pylint: disable=unidiomatic-typecheck
        for transformation in transformations
    ))

    # sort the transformations to make their order canonical
    transformations = sorted(
        transformations,
        key=lambda transformation: transformation.locations()
    )

    # check the locations to make sure that the correct operations are selected
    self.assertTupleEqual(transformations[0].locations(), (0, 2))
    self.assertTupleEqual(transformations[1].locations(), (2, 5))


class CompressLocalOperationsTest(parameterized.TestCase):

  @parameterized.parameters([
      [
          [
              circuit.RotZGate(0.0)
          ],
          []
      ],
      [
          [
              circuit.RotZGate(0.815),
              circuit.RotZGate(0.4711)
          ],
          [
              circuit.RotZGate(0.815 + 0.4711)
          ]
      ],
      [
          [
              circuit.RotZGate(0.815),
              circuit.RotZGate(0.4711),
              circuit.RotZGate(0.42)
          ],
          [
              circuit.RotZGate(0.815 + 0.4711 + 0.42)
          ]
      ],
      [
          [
              circuit.PhasedXGate(0.815, 0.4711),
              circuit.PhasedXGate(0.42, 0.4711)
          ],
          [
              circuit.PhasedXGate(0.815 + 0.42, 0.4711)
          ]
      ],
      [
          [
              circuit.PhasedXGate(0.815, 0.4711),
              circuit.RotZGate(0.0),  # identity
              circuit.RotZGate(0.42)
          ],
          [
              circuit.PhasedXGate(0.815, 0.4711),
              circuit.RotZGate(0.42)
          ]
      ],
      [
          [
              circuit.PhasedXGate(0.815, 0.4711),
              circuit.RotZGate(0.0)
          ],
          [
              circuit.PhasedXGate(0.815, 0.4711)
          ]
      ]
  ])
  def test_positive_for_given_decomposition(self, gates_in, gates_out_expected):
    # preparation workpauli_transform = np.eye(3)
    arch = architecture.XmonArchitecture()
    rule = rules.CompressLocalOperations(arch)
    qubit = np.random.randint(10)

    operations_in = tuple(circuit.Operation(gate, [qubit]) for gate in gates_in)

    # check type and value for rule.accept(...)
    is_accepted = rule.accept(operations_in)
    self.assertIs(type(is_accepted), bool)
    self.assertTrue(is_accepted)

    # call rule.perform(...)
    operations_out = tuple(rule.perform(operations_in))

    # check types for operations_out
    self.assertTrue(all(
        type(operation) is circuit.Operation  # don't want possible subtypes, so pylint: disable=unidiomatic-typecheck
        for operation in operations_out
    ))

    # check the qubits for operations_out
    self.assertTrue(all(
        operation.get_qubits() == (qubit,)
        for operation in operations_out
    ))

    # check the length of operations_out
    self.assertLen(operations_out, len(gates_out_expected))

    # check that gate types and operators match the expectation
    self.assertTrue(all(
        type(operation.get_gate()) is type(gate)
        for operation, gate in zip(operations_out, gates_out_expected)
    ))
    self.assertTrue(all(
        np.allclose(operation.get_gate().get_operator(), gate.get_operator())
        for operation, gate in zip(operations_out, gates_out_expected)
    ))

  @parameterized.parameters([
      [
          circuit.RotZGate(0.42),
          circuit.PhasedXGate(0.815, 0.4711),
          circuit.RotZGate(0.137)
      ],
      [
          circuit.PhasedXGate(0.815, 0.4711),
          circuit.PhasedXGate(0.42, 0.137),
          circuit.PhasedXGate(0.37, 0.73)
      ]
  ])
  def test_positive_for_unknown_decomposition(self, *gates_in):
    # preparation work
    arch = architecture.XmonArchitecture()
    rule = rules.CompressLocalOperations(arch)
    qubit = np.random.randint(10)

    operations_in = tuple(circuit.Operation(gate, [qubit]) for gate in gates_in)

    # check type and value for rule.accept(...)
    is_accepted = rule.accept(operations_in)
    self.assertIs(type(is_accepted), bool)
    self.assertTrue(is_accepted)

    # call rule.perform(...)
    operations_out = tuple(rule.perform(operations_in))

    # check types for operations_out
    self.assertTrue(all(
        type(operation) is circuit.Operation  # don't want possible subtypes, so pylint: disable=unidiomatic-typecheck
        for operation in operations_out
    ))

    # check the qubits for operations_out
    self.assertTrue(all(
        operation.get_qubits() == (qubit,)
        for operation in operations_out
    ))

    # construct the expected output gates (note that for the computation of
    # pauli_transform, the order for the factors needs to be reversed here)
    pauli_transform = np.eye(3)
    for gate in gates_in:
      pauli_transform = np.dot(gate.get_pauli_transform(), pauli_transform)
    gates_out_expected = arch.decompose_single_qubit_gate(pauli_transform)

    # check the length of operations_out
    self.assertLen(operations_out, len(gates_out_expected))
    self.assertTrue(all(
        type(operation.get_gate()) is type(gate)
        for operation, gate in zip(operations_out, gates_out_expected)
    ))
    self.assertTrue(all(
        np.allclose(operation.get_gate().get_operator(), gate.get_operator())
        for operation, gate in zip(operations_out, gates_out_expected)
    ))

  @parameterized.parameters([
      [
          circuit.PhasedXGate(0.815, 0.4711),
          circuit.RotZGate(0.42)
      ],
      [
          circuit.RotZGate(0.42),
          circuit.PhasedXGate(0.815, 0.4711)
      ],
      [
          circuit.PhasedXGate(0.815, 0.4711),
          circuit.RotZGate(0.42),
      ]
  ])
  def test_negative(self, *gates):
    # preparation work
    rule = rules.CompressLocalOperations(architecture.XmonArchitecture())

    qubit = np.random.randint(10)
    operations = tuple(circuit.Operation(gate, [qubit]) for gate in gates)

    # check type and value for rule.accept(...)
    is_accepted = rule.accept(operations)
    self.assertIs(type(is_accepted), bool)
    self.assertFalse(is_accepted)

    # check that rule.perform(...) raises a RuleNotApplicableError
    with self.assertRaises(rules.RuleNotApplicableError):
      rule.perform(operations)

  def test_transformations_from_scanner_with_compressable_groups(self):
    # preparation work
    rule = rules.CompressLocalOperations(architecture.XmonArchitecture())
    scanner = rules.CircuitScanner(circuit.Circuit(4, [
        circuit.Operation(circuit.ControlledZGate(), [0, 1]),
        circuit.Operation(circuit.PhasedXGate(0.815, 0.42), [0]),
        circuit.Operation(circuit.ControlledZGate(), [1, 2]),
        circuit.Operation(circuit.RotZGate(0.14), [3]),
        circuit.Operation(circuit.PhasedXGate(0.37, 0.73), [2]),
        circuit.Operation(circuit.RotZGate(0.15), [3]),
        circuit.Operation(circuit.PhasedXGate(0.4711, 0.42), [0]),
        circuit.Operation(circuit.ControlledZGate(), [0, 1]),
        circuit.Operation(circuit.RotZGate(0.654), [2]),
        circuit.Operation(circuit.RotZGate(0.16), [3]),
        circuit.Operation(circuit.RotZGate(0.1234), [0]),
        circuit.Operation(circuit.PhasedXGate(0.137, 0.1248), [2]),
    ]))

    # call the function to be tested
    transformations = tuple(rule.transformations_from_scanner(scanner))

    # check the length of transformations
    self.assertLen(transformations, 3)
    self.assertTrue(all(
        type(transformation) is transform.GroupTransformation  # don't want possible subtypes, so pylint: disable=unidiomatic-typecheck
        for transformation in transformations
    ))

    # sort the transformations to make their order canonical
    transformations = sorted(
        transformations,
        key=lambda transformation: transformation.locations()
    )

    # check the locations to make sure that the correct operations are selected
    self.assertTupleEqual(transformations[0].locations(), (1, 6))
    self.assertTupleEqual(transformations[1].locations(), (3, 5, 9))
    self.assertTupleEqual(transformations[2].locations(), (4, 8, 11))

  def test_transformations_from_scanner_with_noncompressable_groups(self):
    # preparation work
    rule = rules.CompressLocalOperations(architecture.XmonArchitecture())
    scanner = rules.CircuitScanner(circuit.Circuit(4, [
        circuit.Operation(circuit.ControlledZGate(), [0, 1]),
        circuit.Operation(circuit.PhasedXGate(0.815, 0.4711), [0]),
        circuit.Operation(circuit.ControlledZGate(), [1, 2]),
        circuit.Operation(circuit.RotZGate(0.14), [3]),
        circuit.Operation(circuit.PhasedXGate(0.37, 0.73), [2]),
        circuit.Operation(circuit.PhasedXGate(0.137, 0.1248), [3]),
        circuit.Operation(circuit.RotZGate(0.42), [0]),
        circuit.Operation(circuit.ControlledZGate(), [0, 1]),
        circuit.Operation(circuit.RotZGate(0.1234), [2]),
        circuit.Operation(circuit.RotZGate(0.1234), [0]),
    ]))

    # call the function to be tested
    transformations = tuple(rule.transformations_from_scanner(scanner))

    # check the length of transformations
    self.assertEmpty(transformations)


class CircuitScannerTest(parameterized.TestCase):

  def test_single_operations(self):
    # preparation work: construct a circuit
    circ = circuit.Circuit(5, [
        _random_operation(0),
        _random_operation(0, 1),
        _random_operation(1),
        _random_operation(1, 2),
        _random_operation(2)
    ])

    # construct the CircuitScanner
    scanner = rules.CircuitScanner(circ)

    # call the function to be tested
    attention_circs = tuple(scanner.single_operations())

    # check the length of attention_circs
    self.assertLen(attention_circs, len(circ))

    # check the types for attention_circs
    self.assertTrue(all(
        type(attention_circ) is transform.AttentionCircuit  # don't want possible subtypes, so pylint: disable=unidiomatic-typecheck
        for attention_circ in attention_circs
    ))

    # check attention_circs[0]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[0],
        transform.AttentionCircuit(
            focus=[circ[0]],
            context=transform.TransformationContext(
                before=circuit.Circuit(circ.get_num_qubits(), None),
                between=circuit.Circuit(circ.get_num_qubits(), None),
                after=circ[1:]
            ),
            locations=[0]
        )
    ))

    # check attention_circs[1]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[1],
        transform.AttentionCircuit(
            focus=[circ[1]],
            context=transform.TransformationContext(
                before=circ[:1],
                between=circuit.Circuit(circ.get_num_qubits(), None),
                after=circ[2:]
            ),
            locations=[1]
        )
    ))

    # check attention_circs[2]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[2],
        transform.AttentionCircuit(
            focus=[circ[2]],
            context=transform.TransformationContext(
                before=circ[:2],
                between=circuit.Circuit(circ.get_num_qubits(), None),
                after=circ[3:]
            ),
            locations=[2]
        )
    ))

    # check attention_circs[3]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[3],
        transform.AttentionCircuit(
            focus=[circ[3]],
            context=transform.TransformationContext(
                before=circ[:3],
                between=circuit.Circuit(circ.get_num_qubits(), None),
                after=circ[4:]
            ),
            locations=[3]
        )
    ))

    # check attention_circs[4]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[4],
        transform.AttentionCircuit(
            focus=[circ[4]],
            context=transform.TransformationContext(
                before=circ[:4],
                between=circuit.Circuit(circ.get_num_qubits(), None),
                after=circuit.Circuit(circ.get_num_qubits(), None)
            ),
            locations=[4]
        )
    ))

  def test_operation_pairs(self):
    # preparation work: construct a circuit
    circ = circuit.Circuit(4, [
        _random_operation(0, 1),
        _random_operation(0),
        _random_operation(1, 2),
        _random_operation(1),
        _random_operation(0, 1),
        _random_operation(3),
        _random_operation(0, 1)
    ])

    # construct the CircuitScanner
    scanner = rules.CircuitScanner(circ)

    # call the function to be tested
    attention_circs = tuple(scanner.operation_pairs())

    # check the length of attention_circs
    self.assertLen(attention_circs, 6)

    # check the types for attention_circs
    self.assertTrue(all(
        type(attention_circ) is transform.AttentionCircuit  # don't want possible subtypes, so pylint: disable=unidiomatic-typecheck
        for attention_circ in attention_circs
    ))

    # sort the attention circuits to make their order canonical
    attention_circs = sorted(
        attention_circs,
        key=lambda attention_circ: attention_circ.locations()
    )

    # check attention_circs[0]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[0],
        transform.AttentionCircuit(
            focus=circ[0, 1].get_operation_sequence(),
            context=transform.TransformationContext(
                before=circuit.Circuit(circ.get_num_qubits(), None),
                between=circuit.Circuit(circ.get_num_qubits(), None),
                after=circ[2:]
            ),
            locations=[0, 1]
        )
    ))

    # check attention_circs[1]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[1],
        transform.AttentionCircuit(
            focus=circ[0, 2].get_operation_sequence(),
            context=transform.TransformationContext(
                before=circuit.Circuit(circ.get_num_qubits(), None),
                between=circuit.Circuit(circ.get_num_qubits(), None),
                after=circ[1, 3:]
            ),
            locations=[0, 2]
        )
    ))

    # check attention_circs[2]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[2],
        transform.AttentionCircuit(
            focus=circ[1, 4].get_operation_sequence(),
            context=transform.TransformationContext(
                before=circ[0, 2, 3],
                between=circuit.Circuit(circ.get_num_qubits(), None),
                after=circ[5:]
            ),
            locations=[1, 4]
        )
    ))

    # check attention_circs[3]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[3],
        transform.AttentionCircuit(
            focus=circ[2, 3].get_operation_sequence(),
            context=transform.TransformationContext(
                before=circ[:2],
                between=circuit.Circuit(circ.get_num_qubits(), None),
                after=circ[4:]
            ),
            locations=[2, 3]
        )
    ))

    # check attention_circs[4]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[4],
        transform.AttentionCircuit(
            focus=circ[3, 4].get_operation_sequence(),
            context=transform.TransformationContext(
                before=circ[:3],
                between=circuit.Circuit(circ.get_num_qubits(), None),
                after=circ[5:]
            ),
            locations=[3, 4]
        )
    ))

    # check attention_circs[5]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[5],
        transform.AttentionCircuit(
            focus=circ[4, 6].get_operation_sequence(),
            context=transform.TransformationContext(
                before=circ[:4],
                between=circ[5:6],
                after=circ[7:]
            ),
            locations=[4, 6]
        )
    ))

  def test_local_groups(self):
    # preparation work: construct a circuit
    circ = circuit.Circuit(4, [
        _random_operation(0, 1),
        _random_operation(0),
        _random_operation(1, 2),
        _random_operation(3),
        _random_operation(2),
        _random_operation(3),
        _random_operation(0),
        _random_operation(0, 1),
        _random_operation(2),
        _random_operation(3),
        _random_operation(0),
        _random_operation(2)
    ])

    # construct the CircuitScanner
    scanner = rules.CircuitScanner(circ)

    # call the function to be tested
    attention_circs = tuple(scanner.local_groups())

    # check the length of attention_circs
    self.assertLen(attention_circs, 4)

    # check the types for attention_circs
    self.assertTrue(all(
        type(attention_circ) is transform.AttentionCircuit  # don't want possible subtypes, so pylint: disable=unidiomatic-typecheck
        for attention_circ in attention_circs
    ))

    # sort the attention circuits to make their order canonical
    attention_circs = sorted(
        attention_circs,
        key=lambda attention_circ: attention_circ.locations()
    )

    # check attention_circs[0]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[0],
        transform.AttentionCircuit(
            focus=circ[1, 6].get_operation_sequence(),
            context=transform.TransformationContext(
                before=circ[:1],
                between=circ[2:6],
                after=circ[7:]
            ),
            locations=[1, 6]
        )
    ))

    # check attention_circs[1]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[1],
        transform.AttentionCircuit(
            focus=circ[3, 5, 9].get_operation_sequence(),
            context=transform.TransformationContext(
                before=circ[:3],
                between=circ[4, 6:9],
                after=circ[10:]
            ),
            locations=[3, 5, 9]
        )
    ))

    # check attention_circs[2]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[2],
        transform.AttentionCircuit(
            focus=circ[4, 8, 11].get_operation_sequence(),
            context=transform.TransformationContext(
                before=circ[:4],
                between=circ[5:8, 9:11],
                after=circuit.Circuit(circ.get_num_qubits(), None)
            ),
            locations=[4, 8, 11]
        )
    ))

    # check attention_circs[3]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[3],
        transform.AttentionCircuit(
            focus=[circ[10]],
            context=transform.TransformationContext(
                before=circ[:10],
                between=circuit.Circuit(circ.get_num_qubits(), None),
                after=circ[11:]
            ),
            locations=[10]
        )
    ))

  def test_circ_type_error(self):
    with self.assertRaisesRegex(
        TypeError,
        r'circ is not a Circuit \(found type: int\)'):
      rules.CircuitScanner(42)


class ScanForSingleOperationsTest(parameterized.TestCase):

  def test_successful(self):
    # preparation work: construct a circuit
    circ = circuit.Circuit(5, [
        _random_operation(0),
        _random_operation(0, 1),
        _random_operation(1),
        _random_operation(1, 2),
        _random_operation(2)
    ])

    # call the function to be tested
    attention_circs = tuple(rules.scan_for_single_operations(circ))

    # check the length of attention_circs
    self.assertLen(attention_circs, len(circ))

    # check the types for attention_circs
    self.assertTrue(all(
        type(attention_circ) is transform.AttentionCircuit  # don't want possible subtypes, so pylint: disable=unidiomatic-typecheck
        for attention_circ in attention_circs
    ))

    # check attention_circs[0]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[0],
        transform.AttentionCircuit(
            focus=[circ[0]],
            context=transform.TransformationContext(
                before=circuit.Circuit(circ.get_num_qubits(), None),
                between=circuit.Circuit(circ.get_num_qubits(), None),
                after=circ[1:]
            ),
            locations=[0]
        )
    ))

    # check attention_circs[1]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[1],
        transform.AttentionCircuit(
            focus=[circ[1]],
            context=transform.TransformationContext(
                before=circ[:1],
                between=circuit.Circuit(circ.get_num_qubits(), None),
                after=circ[2:]
            ),
            locations=[1]
        )
    ))

    # check attention_circs[2]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[2],
        transform.AttentionCircuit(
            focus=[circ[2]],
            context=transform.TransformationContext(
                before=circ[:2],
                between=circuit.Circuit(circ.get_num_qubits(), None),
                after=circ[3:]
            ),
            locations=[2]
        )
    ))

    # check attention_circs[3]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[3],
        transform.AttentionCircuit(
            focus=[circ[3]],
            context=transform.TransformationContext(
                before=circ[:3],
                between=circuit.Circuit(circ.get_num_qubits(), None),
                after=circ[4:]
            ),
            locations=[3]
        )
    ))

    # check attention_circs[4]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[4],
        transform.AttentionCircuit(
            focus=[circ[4]],
            context=transform.TransformationContext(
                before=circ[:4],
                between=circuit.Circuit(circ.get_num_qubits(), None),
                after=circuit.Circuit(circ.get_num_qubits(), None)
            ),
            locations=[4]
        )
    ))

  def test_circ_type_error(self):
    with self.assertRaisesRegex(
        TypeError,
        r'circ is not a Circuit \(found type: int\)'):
      # next because function is lazy
      next(rules.scan_for_single_operations(42))


class ScanForOperationPairsTest(parameterized.TestCase):

  def test_successful(self):
    # preparation work: construct a circuit
    circ = circuit.Circuit(4, [
        _random_operation(0, 1),
        _random_operation(0),
        _random_operation(1, 2),
        _random_operation(1),
        _random_operation(0, 1),
        _random_operation(3),
        _random_operation(0, 1)
    ])

    # call the function to be tested
    attention_circs = tuple(rules.scan_for_operation_pairs(circ))

    # check the length of attention_circs
    self.assertLen(attention_circs, 6)

    # check the types for attention_circs
    self.assertTrue(all(
        type(attention_circ) is transform.AttentionCircuit  # don't want possible subtypes, so pylint: disable=unidiomatic-typecheck
        for attention_circ in attention_circs
    ))

    # sort the attention circuits to make their order canonical
    attention_circs = sorted(
        attention_circs,
        key=lambda attention_circ: attention_circ.locations()
    )

    # check attention_circs[0]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[0],
        transform.AttentionCircuit(
            focus=circ[0, 1].get_operation_sequence(),
            context=transform.TransformationContext(
                before=circuit.Circuit(circ.get_num_qubits(), None),
                between=circuit.Circuit(circ.get_num_qubits(), None),
                after=circ[2:]
            ),
            locations=[0, 1]
        )
    ))

    # check attention_circs[1]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[1],
        transform.AttentionCircuit(
            focus=circ[0, 2].get_operation_sequence(),
            context=transform.TransformationContext(
                before=circuit.Circuit(circ.get_num_qubits(), None),
                between=circuit.Circuit(circ.get_num_qubits(), None),
                after=circ[1, 3:]
            ),
            locations=[0, 2]
        )
    ))

    # check attention_circs[2]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[2],
        transform.AttentionCircuit(
            focus=circ[1, 4].get_operation_sequence(),
            context=transform.TransformationContext(
                before=circ[0, 2, 3],
                between=circuit.Circuit(circ.get_num_qubits(), None),
                after=circ[5:]
            ),
            locations=[1, 4]
        )
    ))

    # check attention_circs[3]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[3],
        transform.AttentionCircuit(
            focus=circ[2, 3].get_operation_sequence(),
            context=transform.TransformationContext(
                before=circ[:2],
                between=circuit.Circuit(circ.get_num_qubits(), None),
                after=circ[4:]
            ),
            locations=[2, 3]
        )
    ))

    # check attention_circs[4]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[4],
        transform.AttentionCircuit(
            focus=circ[3, 4].get_operation_sequence(),
            context=transform.TransformationContext(
                before=circ[:3],
                between=circuit.Circuit(circ.get_num_qubits(), None),
                after=circ[5:]
            ),
            locations=[3, 4]
        )
    ))

    # check attention_circs[5]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[5],
        transform.AttentionCircuit(
            focus=circ[4, 6].get_operation_sequence(),
            context=transform.TransformationContext(
                before=circ[:4],
                between=circ[5:6],
                after=circ[7:]
            ),
            locations=[4, 6]
        )
    ))

  def test_circ_type_error(self):
    with self.assertRaisesRegex(
        TypeError,
        r'circ is not a Circuit \(found type: int\)'):
      next(rules.scan_for_operation_pairs(42))  # next because function is lazy


class ScanForLocalGroupsTest(parameterized.TestCase):

  def test_successful(self):
    # preparation work: construct a circuit
    circ = circuit.Circuit(4, [
        _random_operation(0, 1),
        _random_operation(0),
        _random_operation(1, 2),
        _random_operation(3),
        _random_operation(2),
        _random_operation(3),
        _random_operation(0),
        _random_operation(0, 1),
        _random_operation(2),
        _random_operation(3),
        _random_operation(0),
        _random_operation(2)
    ])

    # call the function to be tested
    attention_circs = tuple(rules.scan_for_local_groups(circ))

    # check the length of attention_circs
    self.assertLen(attention_circs, 4)

    # check the types for attention_circs
    self.assertTrue(all(
        type(attention_circ) is transform.AttentionCircuit  # don't want possible subtypes, so pylint: disable=unidiomatic-typecheck
        for attention_circ in attention_circs
    ))

    # sort the attention circuits to make their order canonical
    attention_circs = sorted(
        attention_circs,
        key=lambda attention_circ: attention_circ.locations()
    )

    # check attention_circs[0]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[0],
        transform.AttentionCircuit(
            focus=circ[1, 6].get_operation_sequence(),
            context=transform.TransformationContext(
                before=circ[:1],
                between=circ[2:6],
                after=circ[7:]
            ),
            locations=[1, 6]
        )
    ))

    # check attention_circs[1]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[1],
        transform.AttentionCircuit(
            focus=circ[3, 5, 9].get_operation_sequence(),
            context=transform.TransformationContext(
                before=circ[:3],
                between=circ[4, 6:9],
                after=circ[10:]
            ),
            locations=[3, 5, 9]
        )
    ))

    # check attention_circs[2]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[2],
        transform.AttentionCircuit(
            focus=circ[4, 8, 11].get_operation_sequence(),
            context=transform.TransformationContext(
                before=circ[:4],
                between=circ[5:8, 9:11],
                after=circuit.Circuit(circ.get_num_qubits(), None)
            ),
            locations=[4, 8, 11]
        )
    ))

    # check attention_circs[3]
    self.assertTrue(_cmp_attention_circs(
        attention_circs[3],
        transform.AttentionCircuit(
            focus=[circ[10]],
            context=transform.TransformationContext(
                before=circ[:10],
                between=circuit.Circuit(circ.get_num_qubits(), None),
                after=circ[11:]
            ),
            locations=[10]
        )
    ))

  def test_circ_type_error(self):
    with self.assertRaisesRegex(
        TypeError,
        r'circ is not a Circuit \(found type: int\)'):
      next(rules.scan_for_local_groups(42))  # next because function is lazy


if __name__ == '__main__':
  absltest.main()
