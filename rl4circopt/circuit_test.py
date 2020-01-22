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

# Lint as: python3
"""Tests for circuit."""

import itertools
from absl.testing import absltest
from absl.testing import parameterized

import cirq
import numpy as np
from scipy import stats
from scipy.spatial import transform

from rl4circopt import circuit


def _check_boolean(test_case, found, expected):
  """Checks that found is a bool and matches the expected result."""
  test_case.assertIs(type(found), bool)
  test_case.assertEqual(found, expected)


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


def _check_unitarity(operator, dim):
  """Checks that operator is a unitary operator."""
  np.testing.assert_allclose(
      np.dot(operator, operator.T.conj()),
      np.eye(dim),
      rtol=1e-5, atol=1e-8
  )


def _testing_angles():
  return [-3, -0.6, 0, 0.0, 0.2, 0.73, 2.0, np.pi, 4, 4.2, 2.0*np.pi, 8.0, 10]


def _random_matrix_gate(num_qubits):
  return circuit.MatrixGate(stats.unitary_group.rvs(2 ** num_qubits))


def _euler_to_dcm(*args, **kwargs):
  return transform.Rotation.from_euler(*args, **kwargs).as_dcm()


def _clifford_group():
  """Constructs the single-qubit Clifford group.

  The Clifford group is defined as follows: For any unitary operator U, we
  consider the transformation

      rho -> rho' = U * rho * U^dagger

  where * denotes the matrix product. If rho is a Pauli operator (in the
  single-qubit case ±pauli_x, ±pauli_z, ±pauli_z), then, in general, rho' does
  not need to be a Pauli operator as well. However, there are some unitary
  operators U which satisfy this condition, i.e. they map each Pauli operator
  into another (or potentially the same) Pauli operator. The set of all those
  unitary operators is called the Clifford group. This definition is valid for
  any number of qubits, but here we care only about the single-qubit case.

  In total, the (single-qubit) Clifford group consists of 24 elements, including
  prominent operations like identity, X flip, Y flip, Z flip, the phase gate
  (rotation around the z axis by 90 degrees) and the Hadamard gate.

  The dimension of the (single-qubit) Clifford group is 2. This means that it is
  possible to find two elements, the "generators", such that all other elements
  can be written as a product of these generators (and that 2 is the minimum
  number for which this is the case). A common choice for this generator set is
  the phase gate and the Hadamard gate.

  Returns:
      a list. Each entry is a tuple, consisting of a unitary operator and the
      corresponding pauli_transform, and describes one Clifford group element.
  """

  # This implementation is built on the definition of a Clifford gate that it
  # must transform Pauli operators to Pauli operators. This means that each
  # column of its corresponding pauli_transform matrix needs to have exactly one
  # non-zero entry which must be either +1 or -1. In addition, this matrix has
  # to be orthogonal. An example for such a matrix is
  #
  #     [  0  ±1   0 ]
  #     [  0   0  ±1 ]
  #     [ ±1   0   0 ]
  #
  # for any combination of ±. We can generate all those matrices by considering
  # all permutations of the rows (or equivalently columns) of the identity
  # matrix and all combinations of plus and minus signs.
  #
  # This is a necessary criterion, but it is not yet sufficient. Only half of
  # the pauli_transform matrices that we get from the recipe above preserve
  # the orientation (handedness of space); these are the interesting ones as
  # they correspond to unitary transformations. The other half inverts
  # orientation; we have to filter these out as they correspond to anti-unitary
  # transformations. To discriminate these cases, we can use the determinant
  # (+1 if orientation is preserved, -1 if inverted).
  #
  # This construction scheme as is works only for a single qubit. For multiple
  # qubits, more constraints on the pauli_transform matrices have to be taken
  # into account, and iterating over all possibilities would soon get
  # infeasible.

  elements = []

  # loop over all combinations of coordinate axis permuations and sign patterns
  for axes, signs in itertools.product(
      itertools.permutations(np.eye(3)),
      itertools.product([1.0, -1.0], repeat=3)):
    pauli_transform = np.stack(axes) * np.array(signs)

    if np.linalg.det(pauli_transform) < 0.0:
      continue  # filter orientation-conserving transformations (rotations)

    rot_vector = transform.Rotation.from_dcm(pauli_transform).as_rotvec()
    rot_angle = np.linalg.norm(rot_vector)  # the rotation angle

    x, y, z = rot_vector

    # Compute (exp is the matrix exponential)
    #
    #     exp(-0.5j * (x * pauli_x + y * pauli_y + z * pauli_z)) =
    #     = exp(-0.5j * rot_angle * pauli_axis) =
    #                 // expand as power series and separate even from odd terms
    #     = cos(0.5 * rot_angle) * eye - i * sin(0.5 * rot_angle) * pauli_axis =
    #     = cos(0.5 * rot_angle) * eye - 0.5i * sinc(0.5 * rot_angle) *
    #                                * (x * pauli_x + y * pauli_y + z * pauli_z)
    #
    # where rot_angle = sqrt(x^2 + y^2 + z^2) and
    #
    #     pauli_axis = (x * pauli_x + y * pauli_y + z * pauli_z) / rot_angle
    #
    # which is chosen such that dot(pauli_axis, pauli_axis) = eye.
    #
    # The reason for the additional `/ np.pi` below is that np.sinc implements
    # the normalized sinc function as used in DSP, whereas the sinc above is the
    # unnormalized sinc function; these two versions differ by a factor of pi in
    # their argument.
    operator = (
        np.cos(0.5 * rot_angle) * np.eye(2)
        - 0.5j * np.sinc(0.5 / np.pi * rot_angle) * np.array([
            [z, x - 1.0j * y],
            [x + 1.0j * y, -z]
        ])
    )

    elements.append((operator, pauli_transform))

  return elements


def _generate_random_anticommuting_operators(num_qubits):
  pauli_operators = np.array([
      [[0.0, 1.0], [1.0, 0.0]],     # pauli_x
      [[0.0, -1.0j], [1.0j, 0.0]],  # pauli_y
      [[1.0, 0.0], [0.0, -1.0]]     # pauli_z
  ])
  gates = np.stack([
      np.kron(pauli, np.eye(2 ** (num_qubits-1)))
      for pauli in pauli_operators
  ])

  # apply the same random unitary to all three gates (leaves commutation/
  # anticommutation relations invariant, but randomizes the operators)
  basis_rotation = stats.unitary_group.rvs(2 ** num_qubits)
  gates = np.matmul(np.matmul(basis_rotation, gates), basis_rotation.T.conj())

  return [
      (gates[idx_a], gates[idx_b])
      for idx_a, idx_b in itertools.product(range(3), repeat=2)
      if idx_a != idx_b
  ]


class CircuitTest(parameterized.TestCase):

  @parameterized.parameters([
      [3, 1],
      [5, 2],
      [4, 3],
      [4, 0]  # empty sequence (this is also allowed!)
  ])
  def test_initializer_and_getters(self, num_qubits_in, num_operations):
    operations_in = (
        circuit.Operation(_random_matrix_gate(1), [2]),
        circuit.Operation(_random_matrix_gate(2), [2, 3]),
        circuit.Operation(_random_matrix_gate(1), [3])
    )[:num_operations]

    # construct the circuit
    circ = circuit.Circuit(num_qubits_in, operations_in)

    # retrieve num_qubits
    num_qubits_out = circ.get_num_qubits()

    # check type and value for num_qubits
    self.assertIs(type(num_qubits_out), int)
    self.assertEqual(num_qubits_out, num_qubits_in)

    # retrieve the length
    length = len(circ)

    # check type and value for length
    self.assertIs(type(length), int)
    self.assertEqual(length, num_operations)

    # retrieve operations
    operations_out = circ.get_operation_sequence()

    # check type for operations, and that its elements are the operations which
    # have been put in
    self.assertIs(type(operations_out), tuple)
    self.assertLen(operations_out, num_operations)
    self.assertTrue(_elementwise_is(operations_out, operations_in))

  def test_initializer_with_none(self):
    # preparation work
    num_qubits_in = 5

    # construct the circuit
    circ = circuit.Circuit(num_qubits_in, None)

    # retrieve num_qubits
    num_qubits_out = circ.get_num_qubits()

    # check type and value for num_qubits
    self.assertIs(type(num_qubits_out), int)
    self.assertEqual(num_qubits_out, num_qubits_in)

    # retrieve the length
    length = len(circ)

    # check type and value for length
    self.assertIs(type(length), int)
    self.assertEqual(length, 0)

    # retrieve operations
    operations_out = circ.get_operation_sequence()

    # check type for operations, and that its elements are the operations which
    # have been put in
    self.assertIs(type(operations_out), tuple)
    self.assertEmpty(operations_out)

  def test_initializer_num_qubits_type_error(self):
    operations = [
        circuit.Operation(_random_matrix_gate(1), [2]),
        circuit.Operation(_random_matrix_gate(2), [2, 3]),
        circuit.Operation(_random_matrix_gate(1), [3])
    ]

    with self.assertRaisesRegex(
        TypeError,
        r'num_qubits is not integer-like \(found type: float\)'):
      circuit.Circuit(4.0, operations)

  def test_initializer_operation_sequence_not_iterable_error(self):
    with self.assertRaisesRegex(
        TypeError,
        r'\'float\' object is not iterable'):
      circuit.Circuit(4, 47.11)

  def test_initializer_illegal_qubit_indices_error(self):
    operations = [
        circuit.Operation(_random_matrix_gate(1), [2]),
        circuit.Operation(_random_matrix_gate(2), [2, 3]),
        circuit.Operation(_random_matrix_gate(1), [3])
    ]

    with self.assertRaisesRegex(
        ValueError,
        r'illegal qubit indices: 3 \[expected from range\(3\)\]'):
      circuit.Circuit(3, operations)

  @parameterized.parameters(range(4))
  def test_iter(self, num_operations):
    # preparation work: define the operations and construct the circuit
    operations_in = (
        circuit.Operation(_random_matrix_gate(1), [2]),
        circuit.Operation(_random_matrix_gate(2), [2, 3]),
        circuit.Operation(_random_matrix_gate(1), [3])
    )[:num_operations]
    circ = circuit.Circuit(4, operations_in)

    # (indirectly) call circ.__iter__
    operations_out = tuple(circ)

    # check length and content of operations_out
    self.assertLen(operations_out, num_operations)
    self.assertTrue(_elementwise_is(operations_out, operations_in))

  def test_add(self):
    # construct two circuits
    num_qubits = 4
    circ_1 = circuit.Circuit(num_qubits, [
        circuit.Operation(_random_matrix_gate(1), [2]),
        circuit.Operation(_random_matrix_gate(2), [2, 3]),
        circuit.Operation(_random_matrix_gate(1), [3])
    ])
    circ_2 = circuit.Circuit(num_qubits, [
        circuit.Operation(_random_matrix_gate(1), [1]),
        circuit.Operation(_random_matrix_gate(2), [1, 2]),
        circuit.Operation(_random_matrix_gate(1), [2]),
        circuit.Operation(_random_matrix_gate(2), [0, 1])
    ])

    # add the circuits
    circ_tot = circ_1 + circ_2

    # check num_qubits
    self.assertEqual(circ_tot.get_num_qubits(), num_qubits)

    # check that the operations of circ_tot are the concatenation of the
    # operations of circ_1 and circ_2
    self.assertTrue(_elementwise_is(
        circ_tot.get_operation_sequence(),
        circ_1.get_operation_sequence() + circ_2.get_operation_sequence()
    ))

  def test_add_type_error(self):
    circ = circuit.Circuit(4, [
        circuit.Operation(_random_matrix_gate(1), [2]),
        circuit.Operation(_random_matrix_gate(2), [2, 3]),
        circuit.Operation(_random_matrix_gate(1), [3])
    ])

    with self.assertRaisesRegex(
        TypeError,
        r'illegal type for other: int \(expected a Circuit\)'):
      circ + 5  # pylint: disable=pointless-statement

  def test_add_inconsistent_num_qubits_error(self):
    circ_1 = circuit.Circuit(4, [
        circuit.Operation(_random_matrix_gate(1), [2]),
        circuit.Operation(_random_matrix_gate(2), [2, 3]),
        circuit.Operation(_random_matrix_gate(1), [3])
    ])
    circ_2 = circuit.Circuit(5, [
        circuit.Operation(_random_matrix_gate(1), [1]),
        circuit.Operation(_random_matrix_gate(2), [1, 2]),
        circuit.Operation(_random_matrix_gate(1), [2]),
        circuit.Operation(_random_matrix_gate(2), [0, 1])
    ])

    with self.assertRaisesRegex(
        ValueError,
        r'number of qubits does not match \(4 vs 5\)'):
      circ_1 + circ_2  # pylint: disable=pointless-statement

  @parameterized.parameters([0, 1, 2, -3, -2, -1])
  def test_single_item(self, index):
    # preparation work: define the operations and construct the circuit
    operation_sequence = (
        circuit.Operation(_random_matrix_gate(1), [2]),
        circuit.Operation(_random_matrix_gate(2), [2, 3]),
        circuit.Operation(_random_matrix_gate(1), [3])
    )
    circ = circuit.Circuit(4, operation_sequence)

    # check __getitem__
    self.assertIs(circ[index], operation_sequence[index])

    # check operation method
    self.assertIs(circ.operation(index), operation_sequence[index])

  @parameterized.parameters([
      slice(None),
      slice(2),
      slice(1, 3),
      slice(2),
      slice(1, -1),
      slice(-2, 3),
      slice(-2, -1),
      slice(1, 1)
  ])
  def test_slicing(self, key):
    # preparation work: define the operations and construct the circuit
    operations_full = (
        circuit.Operation(_random_matrix_gate(1), [2]),
        circuit.Operation(_random_matrix_gate(2), [2, 3]),
        circuit.Operation(_random_matrix_gate(1), [3])
    )
    circ_full = circuit.Circuit(4, operations_full)
    operations_extracted = operations_full[key]

    # extracting slice using __getitem__
    circ_1 = circ_full[key]

    # check type and operations for circ_1
    self.assertIs(type(circ_1), circuit.Circuit)
    self.assertTrue(_elementwise_is(
        circ_1.get_operation_sequence(),
        operations_extracted
    ))

    # extracting slice using extract_slice
    circ_2 = circ_full.extract_slice(key)

    # check type and operations for circ_2
    self.assertIs(type(circ_2), circuit.Circuit)
    self.assertTrue(_elementwise_is(
        circ_2.get_operation_sequence(),
        operations_extracted
    ))

  @parameterized.parameters(itertools.product(
      [
          (),
          (0,),
          (0, 2),
          (0, -1),
          (-1, -2, -3)
      ],
      [list, np.array]
  ))
  def test_arbitrary_items(self, keys_value, keys_type):
    # preparation work: define the operations and construct the circuit
    operations_full = (
        circuit.Operation(_random_matrix_gate(1), [2]),
        circuit.Operation(_random_matrix_gate(2), [2, 3]),
        circuit.Operation(_random_matrix_gate(1), [3])
    )
    circ_full = circuit.Circuit(4, operations_full)
    operations_extracted = [operations_full[key] for key in keys_value]

    # extracting selection using __getitem__
    circ_1 = circ_full[keys_type(keys_value)]

    # check type and operations for circ_1
    self.assertIs(type(circ_1), circuit.Circuit)
    self.assertTrue(_elementwise_is(
        circ_1.get_operation_sequence(),
        operations_extracted
    ))

    # extracting selection using subcircuit
    circ_2 = circ_full.subcircuit(keys_type(keys_value))

    # check type and operations for circ_2
    self.assertIs(type(circ_2), circuit.Circuit)
    self.assertTrue(_elementwise_is(
        circ_2.get_operation_sequence(),
        operations_extracted
    ))

  def test_items_multiple_keys(self):
    # preparation work: define the operations and construct the circuit
    operations_full = (
        circuit.Operation(_random_matrix_gate(1), [2]),
        circuit.Operation(_random_matrix_gate(2), [2, 3]),
        circuit.Operation(_random_matrix_gate(1), [3]),
        circuit.Operation(_random_matrix_gate(2), [3, 4]),
        circuit.Operation(_random_matrix_gate(1), [4]),
        circuit.Operation(_random_matrix_gate(2), [4, 5]),
        circuit.Operation(_random_matrix_gate(1), [5]),
        circuit.Operation(_random_matrix_gate(2), [5, 6]),
        circuit.Operation(_random_matrix_gate(1), [6]),
        circuit.Operation(_random_matrix_gate(2), [6, 7]),
        circuit.Operation(_random_matrix_gate(1), [7])
    )
    circ_full = circuit.Circuit(10, operations_full)

    # calling __getitem__
    circ = circ_full[1:3, [8, 9, -4], 5]

    # check type and operations for circ
    self.assertIs(type(circ), circuit.Circuit)
    self.assertTrue(_elementwise_is(
        circ.get_operation_sequence(),
        operations_full[1:3] + (
            operations_full[8],
            operations_full[9],
            operations_full[-4],
            operations_full[5]
        )
    ))

  def test_getitem_single_key_noniterable_type_error(self):
    circ = circuit.Circuit(4, [
        circuit.Operation(_random_matrix_gate(1), [2]),
        circuit.Operation(_random_matrix_gate(2), [2, 3]),
        circuit.Operation(_random_matrix_gate(1), [3])
    ])

    with self.assertRaisesRegex(
        TypeError,
        r'unsupported key type: float'):
      circ[47.11]  # pylint: disable=pointless-statement

  def test_getitem_multiple_keys_noniterable_type_error(self):
    circ = circuit.Circuit(4, [
        circuit.Operation(_random_matrix_gate(1), [2]),
        circuit.Operation(_random_matrix_gate(2), [2, 3]),
        circuit.Operation(_random_matrix_gate(1), [3])
    ])

    with self.assertRaisesRegex(
        TypeError,
        r'unsupported key type: float'):
      circ[47.11, 0.815]  # pylint: disable=pointless-statement

  def test_getitem_single_key_iterable_type_error(self):
    circ = circuit.Circuit(4, [
        circuit.Operation(_random_matrix_gate(1), [2]),
        circuit.Operation(_random_matrix_gate(2), [2, 3]),
        circuit.Operation(_random_matrix_gate(1), [3])
    ])

    with self.assertRaisesRegex(
        TypeError,
        r'unsupported key type: str'):
      circ['hello']  # pylint: disable=pointless-statement

  def test_getitem_multiple_keys_iterable_type_error(self):
    circ = circuit.Circuit(4, [
        circuit.Operation(_random_matrix_gate(1), [2]),
        circuit.Operation(_random_matrix_gate(2), [2, 3]),
        circuit.Operation(_random_matrix_gate(1), [3])
    ])

    with self.assertRaisesRegex(
        TypeError,
        r'unsupported key type: str'):
      circ['hello', 'world']  # pylint: disable=pointless-statement

  def test_operation_key_type_error(self):
    circ = circuit.Circuit(4, [
        circuit.Operation(_random_matrix_gate(1), [2]),
        circuit.Operation(_random_matrix_gate(2), [2, 3]),
        circuit.Operation(_random_matrix_gate(1), [3])
    ])

    with self.assertRaisesRegex(
        TypeError,
        r'key is not integer-like \(found type: float\)'):
      circ.operation(47.11)

  def test_extract_slice_key_type_error(self):
    circ = circuit.Circuit(4, [
        circuit.Operation(_random_matrix_gate(1), [2]),
        circuit.Operation(_random_matrix_gate(2), [2, 3]),
        circuit.Operation(_random_matrix_gate(1), [3])
    ])

    with self.assertRaisesRegex(
        TypeError,
        r'key is not a slice \(found type: float\)'):
      circ.extract_slice(47.11)

  def test_subcircuit_noniterable_key_type_error(self):
    circ = circuit.Circuit(4, [
        circuit.Operation(_random_matrix_gate(1), [2]),
        circuit.Operation(_random_matrix_gate(2), [2, 3]),
        circuit.Operation(_random_matrix_gate(1), [3])
    ])

    with self.assertRaisesRegex(
        TypeError,
        r'key is not an iterable of int \(found type: float\)'):
      circ.subcircuit(47.11)

  def test_subcircuit_iterable_key_type_error(self):
    circ = circuit.Circuit(4, [
        circuit.Operation(_random_matrix_gate(1), [2]),
        circuit.Operation(_random_matrix_gate(2), [2, 3]),
        circuit.Operation(_random_matrix_gate(1), [3])
    ])

    with self.assertRaisesRegex(
        TypeError,
        r'key is not an iterable of int \(found type: str\)'):
      circ.subcircuit('hello')

  def test_schedule_and_depth(self):
    # preparation work: construct the circuit and define the expected moments
    circ = circuit.Circuit(4, [
        circuit.Operation(_random_matrix_gate(1), [2]),
        circuit.Operation(_random_matrix_gate(2), [0, 1]),
        circuit.Operation(_random_matrix_gate(2), [1, 2]),
        circuit.Operation(_random_matrix_gate(1), [1]),
        circuit.Operation(_random_matrix_gate(1), [0]),
        circuit.Operation(_random_matrix_gate(2), [0, 1])
    ])
    moments_expected = [0, 0, 1, 2, 1, 3]
    depth_expected = np.max(moments_expected) + 1

    # call circ.depth()
    depth_1 = circ.depth()

    # check type and value for depth_1
    self.assertIs(type(depth_1), int)
    self.assertEqual(depth_1, depth_expected)

    # call circ.schedule()
    depth_2, moments = circ.schedule()

    # check type and value for depth_2
    self.assertIs(type(depth_2), int)
    self.assertEqual(depth_2, depth_expected)

    # check type and value for moments
    self.assertIs(type(moments), np.ndarray)
    self.assertTupleEqual(moments.shape, (len(circ),))
    self.assertEqual(moments.dtype, np.int64)
    np.testing.assert_array_equal(moments, moments_expected)


class OperationTest(parameterized.TestCase):

  @parameterized.parameters([
      [(42,)],
      [(47, 11)],
      [(1, 3, 7)]
  ])
  def test_initializer_and_getters(self, qubits_in):
    # preparation work
    num_qubits_in = len(qubits_in)
    gate = _random_matrix_gate(num_qubits_in)

    # construct the operation
    operation = circuit.Operation(gate, qubits_in)

    # retrieve the gate and check that it is the one which has been put in
    self.assertIs(operation.get_gate(), gate)

    # retrieve qubits
    qubits_out = operation.get_qubits()

    # check type and value for qubits
    self.assertIs(type(qubits_out), tuple)
    self.assertTrue(all(type(qubit) == int for qubit in qubits_out))  # want only int and not any possible subtype, so pylint: disable=unidiomatic-typecheck
    self.assertEqual(qubits_out, qubits_in)

    # retrieve num_qubits
    num_qubits_out = operation.get_num_qubits()

    # check type and value for num_qubits
    self.assertIs(type(num_qubits_out), int)
    self.assertEqual(num_qubits_out, num_qubits_in)

  def test_initializer_gate_type_error(self):
    with self.assertRaisesRegex(
        TypeError,
        r'gate is not a Gate \(found type: range\)'):
      circuit.Operation(range(42), [47, 11])

  def test_initializer_non_integer_qubits_error(self):
    with self.assertRaisesRegex(
        TypeError,
        r'qubit is not integer-like \(found type: float\)'):
      circuit.Operation(circuit.MatrixGate(np.eye(4)), [2, 3.0])

  def test_initializer_duplicate_qubits_error(self):
    with self.assertRaisesRegex(
        ValueError,
        r'qubits \(47, 11, 47\) contain duplicate values'):
      circuit.Operation(circuit.MatrixGate(np.eye(8)), [47, 11, 47])

  def test_initializer_negative_qubits_error(self):
    with self.assertRaisesRegex(
        ValueError,
        r'illegal qubit indices: -7, -5 \(must be non-negative\)'):
      circuit.Operation(circuit.MatrixGate(np.eye(4)), [-5, -7])

  def test_initializer_num_qubits_error(self):
    with self.assertRaisesRegex(
        ValueError,
        r'num_qubits of gate does not match len\(qubits\) \[2 vs 1\]'):
      circuit.Operation(circuit.MatrixGate(np.eye(4)), [42])

  @parameterized.parameters([
      [(42,)],
      [(47, 11)],
      [(1, 3, 7)]
  ])
  def test_replace_gate(self, qubits):
    # preparation work
    num_qubits = len(qubits)
    placeholder_gate = _random_matrix_gate(num_qubits)
    replacement_gate = _random_matrix_gate(num_qubits)

    # construct the operation
    initial_operation = circuit.Operation(placeholder_gate, qubits)
    operation = initial_operation.replace_gate(replacement_gate)

    # check that the number of qubits did not change
    self.assertEqual(operation.get_num_qubits(), num_qubits)

    # retrieve the gate and check that it is the replacement_gate
    self.assertIs(operation.get_gate(), replacement_gate)

    # check qubits
    self.assertTupleEqual(operation.get_qubits(), qubits)

  def test_replace_gate_type_error(self):
    initial_operation = circuit.Operation(circuit.MatrixGate(np.eye(2)), [42])

    with self.assertRaisesRegex(
        TypeError,
        r'gate is not a Gate \(found type: range\)'):
      initial_operation.replace_gate(range(42))

  def test_replace_gate_num_qubits_error(self):
    initial_operation = circuit.Operation(circuit.MatrixGate(np.eye(2)), [42])

    with self.assertRaisesRegex(
        ValueError,
        r'num_qubits of gate does not match len\(qubits\) \[2 vs 1\]'):
      initial_operation.replace_gate(circuit.MatrixGate(np.eye(4)))

  @parameterized.parameters([
      [(42,), (21,)],
      [(47, 11), (12, 24)],
      [(1, 3, 7), (2, 5, 4)]
  ])
  def test_replace_qubits(self, placeholder_qubits, replacement_qubits):
    # preparation work
    num_qubits = len(placeholder_qubits)
    gate = _random_matrix_gate(num_qubits)

    # construct the operation
    initial_operation = circuit.Operation(gate, placeholder_qubits)
    operation = initial_operation.replace_qubits(replacement_qubits)

    # check that the number of qubits did not change
    self.assertEqual(operation.get_num_qubits(), num_qubits)

    # retrieve the qubits and check that they match the replacement_qubits
    self.assertTupleEqual(operation.get_qubits(), replacement_qubits)

    # check gate
    self.assertIs(operation.get_gate(), gate)

  def test_replace_qubits_non_integer_qubits_error(self):
    initial_operation = circuit.Operation(circuit.MatrixGate(np.eye(2)), [42])

    with self.assertRaisesRegex(
        TypeError,
        r'qubit is not integer-like \(found type: float\)'):
      initial_operation.replace_qubits([3.0])

  def test_replace_qubits_duplicate_qubits_error(self):
    initial_operation = circuit.Operation(
        circuit.MatrixGate(np.eye(8)),
        [1, 3, 7]
    )

    with self.assertRaisesRegex(
        ValueError,
        r'qubits \(19, 4, 19\) contain duplicate values'):
      initial_operation.replace_qubits([19, 4, 19])

  def test_replace_qubits_negative_qubits_error(self):
    initial_operation = circuit.Operation(circuit.MatrixGate(np.eye(2)), [42])

    with self.assertRaisesRegex(
        ValueError,
        r'illegal qubit indices: -7 \(must be non-negative\)'):
      initial_operation.replace_qubits([-7])

  def test_replace_qubits_num_qubits_error(self):
    initial_operation = circuit.Operation(circuit.MatrixGate(np.eye(2)), [42])

    with self.assertRaisesRegex(
        ValueError,
        r'num_qubits of gate does not match len\(qubits\) \[1 vs 2\]'):
      initial_operation.replace_qubits([47, 11])

  @parameterized.parameters(itertools.product([1, 2, 3], [False, True]))
  def test_permute_qubits_trivial(self, num_qubits, inverse):
    # preparation work
    gate = _random_matrix_gate(num_qubits)

    # construct the operation
    initial_operation = circuit.Operation(
        gate,
        np.random.permutation(10)[:num_qubits]
    )
    operation = initial_operation.permute_qubits(
        range(num_qubits),
        inverse=inverse
    )

    # check that operation is the initial_operation
    self.assertIs(operation, initial_operation)

  @parameterized.parameters([
      [(47, 11), (1, 0), False, (11, 47)],
      [(47, 11), (1, 0), True, (11, 47)],
      [(47, 11, 42), (2, 1, 0), False, (42, 11, 47)],
      [(47, 11, 42), (2, 1, 0), True, (42, 11, 47)],
      [(47, 11, 42), (1, 2, 0), False, (11, 42, 47)],
      [(47, 11, 42), (1, 2, 0), True, (42, 47, 11)]
  ])
  def test_permute_qubits(self, qubits_in, permutation, inverse, qubits_out):
    # preparation work
    num_qubits = len(qubits_in)
    gate = _random_matrix_gate(num_qubits)

    # construct the operation
    initial_operation = circuit.Operation(gate, qubits_in)
    operation = initial_operation.permute_qubits(permutation, inverse=inverse)

    # check that the number of qubits did not change
    self.assertEqual(operation.get_num_qubits(), num_qubits)

    # retrieve the qubits and check that they match the expectation
    self.assertTupleEqual(operation.get_qubits(), qubits_out)

    # check gate
    self.assertIs(operation.get_gate(), gate)

  def test_permute_qubits_illegal_permutation_length_error(self):
    operation = circuit.Operation(circuit.MatrixGate(np.eye(8)), [2, 3, 5])

    with self.assertRaisesRegex(
        ValueError,
        r'illegal shape for permutation: \(4,\) \[expected: \(3,\)\]'):
      operation.permute_qubits(np.arange(4))

  def test_permute_qubits_illegal_permutation_ndim_error(self):
    operation = circuit.Operation(circuit.MatrixGate(np.eye(8)), [2, 3, 5])

    with self.assertRaisesRegex(
        ValueError,
        r'illegal shape for permutation: \(2, 5\) \[expected: \(3,\)\]'):
      operation.permute_qubits(np.random.randint(3, size=[2, 5]))

  def test_permute_qubits_permutation_entries_out_of_range_error(self):
    operation = circuit.Operation(circuit.MatrixGate(np.eye(8)), [2, 3, 5])

    with self.assertRaisesRegex(
        ValueError,
        r'not a valid permutation: \[1 2 3\]'):
      operation.permute_qubits(np.arange(1, 4))

  def test_permute_qubits_not_actually_a_permutation_error(self):
    operation = circuit.Operation(circuit.MatrixGate(np.eye(8)), [2, 3, 5])

    with self.assertRaisesRegex(
        ValueError,
        r'not a valid permutation: \[2 2 2\]'):
      operation.permute_qubits([2, 2, 2])

  @parameterized.parameters([
      [(47, 11), (42,)],
      [(42,), (47, 11)],
      [(42,), (47,)]
  ])
  def test_commutes_trivially_positive(self, qubits_a, qubits_b):
    operation_a = circuit.Operation(
        circuit.MatrixGate(np.eye(2 ** len(qubits_a))),
        qubits_a
    )
    operation_b = circuit.Operation(
        circuit.MatrixGate(np.eye(2 ** len(qubits_b))),
        qubits_b
    )

    _check_boolean(
        self,
        operation_a.commutes_trivially_with(operation_b),
        True
    )

  @parameterized.parameters([
      [(47, 11), (47,)],
      [(47, 11), (11,)],
      [(47,), (47, 11)],
      [(11,), (47, 11)],
      [(42,), (42,)],
      [(1, 2, 3), (5, 2)]
  ])
  def test_commutes_trivially_negative(self, qubits_a, qubits_b):
    operation_a = circuit.Operation(
        circuit.MatrixGate(np.eye(2 ** len(qubits_a))),
        qubits_a
    )
    operation_b = circuit.Operation(
        circuit.MatrixGate(np.eye(2 ** len(qubits_b))),
        qubits_b
    )

    _check_boolean(
        self,
        operation_a.commutes_trivially_with(operation_b),
        False
    )

  def test_commutes_trivially_type_error(self):
    operation = circuit.Operation(circuit.MatrixGate(np.eye(4)), [47, 11])

    with self.assertRaisesRegex(
        TypeError,
        r'unexpected type for other: range \(expected an Operation\)'):
      operation.commutes_trivially_with(range(42))


class MatrixGateTest(parameterized.TestCase):

  @parameterized.parameters(_clifford_group())
  def test_initializer_on_clifford_gates(self, operator, pauli_transform):
    gate = circuit.MatrixGate(operator.copy())

    gate_num_qubits = gate.get_num_qubits()
    gate_operator = gate.get_operator()
    gate_pauli_transform = gate.get_pauli_transform()

    self.assertIs(type(gate_num_qubits), int)
    self.assertEqual(gate_num_qubits, 1)

    self.assertIs(type(gate_operator), np.ndarray)
    self.assertEqual(gate_operator.dtype, complex)
    self.assertTrue(np.array_equal(gate_operator, operator))

    self.assertIs(type(gate_pauli_transform), np.ndarray)
    self.assertEqual(gate_pauli_transform.dtype, float)
    self.assertTupleEqual(gate_pauli_transform.shape, (3, 3))
    np.testing.assert_allclose(  # check orthogonality
        np.dot(gate_pauli_transform, gate_pauli_transform.T),
        np.eye(3),
        rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        gate_pauli_transform,
        pauli_transform,
        rtol=1e-5, atol=1e-8
    )

  def test_operator_ndim_in_constructor(self):
    with self.assertRaisesRegex(
        ValueError,
        r'operator must be a 2D array \(found: ndim=3\)'):
      circuit.MatrixGate(np.random.randn(5, 4, 3))

  def test_operator_square_in_constructor(self):
    with self.assertRaisesRegex(
        ValueError,
        r'operator must be a square matrix \[found: shape=\(8, 4\)\]'):
      circuit.MatrixGate(np.random.randn(8, 4))

  def test_operator_dim_power_of_two_in_constructor(self):
    with self.assertRaisesRegex(
        ValueError,
        r'dimension of operator must be a power of 2 \(found: dim=7\)'):
      circuit.MatrixGate(np.random.randn(7, 7))

  def test_operator_unitary_in_constructor(self):
    with self.assertRaisesRegex(
        ValueError,
        r'operator is not unitary'):
      circuit.MatrixGate(np.full([8, 8], 0.2))

  @parameterized.parameters(1, 2, 3, 4)
  def test_initializer_on_random_gates(self, num_qubits):
    # generate a random unitary
    operator = stats.unitary_group.rvs(2 ** num_qubits)

    gate = circuit.MatrixGate(operator.copy())

    gate_num_qubits = gate.get_num_qubits()
    gate_operator = gate.get_operator()
    gate_pauli_transform = gate.get_pauli_transform()

    self.assertIs(type(gate_num_qubits), int)
    self.assertEqual(gate_num_qubits, num_qubits)

    self.assertIs(type(gate_operator), np.ndarray)
    self.assertEqual(gate_operator.dtype, complex)
    self.assertTrue(np.array_equal(gate_operator, operator))

    self.assertIs(type(gate_pauli_transform), np.ndarray)
    self.assertEqual(gate_pauli_transform.dtype, float)
    self.assertTupleEqual(
        gate_pauli_transform.shape,
        (4 ** num_qubits - 1, 4 ** num_qubits - 1)
    )
    np.testing.assert_allclose(  # check orthogonality
        np.dot(gate_pauli_transform, gate_pauli_transform.T),
        np.eye(4**num_qubits-1),
        rtol=1e-5, atol=1e-8
    )

    # define 1-qubit Pauli group
    pauli_1 = np.array([
        [[1.0, 0.0], [0.0, 1.0]],     # identity
        [[0.0, 1.0], [1.0, 0.0]],     # pauli_x
        [[0.0, -1.0j], [1.0j, 0.0]],  # pauli_y
        [[1.0, 0.0], [0.0, -1.0]]     # pauli_z
    ])

    # construct multi-qubit Pauli group
    pauli_n = pauli_1
    for _ in range(num_qubits-1):
      pauli_n = np.kron(pauli_n, pauli_1)

    coeffs_in = np.random.randn(4**num_qubits-1)
    coeffs_out = np.dot(gate_pauli_transform, coeffs_in)

    # might violate some properties of a physical density matrix (like
    # positivity of the eigenvalues), but good enough for testing here
    rho_in = np.tensordot(coeffs_in, pauli_n[1:], axes=[0, 0])
    rho_out = np.dot(operator, rho_in).dot(operator.T.conj())

    # check whether the Pauli coefficients are transformed as expected
    np.testing.assert_allclose(
        rho_out,
        np.tensordot(coeffs_out, pauli_n[1:], axes=[0, 0]),
        rtol=1e-5, atol=1e-8
    )

  @parameterized.parameters(1, 2, 3)
  def test_eq(self, num_qubits):
    # generate a random unitary
    operator = stats.unitary_group.rvs(2 ** num_qubits)

    gate_a = circuit.MatrixGate(operator)
    gate_b = circuit.MatrixGate(np.exp(2.0j*np.pi*np.random.rand()) * operator)
    gate_c = circuit.MatrixGate(np.roll(operator, 1, axis=0))

    self.assertEqual(gate_a, gate_a)
    self.assertEqual(gate_a, gate_b)
    self.assertNotEqual(gate_a, gate_c)

  @parameterized.parameters(1, 2, 3)
  def test_identity_always(self, num_qubits):
    gate = circuit.MatrixGate(np.eye(2 ** num_qubits))

    _check_boolean(
        self,
        gate.is_identity(phase_invariant=False),
        True
    )
    _check_boolean(
        self,
        gate.is_identity(phase_invariant=True),
        True
    )

  @parameterized.parameters(1, 2, 3)
  def test_identity_only_phase_invariant(self, num_qubits):
    gate = circuit.MatrixGate(np.exp(2.0j * np.pi * np.random.rand())
                              * np.eye(2 ** num_qubits))

    _check_boolean(
        self,
        gate.is_identity(phase_invariant=False),
        False
    )
    _check_boolean(
        self,
        gate.is_identity(phase_invariant=True),
        True
    )

  @parameterized.parameters(1, 2, 3)
  def test_identity_never(self, num_qubits):
    gate = circuit.MatrixGate(np.roll(np.eye(2 ** num_qubits), 1, axis=0))

    _check_boolean(
        self,
        gate.is_identity(phase_invariant=False),
        False
    )
    _check_boolean(
        self,
        gate.is_identity(phase_invariant=True),
        False
    )

  @parameterized.parameters(1, 2, 3)
  def test_cancels_always(self, num_qubits):
    # generate a random unitary
    operator = stats.unitary_group.rvs(2 ** num_qubits)

    gate_a = circuit.MatrixGate(operator)
    gate_b = circuit.MatrixGate(operator.T.conj())

    _check_boolean(
        self,
        gate_a.cancels_with(gate_b, phase_invariant=False),
        True
    )
    _check_boolean(
        self,
        gate_a.cancels_with(gate_b, phase_invariant=True),
        True
    )

  @parameterized.parameters(1, 2, 3)
  def test_cancels_only_phase_invariant(self, num_qubits):
    # generate a random unitary
    operator = stats.unitary_group.rvs(2 ** num_qubits)

    gate_a = circuit.MatrixGate(operator)
    gate_b = circuit.MatrixGate(np.exp(2.0j*np.pi*np.random.rand())
                                * operator.T.conj())

    _check_boolean(
        self,
        gate_a.cancels_with(gate_b, phase_invariant=False),
        False
    )
    _check_boolean(
        self,
        gate_a.cancels_with(gate_b, phase_invariant=True),
        True
    )

  @parameterized.parameters(1, 2, 3)
  def test_cancels_never(self, num_qubits):
    # generate a random unitary
    operator = stats.unitary_group.rvs(2 ** num_qubits)

    gate_a = circuit.MatrixGate(operator)
    gate_b = circuit.MatrixGate(np.roll(operator, 1, axis=0))

    _check_boolean(
        self,
        gate_a.cancels_with(gate_b, phase_invariant=False),
        False
    )
    _check_boolean(
        self,
        gate_a.cancels_with(gate_b, phase_invariant=True),
        False
    )

  @parameterized.parameters(False, True)
  def test_cancels_type_error(self, phase_invariant):
    gate = circuit.MatrixGate(np.eye(2))

    with self.assertRaisesRegex(
        TypeError,
        r'unexpected type for other: range \(expected a Gate\)'):
      gate.cancels_with(range(42), phase_invariant=phase_invariant)

  @parameterized.parameters(False, True)
  def test_cancels_num_qubits_error(self, phase_invariant):
    gate_a = circuit.MatrixGate(np.eye(2))
    gate_b = circuit.MatrixGate(np.eye(4))

    with self.assertRaisesRegex(
        ValueError,
        r'cancellation relation not well-defined because the number of qubits'
        r' does not match \(1 vs 2\)'):
      gate_a.cancels_with(gate_b, phase_invariant=phase_invariant)

  @parameterized.parameters(1, 2, 3)
  def test_commutation(self, num_qubits):
    # generate a random unitary
    eigenbasis = stats.unitary_group.rvs(2 ** num_qubits)

    # construct two unitary gates which are diagonal in the same eigenbasis
    # and therefore commute
    gate_a = circuit.MatrixGate(
        np.dot(
            eigenbasis,
            np.diag(np.exp(2.0j*np.pi*np.random.randn(2 ** num_qubits)))
        ).dot(eigenbasis.T.conj())
    )
    gate_b = circuit.MatrixGate(
        np.dot(
            eigenbasis,
            np.diag(np.exp(2.0j*np.pi*np.random.randn(2 ** num_qubits)))
        ).dot(eigenbasis.T.conj())
    )

    _check_boolean(
        self,
        gate_a.commutes_with(gate_b, phase_invariant=False),
        True
    )
    _check_boolean(
        self,
        gate_a.commutes_with(gate_b, phase_invariant=True),
        True
    )

  @parameterized.parameters(1, 2, 3)
  def test_commutation_with_identity(self, num_qubits):
    # generate a random unitary
    gate = _random_matrix_gate(num_qubits)

    identity = circuit.MatrixGate(np.eye(2 ** num_qubits))

    _check_boolean(
        self,
        gate.commutes_with(identity, phase_invariant=False),
        True  # every gate commutes with identity
    )
    _check_boolean(
        self,
        gate.commutes_with(identity, phase_invariant=True),
        True  # every gate commutes with identity
    )

  @parameterized.parameters(itertools.chain.from_iterable(
      _generate_random_anticommuting_operators(num_qubits)
      for num_qubits in (1, 2, 3)
  ))
  def test_anticommutation(self, operator_a, operator_b):
    gate_a = circuit.MatrixGate(operator_a)
    gate_b = circuit.MatrixGate(operator_b)

    _check_boolean(
        self,
        gate_a.commutes_with(gate_b, phase_invariant=False),
        False
    )
    _check_boolean(
        self,
        gate_a.commutes_with(gate_b, phase_invariant=True),
        True
    )

  @parameterized.parameters(1, 2, 3)
  def test_odd_commutation(self, num_qubits):
    # generate two random unitaries
    operator_a = stats.unitary_group.rvs(2 ** num_qubits)
    operator_b = stats.unitary_group.rvs(2 ** num_qubits)

    gate_a = circuit.MatrixGate(operator_a)
    gate_b = circuit.MatrixGate(operator_b)

    _check_boolean(
        self,
        gate_a.commutes_with(gate_b, phase_invariant=False),
        np.allclose(  # almost certainly False, but let's better check this
            np.dot(operator_a, operator_b),
            np.dot(operator_b, operator_a)
        )
    )
    _check_boolean(
        self,
        gate_a.commutes_with(gate_b, phase_invariant=True),
        np.allclose(  # almost certainly False, but let's better check this
            circuit.compute_pauli_transform(np.dot(operator_a, operator_b)),
            circuit.compute_pauli_transform(np.dot(operator_b, operator_a))
        )
    )

  @parameterized.parameters(False, True)
  def test_commutes_type_error(self, phase_invariant):
    gate = circuit.MatrixGate(np.eye(2))

    with self.assertRaisesRegex(
        TypeError,
        r'unexpected type for other: range \(expected a Gate\)'):
      gate.commutes_with(range(42), phase_invariant=phase_invariant)

  @parameterized.parameters(False, True)
  def test_commutes_num_qubits_error(self, phase_invariant):
    gate_a = circuit.MatrixGate(np.eye(2))
    gate_b = circuit.MatrixGate(np.eye(4))

    with self.assertRaisesRegex(
        ValueError,
        r'commutation relation not well-defined because the number of qubits'
        r' does not match \(1 vs 2\)'):
      gate_a.commutes_with(gate_b, phase_invariant=phase_invariant)

  @parameterized.parameters(itertools.product(
      [
          # all permutations between 1 and 4 qubits; can be generated with:
          #
          #     itertools.chain.from_iterable(
          #         itertools.permutations(range(num_qubits))
          #         for num_qubits in [1, 2, 3, 4]
          #     )
          (0,),
          (0, 1),
          (1, 0),
          (0, 1, 2),
          (0, 2, 1),
          (1, 0, 2),
          (1, 2, 0),
          (2, 0, 1),
          (2, 1, 0),
          (0, 1, 2, 3),
          (0, 1, 3, 2),
          (0, 2, 1, 3),
          (0, 2, 3, 1),
          (0, 3, 1, 2),
          (0, 3, 2, 1),
          (1, 0, 2, 3),
          (1, 0, 3, 2),
          (1, 2, 0, 3),
          (1, 2, 3, 0),
          (1, 3, 0, 2),
          (1, 3, 2, 0),
          (2, 0, 1, 3),
          (2, 0, 3, 1),
          (2, 1, 0, 3),
          (2, 1, 3, 0),
          (2, 3, 0, 1),
          (2, 3, 1, 0),
          (3, 0, 1, 2),
          (3, 0, 2, 1),
          (3, 1, 0, 2),
          (3, 1, 2, 0),
          (3, 2, 0, 1),
          (3, 2, 1, 0)
      ],
      [True, False]
  ))
  def test_permute_qubits(self, permutation, inverse):
    num_qubits = len(permutation)

    # generate a random unitary operator
    original_operator = stats.unitary_group.rvs(2 ** num_qubits)

    original_gate = circuit.MatrixGate(original_operator)
    permuted_gate = original_gate.permute_qubits(permutation, inverse=inverse)

    self.assertIsInstance(permuted_gate, circuit.Gate)
    self.assertEqual(permuted_gate.get_num_qubits(), num_qubits)

    # check consistency with function circuit.permute_qubits(...) which is
    # trusted from PermuteQubitsTest
    np.testing.assert_allclose(
        permuted_gate.get_operator(),
        circuit.permute_qubits(original_operator, permutation, inverse=inverse),
        rtol=1e-5, atol=1e-8
    )

  def test_permute_qubits_illegal_permutation_length_error(self):
    gate = circuit.MatrixGate(np.eye(8))
    with self.assertRaisesRegex(
        ValueError,
        r'illegal shape for permutation: \(4,\) \[expected: \(3,\)\]'):
      gate.permute_qubits(np.arange(4))

  def test_permute_qubits_illegal_permutation_ndim_error(self):
    gate = circuit.MatrixGate(np.eye(8))
    with self.assertRaisesRegex(
        ValueError,
        r'illegal shape for permutation: \(2, 5\) \[expected: \(3,\)\]'):
      gate.permute_qubits(np.random.randint(3, size=[2, 5]))

  def test_permute_qubits_permutation_entries_out_of_range_error(self):
    gate = circuit.MatrixGate(np.eye(8))
    with self.assertRaisesRegex(
        ValueError,
        r'not a valid permutation: \[1 2 3\]'):
      gate.permute_qubits(np.arange(1, 4))

  def test_permute_qubits_not_actually_a_permutation_error(self):
    gate = circuit.MatrixGate(np.eye(8))
    with self.assertRaisesRegex(
        ValueError,
        r'not a valid permutation: \[2 2 2\]'):
      gate.permute_qubits([2, 2, 2])

  @parameterized.parameters([
      [(0, 1, 2), lambda u0, u1, u2: (u0, u1, u2)],
      [(1, 2, 0), lambda u0, u1, u2: (u2, u0, u1)],
      [(0, 1, 2), lambda u0, u1, u2: (u0, u1, u2, np.eye(2))],
      [(1, 2, 0), lambda u0, u1, u2: (u2, u0, u1, np.eye(2))],
      [(1, 3, 2), lambda u0, u1, u2: (np.eye(2), u0, u2, u1)],
      [(3, 1, 2), lambda u0, u1, u2: (np.eye(2), u1, u2, u0)],
      [(0, 3, 2), lambda u0, u1, u2: (u0, np.eye(2), u2, u1)],
      [(3, 0, 2), lambda u0, u1, u2: (u1, np.eye(2), u2, u0)]
  ])
  def test_apply_on_didactic_random_product_operators(self, active_qubits,
                                                      expectation):
    # making sure that what is claimed in the docstring actually holds

    # generate three random single-qubit operators
    u0 = stats.unitary_group.rvs(2)
    u1 = stats.unitary_group.rvs(2)
    u2 = stats.unitary_group.rvs(2)

    expectation = expectation(u0, u1, u2)
    num_total_qubits = len(expectation)

    # construct the original (unextended) gate
    gate_orig = circuit.MatrixGate(np.kron(np.kron(u0, u1), u2))

    # call the function to be tested
    gate_mod = gate_orig.apply_on(active_qubits, num_total_qubits)

    # check that gate_mod is a Gate with the correct number of qubits
    self.assertIsInstance(gate_mod, circuit.Gate)
    self.assertEqual(gate_mod.get_num_qubits(), num_total_qubits)

    # compare the operator of gate_mod to the Kronecker product of expectation
    operator_expected = np.eye(1)
    for local_op in expectation:
      operator_expected = np.kron(operator_expected, local_op)
    np.testing.assert_allclose(
        gate_mod.get_operator(),
        operator_expected,
        rtol=1e-5, atol=1e-8
    )

  @parameterized.parameters([
      [(0,), 1],
      [(0,), 2],
      [(1,), 2],
      [(0,), 3],
      [(1,), 3],
      [(2,), 3],
      [(0, 1), 2],
      [(1, 0), 2],
      [(0, 1), 3],
      [(0, 2), 3],
      [(1, 0), 3],
      [(1, 2), 3],
      [(2, 0), 3],
      [(2, 1), 3],
      [(0, 1, 2), 3],
      [(0, 2, 1), 3],
      [(1, 0, 2), 3],
      [(1, 2, 0), 3],
      [(2, 0, 1), 3],
      [(2, 1, 0), 3]
  ])
  def test_apply_on_extensive_random_product_operators(self, active_qubits,
                                                       num_total_qubits):
    # generate random unitary single-qubit operators associated to the active
    # qubits
    local_ops = {qubit: stats.unitary_group.rvs(2) for qubit in active_qubits}

    # construct the original (unextended) gate
    operator_orig = np.eye(1)
    for qubit in active_qubits:
      operator_orig = np.kron(operator_orig, local_ops[qubit])
    gate_orig = circuit.MatrixGate(operator_orig)

    # call the function to be tested
    gate_mod = gate_orig.apply_on(active_qubits, num_total_qubits)

    # check that gate_mod is a Gate with the correct number of qubits
    self.assertIsInstance(gate_mod, circuit.Gate)
    self.assertEqual(gate_mod.get_num_qubits(), num_total_qubits)

    # compare the operator of gate_mod to the manually constructed expectation
    operator_expected = np.eye(1)
    for qubit in range(num_total_qubits):
      operator_expected = np.kron(
          operator_expected,
          local_ops[qubit] if qubit in active_qubits else np.eye(2)
      )
    np.testing.assert_allclose(
        gate_mod.get_operator(),
        operator_expected,
        rtol=1e-5, atol=1e-8
    )

  @parameterized.parameters(itertools.permutations(range(3), 2))
  def test_apply_on_with_cnot(self, src_qubit, tgt_qubit, num_total_qubits=3):
    # construct the original (unextended) gate
    cnot_gate = circuit.MatrixGate(np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ]))

    # call the function to be tested
    mod_gate = cnot_gate.apply_on([src_qubit, tgt_qubit], num_total_qubits)

    # check that gate_mod is a Gate with the correct number of qubits
    self.assertIsInstance(mod_gate, circuit.Gate)
    self.assertEqual(mod_gate.get_num_qubits(), num_total_qubits)

    # compare the operator of gate_mod to the manually constructed expectation
    bit_pattern = np.array(list(itertools.product(range(2),
                                                  repeat=num_total_qubits)))
    bit_pattern[:, tgt_qubit] ^= bit_pattern[:, src_qubit]

    operator_expected = np.zeros([8, 8])
    for row, col in enumerate(np.dot(
        bit_pattern,
        np.flip(2 ** np.arange(num_total_qubits)))):
      operator_expected[row, col] = 1.0
    np.testing.assert_allclose(
        mod_gate.get_operator(),
        operator_expected,
        rtol=1e-5, atol=1e-8
    )

  # TODO(tfoesel): add test for another entangling gate, e.g. FSIM gate

  @parameterized.parameters(1, 2, 3)
  def test_apply_on_trivial_case(self, num_qubits):
    # generate a random unitary
    gate = _random_matrix_gate(num_qubits)

    self.assertIs(gate, gate.apply_on(range(num_qubits), num_qubits))

  def test_apply_on_wrong_ndim_active_qubits(self):
    with self.assertRaisesRegex(
        TypeError,
        r'active_qubits is not a sequence of int \[shape: \(3, 5\)\]'):
      circuit.MatrixGate(np.eye(4)).apply_on(np.arange(15).reshape(3, 5), 17)

  def test_apply_on_total_num_qubits_type_error(self):
    with self.assertRaisesRegex(
        TypeError,
        r'total_num_qubits is not integer-like \(found type: float\)'):
      circuit.MatrixGate(np.eye(4)).apply_on([1, 2], 3.0)

  def test_apply_on_too_less_active_qubits(self):
    with self.assertRaisesRegex(
        ValueError,
        r'illegal length for active_qubits: 1 \(expected: 2\)'):
      circuit.MatrixGate(np.eye(4)).apply_on([1], 2)

  def test_apply_on_too_many_active_qubits(self):
    with self.assertRaisesRegex(
        ValueError,
        r'illegal length for active_qubits: 3 \(expected: 2\)'):
      circuit.MatrixGate(np.eye(4)).apply_on([1, 3, 0], 4)

  def test_apply_on_illegal_num_qubits(self):
    with self.assertRaisesRegex(
        ValueError,
        r'number of qubits cannot be reduced \(from 2 to 1\)'):
      circuit.MatrixGate(np.eye(4)).apply_on([0], 1)

  def test_apply_on_active_qubits_out_of_range(self):
    with self.assertRaisesRegex(
        ValueError,
        r'illegal value for active_qubits: \[9 4\] \(expected a 2-length'
        r' permutation of range\(7\)\)'):
      # if total_num_qubits == 7, then there must not be a 9 in the
      # active_qubits argument (all must be from range(7))
      circuit.MatrixGate(np.eye(4)).apply_on([9, 4], 7)

  def test_apply_on_duplicate_active_qubits(self):
    with self.assertRaisesRegex(
        ValueError,
        r'illegal value for active_qubits: \[5 5\] \(expected a 2-length'
        r' permutation of range\(7\)\)'):
      circuit.MatrixGate(np.eye(4)).apply_on([5, 5], 7)


class PhasedXGateTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product(_testing_angles(), repeat=2))
  def test_initializer(self, rotation_angle, phase_angle):
    # construct the PhasedXGate
    gate = circuit.PhasedXGate(rotation_angle, phase_angle)

    # retrieve num_qubits from the gate
    num_qubits = gate.get_num_qubits()

    # check type and value for num_qubits
    self.assertIs(type(num_qubits), int)
    self.assertEqual(num_qubits, 1)

    # retrieve the rotation angle from the gate
    gate_rotation_angle = gate.get_rotation_angle()

    # check type of the obtained rotation angle
    self.assertIs(type(gate_rotation_angle), float)

    # retrieve the phase angle from the gate
    gate_phase_angle = gate.get_phase_angle()

    # check type of the obtained phase angle
    self.assertIs(type(gate_phase_angle), float)

    # check values for the obtained rotation and phase angle
    np.testing.assert_allclose(
        _euler_to_dcm(
            'zxz',
            [-gate_phase_angle, gate_rotation_angle, gate_phase_angle]
        ),
        _euler_to_dcm(
            'zxz',
            [-phase_angle, rotation_angle, phase_angle]
        ),
        rtol=1e-5, atol=1e-8
    )

  def test_initializer_rotation_angle_type_error(self):
    with self.assertRaisesRegex(TypeError, r'can\'t convert complex to float'):
      circuit.PhasedXGate(42.0 + 47.11j, 0.815)

  def test_initializer_phase_angle_type_error(self):
    with self.assertRaisesRegex(TypeError, r'can\'t convert complex to float'):
      circuit.PhasedXGate(0.815, 42.0 + 47.11j)

  @parameterized.parameters(itertools.product(_testing_angles(), repeat=2))
  def test_vs_cirq(self, rotation_angle, phase_angle):
    # construct the PhasedXGate
    gate = circuit.PhasedXGate(rotation_angle, phase_angle)

    # construct the equivalent gate in Cirq
    cirq_gate = cirq.PhasedXPowGate(
        exponent=rotation_angle / np.pi,
        phase_exponent=phase_angle / np.pi
    )

    # check that they match (up to a potential global phase)
    np.testing.assert_allclose(
        gate.get_pauli_transform(),  # trusted from its unit test below
        circuit.compute_pauli_transform(cirq.unitary(cirq_gate)),
        rtol=1e-5, atol=1e-8
    )

  def test_flip_x(self):
    # construct the PhasedXGate
    gate = circuit.PhasedXGate.flip_x()

    # check the type of gate
    self.assertIs(type(gate), circuit.PhasedXGate)

    # check the pauli_transform of gate
    np.testing.assert_allclose(
        gate.get_pauli_transform(),  # trusted from its unit test below
        np.diag([1.0, -1.0, -1.0]),
        rtol=1e-5, atol=1e-8
    )

  def test_flip_y(self):
    # construct the PhasedXGate
    gate = circuit.PhasedXGate.flip_y()

    # check the type of gate
    self.assertIs(type(gate), circuit.PhasedXGate)

    # check the pauli_transform of gate
    np.testing.assert_allclose(
        gate.get_pauli_transform(),  # trusted from its unit test below
        np.diag([-1.0, 1.0, -1.0]),
        rtol=1e-5, atol=1e-8
    )

  @parameterized.parameters(_testing_angles())
  def test_rot_x(self, rotation_angle):
    # construct the PhasedXGate
    gate = circuit.PhasedXGate.rot_x(rotation_angle)

    # check the type of gate
    self.assertIs(type(gate), circuit.PhasedXGate)

    # check the pauli_transform of gate
    np.testing.assert_allclose(
        gate.get_pauli_transform(),  # trusted from its unit test below
        _euler_to_dcm('x', rotation_angle),
        rtol=1e-5, atol=1e-8
    )

  def test_rot_x_rotation_angle_type_error(self):
    with self.assertRaisesRegex(TypeError, r'can\'t convert complex to float'):
      circuit.PhasedXGate.rot_x(42.0 + 47.11j)

  @parameterized.parameters(_testing_angles())
  def test_rot_y(self, rotation_angle):
    # construct the PhasedXGate
    gate = circuit.PhasedXGate.rot_y(rotation_angle)

    # check the type of gate
    self.assertIs(type(gate), circuit.PhasedXGate)

    # check the pauli_transform of gate
    np.testing.assert_allclose(
        gate.get_pauli_transform(),  # trusted from its unit test below
        _euler_to_dcm('y', rotation_angle),
        rtol=1e-5, atol=1e-8
    )

  def test_rot_y_rotation_angle_type_error(self):
    with self.assertRaisesRegex(TypeError, r'can\'t convert complex to float'):
      circuit.PhasedXGate.rot_y(42.0 + 47.11j)

  def test_shift_rotation_angle(self):
    # draw three random angles
    base_rotation_angle = 2.0 * np.pi * np.random.rand()
    added_rotation_angle = 2.0 * np.pi * np.random.rand()
    phase_angle = 2.0 * np.pi * np.random.rand()

    # preparation work
    base_gate = circuit.PhasedXGate(base_rotation_angle, phase_angle)

    # call the function to be tested
    gate = base_gate.shift_rotation_angle(added_rotation_angle)

    # check the operator of the obtained gate
    expected_gate = circuit.PhasedXGate(
        base_rotation_angle + added_rotation_angle,
        phase_angle
    )
    np.testing.assert_allclose(
        gate.get_operator(),
        expected_gate.get_operator()
    )

  def test_shift_rotation_angle_type_error(self):
    gate = circuit.PhasedXGate(0.815, 0.137)

    with self.assertRaisesRegex(TypeError, r'can\'t convert complex to float'):
      gate.shift_rotation_angle(42.0 + 47.11j)

  def test_shift_phase_angle(self):
    # draw three random angles
    rotation_angle = 2.0 * np.pi * np.random.rand()
    base_phase_angle = 2.0 * np.pi * np.random.rand()
    added_phase_angle = 2.0 * np.pi * np.random.rand()

    # preparation work
    base_gate = circuit.PhasedXGate(rotation_angle, base_phase_angle)

    # call the function to be tested
    gate = base_gate.shift_phase_angle(added_phase_angle)

    # check the operator of the obtained gate
    expected_gate = circuit.PhasedXGate(
        rotation_angle,
        base_phase_angle + added_phase_angle
    )
    np.testing.assert_allclose(
        gate.get_operator(),
        expected_gate.get_operator()
    )

  def test_shift_phase_angle_type_error(self):
    gate = circuit.PhasedXGate(0.815, 0.137)

    with self.assertRaisesRegex(TypeError, r'can\'t convert complex to float'):
      gate.shift_phase_angle(42.0 + 47.11j)

  @parameterized.parameters(itertools.product(_testing_angles(), repeat=2))
  def test_operator(self, rotation_angle, phase_angle):
    # preparation work: construct the PhasedXGate
    gate = circuit.PhasedXGate(rotation_angle, phase_angle)

    # retrieve the operator from the gate
    operator = gate.get_operator()

    # check type for the obtained operator
    self.assertIs(type(operator), np.ndarray)

    # check dtype and shape for the obtained operator
    self.assertEqual(operator.dtype, complex)
    self.assertTupleEqual(operator.shape, (2, 2))

    # check the value for the obtained operator
    _check_unitarity(operator, 2)
    np.testing.assert_allclose(
        circuit.compute_pauli_transform(operator),
        _euler_to_dcm('zxz', [-phase_angle, rotation_angle, phase_angle]),
        rtol=1e-5, atol=1e-8
    )

  @parameterized.parameters(itertools.product(_testing_angles(), repeat=2))
  def test_pauli_transform(self, rotation_angle, phase_angle):
    # preparation work: construct the PhasedXGate
    gate = circuit.PhasedXGate(rotation_angle, phase_angle)

    # retrieve pauli_transform from the gate
    pauli_transform = gate.get_pauli_transform()

    # check type for the obtained pauli_transform
    self.assertIs(type(pauli_transform), np.ndarray)

    # check dtype and shape for the obtained pauli_transform
    self.assertEqual(pauli_transform.dtype, float)
    self.assertTupleEqual(pauli_transform.shape, (3, 3))

    # check the value for the obtained operator
    np.testing.assert_allclose(  # check orthogonality
        np.dot(pauli_transform, pauli_transform.T),
        np.eye(3),
        rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        pauli_transform,
        _euler_to_dcm('zxz', [-phase_angle, rotation_angle, phase_angle]),
        rtol=1e-5, atol=1e-8
    )

  @parameterized.parameters(itertools.product(_testing_angles(), repeat=2))
  def test_identity_for_testing_angles(self, rotation_angle, phase_angle):
    # preparation work: construct the PhasedXGate and an equivalent MatrixGate
    gate = circuit.PhasedXGate(rotation_angle, phase_angle)
    clone = circuit.MatrixGate(gate.get_operator())

    # for gate.is_identity(...), check type and consistency with
    # clone.is_identity(...) (which is trusted from the unit test for
    # MatrixGate) with both options for phase_invariant
    _check_boolean(
        self,
        gate.is_identity(phase_invariant=False),
        clone.is_identity(phase_invariant=False)
    )
    _check_boolean(
        self,
        gate.is_identity(phase_invariant=True),
        clone.is_identity(phase_invariant=True)
    )

  @parameterized.parameters(itertools.product(
      2.0*np.pi*np.arange(-3, 4),
      _testing_angles()
  ))
  def test_identity_for_full_rotations(self, rotation_angle, phase_angle):
    # preparation work: construct the PhasedXGate
    gate = circuit.PhasedXGate(rotation_angle, phase_angle)

    # for gate.is_identity(phase_invariant=False), check type and consistency
    # with clone.is_identity(...) (which is trusted from the unit test for
    # MatrixGate)
    _check_boolean(
        self,
        gate.is_identity(phase_invariant=False),
        circuit.MatrixGate(gate.get_operator()).is_identity()
    )

    # for gate.is_identity(phase_invariant=True), check type and value
    _check_boolean(
        self,
        gate.is_identity(phase_invariant=True),
        True
    )


class RotZGateTest(parameterized.TestCase):

  @parameterized.parameters(_testing_angles())
  def test_initializer_and_getter(self, rotation_angle):
    # construct the RotZGate
    gate = circuit.RotZGate(rotation_angle)

    # retrieve num_qubits from the gate
    num_qubits = gate.get_num_qubits()

    # check type and value for num_qubits
    self.assertIs(type(num_qubits), int)
    self.assertEqual(num_qubits, 1)

    # retrieve the rotation angle from the gate
    gate_rotation_angle = gate.get_rotation_angle()

    # check type and value of the obtained rotation angle
    self.assertIs(type(gate_rotation_angle), float)
    self.assertTrue(np.isclose(
        np.exp(1.0j * gate_rotation_angle),
        np.exp(1.0j * rotation_angle)
    ))

  def test_initializer_rotation_angle_type_error(self):
    with self.assertRaisesRegex(TypeError, r'can\'t convert complex to float'):
      circuit.RotZGate(42.0 + 47.11j)

  @parameterized.parameters(_testing_angles())
  def test_vs_cirq(self, rotation_angle):
    # construct the RotZGate
    gate = circuit.RotZGate(rotation_angle)

    # construct the equivalent gate in Cirq
    cirq_gate = cirq.ZPowGate(exponent=rotation_angle / np.pi)

    # check that they match (up to a potential global phase)
    np.testing.assert_allclose(
        gate.get_pauli_transform(),  # trusted from its unit test below
        circuit.compute_pauli_transform(cirq.unitary(cirq_gate)),
        rtol=1e-5, atol=1e-8
    )

  def test_shift_rotation_angle(self):
    # draw two random angles
    base_angle = 2.0 * np.pi * np.random.rand()
    added_angle = 2.0 * np.pi * np.random.rand()

    # preparation work
    base_gate = circuit.RotZGate(base_angle)

    # call the function to be tested
    gate = base_gate.shift_rotation_angle(added_angle)

    # check the operator of the obtained gate
    expected_gate = circuit.RotZGate(base_angle + added_angle)
    np.testing.assert_allclose(
        gate.get_operator(),
        expected_gate.get_operator()
    )

  def test_shift_rotation_angle_type_error(self):
    gate = circuit.RotZGate(0.815)

    with self.assertRaisesRegex(TypeError, r'can\'t convert complex to float'):
      gate.shift_rotation_angle(42.0 + 47.11j)

  @parameterized.parameters(_testing_angles())
  def test_operator(self, rotation_angle):
    # preparation work: construct the RotZGate
    gate = circuit.RotZGate(rotation_angle)

    # retrieve the operator from the gate
    operator = gate.get_operator()

    # check type for the obtained operator
    self.assertIs(type(operator), np.ndarray)

    # check dtype and shape for the obtained operator
    self.assertEqual(operator.dtype, complex)
    self.assertTupleEqual(operator.shape, (2, 2))

    # check the value for the obtained operator
    _check_unitarity(operator, 2)
    np.testing.assert_allclose(
        circuit.compute_pauli_transform(operator),
        _euler_to_dcm('z', rotation_angle),
        rtol=1e-5, atol=1e-8
    )

  @parameterized.parameters(_testing_angles())
  def test_pauli_transform(self, rotation_angle):
    # preparation work: construct the RotZGate
    gate = circuit.RotZGate(rotation_angle)

    # retrieve pauli_transform from the gate
    pauli_transform = gate.get_pauli_transform()

    # check type for the obtained pauli_transform
    self.assertIs(type(pauli_transform), np.ndarray)

    # check dtype and shape for the obtained pauli_transform
    self.assertEqual(pauli_transform.dtype, float)
    self.assertTupleEqual(pauli_transform.shape, (3, 3))

    # check the value for the obtained pauli_transform
    np.testing.assert_allclose(  # check orthogonality
        np.dot(pauli_transform, pauli_transform.T),
        np.eye(3),
        rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        pauli_transform,
        _euler_to_dcm('z', rotation_angle),
        rtol=1e-5, atol=1e-8
    )

  @parameterized.parameters(_testing_angles())
  def test_identity_for_testing_angles(self, rotation_angle):
    # preparation work: construct the RotZGate and an equivalent MatrixGate
    gate = circuit.RotZGate(rotation_angle)
    clone = circuit.MatrixGate(gate.get_operator())

    # for gate.is_identity(...), check type and consistency with
    # clone.is_identity(...) (which is trusted from the unit test for
    # MatrixGate) with both options for phase_invariant
    _check_boolean(
        self,
        gate.is_identity(phase_invariant=False),
        clone.is_identity(phase_invariant=False)
    )
    _check_boolean(
        self,
        gate.is_identity(phase_invariant=True),
        clone.is_identity(phase_invariant=True)
    )

  @parameterized.parameters(2.0*np.pi*np.arange(-3, 4))
  def test_identity_for_full_rotations(self, rotation_angle):
    # preparation work: construct the RotZGate
    gate = circuit.RotZGate(rotation_angle)

    # for gate.is_identity(phase_invariant=False), check type and consistency
    # with clone.is_identity(...) (which is trusted from the unit test for
    # MatrixGate)
    _check_boolean(
        self,
        gate.is_identity(phase_invariant=False),
        circuit.MatrixGate(gate.get_operator()).is_identity()
    )

    # for gate.is_identity(phase_invariant=True), check type and value
    _check_boolean(
        self,
        gate.is_identity(phase_invariant=True),
        True
    )


class ControlledZGateTest(parameterized.TestCase):

  def test_initializer(self):
    num_qubits = circuit.ControlledZGate().get_num_qubits()

    self.assertIs(type(num_qubits), int)
    self.assertEqual(num_qubits, 2)

  def test_vs_cirq(self):
    # construct the ControlledZGate
    gate = circuit.ControlledZGate()

    # construct the equivalent gate in Cirq
    cirq_gate = cirq.CZPowGate(exponent=1.0)

    # check that they match (up to a potential global phase)
    np.testing.assert_allclose(
        gate.get_pauli_transform(),  # trusted from its unit test below
        circuit.compute_pauli_transform(cirq.unitary(cirq_gate)),
        rtol=1e-5, atol=1e-8
    )

  def test_operator(self):
    operator = circuit.ControlledZGate().get_operator()

    self.assertIs(type(operator), np.ndarray)
    self.assertEqual(operator.dtype, complex)
    self.assertTupleEqual(operator.shape, (4, 4))
    _check_unitarity(operator, 4)
    np.testing.assert_allclose(
        circuit.compute_pauli_transform(operator),
        circuit.compute_pauli_transform(np.diag([1.0, 1.0, 1.0, -1.0])),
        rtol=1e-5, atol=1e-8
    )

  def test_pauli_transform(self):
    pauli_transform = circuit.ControlledZGate().get_pauli_transform()

    self.assertIs(type(pauli_transform), np.ndarray)
    self.assertEqual(pauli_transform.dtype, float)
    self.assertTupleEqual(pauli_transform.shape, (15, 15))
    np.testing.assert_allclose(  # check orthogonality
        np.dot(pauli_transform, pauli_transform.T),
        np.eye(15),
        rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        pauli_transform,
        circuit.compute_pauli_transform(np.diag([1.0, 1.0, 1.0, -1.0])),
        rtol=1e-5, atol=1e-8
    )

  def test_identity(self):
    gate = circuit.ControlledZGate()

    _check_boolean(
        self,
        gate.is_identity(phase_invariant=False),
        False
    )
    _check_boolean(
        self,
        gate.is_identity(phase_invariant=True),
        False
    )

  @parameterized.parameters(itertools.product([(0, 1), (1, 0)], [False, True]))
  def test_permute_qubits(self, permutation, inverse):
    gate = circuit.ControlledZGate()
    self.assertIs(gate, gate.permute_qubits(permutation, inverse=inverse))


class ComputePauliTransformTest(parameterized.TestCase):

  @parameterized.parameters(1, 2, 3, 4)
  def test_random_gates(self, num_qubits):
    # generate a random unitary
    operator = stats.unitary_group.rvs(2 ** num_qubits)

    pauli_transform = circuit.compute_pauli_transform(operator.copy())

    self.assertTupleEqual(
        pauli_transform.shape,
        (4 ** num_qubits - 1, 4 ** num_qubits - 1)
    )
    np.testing.assert_allclose(  # check orthogonality
        np.dot(pauli_transform, pauli_transform.T),
        np.eye(4**num_qubits-1),
        rtol=1e-5, atol=1e-8
    )

    # define 1-qubit Pauli group
    pauli_1 = np.array([
        [[1.0, 0.0], [0.0, 1.0]],     # identity
        [[0.0, 1.0], [1.0, 0.0]],     # pauli_x
        [[0.0, -1.0j], [1.0j, 0.0]],  # pauli_y
        [[1.0, 0.0], [0.0, -1.0]]     # pauli_z
    ])

    # construct multi-qubit Pauli group
    pauli_n = pauli_1
    for _ in range(num_qubits-1):
      pauli_n = np.kron(pauli_n, pauli_1)

    coeffs_in = np.random.randn(4**num_qubits-1)
    coeffs_out = np.dot(pauli_transform, coeffs_in)

    # might violate some properties of a physical density matrix (like
    # positivity of the eigenvalues), but good enough for testing here
    rho_in = np.tensordot(coeffs_in, pauli_n[1:], axes=[0, 0])
    rho_out = np.dot(operator, rho_in).dot(operator.T.conj())

    # check whether the Pauli coefficients are transformed as expected
    np.testing.assert_allclose(
        rho_out,
        np.tensordot(coeffs_out, pauli_n[1:], axes=[0, 0]),
        rtol=1e-5, atol=1e-8
    )


class PermuteQubitsTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product([1, 2, 3, 4, 5], [False, True]))
  def test_random_product_gates(self, num_qubits, inverse):
    permutation = np.random.permutation(num_qubits)

    # generate a bunch of random single-qubit unitaries
    operators = [stats.unitary_group.rvs(2) for _ in range(num_qubits)]

    operator_original = np.eye(1)
    for operator in operators:
      operator_original = np.kron(operator_original, operator)

    operator_expected = np.eye(1)
    for idx in np.argsort(permutation) if inverse else permutation:
      operator_expected = np.kron(operator_expected, operators[idx])

    operator_permuted = circuit.permute_qubits(
        operator_original.copy(),
        permutation.copy(),
        inverse=inverse
    )

    self.assertIs(type(operator_permuted), np.ndarray)
    self.assertEqual(operator_permuted.dtype, complex)
    self.assertTupleEqual(
        operator_permuted.shape,
        (2 ** num_qubits, 2 ** num_qubits)
    )
    _check_unitarity(operator_permuted, 2 ** num_qubits)
    np.testing.assert_allclose(
        operator_permuted,
        operator_expected,
        rtol=1e-5, atol=1e-8
    )

  @parameterized.parameters(False, True)
  def test_random_two_qubit_gate(self, inverse):
    # generate a random unitary operator on two qubits
    op_orig = stats.unitary_group.rvs(4)

    operator_expected = np.array([
        [op_orig[0, 0], op_orig[0, 2], op_orig[0, 1], op_orig[0, 3]],
        [op_orig[2, 0], op_orig[2, 2], op_orig[2, 1], op_orig[2, 3]],
        [op_orig[1, 0], op_orig[1, 2], op_orig[1, 1], op_orig[1, 3]],
        [op_orig[3, 0], op_orig[3, 2], op_orig[3, 1], op_orig[3, 3]]
    ])

    operator_permuted = circuit.permute_qubits(
        op_orig,
        np.array([1, 0]),
        inverse=inverse
    )

    self.assertIs(type(operator_permuted), np.ndarray)
    self.assertEqual(operator_permuted.dtype, complex)
    self.assertTupleEqual(operator_permuted.shape, (4, 4))
    _check_unitarity(operator_permuted, 4)
    np.testing.assert_allclose(
        operator_permuted,
        operator_expected,
        rtol=1e-5, atol=1e-8
    )

  @parameterized.parameters(False, True)
  def test_invert_cnot(self, inverse):
    operator_permuted = circuit.permute_qubits(
        np.array([  # CNOT gate as usual (i.e. source qubit first)
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0]
        ]),
        np.array([1, 0]),
        inverse=inverse
    )

    self.assertIs(type(operator_permuted), np.ndarray)
    self.assertEqual(operator_permuted.dtype, complex)
    self.assertTupleEqual(operator_permuted.shape, (4, 4))
    _check_unitarity(operator_permuted, 4)
    np.testing.assert_allclose(
        operator_permuted,
        np.array([  # CNOT gate inverted (i.e. target qubit first)
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ]),
        rtol=1e-5, atol=1e-8
    )

  @parameterized.parameters(False, True)
  def test_permute_fredkin(self, inverse):
    operator_permuted = circuit.permute_qubits(
        np.array([  # Fredkin gate as usual (i.e. 1st qubit controls swap of 2nd
                    # and 3rd qubit)
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ]),
        np.array([1, 2, 0]),
        inverse=inverse
    )

    self.assertIs(type(operator_permuted), np.ndarray)
    self.assertEqual(operator_permuted.dtype, complex)
    self.assertTupleEqual(operator_permuted.shape, (8, 8))
    _check_unitarity(operator_permuted, 8)

    if inverse:
      operator_expected = np.array([
          # modified Fredkin gate: 2nd qubit controls swap of 1st and 3rd qubit
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
      ])
    else:
      operator_expected = np.array([
          # modified Fredkin gate: 3rd qubit controls swap of 1st and 2nd qubit
          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
      ])

    np.testing.assert_allclose(
        operator_permuted,
        operator_expected,
        rtol=1e-5, atol=1e-8
    )

  def test_operator_ndim_error(self):
    with self.assertRaisesRegex(
        ValueError,
        r'operator must be a 2D array \(found: ndim=3\)'):
      circuit.permute_qubits(np.random.randn(5, 4, 3), np.arange(3))

  def test_operator_square_error(self):
    with self.assertRaisesRegex(
        ValueError,
        r'operator must be a square matrix \[found: shape=\(8, 4\)\]'):
      circuit.permute_qubits(np.random.randn(8, 4), np.arange(3))

  def test_operator_dim_power_of_two_error(self):
    with self.assertRaisesRegex(
        ValueError,
        r'dimension of operator must be a power of 2 \(found: dim=7\)'):
      circuit.permute_qubits(np.random.randn(7, 7), np.arange(3))

  def test_operator_unitary_error(self):
    with self.assertRaisesRegex(
        ValueError,
        r'operator is not unitary'):
      circuit.permute_qubits(np.full([8, 8], 0.2), np.arange(3))

  def test_illegal_permutation_length_error(self):
    with self.assertRaisesRegex(
        ValueError,
        r'illegal shape for permutation: \(4,\) \[expected: \(3,\)\]'):
      circuit.permute_qubits(np.eye(8), np.arange(4))

  def test_illegal_permutation_ndim_error(self):
    with self.assertRaisesRegex(
        ValueError,
        r'illegal shape for permutation: \(2, 5\) \[expected: \(3,\)\]'):
      circuit.permute_qubits(np.eye(8), np.random.randint(7, size=[2, 5]))

  def test_permutation_entries_out_of_range_error(self):
    with self.assertRaisesRegex(
        ValueError,
        r'not a valid permutation: \[1 2 3\]'):
      circuit.permute_qubits(np.eye(8), np.arange(1, 4))

  def test_not_actually_a_permutation_error(self):
    with self.assertRaisesRegex(
        ValueError,
        r'not a valid permutation: \[2 2 2\]'):
      circuit.permute_qubits(np.eye(8), [2, 2, 2])


class ExtendOperatorTest(parameterized.TestCase):

  @parameterized.parameters([
      # can be generated with:
      #
      #     [
      #         [is_qubit_active]
      #         for is_qubit_active in itertools.chain.from_iterable(
      #             itertools.product([True, False], repeat=num_qubits)
      #             for num_qubits in [1, 2, 3, 4]
      #         )
      #         if any(is_qubit_active)
      #     ]
      [(True,)],
      [(True, True)],
      [(True, False)],
      [(False, True)],
      [(True, True, True)],
      [(True, True, False)],
      [(True, False, True)],
      [(True, False, False)],
      [(False, True, True)],
      [(False, True, False)],
      [(False, False, True)],
      [(True, True, True, True)],
      [(True, True, True, False)],
      [(True, True, False, True)],
      [(True, True, False, False)],
      [(True, False, True, True)],
      [(True, False, True, False)],
      [(True, False, False, True)],
      [(True, False, False, False)],
      [(False, True, True, True)],
      [(False, True, True, False)],
      [(False, True, False, True)],
      [(False, True, False, False)],
      [(False, False, True, True)],
      [(False, False, True, False)],
      [(False, False, False, True)]
  ])
  def test_random_product_operators(self, is_qubit_active):
    active_qubits, = np.where(is_qubit_active)
    num_total_qubits = len(is_qubit_active)

    # generate random single-qubit operators associated to the active qubits
    local_ops = {
        qubit: np.dot(np.random.randn(2, 2, 2), [1.0, 1.0j])
        for qubit in active_qubits
    }

    # construct the original (unextended) operator
    operator_orig = np.eye(1)
    for qubit in active_qubits:
      operator_orig = np.kron(operator_orig, local_ops[qubit])

    # call the function to be tested
    operator_extended = circuit.extend_operator(operator_orig, is_qubit_active)

    # check that operator_extended is a np.ndarray with correct shape and dtype
    self.assertIs(type(operator_extended), np.ndarray)
    self.assertEqual(operator_extended.dtype, complex)
    self.assertTupleEqual(
        operator_extended.shape,
        (2 ** num_total_qubits, 2 ** num_total_qubits)
    )

    # compare operator_extended to the manually constructed expectation
    operator_expected = np.eye(1)
    for qubit in range(num_total_qubits):
      operator_expected = np.kron(
          operator_expected,
          local_ops[qubit] if qubit in active_qubits else np.eye(2)
      )
    np.testing.assert_allclose(
        operator_extended,
        operator_expected,
        rtol=1e-5, atol=1e-8
    )

  @parameterized.parameters([
      # can be generated with:
      #
      #     [
      #         [perm]
      #         for num_qubits in [2, 3, 4]
      #         for perm in sorted(set(itertools.permutations(
      #             2 * [True]
      #             + (num_qubits - 2) * [False]
      #         )), reverse=True)
      #     ]
      [(True, True)],
      [(True, True, False)],
      [(True, False, True)],
      [(False, True, True)],
      [(True, True, False, False)],
      [(True, False, True, False)],
      [(True, False, False, True)],
      [(False, True, True, False)],
      [(False, True, False, True)],
      [(False, False, True, True)]
  ])
  def test_cnot(self, is_qubit_active):
    (src_qubit, tgt_qubit), = np.where(is_qubit_active)
    num_total_qubits = len(is_qubit_active)

    # call the function to be tested
    operator_extended = circuit.extend_operator(
        np.array([  # CNOT operation
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0]
        ]),
        is_qubit_active
    )

    # check that operator_extended is a np.ndarray with correct shape and dtype
    self.assertIs(type(operator_extended), np.ndarray)
    self.assertEqual(operator_extended.dtype, complex)
    self.assertTupleEqual(
        operator_extended.shape,
        (2 ** num_total_qubits, 2 ** num_total_qubits)
    )

    # compare operator_extended to the manually constructed expectation
    bit_pattern = np.array(list(itertools.product(range(2),
                                                  repeat=num_total_qubits)))
    bit_pattern[:, tgt_qubit] ^= bit_pattern[:, src_qubit]

    operator_expected = np.zeros([2 ** num_total_qubits, 2 ** num_total_qubits])
    for row, col in enumerate(np.dot(
        bit_pattern,
        np.flip(2 ** np.arange(num_total_qubits)))):
      operator_expected[row, col] = 1.0

    np.testing.assert_allclose(
        operator_extended,
        operator_expected,
        rtol=1e-5, atol=1e-8
    )

  def test_dtype_for_active_qubits_error(self):
    with self.assertRaisesRegex(
        TypeError,
        r'is_qubit_active is not a sequence of bool \[int64 cannot be casted'
        r' safely to bool\]'):
      circuit.extend_operator(np.eye(4), np.arange(2))

  def test_wrong_ndim_for_operator_error(self):
    with self.assertRaisesRegex(
        ValueError,
        r'operator must be a 2D array \(found: ndim=3\)'):
      circuit.extend_operator(np.random.randn(2, 3, 4), [True, False])

  def test_nonsquare_operator_error(self):
    with self.assertRaisesRegex(
        ValueError,
        r'operator must be a square matrix \[found: shape=\(8, 4\)\]'):
      circuit.extend_operator(np.random.randn(8, 4), [True, False])

  def test_operator_dim_not_power_of_two_error(self):
    with self.assertRaisesRegex(
        ValueError,
        r'dimension of operator must be a power of 2 \(found: dim=3\)'):
      circuit.extend_operator(np.eye(3), [True, False])

  def test_wrong_ndim_for_active_qubits_error(self):
    with self.assertRaisesRegex(
        TypeError,
        r'is_qubit_active is not a sequence of bool \(found: ndim=2\)'):
      circuit.extend_operator(
          np.eye(4),
          np.array([False, True, True, False]).reshape(2, 2)
      )

  def test_inconsistent_number_of_active_qubits_error(self):
    with self.assertRaisesRegex(
        ValueError,
        r'1 active qubit\(s\) not compatible with operator dimension 4 \[i.e. 2'
        r' qubit\(s\)\]'):
      circuit.extend_operator(np.eye(4), [False, True, False])


class TestRelations(parameterized.TestCase):

  # TODO(tfoesel): simplify the structure for the parameters.

  @parameterized.parameters(itertools.product(
      [False, True],
      itertools.chain(
          itertools.product(
              [
                  circuit.PhasedXGate(0.0, 0.815),
                  circuit.PhasedXGate(4.0*np.pi, 0.815)
              ],
              [
                  circuit.PhasedXGate(0.0, 0.137),
                  circuit.PhasedXGate(4.0*np.pi, 0.137)
              ]
          ),
          itertools.product(
              [
                  circuit.PhasedXGate(2.0*np.pi, 0.815),
                  circuit.PhasedXGate(6.0*np.pi, 0.815)
              ],
              [
                  circuit.PhasedXGate(2.0*np.pi, 0.137),
                  circuit.PhasedXGate(6.0*np.pi, 0.137)
              ]
          ),
          itertools.product(
              [circuit.PhasedXGate(0.4711, 0.815)],
              [
                  circuit.PhasedXGate(0.4711, 0.815+np.pi),
                  circuit.PhasedXGate(0.4711, 0.815+3.0*np.pi),
                  circuit.PhasedXGate(0.4711+4.0*np.pi, 0.815+np.pi),
                  circuit.PhasedXGate(0.4711+4.0*np.pi, 0.815+3.0*np.pi)
              ]
          ),
          itertools.product(
              [circuit.PhasedXGate(0.4711, 0.815)],
              [
                  circuit.PhasedXGate(-0.4711, 0.815),
                  circuit.PhasedXGate(4.0*np.pi-0.4711, 0.815),
                  circuit.PhasedXGate(-0.4711, 2.0*np.pi+0.815),
                  circuit.PhasedXGate(4.0*np.pi-0.4711, 2.0*np.pi+0.815)
              ]
          )
      )
  ))
  def test_phased_x_vs_phased_x_a(self, cnvt_to_ops, gates):
    gate_a, gate_b = gates

    if cnvt_to_ops:
      obj_a = circuit.Operation(gate_a, [42])
      obj_b = circuit.Operation(gate_b, [42])
    else:
      obj_a = gate_a
      obj_b = gate_b

    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=False),
        True
    )
    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=True),
        True
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=False),
        True
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=True),
        True
    )

  @parameterized.parameters(itertools.product(
      [False, True],
      itertools.chain(
          itertools.product(
              [
                  circuit.PhasedXGate(0.0, 0.815),
                  circuit.PhasedXGate(4.0*np.pi, 0.815)
              ],
              [
                  circuit.PhasedXGate(2.0*np.pi, 0.137),
                  circuit.PhasedXGate(6.0*np.pi, 0.137)
              ]
          ),
          itertools.product(
              [circuit.PhasedXGate(0.4711, 0.815)],
              [
                  circuit.PhasedXGate(0.4711+2.0*np.pi, 0.815+np.pi),
                  circuit.PhasedXGate(0.4711+6.0*np.pi, 0.815+np.pi),
                  circuit.PhasedXGate(0.4711+2.0*np.pi, 0.815+3.0*np.pi),
                  circuit.PhasedXGate(0.4711+6.0*np.pi, 0.815+3.0*np.pi)
              ]
          ),
          itertools.product(
              [circuit.PhasedXGate(0.4711, 0.815)],
              [
                  circuit.PhasedXGate(2.0*np.pi-0.4711, 0.815),
                  circuit.PhasedXGate(2.0*np.pi-0.4711, 0.815+2.0*np.pi),
                  circuit.PhasedXGate(6.0*np.pi-0.4711, 0.815),
                  circuit.PhasedXGate(6.0*np.pi-0.4711, 0.815+2.0*np.pi)
              ]
          )
      )
  ))
  def test_phased_x_vs_phased_x_b(self, cnvt_to_ops, gates):
    gate_a, gate_b = gates

    if cnvt_to_ops:
      obj_a = circuit.Operation(gate_a, [42])
      obj_b = circuit.Operation(gate_b, [42])
    else:
      obj_a = gate_a
      obj_b = gate_b

    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=False),
        False
    )
    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=True),
        True
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=False),
        True
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=True),
        True
    )

  @parameterized.parameters(itertools.product(
      [False, True],
      [
          [
              circuit.PhasedXGate(0.4711, 0.815),
              circuit.PhasedXGate(0.137, 0.815)
          ],
          [
              circuit.PhasedXGate(0.4711, 0.815),
              circuit.PhasedXGate(0.137, 0.815+np.pi)
          ]
      ]
  ))
  def test_phased_x_vs_phased_x_c(self, cnvt_to_ops, gates):
    gate_a, gate_b = gates

    if cnvt_to_ops:
      obj_a = circuit.Operation(gate_a, [42])
      obj_b = circuit.Operation(gate_b, [42])
    else:
      obj_a = gate_a
      obj_b = gate_b

    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=False),
        False
    )
    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=True),
        False
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=False),
        True
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=True),
        True
    )

  @parameterized.parameters(itertools.product(
      [False, True],
      itertools.product(
          [
              circuit.PhasedXGate(np.pi, 0.815),
              circuit.PhasedXGate(3.0*np.pi, 0.815)
          ],
          [
              circuit.PhasedXGate(np.pi, 0.815+0.5*np.pi),
              circuit.PhasedXGate(3.0*np.pi, 0.815+0.5*np.pi),
              circuit.PhasedXGate(np.pi, 0.815+1.5*np.pi),
              circuit.PhasedXGate(3.0*np.pi, 0.815+1.5*np.pi)
          ]
      )
  ))
  def test_phased_x_vs_phased_x_d(self, cnvt_to_ops, gates):
    gate_a, gate_b = gates

    if cnvt_to_ops:
      obj_a = circuit.Operation(gate_a, [42])
      obj_b = circuit.Operation(gate_b, [42])
    else:
      obj_a = gate_a
      obj_b = gate_b

    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=False),
        False
    )
    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=True),
        False
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=False),
        False
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=True),
        True
    )

  @parameterized.parameters(True, False)
  def test_phased_x_vs_phased_x_e(self, cnvt_to_ops):
    gate_a = circuit.PhasedXGate(0.4711, 0.815)
    gate_b = circuit.PhasedXGate(0.42, 0.137)

    if cnvt_to_ops:
      obj_a = circuit.Operation(gate_a, [42])
      obj_b = circuit.Operation(gate_b, [42])
    else:
      obj_a = gate_a
      obj_b = gate_b

    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=False),
        False
    )
    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=True),
        False
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=False),
        False
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=True),
        False
    )

  @parameterized.parameters(itertools.product(
      [False, True],
      [
          [circuit.RotZGate(0.0), circuit.RotZGate(0.0)],
          [circuit.RotZGate(0.0), circuit.RotZGate(2.0*np.pi)],
          [circuit.RotZGate(0.4), circuit.RotZGate(-0.4)],
          [circuit.RotZGate(-0.5), circuit.RotZGate(2.0*np.pi+0.5)]
      ]
  ))
  def test_rot_z_vs_rot_z_a(self, cnvt_to_ops, gates):
    gate_a, gate_b = gates

    if cnvt_to_ops:
      obj_a = circuit.Operation(gate_a, [42])
      obj_b = circuit.Operation(gate_b, [42])
    else:
      obj_a = gate_a
      obj_b = gate_b

    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=False),
        True
    )
    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=True),
        True
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=False),
        True
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=True),
        True
    )

  @parameterized.parameters([False, True])
  def test_rot_z_vs_rot_z_b(self, cnvt_to_ops):
    gate_a = circuit.RotZGate(0.2)
    gate_b = circuit.RotZGate(-0.6)

    if cnvt_to_ops:
      obj_a = circuit.Operation(gate_a, [42])
      obj_b = circuit.Operation(gate_b, [42])
    else:
      obj_a = gate_a
      obj_b = gate_b

    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=False),
        False
    )
    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=True),
        False
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=False),
        True
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=True),
        True
    )

  @parameterized.parameters([False, True])
  def test_controlled_z_vs_controlled_z_a(self, cnvt_to_ops):
    gate_a = circuit.ControlledZGate()
    gate_b = circuit.ControlledZGate()

    if cnvt_to_ops:
      obj_a = circuit.Operation(gate_a, [47, 11])
      obj_b = circuit.Operation(gate_b, [47, 11])
    else:
      obj_a = gate_a
      obj_b = gate_b

    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=False),
        True
    )
    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=True),
        True
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=False),
        True
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=True),
        True
    )

  def test_controlled_z_vs_controlled_z_b(self):
    operation_a = circuit.Operation(circuit.ControlledZGate(), [47, 11])
    operation_b = circuit.Operation(circuit.ControlledZGate(), [47, 42])

    _check_boolean(
        self,
        operation_a.cancels_with(operation_b, phase_invariant=False),
        False
    )
    _check_boolean(
        self,
        operation_a.cancels_with(operation_b, phase_invariant=True),
        False
    )
    _check_boolean(
        self,
        operation_a.commutes_with(operation_b, phase_invariant=False),
        True
    )
    _check_boolean(
        self,
        operation_a.commutes_with(operation_b, phase_invariant=True),
        True
    )

  @parameterized.parameters(itertools.product(
      [False, True],
      [
          circuit.PhasedXGate(0.0, 0.0),
          circuit.PhasedXGate(4.0*np.pi, 0.0),
          circuit.PhasedXGate(0.0, 0.815),
          circuit.PhasedXGate(4.0*np.pi, 0.815)
      ],
      [
          circuit.RotZGate(0.0),
          circuit.RotZGate(2.0*np.pi)
      ]
  ))
  def test_phased_x_vs_rot_z_a(self, cnvt_to_ops, phased_x_gate, rot_z_gate):
    if cnvt_to_ops:
      obj_a = circuit.Operation(phased_x_gate, [42])
      obj_b = circuit.Operation(rot_z_gate, [42])
    else:
      obj_a = phased_x_gate
      obj_b = rot_z_gate

    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=False),
        True
    )
    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=True),
        True
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=False),
        True
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=True),
        True
    )

  @parameterized.parameters(itertools.product(
      [False, True],
      [
          circuit.PhasedXGate(2.0*np.pi, 0.0),
          circuit.PhasedXGate(6.0*np.pi, 0.0),
          circuit.PhasedXGate(2.0*np.pi, 0.815),
          circuit.PhasedXGate(6.0*np.pi, 0.815)
      ],
      [
          circuit.RotZGate(0.0),
          circuit.RotZGate(2.0*np.pi)
      ]
  ))
  def test_phased_x_vs_rot_z_b(self, cnvt_to_ops, phased_x_gate, rot_z_gate):
    if cnvt_to_ops:
      obj_a = circuit.Operation(phased_x_gate, [42])
      obj_b = circuit.Operation(rot_z_gate, [42])
    else:
      obj_a = phased_x_gate
      obj_b = rot_z_gate

    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=False),
        False
    )
    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=True),
        True
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=False),
        True
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=True),
        True
    )

  @parameterized.parameters(itertools.product(
      [False, True],
      [
          circuit.PhasedXGate(np.pi, 0.0),
          circuit.PhasedXGate(3.0*np.pi, 0.0),
          circuit.PhasedXGate(np.pi, 0.815),
          circuit.PhasedXGate(3.0*np.pi, 0.815)
      ],
      [
          circuit.RotZGate(np.pi),
          circuit.RotZGate(3.0*np.pi)
      ]
  ))
  def test_phased_x_vs_rot_z_c(self, cnvt_to_ops, phased_x_gate, rot_z_gate):
    if cnvt_to_ops:
      obj_a = circuit.Operation(phased_x_gate, [42])
      obj_b = circuit.Operation(rot_z_gate, [42])
    else:
      obj_a = phased_x_gate
      obj_b = rot_z_gate

    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=False),
        False
    )
    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=True),
        False
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=False),
        False
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=True),
        True
    )

  @parameterized.parameters(itertools.product(
      [False, True],
      itertools.chain(
          itertools.product(
              [
                  circuit.PhasedXGate(np.pi, 0.0),
                  circuit.PhasedXGate(3.0*np.pi, 0.0),
                  circuit.PhasedXGate(np.pi, 0.815),
                  circuit.PhasedXGate(3.0*np.pi, 0.815)
              ],
              [
                  circuit.RotZGate(0.0),
                  circuit.RotZGate(2.0*np.pi)
              ]
          ),
          itertools.product(
              [
                  circuit.PhasedXGate(0.0, 0.0),
                  circuit.PhasedXGate(2.0*np.pi, 0.0),
                  circuit.PhasedXGate(4.0*np.pi, 0.0),
                  circuit.PhasedXGate(0.0, 0.815),
                  circuit.PhasedXGate(2.0*np.pi, 0.815),
                  circuit.PhasedXGate(4.0*np.pi, 0.815)
              ],
              [
                  circuit.RotZGate(np.pi),
                  circuit.RotZGate(3.0*np.pi)
              ]
          ),
          itertools.product(
              [
                  circuit.PhasedXGate(0.0, 0.0),
                  circuit.PhasedXGate(2.0*np.pi, 0.0),
                  circuit.PhasedXGate(4.0*np.pi, 0.0),
                  circuit.PhasedXGate(0.0, 0.815),
                  circuit.PhasedXGate(2.0*np.pi, 0.815),
                  circuit.PhasedXGate(4.0*np.pi, 0.815)
              ],
              [circuit.RotZGate(47.11)]
          ),
          itertools.product(
              [
                  circuit.PhasedXGate(47.11, 0.0),
                  circuit.PhasedXGate(47.11, 0.815)
              ],
              [
                  circuit.RotZGate(0.0),
                  circuit.RotZGate(2.0*np.pi)
              ]
          )
      )
  ))
  def test_phased_x_vs_rot_z_d(self, cnvt_to_ops, gates):
    phased_x_gate, rot_z_gate = gates

    if cnvt_to_ops:
      obj_a = circuit.Operation(phased_x_gate, [42])
      obj_b = circuit.Operation(rot_z_gate, [42])
    else:
      obj_a = phased_x_gate
      obj_b = rot_z_gate

    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=False),
        False
    )
    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=True),
        False
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=False),
        True
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=True),
        True
    )

  @parameterized.parameters(itertools.product(
      [False, True],
      itertools.chain(
          itertools.product(
              [
                  circuit.PhasedXGate(0.42, 0.0),
                  circuit.PhasedXGate(0.42, 0.815)
              ],
              [
                  circuit.RotZGate(np.pi),
                  circuit.RotZGate(3.0*np.pi)
              ]
          ),
          itertools.product(
              [
                  circuit.PhasedXGate(np.pi, 0.0),
                  circuit.PhasedXGate(3.0*np.pi, 0.0),
                  circuit.PhasedXGate(0.137, 0.0),
                  circuit.PhasedXGate(np.pi, 0.815),
                  circuit.PhasedXGate(3.0*np.pi, 0.815),
                  circuit.PhasedXGate(0.137, 0.815)
              ],
              [circuit.RotZGate(0.42)]
          )
      )
  ))
  def test_phased_x_vs_rot_z_e(self, cnvt_to_ops, gates):
    phased_x_gate, rot_z_gate = gates

    if cnvt_to_ops:
      obj_a = circuit.Operation(phased_x_gate, [42])
      obj_b = circuit.Operation(rot_z_gate, [42])
    else:
      obj_a = phased_x_gate
      obj_b = rot_z_gate

    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=False),
        False
    )
    _check_boolean(
        self,
        obj_a.cancels_with(obj_b, phase_invariant=True),
        False
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=False),
        False
    )
    _check_boolean(
        self,
        obj_a.commutes_with(obj_b, phase_invariant=True),
        False
    )

  @parameterized.parameters(itertools.product(
      range(2),
      [
          circuit.PhasedXGate(0.0, 0.0),
          circuit.PhasedXGate(2.0*np.pi, 0.0),
          circuit.PhasedXGate(0.0, 0.815),
          circuit.PhasedXGate(2.0*np.pi, 0.815)
      ]
  ))
  def test_phased_x_vs_controlled_z_a(self, idx, phased_x_gate):
    qubits = [47, 11]

    operation_a = circuit.Operation(phased_x_gate, qubits[idx:idx+1])
    operation_b = circuit.Operation(circuit.ControlledZGate(), qubits)

    _check_boolean(
        self,
        operation_a.cancels_with(operation_b, phase_invariant=False),
        False
    )
    _check_boolean(
        self,
        operation_a.cancels_with(operation_b, phase_invariant=True),
        False
    )
    _check_boolean(
        self,
        operation_a.commutes_with(operation_b, phase_invariant=False),
        True
    )
    _check_boolean(
        self,
        operation_a.commutes_with(operation_b, phase_invariant=True),
        True
    )

  @parameterized.parameters(itertools.product(
      range(2),
      [
          circuit.PhasedXGate(0.4, 0.0),
          circuit.PhasedXGate(0.4, 0.815)
      ]
  ))
  def test_phased_x_vs_controlled_z_b(self, idx, phased_x_gate):
    qubits = [47, 11]

    operation_a = circuit.Operation(phased_x_gate, qubits[idx:idx+1])
    operation_b = circuit.Operation(circuit.ControlledZGate(), qubits)

    _check_boolean(
        self,
        operation_a.cancels_with(operation_b, phase_invariant=False),
        False
    )
    _check_boolean(
        self,
        operation_a.cancels_with(operation_b, phase_invariant=True),
        False
    )
    _check_boolean(
        self,
        operation_a.commutes_with(operation_b, phase_invariant=False),
        False
    )
    _check_boolean(
        self,
        operation_a.commutes_with(operation_b, phase_invariant=True),
        False
    )

  @parameterized.parameters(itertools.product(
      range(2),
      [
          circuit.RotZGate(0.0),
          circuit.RotZGate(0.7),
          circuit.RotZGate(2.0*np.pi)
      ]
  ))
  def test_rot_z_vs_controlled_z(self, idx, rot_z_gate):
    qubits = [47, 11]

    operation_a = circuit.Operation(rot_z_gate, qubits[idx:idx+1])
    operation_b = circuit.Operation(circuit.ControlledZGate(), qubits)

    _check_boolean(
        self,
        operation_a.cancels_with(operation_b, phase_invariant=False),
        False
    )
    _check_boolean(
        self,
        operation_a.cancels_with(operation_b, phase_invariant=True),
        False
    )
    _check_boolean(
        self,
        operation_a.commutes_with(operation_b, phase_invariant=False),
        True
    )
    _check_boolean(
        self,
        operation_a.commutes_with(operation_b, phase_invariant=True),
        True
    )

if __name__ == '__main__':
  absltest.main()
