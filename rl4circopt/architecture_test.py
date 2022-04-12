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

"""Tests for architecture."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
from scipy.spatial import transform

from rl4circopt import architecture
from rl4circopt import circuit


def _check_boolean(test_case, found, expected):
  """Checks that found is a bool and matches the expected result."""
  test_case.assertIs(type(found), bool)
  test_case.assertEqual(found, expected)


class XmonArchitectureTest(parameterized.TestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.xmon_arch = architecture.XmonArchitecture()

  @parameterized.parameters([
      [[]],
      [[circuit.RotZGate(0.42)]],
      [[circuit.PhasedXGate(0.4711, 0.137)]],
      [[circuit.RotZGate(0.42), circuit.PhasedXGate(0.4711, 0.137)]],
      [[circuit.PhasedXGate(0.4711, 0.137), circuit.RotZGate(0.42)]]
  ])
  def test_can_optimize_single_qubit_group_negative(self, gates):
    # check type and value for xmon_arch.can_optimize_single_qubit_group(...)
    _check_boolean(
        self,
        self.xmon_arch.can_optimize_single_qubit_group(gates),
        False
    )

    # construct the pauli_transform for the full gate sequence (note that the
    # order for the factors needs to be reversed here)
    pauli_transform = np.eye(3)
    for gate in gates:
      pauli_transform = np.dot(gate.get_pauli_transform(), pauli_transform)

    # check that xmon_arch.decompose_single_qubit_gate(...) gives a gate
    # sequence of equal length (the result of that function does not need to
    # match the input sequence because the order of PhasedX and RotZ is
    # arbitrary, so this check would be too strict)
    self.assertEqual(
        len(self.xmon_arch.decompose_single_qubit_gate(pauli_transform)),
        len(gates)
    )

  def test_can_optimize_single_qubit_group_debatable(self, *gates):
    # compare the TODO at the beginning of class XmonArchitecture

    gates = (
        circuit.PhasedXGate(0.4711, 0.42),
        circuit.PhasedXGate(0.815, 0.137)
    )

    # check type and value for xmon_arch.can_optimize_single_qubit_group(...)
    _check_boolean(
        self,
        self.xmon_arch.can_optimize_single_qubit_group(gates),
        True
    )

    # construct the pauli_transform for the full gate sequence (note that the
    # order for the factors needs to be reversed here)
    pauli_transform = np.eye(3)
    for gate in gates:
      pauli_transform = np.dot(gate.get_pauli_transform(), pauli_transform)

    # check that xmon_arch.decompose_single_qubit_gate(...) gives a gate
    # sequence of equal length (the result of that function will not match the
    # input sequence because it decomposes it into a PhasedX and a RotZ gate)
    self.assertLen(
        self.xmon_arch.decompose_single_qubit_gate(pauli_transform),
        2
    )

  @parameterized.parameters(itertools.chain(
      [[circuit.RotZGate(0.0)]],

      [[circuit.PhasedXGate(0.0, 0.0)]],
      [[circuit.PhasedXGate(0.0, 0.42)]],
      [[circuit.PhasedXGate(0.0, 0.5*np.pi)]],

      itertools.product(
          [circuit.RotZGate(0.0)],
          [
              circuit.PhasedXGate(0.0, 0.0),
              circuit.PhasedXGate(0.0, 0.42),
              circuit.PhasedXGate(0.0, 0.5*np.pi)
          ]
      ),
      itertools.product(
          [
              circuit.PhasedXGate(0.0, 0.0),
              circuit.PhasedXGate(0.0, 0.42),
              circuit.PhasedXGate(0.0, 0.5*np.pi)
          ],
          [circuit.RotZGate(0.0)]
      ),

      itertools.product(
          [circuit.RotZGate(0.137)],
          [
              circuit.PhasedXGate(0.0, 0.0),
              circuit.PhasedXGate(0.0, 0.42),
              circuit.PhasedXGate(0.0, 0.5*np.pi)
          ]
      ),
      itertools.product(
          [
              circuit.PhasedXGate(0.0, 0.0),
              circuit.PhasedXGate(0.0, 0.42),
              circuit.PhasedXGate(0.0, 0.5*np.pi)
          ],
          [circuit.RotZGate(0.137)]
      ),

      itertools.product(
          [circuit.RotZGate(0.0)],
          [
              circuit.PhasedXGate(0.4711, 0.0),
              circuit.PhasedXGate(0.4711, 0.42),
              circuit.PhasedXGate(0.4711, 0.5*np.pi)
          ]
      ),
      itertools.product(
          [
              circuit.PhasedXGate(0.4711, 0.0),
              circuit.PhasedXGate(0.4711, 0.42),
              circuit.PhasedXGate(0.4711, 0.5*np.pi)
          ],
          [circuit.RotZGate(0.0)]
      ),

      [[circuit.RotZGate(0.137), circuit.RotZGate(0.137)]],
      [[circuit.RotZGate(0.137), circuit.RotZGate(0.42)]],

      [
          [  # pylint: disable=g-complex-comprehension
              circuit.PhasedXGate(0.4711, phase_angle),
              circuit.PhasedXGate(0.42, phase_angle)
          ]
          for phase_angle in (0.0, 0.42, 0.5*np.pi)
      ],

      [[circuit.RotZGate(0.137), circuit.PhasedXGate(np.pi, 0.42)]],
      [[circuit.PhasedXGate(np.pi, 0.42), circuit.RotZGate(0.137)]],

      [[
          circuit.RotZGate(0.815),
          circuit.PhasedXGate(np.pi, 0.42),
          circuit.RotZGate(0.137)
      ]]
  ))
  def test_can_optimize_single_qubit_group_positive(self, *gates):
    gates = tuple(gates)

    # check type and value for xmon_arch.can_optimize_single_qubit_group(...)
    _check_boolean(
        self,
        self.xmon_arch.can_optimize_single_qubit_group(gates),
        True
    )

    # construct the pauli_transform for the full gate sequence (note that the
    # order for the factors needs to be reversed here)
    pauli_transform = np.eye(3)
    for gate in gates:
      pauli_transform = np.dot(gate.get_pauli_transform(), pauli_transform)

    # check that xmon_arch.decompose_single_qubit_gate(...) can construct a
    # sequence with *less* gates
    self.assertLess(
        len(self.xmon_arch.decompose_single_qubit_gate(pauli_transform)),
        len(gates)
    )

  def test_can_optimize_single_qubit_group_error_not_iterable(self):
    with self.assertRaisesRegex(
        TypeError,
        r'\'int\' object is not iterable'):
      self.xmon_arch.can_optimize_single_qubit_group(42)

  def test_can_optimize_single_qubit_group_error_not_gate(self):
    with self.assertRaisesRegex(
        TypeError,
        r'illegal types found in gates: range \(should be subtypes of Gate\)'):
      self.xmon_arch.can_optimize_single_qubit_group([range(42)])

  @parameterized.parameters([
      (
          np.eye(3),
          []
      ),
      (
          transform.Rotation.from_euler('z', 0.42).as_dcm(),
          [circuit.RotZGate]
      ),
      (
          transform.Rotation.from_euler('zxz', [0.42, 0.4711, -0.42]).as_dcm(),
          [circuit.PhasedXGate]
      ),
      (
          transform.Rotation.from_euler('zxz', [0.42, np.pi, -0.42]).as_dcm(),
          [circuit.PhasedXGate]
      ),
      (
          transform.Rotation.from_euler('zxz', [0.42, 0.4711, 0.137]).as_dcm(),
          [circuit.PhasedXGate, circuit.RotZGate]
      )
  ])
  def test_decompose_single_qubit_gate(self, pauli_transform, gate_types):
    # call the method to be tested
    gates = self.xmon_arch.decompose_single_qubit_gate(pauli_transform.copy())

    # check that gates is a list of Gate instances
    self.assertIsInstance(gates, list)
    self.assertTrue(all(isinstance(gate, circuit.Gate) for gate in gates))

    # check that the gates reconstruct the correct target operation
    pauli_reconstr = np.eye(3)
    for gate in gates:
      pauli_reconstr = np.dot(gate.get_pauli_transform(), pauli_reconstr)
    np.testing.assert_allclose(
        pauli_transform,
        pauli_reconstr,
        rtol=1e-5, atol=1e-8
    )

    # check that gates has the expected length
    self.assertEqual(len(gates), len(gate_types))

    # check that the gates match the expected types
    self.assertTrue(all(
        isinstance(gate, gate_type)
        for gate, gate_type in zip(gates, gate_types)
    ))

  def test_decompose_single_qubit_gate_error_no_array(self):
    with self.assertRaisesRegex(
        TypeError,
        r'slice cannot be converted to np.array'):
      self.xmon_arch.decompose_single_qubit_gate(slice(42))

  def test_decompose_single_qubit_gate_error_illegal_dtype(self):
    with self.assertRaisesRegex(
        TypeError,
        r'illegal dtype for pauli_transform: complex128 \(must be safely'
        r' castable to float\)'):
      self.xmon_arch.decompose_single_qubit_gate(np.eye(3, dtype=complex))

  def test_decompose_single_qubit_gate_error_illegal_shape(self):
    with self.assertRaisesRegex(
        ValueError,
        r'illegal shape for pauli_transform: \(3, 5\) \[expected: \(3, 3\)\]'):
      self.xmon_arch.decompose_single_qubit_gate(np.random.randn(3, 5))

  def test_decompose_single_qubit_gate_error_not_orthogonal(self):
    with self.assertRaisesRegex(
        ValueError,
        r'pauli_transform is not an orthogonal matrix'):
      self.xmon_arch.decompose_single_qubit_gate(np.ones([3, 3]))

if __name__ == '__main__':
  absltest.main()
