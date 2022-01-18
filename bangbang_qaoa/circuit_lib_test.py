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

"""Tests for circuit_lib."""

import math

from absl.testing import absltest
from absl.testing import parameterized
import cirq

from bangbang_qaoa import circuit_lib


class CircuitLibTest(parameterized.TestCase):

  def test_generate_x_hamiltonian_exponential(self):
    self.assertCountEqual(
        circuit_lib.generate_x_hamiltonian_exponential(4, 0.5),
        [
            cirq.XPowGate(exponent=-1.0 / math.pi, global_shift=-0.5).on(
                cirq.LineQubit(0)),
            cirq.XPowGate(exponent=-1.0 / math.pi, global_shift=-0.5).on(
                cirq.LineQubit(1)),
            cirq.XPowGate(exponent=-1.0 / math.pi, global_shift=-0.5).on(
                cirq.LineQubit(2)),
            cirq.XPowGate(exponent=-1.0 / math.pi, global_shift=-0.5).on(
                cirq.LineQubit(3)),
        ])

  @parameterized.parameters(
      (
          circuit_lib.HamiltonianType.CONSTRAINT,
          circuit_lib.HamiltonianType.X
      ),
      (
          circuit_lib.HamiltonianType.X,
          circuit_lib.HamiltonianType.CONSTRAINT
      ),
  )
  def test_switch_hamiltonian_type(self, input_type, expected_output):
    self.assertEqual(circuit_lib.switch_hamiltonian_type(input_type),
                     expected_output)

  def test_bangbang_compressor(self):
    compressed_protocol = iter(
        circuit_lib.bangbang_compressor([
            circuit_lib.HamiltonianType.CONSTRAINT,
            circuit_lib.HamiltonianType.CONSTRAINT,
            circuit_lib.HamiltonianType.CONSTRAINT,
            circuit_lib.HamiltonianType.X,
            circuit_lib.HamiltonianType.CONSTRAINT,
            circuit_lib.HamiltonianType.X,
            circuit_lib.HamiltonianType.X,
        ]))

    time, hamiltonian_type = next(compressed_protocol)
    self.assertEqual(time, 3)
    self.assertEqual(hamiltonian_type, circuit_lib.HamiltonianType.CONSTRAINT)

    time, hamiltonian_type = next(compressed_protocol)
    self.assertEqual(time, 1)
    self.assertEqual(hamiltonian_type, circuit_lib.HamiltonianType.X)

    time, hamiltonian_type = next(compressed_protocol)
    self.assertEqual(time, 1)
    self.assertEqual(hamiltonian_type, circuit_lib.HamiltonianType.CONSTRAINT)

    time, hamiltonian_type = next(compressed_protocol)
    self.assertEqual(time, 2)
    self.assertEqual(hamiltonian_type, circuit_lib.HamiltonianType.X)

    with self.assertRaises(StopIteration):
      next(compressed_protocol)

  @parameterized.parameters(
      ([], '[]'),
      ([circuit_lib.HamiltonianType.CONSTRAINT], '[1]'),
      ([circuit_lib.HamiltonianType.X, circuit_lib.HamiltonianType.X], '[0, 0]')
  )
  def test_protocol_to_string(self, protocol, expected_string):
    self.assertEqual(circuit_lib.protocol_to_string(protocol), expected_string)

  @parameterized.parameters(
      ('[]', []),
      ('[1]', [1]),
      ('[0, 0]', [0, 0]),
  )
  def test_string_to_protocol(self, protocol_string, expected_protocol):
    self.assertListEqual(circuit_lib.string_to_protocol(protocol_string),
                         expected_protocol)

if __name__ == '__main__':
  absltest.main()
