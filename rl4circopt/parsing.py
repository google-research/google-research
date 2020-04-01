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
"""Tool for parsing Gate types.

This module currently contains two functions, parse_gates(...) and
parse_operations(...), which implement gate parsing for sequences of Gate and
Operation instances, respectively.
"""

import numpy as np
import scipy.spatial

from rl4circopt import circuit


# TODO(tfoesel): improve design for this module and add unit tests


def check_gates(gates, *gate_types):
  """Checks whether the gates match the expected types."""
  return parse_gates(gates, *gate_types) is not None


def check_operations(operations, *gate_types):
  """Checks whether the gates of the operations match the expected types."""
  return parse_operations(operations, *gate_types) is not None


def parse_gates(gates, *gate_types):
  """Parses gates into expected gate types."""

  if len(gates) != len(gate_types):
    raise ValueError('inconsistent length of gates and gate_types (%d vs %d)'
                     %(len(gates), len(gate_types)))

  parsed_gates = []

  for gate_in, gate_type in zip(gates, gate_types):
    if not isinstance(gate_in, circuit.Gate):
      raise TypeError('%s is not a Gate'%type(gate_in).__name__)
    if not isinstance(gate_type, type):
      raise TypeError('%s instance is not a type'%type(gate_type).__name__)
    if not issubclass(gate_type, circuit.Gate):
      raise TypeError('%s is not a Gate type'%gate_type.__name__)

    if gate_type == circuit.PhasedXGate:
      if gate_in.get_num_qubits() != 1:
        return None
      elif isinstance(gate_in, circuit.PhasedXGate):
        gate_out = gate_in
      elif isinstance(gate_in, circuit.RotZGate):
        if gate_in.is_identity(phase_invariant=True):
          gate_out = circuit.PhasedXGate(0.0, 0.0)
        else:
          return None
      else:
        pauli_transform = gate_in.get_pauli_transform()
        if np.isclose(pauli_transform[2, 2], -1.0):
          gate_out = circuit.PhasedXGate(
              np.pi,
              0.5 * np.arctan2(pauli_transform[0, 1], pauli_transform[0, 0])
          )
        else:
          rotation = scipy.spatial.transform.Rotation.from_dcm(pauli_transform)
          alpha, beta, gamma = rotation.as_euler('zxz')
          if np.isclose(alpha, -gamma):
            gate_out = circuit.PhasedXGate(beta, -alpha)
          else:
            return None
    elif gate_type == circuit.RotZGate:
      if gate_in.get_num_qubits() != 1:
        return None
      elif isinstance(gate_in, circuit.RotZGate):
        gate_out = gate_in
      elif isinstance(gate_in, circuit.PhasedXGate):
        if gate_in.is_identity(phase_invariant=True):
          gate_out = circuit.RotZGate(0.0)
        else:
          return None
      else:
        pauli_transform = gate_in.get_pauli_transform()
        if np.isclose(pauli_transform[2, 2], 1.0):
          gate_out = circuit.RotZGate(np.arctan2(
              pauli_transform[1, 0],
              pauli_transform[0, 0]
          ))
        else:
          return None
    elif gate_type == circuit.ControlledZGate:
      if gate_in.get_num_qubits() != 2:
        return None
      elif isinstance(gate_in, circuit.ControlledZGate):
        gate_out = gate_in
      elif gate_in == circuit.ControlledZGate():
        gate_out = circuit.ControlledZGate()
      else:
        return None
    else:
      raise ValueError('unknown Gate type: %s'%gate_type.__name__)

    assert isinstance(gate_out, gate_type)

    parsed_gates.append(gate_out)

  assert len(parsed_gates) == len(gates)

  return parsed_gates


def parse_operations(operations, *gate_types):
  """Parse operations into expected gate types."""

  if len(operations) != len(gate_types):
    raise ValueError('inconsistent length of operations and gate_types'
                     ' (%d vs %d)'%(len(operations), len(gate_types)))

  for operation in operations:
    if not isinstance(operation, circuit.Operation):
      raise TypeError('%s is not an Operation'%type(operation).__name__)

  parsed_gates = parse_gates(
      [operation.get_gate() for operation in operations],
      *gate_types
  )

  if parsed_gates is None:
    return None
  else:
    parsed_operations = []

    for operation, parsed_gate in zip(operations, parsed_gates):
      if operation.get_gate() is not parsed_gate:
        operation = circuit.Operation(parsed_gate, operation.get_qubits())
      parsed_operations.append(operation)

    return parsed_operations
