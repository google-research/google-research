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
"""Conversion functions between Cirq and this package.

Converts between the following types:
- gates:
  * circuit.PhasedXGate <-> cirq.PhasedXPowGate
  * circuit.RotZGate <-> cirq.ZPowGate
  * circuit.ControlledZGate <-> cirq.CZPowGate
- operations: circuit.Operation <-> cirq.GateOperation
- circuits: circuit.Circuit <-> cirq.Circuit
"""

import itertools

import cirq
import numpy as np

from rl4circopt import circuit


def import_from_cirq(obj):
  """Imports a gate, operation or circuit from Cirq.

  Args:
      obj: the Cirq object to be imported.

  Returns:
      the imported object (an instance of circuit.Circuit, circuit.Operation, or
      a subclass of circuit.Gate).

  Raises:
      TypeError: if import is not supported for the given type.
      ValueError: if the object cannot be imported successfully.
  """

  if isinstance(obj, cirq.PhasedXPowGate):
    return circuit.PhasedXGate(obj.exponent * np.pi, obj.phase_exponent * np.pi)
  elif isinstance(obj, cirq.ZPowGate):
    return circuit.RotZGate(obj.exponent * np.pi)
  elif isinstance(obj, cirq.CZPowGate):
    if not np.isclose(np.mod(obj.exponent, 2.0), 1.0):
      raise ValueError('partial ControlledZ gates are not supported')
    return circuit.ControlledZGate()
  elif isinstance(obj, cirq.MatrixGate):
    return circuit.MatrixGate(cirq.unitary(obj))
  elif isinstance(obj, cirq.GateOperation):
    return circuit.Operation(
        import_from_cirq(obj.gate),
        [qubit.x for qubit in obj.qubits]
    )
  elif isinstance(obj, cirq.Circuit):
    qubits = obj.all_qubits()
    if not all(isinstance(qubit, cirq.LineQubit) for qubit in qubits):
      qubit_types = set(type(qubit) for qubit in qubits)
      qubit_types = sorted(qubit_type.__name__ for qubit_type in qubit_types)
      raise ValueError('import is supported for circuits on LineQubits only'
                       ' [found qubit type(s): %s]'%', '.join(qubit_types))
    return circuit.Circuit(
        max(qubit.x for qubit in qubits) + 1,
        [
            import_from_cirq(operation)
            for operation in itertools.chain.from_iterable(obj)
        ]
    )
  else:
    raise TypeError('unknown type: %s'%type(obj).__name__)


def export_to_cirq(obj):
  """Imports a gate, operation or circuit from Cirq.

  Args:
      obj: the object to be exported to Cirq.

  Returns:
      the exported Cirq object (an instance of cirq.Circuit, cirq.GateOperation,
      or a subclass of cirq.Gate).

  Raises:
      TypeError: if export is not supported for the given type.
      ValueError: if the object cannot be exported successfully.
  """

  if isinstance(obj, circuit.PhasedXGate):
    return cirq.PhasedXPowGate(
        exponent=obj.get_rotation_angle() / np.pi,
        phase_exponent=obj.get_phase_angle() / np.pi
    )
  elif isinstance(obj, circuit.RotZGate):
    return cirq.ZPowGate(exponent=obj.get_rotation_angle() / np.pi)
  elif isinstance(obj, circuit.ControlledZGate):
    return cirq.CZPowGate(exponent=1.0)
  elif isinstance(obj, circuit.MatrixGate):
    return cirq.MatrixGate(obj.get_operator())
  elif isinstance(obj, circuit.Operation):
    return cirq.GateOperation(
        export_to_cirq(obj.get_gate()),
        [cirq.LineQubit(qubit) for qubit in obj.get_qubits()]
    )
  elif isinstance(obj, circuit.Circuit):
    return cirq.Circuit(export_to_cirq(operation) for operation in obj)
  else:
    raise TypeError('unknown type: %s'%type(obj).__name__)
