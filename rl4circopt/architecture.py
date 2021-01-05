# coding=utf-8
# Copyright 2021 The Google Research Authors.
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
"""Definition of architecture-specific aspects of the circuit optimization.

Currently, only the Xmon architecture is implemented.
"""

from typing import List, Sequence

import numpy as np
import scipy.spatial.transform

from rl4circopt import circuit
from rl4circopt import parsing


class XmonArchitecture:
  """Definition of Xmon-specific aspects of the circuit optimization."""

  # TODO(tfoesel): Is the following behavior the expected one?
  #
  # Currently, a pair of PhasedX gates (both non-trivial, with different phase
  # angles) is considered non-optimal because it does not follow the format "at
  # most one PhasedX and one RotZ", but the gate sequence which it gets
  # converted to, namely a PhasedX and a RotZ gate, is not shorter than this.

  def can_optimize_single_qubit_group(self,
                                      gates):
    """Checks whether a group of single-qubit gates can be optimized.

    Args:
        gates: the sequence of gates (must all be either PhasedX or RotZ gates).

    Returns:
        returns: a bool indicating whether the sequence is not yet optimal.

    Raises:
        TypeError: if gates is not an iterable over Gate instances.
    """
    gates = tuple(gates)

    if not all(isinstance(gate, circuit.Gate) for gate in gates):
      raise TypeError(
          'illegal types found in gates: %s (should be subtypes of Gate)'
          %', '.join(sorted(set(
              type(gate).__name__
              for gate in gates
              if not isinstance(gate, circuit.Gate)
          )))
      )

    if any(gate.is_identity(phase_invariant=True) for gate in gates):
      return True

    if len(gates) <= 1:
      return False  # we already know that it's not identity (if length is 1)
    elif len(gates) == 2:
      parsed = parsing.parse_gates(gates, circuit.PhasedXGate, circuit.RotZGate)
      if parsed is not None:
        return bool(np.isclose(  # check whether the PhasedX is a flip
            np.cos(parsed[0].get_rotation_angle()),
            -1.0
        ))

      parsed = parsing.parse_gates(gates, circuit.RotZGate, circuit.PhasedXGate)
      if parsed is not None:
        return bool(np.isclose(  # check whether the PhasedX is a flip
            np.cos(parsed[1].get_rotation_angle()),
            -1.0
        ))

      return True  # either two PhasedX gates or two RotZ gates
    else:
      return True

  def decompose_single_qubit_gate(self,
                                  pauli_transform
                                 ):
    """Decomposes an arbitrary single-qubit gate into native Xmon gates.

    Args:
        pauli_transform: np.ndarray with shape (3, 3) and dtype float.
            The pauli_transform representing the (single-qubit) gate to be
            decomposed.

    Returns:
        a sequence of native Xmon gates that are equivalent to the input,
        consisting of a PhasedX and/or a RotZ gate

    Raises:
        TypeError: if pauli_transform is not a np.array with real dtype, and
            cannot be casted to one.
        ValueError: if pauli_transform does not have the correct shape or if
            pauli_transform is not an orthogonal matrix
    """

    if not hasattr(pauli_transform, '__getitem__'):
      raise TypeError('%s cannot be converted to np.array'
                      %type(pauli_transform).__name__)

    pauli_transform = np.array(pauli_transform)

    try:
      pauli_transform = pauli_transform.astype(float, casting='safe')
    except TypeError:
      raise TypeError('illegal dtype for pauli_transform: %s (must be safely'
                      ' castable to float)'%pauli_transform.dtype)

    if not np.array_equal(pauli_transform.shape, [3, 3]):
      raise ValueError(
          'illegal shape for pauli_transform: %s [expected: (3, 3)]'
          %str(pauli_transform.shape)
      )
    if not np.allclose(np.dot(pauli_transform, pauli_transform.T), np.eye(3)):
      raise ValueError('pauli_transform is not an orthogonal matrix')

    if np.allclose(pauli_transform, np.eye(3)):
      return []

    # We want to decompose a given single-qubit gate U into PhasedX and RotZ
    # gates. For this, we make the following ansatz:
    #
    #     ───U───   ≡   ───PhasedX(beta, -alpha)───RotZ(phi)───
    #
    # To obtain the parameters of the PhasedX and RotZ gate, we perform an Euler
    # angle decomposition for the pauli_transform of U.
    #
    # This can be motivated as follows. We have
    #
    #     PhasedX(beta, -alpha).pauli_transform ==
    #         == dot(Rz(-alpha), Rx(beta), Rz(alpha))
    #     RotZ(phi).pauli_transform == Rz(phi)
    #
    # where Rx and Rz are 3D rotations around the x and z axis, respectively.
    # From this, we get:
    #
    #     dot(RotZ(phi), PhasedX(alpha, beta)).pauli_transform ==
    #         == dot(Rz(phi-alpha), Rx(beta), Rz(alpha))
    #
    # The right-hand side is exactly the Euler angle decomposition with
    # signature 'zxz' and angles [alpha, beta, phi-alpha], which gives us a
    # direct way to extract the parameters for the two gates. Below, the angle
    # gamma will substitute phi-alpha.
    #
    # Note 1: The fact that this Euler angle decomposition is possible for every
    # SO(3) rotation, i.e. every possible Pauli transform, is one way to prove
    # universality for the combination of the PhasedX and RotZ gate.
    #
    # Note 2: In the calculation above, we have reversed two times the order of
    # the operations. This is because in circuit diagrams and for the Euler
    # angle decomposition (following the convention used in the corresponding
    # Scipy function), the first operation is left, whereas when multiplying the
    # associated operators, the first operation is right such that this is the
    # one applied first to the ket state (a vector multiplied from the right).

    # The two if-branches are special cases where the Euler decomposition into
    # 'zxz' is not anymore unique; here, we fix a choice that corresponds to
    # either just a PhasedX gate or just a RotZ gate, and thus minimizes the
    # total number of gates.
    #
    # In addition, the algorithm for Euler decomposition used in the else
    # branch seems to run into numerical precision issues for these cases; as
    # a positive side effect, these issues are circumvented.
    if np.isclose(pauli_transform[2, 2], 1.0):
      gamma = 0.5 * np.arctan2(pauli_transform[1, 0], pauli_transform[0, 0])
      beta = 0.0
      alpha = gamma
    elif np.isclose(pauli_transform[2, 2], -1.0):
      gamma = 0.5 * np.arctan2(pauli_transform[1, 0], pauli_transform[0, 0])
      beta = np.pi
      alpha = -gamma
    else:
      rot = scipy.spatial.transform.Rotation.from_dcm(pauli_transform)
      alpha, beta, gamma = rot.as_euler('zxz')

    gates = []
    if not np.isclose(np.cos(beta), 1.0):
      gates.append(circuit.PhasedXGate(beta, -alpha))
    if not np.isclose(np.cos(alpha + gamma), 1.0):
      gates.append(circuit.RotZGate(alpha + gamma))
    return gates
