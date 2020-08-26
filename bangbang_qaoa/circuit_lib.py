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

"""Abstract functions and classes used to build the QAOA circuit in cirq.

Quantum Approximate Optimization Algorithm (QAOA) provides a recipe on how to
parametrized a trial wavefunction of the ground state which can be solved
variationally.

Original paper:
A Quantum Approximate Optimization Algorithm
https://arxiv.org/abs/1411.4028


Different from the standard QAOA algorithm, we will allocate chunks of time to
apply either Hamiltonian, and optimize over these decisions.
"""

import abc
import ast
import enum
import itertools
from typing import Any, List, Tuple

import cirq
import numpy as np


def generate_x_hamiltonian_exponential(num_qubits, time
                                      ):
  r"""Returns the list of all operators for the exponential of the X Hamiltonian.

  Applies exp(i*time*\sum_i X_i) as a Cirq circuit.

  Args:
    num_qubits: Integer, Apply XPowGate to each qubit in range(num_qubits).
    time: Float, how long to apply the Hamiltonian.

  Returns:
    List of all cirq operators for the X Hamiltonian defined by the time.
  """
  # Corrects for the fact that Rx performs e^(-i*X*t/2)
  return list(cirq.rx(-2 * time).on_each(
      *[cirq.LineQubit(i) for i in range(num_qubits)]
  ))


class HamiltonianType(enum.IntEnum):
  X = 0
  CONSTRAINT = 1


def switch_hamiltonian_type(hamiltonian_type
                           ):
  """Converts to the other HamiltonianType value.

  Args:
    hamiltonian_type: HamiltonianType, the type to get flipped

  Returns:
    The opposite HamiltonianType
  """
  if hamiltonian_type is HamiltonianType.CONSTRAINT:
    return HamiltonianType.X
  else:
    return HamiltonianType.CONSTRAINT


def bangbang_compressor(bangbang_protocol
                       ):
  """Compresses the bang bang protocol.

  Merges chunks of contiguous bangbang chunks into a Tuple of duration (in
  number of chunks) and which Hamiltonian to apply.

  Args:
    bangbang_protocol: List of HamiltonianType values, determines which
        Hamiltonian should be applied at the corresponding chunk.

  Returns:
    List of Tuples containing the Hamiltonian type and the number of chunks to
    apply the Hamiltonian type for.
  """
  current_mode = None
  compressed_protocol = []
  chunk_counter = 0
  for protocol_mode in bangbang_protocol:
    if current_mode is None:
      current_mode = protocol_mode
      chunk_counter = 1
    elif current_mode == protocol_mode:
      chunk_counter += 1
    else:
      compressed_protocol.append((chunk_counter, current_mode))
      current_mode = protocol_mode
      chunk_counter = 1
  # Append what's left over
  if chunk_counter > 0:
    compressed_protocol.append((chunk_counter, current_mode))
  return compressed_protocol


def protocol_to_string(bangbang_protocol):
  """Converts a bang-bang protocol into a string.

  Args:
    bangbang_protocol: List of circuit_lib.HamiltonianType

  Returns:
    String that represents the bangbang_protocol
  """
  return str([int(hamiltonian_type) for hamiltonian_type in bangbang_protocol])


def string_to_protocol(protocol_string):
  """Converts a string into a bang-bang protocol.

  Args:
    protocol_string: String, represents the bangbang_protocol

  Returns:
    List of HamiltonianType, representing the bang-bang protocol.
  """
  return ast.literal_eval(protocol_string)


class BangBangProtocolCircuit(abc.ABC):
  """QAOA circuit for bang-bang protocols.

  This circuit will divide the given time into a series of chunks that will
  be allocated to one of the two Hamiltonians.

  Attributes:
    chunk_time: Positive float, amount of time allocated to each chunk.
    num_qubits: Positive int, the number of qubits in the simulation
    hamiltonian_diagonal: List of floats, the diagonal of the target
        Hamiltonian, that determines the evaluations of each measurement.
        Note: Should be defined by child class using get_hamiltonian_diagonal().
    simulator: cirq.Simulator, the simulator to run the circuit.
  """

  def __init__(self, chunk_time, num_qubits):
    """Initializer.

    This class creates QAOA circuit in cirq.

    Args:
      chunk_time: Positive float, amount of time allocated to each chunk.
      num_qubits: Positive int, number of qubits in the simulation.

    Raises:
      ValueError: If num_qubits or chunk_time are not positive.
    """
    if chunk_time < 0.0:
      raise ValueError('chunk_time must be positive, not %f.' % chunk_time)
    self.chunk_time = chunk_time
    if num_qubits < 1:
      raise ValueError('num_qubits must be positive, not %d' % num_qubits)
    self.num_qubits = num_qubits
    self.simulator = cirq.Simulator()
    # This attribute must be defined by calling get_hamiltonian_diagonal().
    self.hamiltonian_diagonal = None

  def get_hamiltonian_diagonal(self):
    """Computes the diagonal elements of the target Hamiltonian.

    Returns:
      List of 2 ** self.num_qubits floats containing the diagonal elements of
      the target Hamiltonian.
    """
    return [
        self.constraint_evaluation(measurement)
        for measurement in itertools.product([0, 1], repeat=self.num_qubits)]

  @abc.abstractmethod
  def generate_constraint_hamiltonian_exponential(self, time
                                                 ):
    """Generates the circuit to apply the constraint Hamiltonian.

    Applies exp(i*time*Hamiltonian) as a Cirq circuit.

    Args:
      time: Float, the amount of time to apply the Hamiltonian.

    Returns:
      List of cirq operations.
    """
    pass

  def qaoa_circuit(self, bangbang_protocol
                  ):
    """Generates the QAOA circuit for the problem given the bang bang protocol.

    Starts with a layer of Hamiltonians to put into uniform superposition. Then
    collapses each chunk into continugous times and then generates the
    corresponding exponentiated Hamiltonians.

    Args:
      bangbang_protocol: List of HamiltonianType values, determines which
          Hamiltonian should be applied at the corresponding chunk.

    Returns:
      The circuit for the bang bang protocol in QAOA format.
    """
    compressed_protocol = []
    for duration, hamiltonian_type in bangbang_compressor(bangbang_protocol):
      compressed_protocol.append((self.chunk_time * duration, hamiltonian_type))

    circuit = cirq.Circuit()
    circuit.append(cirq.H.on_each(
        *[cirq.LineQubit(i) for i in range(self.num_qubits)]
    ))
    for time, hamiltonian_type in compressed_protocol:
      if time == 0.0:
        continue
      if hamiltonian_type == HamiltonianType.CONSTRAINT:
        circuit.append(self.generate_constraint_hamiltonian_exponential(time))
      else:
        circuit.append(generate_x_hamiltonian_exponential(self.num_qubits,
                                                          time))
    return circuit

  def get_wavefunction(self, bangbang_protocol):
    """Gets wavefunction from the circuit.

    Adapted from:
      biology.research.collaborations.xc.qaoa.

    Note in real quantum hardware, wavefunction cannot be obtained directly. It
    can only be sampled from measurements.

    Args:
      bangbang_protocol: List of HamiltonianType values, determines which
          Hamiltonian should be applied at the corresponding chunk.

    Returns:
      Complex numpy array with shape [2 ** self.num_qubits].
    """
    circuit = self.qaoa_circuit(bangbang_protocol)
    return self.simulator.simulate(circuit).final_state

  def get_probabilities(self, wavefunction):
    """Gets the probabilities of states from wavefunction.

    Adapted from:
      biology.research.collaborations.xc.qaoa.

    Args:
      wavefunction: Complex numpy array with shape [n_dim].

    Returns:
      Float numpy array with shape [2 ** self.num_qubits].

    Raises:
      ValueError: If the shape of wavefunction is not
      [2 ** self.num_qubits] or the wavefunction is not normalized.
    """
    if wavefunction.shape != (2 ** self.num_qubits,):
      raise ValueError(
          'The shape of wavefunction should be (%d,) but got %s'
          % (2 ** self.num_qubits, wavefunction.shape))
    probabilities = np.abs(wavefunction) ** 2
    norm = np.sum(probabilities)
    if not np.isclose(norm, 1., atol=0.001):
      raise ValueError(
          'Wavefunction should be normalized to 1 but got %4.6f' % norm)
    return probabilities

  @abc.abstractmethod
  def constraint_evaluation(self, measurement):
    """Gets the evaluation of that measurement on the given problem.

    Args:
      measurement: Numpy array of integers, corresponds to the measured value.

    Returns:
      Float.
    """
    pass

  def get_constraint_expectation(self, wavefunction):
    """Gets the energy of the wavefunction with the Hamiltonian.

    Computes the expectation value of the wavefunction with the target
    Hamiltonian. Assumes the Hamiltonian is diagonal in the computational basis.

    Args:
      wavefunction: Complex numpy array with shape [n_dim].

    Returns:
      Float.
    """
    probabilities = self.get_probabilities(wavefunction)
    return np.dot(probabilities, self.hamiltonian_diagonal)
