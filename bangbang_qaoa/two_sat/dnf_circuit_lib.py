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

"""Functions and Classes used to build the QAOA circuit in cirq for 2-SAT.

Quantum Approximate Optimization Algorithm (QAOA) provides a recipe on how to
parametrized a trial wavefunction of the ground state which can be solved
variationally.

Original paper:
A Quantum Approximate Optimization Algorithm
https://arxiv.org/abs/1411.4028


Different from the standard QAOA algorithm, we will allocate chunks of time to
apply either Hamiltonian, and optimize over these decisions.
"""

import itertools
import math
from typing import List

import cirq

from bangbang_qaoa import circuit_lib
from bangbang_qaoa.two_sat import dnf_lib


def _get_sign(is_negative, other_is_negative = False):
  """Returns the sign to multiply by time.

  Args:
    is_negative: Boolean. If True, multiply by -1.
    other_is_negative: Additional argument. If True, multiply by -1.
                       Default value False

  Returns:
    -1 or 1 integer. -1 if only one of is_negative values is True
  """
  return (1 - 2 * is_negative) * (1 - 2 * other_is_negative)


def generate_clause_hamiltonian_exponential(clause, time
                                           ):
  """Generates the operators for this clause.

  Applies e^(iZt) on each qubit, and e^(i ZZ t) on both. The sign of t gets
  flipped if the variable is negated.

  Args:
    clause: dnf_lib.Clause, the clause that defines the local Hamiltonian
    time: Float, how long to apply the Hamiltonian

  Returns:
    A list of operations corresponding to each of the gates being applied by
    this clause.
  """
  # Corrects for the fact that ZPowGate and ZZPowGate perform e^(-i*\pi*Z*t/2)
  # up to global phase.
  time = -2 * time / math.pi
  return [
      cirq.ZPowGate(
          exponent=_get_sign(clause.is_negative1) * time,
          global_shift=-0.5).on(cirq.LineQubit(clause.index1)),
      cirq.ZPowGate(
          exponent=_get_sign(clause.is_negative2) * time,
          global_shift=-0.5).on(cirq.LineQubit(clause.index2)),
      cirq.ZZPowGate(
          exponent=_get_sign(clause.is_negative1, clause.is_negative2) * time,
          global_shift=-0.5).on(cirq.LineQubit(clause.index1),
                                cirq.LineQubit(clause.index2))]


def generate_dnf_hamiltonian_exponential(dnf, time
                                        ):
  """Returns the list of all operators for the DNF.

  Iterates over the list of operators produced by each clause and flattens.

  Args:
    dnf: dnf_lib.DNF, the dnf that defines the local Hamiltonian.
    time: Float, how long to apply the Hamiltonian.

  Returns:
    List of all cirq operators for the Hamiltonian defined by the DNF.
  """
  clause_operators_list = [
      generate_clause_hamiltonian_exponential(clause, time)
      for clause in dnf.clauses
  ]
  return list(itertools.chain.from_iterable(clause_operators_list))


class BangBangProtocolCircuit(circuit_lib.BangBangProtocolCircuit):
  """Modified QAOA circuit generator for bang-bang protocols.

  This circuit will divide the given time into a series of chunks that will
  be allocated to one of the two Hamiltonians.

  Attributes:
    chunk_time: Positive float, amount of time allocated to each chunk.
    dnf: dnf_lib.DNF, the 2-DNF we are trying to find the best solution to.
    hamiltonian_diagonal: List of floats, the diagonal of the target
        Hamiltonian, that determines the evaluations of each measurement.
    simulator: cirq.Simulator, the simulator to run the circuit.
  """

  def __init__(self, chunk_time, dnf):
    """Initializer.

    This class creates QAOA circuit in cirq.

    Args:
      chunk_time: Positive float, amount of time allocated to each chunk.
      dnf: dnf_lib.DNF, the 2-DNF we are trying to find the best solution to.
    """
    super(BangBangProtocolCircuit, self).__init__(chunk_time, dnf.num_literals)
    self.dnf = dnf
    self.hamiltonian_diagonal = self.get_hamiltonian_diagonal()

  def generate_constraint_hamiltonian_exponential(self, time):
    """Returns the time evolution of the Hamiltonian for the DNF.

    Iterates over the list of operators produced by each clause and flattens.

    Args:
      time: Float, how long to apply the Hamiltonian.

    Returns:
      List of all cirq operators for the Hamiltonian defined by the DNF.
    """
    return generate_dnf_hamiltonian_exponential(self.dnf, time)

  def constraint_evaluation(self, measurement):
    """Returns the fraction of clauses were satisfied compared to optimal.

    For any given 2-DNF, we want to maximize the probability of measuring
    literal assignments that return large values.

    Args:
      measurement: List of 0s and 1s, corresponding to a binary value of a
          measurement. Each 0 or 1 corresponds to a truth value of a literal
          assignement, where 0 is False and 1 is True.

    Returns:
      Float in interval [0, 1], the number of clauses satisfied by this literal
      assignment divided by the number of clauses satisfied by the optimal
      assignment.
    """
    return (self.dnf.get_num_clauses_satisfied(measurement) /
            self.dnf.optimal_num_satisfied)
