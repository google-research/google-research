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
"""Quantum circuit transformation rules.

TransformationRule is the abstract base class for all transformation rules. In
addition, this module contains various subclasses which implement concrete
transformation rules:
* InvertCnot
* CancelOperations
* ExchangeCommutingOperations
* ExchangePhasedXwithRotZ
* ExchangePhasedXwithControlledZ
* CompressLocalOperations

TransformationRuleSet allows to group several of those into one single
TransformationRule.

CircuitScanner is a helper class to scan a circuit for subgroups on which these
transformation rules can potentially be applied, e.g. pairs of operations for
PairTransformationRule.
"""

import abc
import copy
import itertools

from typing import Iterator, Iterable, Tuple, Union

import numpy as np

from rl4circopt import circuit
from rl4circopt import parsing
from rl4circopt import transform


# TODO(tfoesel):
# Think of how to deal with cases where the operations in the focus don't
# satisfy certain sanity constraints, in particular:
# * for PointTransformationRule and PairTransformationRule: if the focus does
#   not have the promised length, i.e. 1 or 2, respectively.
# * for PairTransformationRule: if the pair trivially commutes
# * for LocalGroupTransformationRule: if not all operations are local, or
#   different operations act on different qubits
# Currently, in some cases RuntimeException is raised, whereas some would go
# undetected.


class TransformationRule(abc.ABC):
  """Base class for all transformation rules."""

  @abc.abstractmethod
  def transformations_from_scanner(self,
                                   scanner
                                  ):
    """Scans a circuit for transformations.

    The circuit is represented by an instance of CircuitScanner and not
    circuit.Circuit. This has the advantage that various TransformationRules can
    be called with the same CircuitScanner, and in this way it allows to reuse
    results of the scanning process. In particular, class TransformationRuleSet
    is making use of this.

    For most users of this class, it might be more convenient to call the method
    scan(...) which takes an instance of Circuit.

    Args:
        scanner: the scanner over the input circuit.

    Returns:
        an iterator over all transformations that can be found.
    """
    pass

  def scan(self, circ):
    """Scans a circuit for transformations.

    Args:
        circ: the input circuit to be scanned.

    Returns:
        an iterator over all transformations that can be found.
    """
    return self.transformations_from_scanner(CircuitScanner(circ))

  def apply_first(self, circ):
    """Applies the first transformation that can be found.

    Args:
        circ: the input circuit.

    Returns:
        the transformed circuit, or None to indicate that no transformation
        could be found.
    """
    scanner = CircuitScanner(circ)

    for transformation in self.transformations_from_scanner(scanner):
      transformed_circ = transformation.perform()
      if transformed_circ is not None:
        return transformed_circ

    return None

  def apply_greedily(self, circ):
    """Applies greedily every transformation that can be found.

    This method applies the first transformation to the input circuit that can
    be found, and then goes on to apply always the first transformation to the
    transformed circuits until no further transformation can be found.

    Be careful! If the transformation rule set is not designed suitably, this
    might very easily run into an infinity loop.

    Args:
        circ: the input circuit.

    Returns:
        the transformed circuit, or the original circuit if no transformation
        could be found.
    """

    while True:
      transformed_circ = self.apply_first(circ)

      if transformed_circ is None:
        break

      # TODO(tfoesel):
      # would be good here to have a check that transformed_circ != circ
      circ = transformed_circ

    return circ


class RuleNotApplicableError(Exception):
  """Indicates that a transformation rule cannot be applied.

  Some subtypes of TransformationRule have an accept(...) and a perform(...)
  method. If perform(...) is called with arguments that would not pass
  accept(...), a RuleNotApplicableError is raised.
  """
  pass


class TransformationRuleSet(TransformationRule):
  """A group of several transformation rules."""

  def __init__(self, *rules):
    if not all(isinstance(rule, TransformationRule) for rule in rules):
      illegal_types = set(
          type(rule)
          for rule in rules
          if not isinstance(rule, TransformationRule)
      )
      raise TypeError(
          'only Operation objects allowed in focus (found types: %s)'
          %', '.join(sorted(illegal_type.__name__
                            for illegal_type in illegal_types))
      )

    self._rules = rules

  def transformations_from_scanner(self, scanner):
    # implements abstract method from parent class TransformationRule
    for rule in self._rules:
      for transformation in rule.transformations_from_scanner(scanner):
        yield transformation


class PointTransformationRule(TransformationRule):
  """A transformation rule which involves a single operation."""

  def transformations_from_scanner(self, scanner):
    # implements abstract method from parent class TransformationRule

    for attention_circ in scanner.single_operations():
      if len(attention_circ) != 1:
        raise RuntimeError()  # cmp. TODO at the beginning of this file
      operation, = attention_circ.focus()
      if self.accept(operation):
        yield transform.PointTransformation(attention_circ, self.perform, self)

  @abc.abstractmethod
  def accept(self, operation):
    """Checks whether this PointTransformationRule is applicable.

    Args:
        operation: the operation for which to check applicability.

    Returns:
        a bool indicating whether this transformation rule is applicable to the
        specified operation.
    """
    pass

  @abc.abstractmethod
  def perform(self,
              operation):
    """Applies this PointTransformationRule.

    Args:
        operation: the operation on which to perform this transformation.

    Returns:
        a sequence containing the modified operations.

    Raises:
        RuleNotApplicableError: if this transformation rule cannot be applied to
            the specified operation. This exception is raised iff
            `self.accept(operation)` gives False.
    """
    pass


class InvertCnot(PointTransformationRule):
  """Invert the direction of a CNOT, creating additional Hadamard gates."""

  hadamard = circuit.MatrixGate(np.sqrt(0.5) * np.array([
      [1.0, 1.0],
      [1.0, -1.0]
  ]))
  cnot = circuit.MatrixGate(np.array([
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 1.0, 0.0]
  ]))
  inverted_cnot = cnot.permute_qubits([1, 0])

  def __init__(self, architecture):
    self.architecture = architecture

  def accept(self, operation):
    # implements abstract method from parent class PointTransformationRule

    gate = operation.get_gate()
    return gate == self.cnot or gate == self.inverted_cnot

  def perform(self, operation):
    # implements abstract method from parent class PointTransformationRule

    if not self.accept(operation):
      raise RuleNotApplicableError

    hadamard_pauli_transform = self.hadamard.get_pauli_transform()

    hadamard_on_both = itertools.product(
        operation.get_qubits(),
        self.architecture.decompose_single_qubit_gate(hadamard_pauli_transform)
    )
    hadamard_on_both = [
        circuit.Operation(gate, [qubit])
        for qubit, gate in hadamard_on_both
    ]

    inverted = operation.permute_qubits([1, 0])

    return hadamard_on_both + [inverted] + hadamard_on_both


class PairTransformationRule(TransformationRule):
  """A transformation rule which involves two operations."""

  def transformations_from_scanner(self, scanner):
    # implements abstract method from parent class TransformationRule

    for attention_circ in scanner.operation_pairs():
      if len(attention_circ) != 2:
        raise RuntimeError()  # cmp. TODO at the beginning of this file
      operation_first, operation_second = attention_circ.focus()
      if self.accept(operation_first, operation_second):
        yield transform.PairTransformation(attention_circ, self.perform, self)

  @abc.abstractmethod
  def accept(self,
             operation_first,
             operation_second):
    """Checks whether this PairTransformationRule is applicable.

    Args:
        operation_first: the first operation of the pair for which to check
            applicability.
        operation_second: the second operation of the pair for which to check
            applicability.

    Returns:
        a bool indicating whether this transformation rule is applicable to the
        specified pair of operations.
    """
    pass

  @abc.abstractmethod
  def perform(self,
              operation_first,
              operation_second
             ):
    """Applies this PairTransformationRule.

    Args:
        operation_first: the first operation of the pair on which to perform
            this transformation.
        operation_second: the second operation of the pair on which to perform
            this transformation.

    Returns:
        operations_out_first: a sequence containing the first part of the
            modified operations.
        operations_out_second: a sequence containing the second part of the
            modified operations.

    Raises:
        RuleNotApplicableError: if this transformation rule cannot be applied to
            the specified pair of operations. This exception is raised iff
            `self.accept(operation_first, operation_second)` gives False.
    """
    pass


class CancelOperations(PairTransformationRule):
  """Transformation rule which eliminates operations that cancel each other."""

  # TODO(tfoesel):
  # Currently, CancelOperations is a PairTransformationRule, even though
  # cancellation is in principle defined for larger groups of operations. The
  # reason why CancelOperations is currently restricted to pairs is that
  # scanning for these larger groups is complicated and not implemented yet. As
  # soon as this has changed, CancelOperations should be extended to a larger
  # number of operations.

  def accept(self, operation_first, operation_second):
    # implements abstract method from parent class PairTransformationRule
    if operation_first.commutes_trivially_with(operation_second):
      raise RuntimeError()  # cmp. TODO at the beginning of this file
    return operation_first.cancels_with(operation_second, phase_invariant=True)

  def perform(self, operation_first, operation_second):
    # implements abstract method from parent class PairTransformationRule
    if self.accept(operation_first, operation_second):
      return [], []
    else:
      raise RuleNotApplicableError


class ExchangeCommutingOperations(PairTransformationRule):
  """Exchange the order of two commuting operations."""

  def accept(self, operation_first, operation_second):
    # implements abstract method from parent class PairTransformationRule
    if operation_first.commutes_trivially_with(operation_second):
      raise RuntimeError()  # cmp. TODO at the beginning of this file
    return operation_first.commutes_with(operation_second, phase_invariant=True)

  def perform(self, operation_first, operation_second):
    # implements abstract method from parent class PairTransformationRule
    if self.accept(operation_first, operation_second):
      return [operation_second], [operation_first]
    else:
      raise RuleNotApplicableError


class ExchangePhasedXwithRotZ(PairTransformationRule):
  """Exchange the order of a PhasedX and a RotZ gate (modifies the PhasedX gate).

  Uses the equivalence between the two circuits

      ───PhasedX(alpha, beta)───RotZ(gamma)───

      ───RotZ(gamma)───PhasedX(alpha, beta-gamma)───

  where alpha, beta and gamma are three arbitrary angles. The order of the
  parameters for the PhasedX gate is:

      PhasedX(rotation_angle, phase_angle)
  """

  def accept(self, operation_first, operation_second):
    # implements abstract method from parent class PairTransformationRule

    if operation_first.commutes_trivially_with(operation_second):
      raise RuntimeError()  # cmp. TODO at the beginning of this file

    return (
        parsing.check_operations(
            [operation_first, operation_second],
            circuit.PhasedXGate, circuit.RotZGate
        )
        or parsing.check_operations(
            [operation_first, operation_second],
            circuit.RotZGate, circuit.PhasedXGate
        )
    )

  def perform(self, operation_first, operation_second):
    # implements abstract method from parent class PairTransformationRule

    if operation_first.commutes_trivially_with(operation_second):
      raise RuntimeError()  # cmp. TODO at the beginning of this file

    parsed = parsing.parse_operations([operation_first, operation_second],
                                      circuit.PhasedXGate, circuit.RotZGate)
    if parsed is not None:
      phased_x, rot_z = parsed
      rot_z_angle = rot_z.get_gate().get_rotation_angle()

      mod_phased_x = phased_x.replace_gate(
          phased_x.get_gate().shift_phase_angle(rot_z_angle)
      )

      return [rot_z], [mod_phased_x]

    parsed = parsing.parse_operations([operation_first, operation_second],
                                      circuit.RotZGate, circuit.PhasedXGate)
    if parsed is not None:
      rot_z, phased_x = parsed
      rot_z_angle = rot_z.get_gate().get_rotation_angle()

      mod_phased_x = phased_x.replace_gate(
          phased_x.get_gate().shift_phase_angle(-rot_z_angle)
      )

      return [mod_phased_x], [rot_z]

    raise RuleNotApplicableError


class ExchangePhasedXwithControlledZ(PairTransformationRule):
  """Exchange the order of a PhasedX flip and a CZ gate (creates an additional Z flip).

  This transformation is only applicable if the rotation angle for the PhasedX
  gate is pi (or equivalent), i.e. if it performs a flip. In this case, we can
  use one of the following circuit equivalence relations, depending on the
  order of the PhasedX and the CZ gate in the input circuit:

      ───PXF───@───      ───@───PXF───      ───────@───PXF───
               |     ≡      |           ≡          |
      ─────────@───      ───@────Z────      ───Z───@─────────

      ───@───PXF───      ───PXF───@───      ───PXF───@───────
         |           ≡            |     ≡            |
      ───@─────────      ────Z────@───      ─────────@───Z───

  Here, PXF is a PhasedX gate whose rotation angle is a pi (or equivalent), Z
  is a Z flip, and "@─@" denotes a CZ gate.
  """

  def __init__(self, architecture):
    self.architecture = architecture

  def accept(self, operation_first, operation_second):
    # implements abstract method from parent class PairTransformationRule

    if operation_first.commutes_trivially_with(operation_second):
      raise RuntimeError()  # cmp. TODO at the beginning of this file

    parsed = parsing.parse_operations(
        [operation_first, operation_second],
        circuit.PhasedXGate, circuit.ControlledZGate
    )
    if parsed is not None:
      phased_x, _ = parsed
      return bool(np.isclose(
          np.cos(phased_x.get_gate().get_rotation_angle()),
          -1.0
      ))

    parsed = parsing.parse_operations(
        [operation_first, operation_second],
        circuit.ControlledZGate, circuit.PhasedXGate
    )
    if parsed is not None:
      _, phased_x = parsed
      return bool(np.isclose(
          np.cos(phased_x.get_gate().get_rotation_angle()),
          -1.0
      ))

    return False

  def perform(self, operation_first, operation_second):
    # implements abstract method from parent class PairTransformationRule

    if not self.accept(operation_first, operation_second):
      raise RuleNotApplicableError

    parsed = parsing.parse_operations(
        [operation_first, operation_second],
        circuit.PhasedXGate, circuit.ControlledZGate
    )
    if parsed is not None:
      phased_x, cz = parsed
    else:
      parsed = parsing.parse_operations(
          [operation_first, operation_second],
          circuit.ControlledZGate, circuit.PhasedXGate
      )
      if parsed is not None:
        cz, phased_x = parsed
      else:
        raise RuleNotApplicableError

    if not np.isclose(np.cos(phased_x.get_gate().get_rotation_angle()), -1.0):
      raise RuleNotApplicableError

    other_qubit, = set(cz.get_qubits()).difference(phased_x.get_qubits())

    z_flip = np.diag([-1.0, -1.0, 1.0])  # the pauli_transform for a Z flip
    z_flip = [
        circuit.Operation(gate, [other_qubit])
        for gate in self.architecture.decompose_single_qubit_gate(z_flip)
    ]

    return [operation_second], z_flip + [operation_first]


class LocalGroupTransformationRule(TransformationRule):
  """Transformation rule for a group of local operations on the same qubit."""

  def transformations_from_scanner(self, scanner):
    # implements abstract method from parent class TransformationRule
    for attention_circ in scanner.local_groups():
      if self.accept(attention_circ.focus()):
        yield transform.GroupTransformation(attention_circ, self.perform, self)

  @abc.abstractmethod
  def accept(self, operations):
    """Checks whether this LocalGroupTransformationRule is applicable.

    Args:
        operations: the group of operations for which to check applicability.

    Returns:
        a bool indicating whether this transformation rule is applicable to the
        specified group of operations.
    """
    pass

  @abc.abstractmethod
  def perform(self,
              operations
             ):
    """Applies this LocalGroupTransformationRule.

    Args:
        operations: the group of operations for which to check applicability.

    Returns:
        a sequence containing the modified operations.
    """
    pass


class CompressLocalOperations(LocalGroupTransformationRule):
  """Compresses a group of subsequent local (i.e. single-qubit) operations."""

  def __init__(self, architecture):
    self.architecture = architecture

  def accept(self, operations):
    # implements abstract method from parent class LocalGroupTransformationRule
    gates = [operation.get_gate() for operation in operations]

    # check whether all operations are local and act on the same gate
    qubits = [operation.get_qubits() for operation in operations]
    if not all(len(qu) == 1 for qu in qubits):
      raise RuntimeError()  # cmp. TODO at the beginning of this file
    qubits = set().union(*qubits)
    if len(qubits) != 1:
      raise RuntimeError()  # cmp. TODO at the beginning of this file

    return self.architecture.can_optimize_single_qubit_group(gates)

  def perform(self, operations):
    # implements abstract method from parent class LocalGroupTransformationRule

    if not self.accept(operations):
      raise RuleNotApplicableError

    qubits = [operation.get_qubits() for operation in operations]
    if not all(len(qu) == 1 for qu in qubits):
      raise RuntimeError()  # cmp. TODO at the beginning of this file
    qubits = set().union(*qubits)
    if len(qubits) != 1:
      raise RuntimeError()  # cmp. TODO at the beginning of this file
    qubit, = qubits

    pauli_transform = np.eye(3)
    for operation in operations:
      pauli_transform = np.dot(
          operation.get_gate().get_pauli_transform(),
          pauli_transform
      )

    gates = self.architecture.decompose_single_qubit_gate(pauli_transform)

    return [circuit.Operation(gate, [qubit]) for gate in gates]


class CircuitScanner:
  """Provides iterators over certain families of AttentionCircuits for a given input circuit.

  For optimal performance, all iterations are lazy (i.e. items are not computed
  before they are actually requested) and memoized (i.e. for repetitive access,
  the items are read from a cache instead of being computed again).
  """

  def __init__(self, circ):
    """Initializes a new CircuitScanner.

    Args:
        circ: the circuit to be scanned.

    Raises:
        TypeError: if circ is not a Circuit.
    """
    if not isinstance(circ, circuit.Circuit):
      raise TypeError('circ is not a Circuit (found type: %s)'
                      %type(circ).__name__)

    self._single_operations, = \
        itertools.tee(scan_for_single_operations(circ), 1)
    self._operation_pairs, = itertools.tee(scan_for_operation_pairs(circ), 1)
    self._local_groups, = itertools.tee(scan_for_local_groups(circ), 1)

  def single_operations(self):
    """Iterates over all attention circuits with exactly one operation in the focus.

    See also: scan_for_single_operations(...).

    For optimal performance, this method is lazy (i.e. items are not computed
    before they are actually requested) and memoized (i.e. for repetitive
    access, the items are read from a cache instead of being computed again).

    Returns:
        an iterator over all attention circuits with exactly one operation in
        the focus.
    """
    return copy.copy(self._single_operations)

  def operation_pairs(self):
    """Iterates over all attention circuits with two aligned operations in the focus.

    See also: scan_for_operation_pairs(...)

    For optimal performance, this method is lazy (i.e. items are not computed
    before they are actually requested) and memoized (i.e. for repetitive
    access, the items are read from a cache instead of being computed again).

    Returns:
        an iterator over all attention circuits with two aligned operations in
        the focus.
    """
    return copy.copy(self._operation_pairs)

  def local_groups(self):  # -> Iterator[transform.AttentionCircuit]:
    """Iterates over all attention circuits with an aligned group of local operations in the focus.

    See also: scan_for_local_groups(...)

    For optimal performance, this method is lazy (i.e. items are not computed
    before they are actually requested) and memoized (i.e. for repetitive
    access, the items are read from a cache instead of being computed again).

    Returns:
        an iterator over all attention circuits with an aligned group of local
        (i.e. single-qubit) operations in the focus.
    """
    return copy.copy(self._local_groups)


def scan_for_single_operations(circ
                              ):
  """Iterates over all attention circuits with exactly one operation in the focus.

  There will be such an AttentionCircuit for each operation in the circuit.

  This function is lazy, i.e. items are not computed before they are actually
  requested.

  Args:
      circ: the circuit to be scanned.

  Yields:
      all attention circuits with exactly one operation in the focus.

  Raises:
      TypeError: if circ is not a Circuit.
  """
  if not isinstance(circ, circuit.Circuit):
    raise TypeError('circ is not a Circuit (found type: %s)'
                    %type(circ).__name__)

  for location, _ in enumerate(circ):
    yield transform.focus_single_operation(circ, location)


def scan_for_operation_pairs(circ
                            ):
  """Iterates over all attention circuits with two aligned operations in the focus.

  For each pair of operations in the circuit which are not separated by other
  operations, there will be one AttentionCircuit with only these two operations
  in the focus. The exact criterion for whether the operations are considered
  to be not separated by other operations is whether it is possible, by only
  exchanging trivially commuting operations, to transform the circuit such that
  there is no operation between them.

  Example: the two operations B and D (marked with a star) in the following
  circuit form such a pair.

      operation A on qubits [0, 1]
      operation B on qubits [1, 3]   (*)
      operation C on qubits [0]
      operation D on qubits [3, 4]   (*)
      operation E on qubits [1, 3]

  Operation C is no problem because it does not act on any qubit affected by B
  and D. Also, all operations before B or after D are no problem anyways.

  Note that "no operation in between acts on one of the qubits affected by one
  of the operations in the pair" is only a sufficient, but not a necessary
  criterion. For example, operations A' and C' in

      operation A' on qubits [1, 3]   (*)
      operation B' on qubits [1]
      operation C' on qubits [3, 4]   (*)

  still form a pair even though operation B' acts on qubit 1 which is affected
  by operation A'. The reason is that operation B' and C' commute trivially, so
  we can arrive at the circuit (A', C', B') for which the pair relation for
  operations A' and C' is obvious.

  This function always takes into account the minimal requirement to form a
  pair.

  This function is lazy, i.e. items are not computed before they are actually
  requested.

  Args:
      circ: the circuit to be scanned.

  Yields:
      all attention circuits with two aligned operations in the focus.

  Raises:
      TypeError: if circ is not a Circuit.
  """
  if not isinstance(circ, circuit.Circuit):
    raise TypeError('circ is not a Circuit (found type: %s)'
                    %type(circ).__name__)

  last_operation_on_qubit = np.full(circ.get_num_qubits(), -1)

  for curr_location, curr_operation in enumerate(circ):
    qubits = np.array(curr_operation.get_qubits())

    # look up the locations for the latest operations (so far) on all qubits
    # affected by curr_operation
    prev_locations = np.unique(last_operation_on_qubit[qubits])
    assert np.all(np.isin(prev_locations, range(-1, curr_location)))
    prev_locations = prev_locations[prev_locations >= 0]

    for prev_location in prev_locations:
      # we have a candidate
      # now, there is no better way than "trial and error"
      try:
        yield transform.focus_operation_pair(circ, prev_location, curr_location)
      except transform.OperationsNotAlignedError:
        continue

    # update last_operation_on_qubit to curr_location for all qubits affected
    # by curr_operation
    last_operation_on_qubit[qubits] = curr_location


def scan_for_local_groups(circ
                         ):
  """Iterates over all attention circuits with an aligned group of local operations in the focus.

  For each maximal group of local (i.e. single-qubit) operations in the
  circuit which act on the same qubit and are not separated by other
  operations, there will be one AttentionCircuit with only these operations in
  the focus. The exact criterion for whether the operations are considered to
  be not separated by other operations is whether between the first and the
  last operation in the group, there is no other (multi-qubit) operation
  which acts on the same qubit. Maximal means that there are no more local
  operations on this qubit which could be attached (either before the first
  or after the last operation in the group).

  Example: the operations B, E and F (marked with a star) in the following
  circuit form such a local group.

      operation A on qubits [0, 1]
      operation B on qubits [2]      (*)
      operation C on qubits [0]
      operation D on qubits [1, 3]
      operation E on qubits [2]      (*)
      operation F on qubits [2]      (*)
      operation G on qubits [2, 3]
      operation H on qubits [2]

  Operations C and D are no problem because the do not affect qubit 2 (the
  qubit which all operations in the group act on). Also, all operations before
  operation B or after operation F, i.e. the first/last operation in the
  group, are no problem anyway.

  The group in this example is maximal. Note that operation H cannot be
  attached because operation G affects qubit 2 (the qubit which all operations
  in the group act on). An example for a non-maximal group is if operation F
  was missing.

  This function is lazy, i.e. items are not computed before they are actually
  requested.

  Args:
      circ: the circuit to be scanned.

  Yields:
      all attention circuits with an aligned group of local operations in the
      focus.

  Raises:
      TypeError: if circ is not a Circuit.
  """
  if not isinstance(circ, circuit.Circuit):
    raise TypeError('circ is not a Circuit (found type: %s)'
                    %type(circ).__name__)

  # for each qubit, allocate a list to store the local operations on it
  candidates = [[] for _ in range(circ.get_num_qubits())]

  for location, operation in enumerate(circ):
    if operation.get_num_qubits() == 1:
      # a local operation was found; we put it into candidates[qubit]
      qubit, = operation.get_qubits()
      candidates[qubit].append(location)
    else:
      # a non-local operation was found
      # this terminates the local groups on all affected qubits
      for qubit in operation.get_qubits():
        candidate = candidates[qubit]
        if candidate:
          yield transform.focus_local_group(circ, candidate)
          candidate.clear()

  # we have now gone through the full circuit
  # the qubits for which the last operation has been local have unterminated
  # candidates; we take care of them in the following
  for candidate in candidates:
    if candidate:
      yield transform.focus_local_group(circ, candidate)
