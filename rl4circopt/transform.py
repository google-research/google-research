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
"""Tools for transforming quantum circuits.

This module contains the following classes:
* Transformation (+ subclasses): representation for concrete transformations of
  quantum circuits.
* AttentionCircuit and TransformationContext: helper classes which make it
  convenient to implement these transformations.
"""

import abc
import itertools
from typing import Callable, Iterable, Sequence, Tuple, Union

import numpy as np

from rl4circopt import circuit


class Transformation(abc.ABC):
  """Base class for all transformations."""

  def __init__(self, attention_circ, rule_id):
    """Initializes a new Transformation.

    Args:
        attention_circ: the AttentionCircuit representing the input to the
            transformation.
        rule_id: an identifier for the underlying rule.

    Raises:
        TypeError: if attention_circ is not an AttentionCircuit.
    """
    if not isinstance(attention_circ, AttentionCircuit):
      raise TypeError(
          'attention_circ is not an AttentionCircuit (found type: %s)'
          %type(attention_circ).__name__
      )

    self._attention_circ = attention_circ
    self._rule_id = rule_id

  @abc.abstractmethod
  def perform(self):
    """Applies this transformation.

    Returns:
        the transformed circuit.
    """
    pass

  def focus(self):
    """Returns the focus of this Transformation.

    The operations in the focus are those which are directly affected by the
    transformation.

    Returns:
        the operations in the focus for this Transformation.
    """
    return self._attention_circ.focus()

  def context(self):
    """Returns the context of this Transformation.

    The context contains those operations which remain untouched under the
    transformation.

    Returns:
        the context of this Transformation.
    """
    return self._attention_circ.context()

  def locations(self):
    """Returns the locations for this Transformation.

    The locations are the indices of the operations in the focus within the
    input circuit. This can be useful meta information for different purposes,
    in particular for implementing handcrafted optimization strategies and for
    interpreting the output of a neural network.

    The locations are optional and can be None.

    Returns:
        the indices of the operations in the focus within the input circuit, or
        None if no locations are set.
    """
    return self._attention_circ.locations()

  def rule_id(self):
    """Returns an identifier for the underlying transformation rule.

    This can be any object which can represent the transformation rule, for
    example an instance of rules.TransformationRule. This can be useful meta
    information for different purposes, in particular for implementing
    handcrafted optimization strategies and for interpreting the output of a
    neural network.

    Returns:
        an identifier for the underlying transformation rule.
    """
    return self._rule_id


class PointTransformation(Transformation):
  """Transforms a circuit by modifying a single operation."""

  def __init__(self,
               attention_circ,
               perform_rule,
               rule_id):
    """Initializes a new PointTransformation.

    Args:
        attention_circ: the AttentionCircuit representing the input to the
            transformation. For a PointTransformation, the size of the focus
            must be 1.
        perform_rule: a callback with signature `operations_out =
            perform_rule(operation_in)` where operation_in is an Operation, and
            operations_out is a sequence of Operation.
        rule_id: an identifier for the underlying rule.

    Raises:
        TypeError: if attention_circ is not an AttentionCircuit.
        ValueError: if the focus of attention_circ does not have size 1.
    """
    super().__init__(attention_circ, rule_id)

    if len(attention_circ) != 1:
      raise ValueError(
          'focus of attention_circ for %s must have size 1 (found: %d)'
          %(type(self).__name__, len(attention_circ))
      )

    self._perform_rule = perform_rule

  def perform(self):
    # implements abstract method from parent class Transformation
    operation_in, = self.focus()  # length checked in initializer
    operations_out = self._perform_rule(operation_in)
    return self.context().inject(operations_out)


class PairTransformation(Transformation):
  """Transforms a circuit by modifying a pair of operations."""

  def __init__(self,
               attention_circ,
               perform_rule,
               rule_id):
    """Initializes a new PairTransformation.

    Args:
        attention_circ: the AttentionCircuit representing the input to the
            transformation. For a PairTransformation, the size of the focus
            must be 2.
        perform_rule: a callback with signature `operations_out_first,
            operations_out_second = perform_rule(operation_in_first,
            operation_in_second)` where operation_in_first and
            operation_in_second are two instances of Operation, and
            operations_out_first and operations_out_second are two sequences of
            Operation.
        rule_id: an identifier for the underlying rule.

    Raises:
        TypeError: if attention_circ is not an AttentionCircuit.
        ValueError: if the focus of attention_circ does not have size 2, or if
            the operations in the focus trivially commute.
    """
    super().__init__(attention_circ, rule_id)

    if len(attention_circ) != 2:
      raise ValueError(
          'focus of attention_circ for %s must have size 2 (found: %d)'
          %(type(self).__name__, len(attention_circ))
      )

    # TODO(tfoesel):
    # consider doing this in the initializer of AttentionCircuit after
    # generalizing this condition to larger groups
    operation_first, operation_second = attention_circ.focus()
    if operation_first.commutes_trivially_with(operation_second):
      raise ValueError(
          'transformation redundant because operations trivially commute')

    self._perform_rule = perform_rule

  def perform(self):
    # implements abstract method from parent class Transformation

    # length for self.focus() checked in initializer
    operation_in_first, operation_in_second = self.focus()

    operations_out_first, operations_out_second = self._perform_rule(
        operation_in_first,
        operation_in_second
    )

    return self.context().inject(operations_out_first, operations_out_second)


class GroupTransformation(Transformation):
  """Transforms a circuit by modifying a group of operations."""

  def __init__(self,
               attention_circ,
               perform_rule,
               rule_id):
    """Initializes a new GroupTransformation.

    Args:
        attention_circ: the AttentionCircuit representing the input to the
            transformation.
        perform_rule: a callback with signature `operations_out =
            perform_rule(operations_in)` where operations_in and operations_out
            are sequences of Operation (operations_in is guaranteed to be a
            tuple).
        rule_id: an identifier for the underlying rule.

    Raises:
        TypeError: if attention_circ is not an AttentionCircuit.
    """
    super().__init__(attention_circ, rule_id)
    self._perform_rule = perform_rule

  def perform(self):
    # implements abstract method from parent class Transformation
    operations_in = self.focus()
    operations_out = self._perform_rule(operations_in)
    return self.context().inject(operations_out)


class AttentionCircuit:
  """Sets the focus on some selected operations within a circuit.

  AttentionCircuit represents a circuit in a format which makes it convenient to
  perform transformations on it. For this purpose, the operations in the
  circuit are separated into the "focus" and the "context": the operations in
  the focus are directly affected by the transformation, whereas those in the
  context remain untouched.
  """

  def __init__(self,
               focus,
               context,
               locations=None):
    """Initializes a new AttentionCircuit.

    Args:
        focus: the operations which will be directly affected by the
            transformation.
        context: a TransformationContext object representing the operations
            which remain untouched by the transformation.
        locations: the indices of the operations in the focus within the input
            circuit. Can be set to None.

    Raises:
        TypeError: if focus is not an iterable of Operations, if context is not
            an instance of TransformationContext, or if locations is neither
            None nor an iterable of int.
        ValueError: if focus is empty, or if locations is not None and does not
            match the length for focus.
    """

    focus = tuple(focus)
    if not focus:
      raise ValueError('focus must not be empty')
    if not all(isinstance(operation, circuit.Operation)
               for operation in focus):
      illegal_types = set(
          type(operation)
          for operation in focus
          if not isinstance(operation, circuit.Operation)
      )
      illegal_types = sorted(
          illegal_type.__name__
          for illegal_type in illegal_types
      )
      raise TypeError(
          'only Operation objects allowed in focus (found types: %s)'
          %', '.join(illegal_types)
      )

    if not isinstance(context, TransformationContext):
      raise TypeError('context is not a TransformationContext (found type: %s)'
                      %type(context).__name__)

    if locations is not None:
      locations = tuple(
          _cast_to_int(location, 'location')
          for location in locations
      )
      if len(focus) != len(locations):
        raise ValueError(
            'inconsistent lengths for focus and locations: %d vs. %d'
            %(len(focus), len(locations))
        )

    self._focus = focus
    self._context = context
    self._locations = locations

  def focus(self):
    """Returns the focus of this AttentionCircuit.

    The operations in the focus are those which are directly affected by the
    transformation.

    Returns:
        the operations in the focus for this AttentionCircuit.
    """
    return self._focus

  def context(self):
    """Returns the context of this AttentionCircuit.

    The context contains those operations which remain untouched under the
    transformation.

    Returns:
        the context of this AttentionCircuit.
    """
    return self._context

  def locations(self):
    """Returns the locations for this AttentionCircuit.

    The locations are the indices of the operations in the focus within the
    input circuit. This can be useful meta information for different purposes,
    in particular for implementing handcrafted optimization strategies and for
    interpreting the output of a neural network.

    The locations are optional and can be None. Therefore, Transformation
    objects must not make use of this information.

    Returns:
        the indices of the operations in the focus within the input circuit, or
        None if no locations are set.
    """
    return self._locations

  def __len__(self):
    """Returns the number of operations in the focus."""
    return len(self._focus)


class TransformationContext:
  """Represents the operations which remain untouched under a operation.

  There are three groups of operations split along the time axis of the circuit,
  each one represented by an instance of Circuit:
  * "before": the operations which are to be executed before the operations in
    the focus.
  * "between": the operations which are to be executed either before, between
    or after the operations in the focus.
  * "after": the operations which are to be executed after the operations in
    the focus.

  (The operations in the focus are those which are directly affected by the
  transformation.)
  """

  def __init__(self,
               before,
               between,
               after):
    """Initializes a new TransformationContext.

    Args:
        before: the subcircuit to be executed before the operations in the
            focus.
        between: the subcircuit to be executed either before, between or after
            the operations in the focus.
        after: the operations to be executed before the operations in the
            focus.

    Raises:
        TypeError: if before, between or after is not a Circuit.
        ValueError: if the number of qubits for before, between or after is
            inconsistent.
    """
    if not all(isinstance(comp, circuit.Circuit)
               for comp in [before, between, after]):
      raise TypeError(
          'before, between and after must be Circuits (found types: %s)'
          %', '.join(type(comp).__name__ for comp in [before, between, after])
      )

    num_qubits = before.get_num_qubits()
    if (num_qubits != between.get_num_qubits()
        or num_qubits != after.get_num_qubits()):
      raise ValueError(
          'inconsistent number of qubits for before, between and after: (%s)'
          %', '.join(str(comp.get_num_qubits())
                     for comp in [before, between, after])
      )

    self._num_qubits = num_qubits
    self._before = before
    self._between = between
    self._after = after

  def before(self):
    """Returns the circuit "before" associated to this TransformationContext.

    This circuit represents the operations which are to be executed before the
    operations in the focus.

    Returns:
        the circuit "before" associated to this TransformationContext.
    """
    return self._before

  def between(self):
    """Returns the circuit "between" associated to this TransformationContext.

    This circuit represents the operations which are to be executed either
    before, between or after the operations in the focus.

    Returns:
        the circuit "between" associated to this TransformationContext.
    """
    return self._between

  def after(self):
    """Returns the circuit "after" associated to this TransformationContext.

    This circuit represents the operations which are to be executed after the
    operations in the focus.

    Returns:
        the circuit "after" associated to this TransformationContext.
    """
    return self._after

  def inject(self,
             operations_first,
             operations_second = None
            ):
    """Constructs a new circuit by inserting the specified operations into this context.

    The output circuit consists of the following parts in the given order:
    1. the circuit "before" associated to this TransformationContext
    2. the elements of operations_first (skipped if this argument is None)
    3. the circuit "between" associated to this TransformationContext
    4. the elements of operations_second (skipped if this argument is None)
    5. the circuit "after" associated to this TransformationContext

    Args:
        operations_first: the operations to be inserted between "before" and
            "between".
        operations_second: the operations to be inserted between "between" and
            "after".

    Returns:
        a new circuit where operations_first is inserted between "before" and
        "between", and operations_second is inserted between "between" and
        "after".

    Raises:
        TypeError: if operations_first and/or operations_seconds is neither None
            nor a sequence of Operations.
    """

    # TypeErrors will be raised in the circuit.Circuit initializers
    return (
        self.before()
        + circuit.Circuit(self._num_qubits, operations_first)
        + self.between()
        + circuit.Circuit(self._num_qubits, operations_second)
        + self.after()
    )


class OperationsNotAlignedError(Exception):
  """Indicates an error because some operations are not aligned.

  Two operations are not aligned iff two they are separated by other operations
  which have to be executed in between. A larger set operations is not aligned
  if it contains at least one pair of non-aligned operations.

  Example: We consider the following operation sequence.

      operation A on qubits [1, 3]
      operation B on qubits [1]
      operation C on qubits [0, 1]

  Operations A and C are not aligned because they are separated by operation B,
  which has to be executed between A and C because both of them act on qubit 1
  and qubit 1 is also affected by operation B. Therefore, operations A and C are
  selected for the focus, an OperationsNotAlignedError should be raised.

  Its main purpose is to indicate that the construction of an AttentionCircuit
  failed because an set of selected operations is invalid.
  """
  pass


def focus_single_operation(circ,
                           location):
  """Constructs an AttentionCircuit with exactly one operation in the focus.

  Args:
      circ: the full circuit.
      location: the index of the operation to be in the focus.

  Returns:
      an AttentionCircuit with exactly one operation in the focus.

  Raises:
      TypeError: if circ is not a Circuit or location is not integer-like.
      IndexError: if location is not a valid index for circ.
  """
  if not isinstance(circ, circuit.Circuit):
    raise TypeError('circ is not a Circuit (found type: %s)'
                    %type(circ).__name__)
  location = _check_and_convert_to_non_negative_circuit_index(
      location,
      len(circ),
      'location'
  )

  return AttentionCircuit(
      focus=[circ[location]],
      context=TransformationContext(
          before=circ[:location],
          between=circuit.Circuit(circ.get_num_qubits(), None),
          after=circ[location + 1:]
      ),
      locations=[location]
  )


def focus_operation_pair(circ,
                         location_first,
                         location_second):
  """Constructs an AttentionCircuit with two operations in the focus.

  Such a transformation is only permitted iff the selected operations are not
  separated by other operations. The exact criterion for "not being separated"
  is whether the given circuit can be transformed, by only exchanging trivially
  commuting operations, into another circuit in which the two operations are
  consecutive, i.e. `new_location_second - new_location_first == 1`.

  Args:
      circ: the full circuit.
      location_first: the index of the first operation of the pair.
      location_second: the index of the second operation of the pair.

  Returns:
      an AttentionCircuit with the two specified operations in the focus.

  Raises:
      TypeError: if circ is not a Circuit, or if location_first or
          location_second is not integer-like.
      IndexError: if location_first and/or location_second is out of range.
      ValueError: if location_first does not refer to an earlier operation than
          location_second.
      OperationsNotAlignedError: if a pair operation is not permitted because
          the selected operations are separated by other operations.
  """
  if not isinstance(circ, circuit.Circuit):
    raise TypeError('circ is not a Circuit (found type: %s)'
                    %type(circ).__name__)

  length = len(circ)
  location_first = _check_and_convert_to_non_negative_circuit_index(
      location_first,
      length,
      'location_first'
  )
  location_second = _check_and_convert_to_non_negative_circuit_index(
      location_second,
      length,
      'location_second'
  )

  if location_first >= location_second:
    raise ValueError('location_first not smaller than location_second:'
                     ' %d (or %d) vs %d (or %d)'%(
                         location_first,
                         location_first - length,
                         location_second,
                         location_second - length
                     ))

  # In the following, we will check whether the selected operations are not
  # separated by other operations. If successful, we obtain as a side effect
  # the segmentation for the other operations (i.e. the "context") into the
  # subcircuits "before", "between" and "after".
  #
  # To this end, we first check whether the selected operation pair trivially
  # commutes with all operations in between. If we find this condition
  # violated, the game is not yet over. For illustration, we consider the
  # following situation (the pair to be checked are the marked operations A and
  # D):
  #
  #   operation A on qubits [1, 3]   (*)
  #   operation B on qubits [1]
  #   operation C on qubits [?]
  #   operation D on qubits [3, 4]   (*)
  #
  # Here, operation B does not commute trivially with operation A. Depending
  # on the qubits that operation C acts on, we can still be successful: if
  # operation C either trivially commutes with operation A and B, or if it
  # trivially commutes with operation D.
  #
  # The general pattern is the following: All operations which do not trivially
  # commute with the first operation of the pair have to be performed after the
  # second operation, as well as all operations which do not trivially commute
  # with those operations that do not trivially commute with the first
  # operation, and so on recursively. Analogically, all operations which do not
  # trivially commute with the second operation of the pair have to be performed
  # before the first operation, as well as all operations which do not trivially
  # commute with those operations that do not trivially commute with the second
  # operation, and so on recursively.
  #
  # To resolve all possible cases, we proceed as follows (first and second
  # operation refers to the selected pair):
  #  * We define a pool of operations. The operations in this pool are the ones
  #    which cannot be moved before the first operation, and thus have to be
  #    moved after the second one.
  #    We build this pool as follows: For the first operation and every new
  #    member to the pool, we will add each operation between this and the
  #    second operation (exclusively) to the pool unless it trivially commutes
  #    with this operation. We go on doing this until the pool consists entirely
  #    of operations which have already been visited.
  #    Example: All the operations below marked with a star end up in the pool.
  #    The second argument tells because of which operation (that it does not
  #    trivially commute with) it has been added to the pool.
  #
  #        operation_first  on qubits [1, 3]
  #        operation A      on qubits [1, 2, 5]   (*, operation_first)
  #        operation B      on qubits [4]
  #        operation C      on qubits [4, 6]
  #        operation D      on qubits [2, 7]      (*, operation A)
  #        operation E      on qubits [4, 5]      (*, operation A)
  #        operation F      on qubits [4, 7]      (*, operation D or E)
  #        operation_second on qubits [???]
  #
  #  * We define a second pool of operations. The operations in this pool are
  #    the ones which cannot be moved after the second operation, and thus have
  #    to be moved before the first one.
  #    We build this pool similar to the first one. The only difference is that
  #    we check operations between the first and the new operation (instead of
  #    the opeations between the new and the second operation).
  #  * A pair transformation for the operations at location_first and
  #    location_second is permitted iff (finally) there is no operation which
  #    belongs to both pools. As soon as we find such an operation, we know that
  #    this transformation is not permitted and terminate.

  # mismatch between the indexing of circ and movement_flags
  offset = location_first + 1

  # movement_flags tells which operation belongs to which pool:
  # movement_flags[idx, 0] (movement_flags[idx, 1]) being set indicates that
  # the operation at circ[idx + offset] needs to be moved before (after) the
  # operation at circ[location_first] (circ[location_second])
  movement_flags = np.full([location_second - location_first - 1, 2], False)

  # unvisited contains the unvisited new members to both pools; every entry is
  # a tuple of the following two items:
  #
  #   (0) the location of the operation
  #   (1) the direction in which to check (0 for towards the start, 1 for
  #       towards the end of the circuit)
  unvisited = [(location_first, 1), (location_second, 0)]

  while unvisited:
    # extract and remove one element from unvisited
    location_to_visit, direction_idx = unvisited.pop()

    # retrieve the corresponding operation
    operation_to_visit = circ.operation(location_to_visit)

    # depending on the direction, select the locations for the operations to
    # check against
    if direction_idx == 0:
      locations_to_check = np.arange(location_first + 1, location_to_visit)
    else:
      locations_to_check = np.arange(location_to_visit + 1, location_second)

    # filter out the operations which are already in the corresponding pool
    locations_to_check = locations_to_check[~movement_flags[
        locations_to_check - offset, direction_idx
    ]]

    # filter out the operations which trivially commute with operation_to_visit
    locations_to_check = locations_to_check[[
        not operation_to_visit.commutes_trivially_with(circ.operation(location))
        for location in locations_to_check
    ]]

    # now, locations_to_check consists only of operations which have to be added
    # to the pool

    if np.any(movement_flags[locations_to_check - offset, 1 - direction_idx]):
      # (at least) one operation is in both pools, i.e. it has to be moved
      # before the operation at circ[location_first] and at the same time also
      # after the operation at circ[location_second], which is obviously not
      # possible; therefore, the pair transformation between the operations at
      # circ[location_first] and circ[location_second] is not permitted
      raise OperationsNotAlignedError

    # add all operations which do not commute trivially with operation_to_visit
    # to the same pool (by setting the same direction_idx)
    movement_flags[locations_to_check - offset, direction_idx] = True
    unvisited.extend((location, direction_idx)
                     for location in locations_to_check)

  # If the function has not already been stopped, i.e. there are no operations
  # for which both movement_flags are set, we know now that the pair
  # transformation is permitted. Dependent on the movement_flags, we now group
  # them into "before", "between" and "after".
  before, = movement_flags[:, 0].nonzero()
  between, = (~np.logical_or(movement_flags[:, 0],
                             movement_flags[:, 1])).nonzero()
  after, = movement_flags[:, 1].nonzero()

  return AttentionCircuit(
      focus=[circ[location_first], circ[location_second]],
      context=TransformationContext(
          before=circ[:location_first, before + offset],
          between=circ[between + offset],
          after=circ[after + offset, location_second+1:]
      ),
      locations=[location_first, location_second]
  )


def focus_local_group(circ,
                      locations):
  """Constructs an AttentionCircuit with an group of neighboring local operations in the focus.

  A transformation on such a group of local (i.e. single-qubit) operations is
  only permitted iff the operations in the focus are neighboring, i.e. not
  separated by other operations, which is equivalent to the condition whether
  between the first and the last operation in the group, there is no other
  (multi-qubit) operation which acts on the same qubit. If this is not the case,
  an OperationsNotAlignedError is raised.

  Args:
      circ: the full circuit.
      locations: the indices of the operations to be in the focus.

  Returns:
      an AttentionCircuit with the selected (local) operations in the focus.

  Raises:
      TypeError: if circ is not a Circuit, or if locations is not a sequence of
          int.
      ValueError: if locations is empty or contains duplicate elements, or if
          the focus does not consist of local operations which all act on the
          same qubit.
      IndexError: if one of the locations is out of bounds.
      OperationsNotAlignedError: if the locations do not correspond to a valid
          local group.
  """
  if not isinstance(circ, circuit.Circuit):
    raise TypeError('circ is not a Circuit (found type: %s)'
                    %type(circ).__name__)

  length = len(circ)
  locations = np.array([
      _check_and_convert_to_non_negative_circuit_index(location, length,
                                                       'location')
      for location in locations
  ])

  num_locations = locations.size

  if num_locations == 0:
    raise ValueError('locations must not be empty')

  locations = np.unique(locations)  # sorts and makes elements unique
  if locations.size != num_locations:
    raise ValueError('locations contains duplicate elements')

  # extract the focus from the input circuit
  focus = circ.subcircuit(locations)

  if not all(operation.get_num_qubits() == 1 for operation in focus):
    raise ValueError('focus contains non-local operations')

  active_qubit = focus.operation(0).get_qubits()
  if not all(operation.get_qubits() == active_qubit for operation in focus[1:]):
    raise ValueError('operations in the focus act on different qubits')

  # every operation between the first and last operation of the focus is put
  # into between as long as this operation does not belong to the focus itself
  between = np.arange(locations[0] + 1, locations[-1])
  between = between[~np.isin(between, locations)]
  between = circ.subcircuit(between)

  # check whether the operations in the focus are aligned
  # For a local group, this is equivalent to that no operation in "between"
  # affects on the qubit that all operations in the focus act on, i.e. every
  # operation in the focus has to commute trivially with every operation in
  # "between".
  if all(op1.commutes_trivially_with(op2)
         for op1, op2 in itertools.product(focus, between)):
    return AttentionCircuit(
        focus.get_operation_sequence(),
        TransformationContext(
            circ[:locations[0]],
            between,
            circ[locations[-1] + 1:]
        ),
        locations=locations
    )
  else:
    raise OperationsNotAlignedError


def _cast_to_int(num, param_name):
  """Casts integer-like objects to int.

  Args:
      num: the number to be casted.
      param_name: the name of the parameter to be displayed in an error
          message.

  Returns:
      num as an int.

  Raises:
      TypeError: if num is not integer-like.
  """

  try:
    # __index__() is the Pythonic way to convert integer-like objects (e.g.
    # np.int64) to an int
    return num.__index__()
  except (TypeError, AttributeError):
    raise TypeError('%s is not integer-like (found type: %s)'
                    %(param_name, type(num).__name__))


def _check_and_convert_to_non_negative_circuit_index(index, length, param_name):
  """Normalizes an index within a circuit.

  Converts index to an int, checks whether it is within the bounds given by the
  circuit length, and translates negative indices.

  Args:
      index: the index for an operation.
      length: the length of the circuit.
      param_name: the name of the parameter to be displayed in an error
          message.

  Returns:
      the normalized (non-negative) index as an int.

  Raises:
      TypeError: if index is not integer-like.
      IndexError: if index is out of bounds.
  """
  index = _cast_to_int(index, param_name)
  if 0 <= index < length:
    return index
  elif -length <= index < 0:
    return index + length
  else:
    raise IndexError('%s %d out of bounds for a Circuit of length %d'
                     %(param_name, index, length))
