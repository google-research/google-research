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
"""Basic representation of a quantum circuit.

This module contains the classes:
  * Gate: an operation on a group of qubits, where the qubits not yet specified
    explicitly
  * Operation: a Gate applied to a specific group of qubits
  * Circuit: a sequence of Operations
These classes are intended to serve as the basis for a quantum circuit
optimization framework.
"""

import abc
import functools
import itertools
from typing import Iterable, Sequence, Tuple, Union

import numpy as np
from scipy.spatial import transform


class Circuit:
  """A quantum circuit.

  Circuit represents a sequence of operations.

  Note that here the operations are not grouped into moments (in contrast to the
  cirq.Circuit class). Doing so would make things more difficult for the
  intended task of circuit optimization.

  The Circuit class is immutable.
  """

  def __init__(self,
               num_qubits,
               operation_sequence,
               skip_checks=False):
    """Initializes a new Circuit.

    Args:
        num_qubits: the number of qubits on which this circuit acts. The qubits
            for all operations must be in range(num_qubits).
        operation_sequence: the operations in the circuit. Can also be set to
            None which yields an empty circuit.
        skip_checks: a flag indicating whether the consistency checks should be
            skipped (defaults to False). Passing skip_checks == True puts the
            responsibility for guaranteeing validity on the caller. This
            argument had to be introduced due to avoid runtime bottlenecks.

    Raises:
        TypeError: if num_qubits is not an integer type of if operation_sequence
            is not an iterable over Operation instances.
        ValueError: if the qubits of the operations are inconsistent with
            num_qubits.
    """
    num_qubits = _cast_to_int(num_qubits, 'num_qubits')

    if operation_sequence is None:
      operation_sequence = ()
    else:
      operation_sequence = tuple(operation_sequence)

      if not skip_checks:
        if not all(isinstance(operation, Operation)
                   for operation in operation_sequence):
          illegal_type_names = sorted(set(
              type(operation).__name__
              for operation in operation_sequence
              if not isinstance(operation, Operation)
          ))
          raise TypeError('found illegal type(s) in operation_sequence: %s '
                          '(expected: Operation)'%', '.join(illegal_type_names))

        if any(max(operation.get_qubits()) >= num_qubits
               for operation in operation_sequence):
          qubits = np.unique(list(itertools.chain.from_iterable(
              operation.get_qubits() for operation in operation_sequence
          )))
          raise ValueError(
              'illegal qubit indices: %s [expected from range(%d)]'%(
                  ', '.join(str(qubit)
                            for qubit in qubits[qubits >= num_qubits]),
                  num_qubits
              )
          )

    self._operation_sequence = operation_sequence
    self._num_qubits = num_qubits

  def get_num_qubits(self):
    return self._num_qubits

  def get_operation_sequence(self):
    return self._operation_sequence

  def operation(self, key):
    """Extracts an operation from this circuit.

    Args:
        key: the index of the operation to extract.

    Returns:
        the selected operation.

    Raises:
        TypeError: if key is not integer-like.
    """
    return self._operation_sequence[_cast_to_int(key, 'key')]

  def depth(self):
    """Computes the depth of this circuit.

    The depth is the number of steps which would be required to execute this
    circuit on a quantum device, taking into account that operations can be
    executed in parallel if they act on distinct sets of qubits.

    Returns:
        the depth of this circuit.
    """
    depth, _ = self.schedule()
    return depth

  def schedule(self):
    """Schedules the operations in this circuit.

    See also: Circuit.depth()

    Returns:
        depth: the depth of this circuit.
        moments: a NumPy array containing the moments for each operation, i.e.
            moment[op_idx] specifies the moment in which the op_idx-th operation
            is scheduled.
    """

    moments = []
    moment_counters = np.full(self.get_num_qubits(), 0)

    for operation in self:
      # To determine the earliest moment into which an operation can be
      # inserted, we consider the latest moments so far for all qubits involved,
      # pick the maximum (bottleneck principle: can't launch the operation
      # before every qubit is available) and increment this number by 1. Then,
      # we set the highest moment for all involved qubits to this value.
      #
      # For illustration, consider the following example:
      #
      #     qubit A:   -@-----        -@-Z---        -@-Z---@-
      #                 |              |              |     |
      #     qubit B:   -@-@-Y-   ->   -@-@-Y-   ->   -@-@-Y-@-
      #                   |              |              |
      #     qubit C:   -X-@---        -X-@---        -X-@-----
      #
      #     moment:     1 2 3          1 2 3          1 2 3 4
      #
      # In the first step, the Z gate acting on qubit A can be inserted in
      # moment 2 because the latest operation on qubit A was in moment 1. For
      # this, it is irrelevant that qubits B and C are already at moment 3 and
      # 2, respectively. In the second step, the CZ gate acting on qubits A and
      # B needs to be inserted in moment 4 because the latest operation on qubit
      # B was in moment 3, even though qubit A would have allowed an earlier
      # execution.
      qubits = np.array(operation.get_qubits())
      moment = np.max(moment_counters[qubits])
      moments.append(moment)
      moment_counters[qubits] = moment + 1

    moments = np.array(moments)

    # again the bottleneck principle: the latest of all moments determines the
    # depth of the circuit
    depth = int(np.max(moment_counters))

    return depth, moments

  def __add__(self, other):
    """Concatenates two circuits.

    Args:
        other: the circuit to append after this circuit.

    Returns:
        a new Circuit instance consisting of the operations of this (1st part)
        and the other (2nd part) circuit.

    Raises:
        TypeError: if other is not a Circuit.
        ValueError: if the number of qubits for the two circuits does not match.
    """
    if not isinstance(other, Circuit):
      raise TypeError('illegal type for other: %s (expected a Circuit)'
                      %type(other).__name__)

    self_num_qubits = self.get_num_qubits()
    other_num_qubits = other.get_num_qubits()

    if self_num_qubits != other_num_qubits:
      raise ValueError('number of qubits does not match (%d vs %d)'
                       %(self_num_qubits, other_num_qubits))

    return Circuit(
        self_num_qubits,
        self.get_operation_sequence() + other.get_operation_sequence(),
        skip_checks=True
    )

  def __len__(self):
    """Returns the number of operations in this circuit."""
    return len(self._operation_sequence)

  def __iter__(self):
    """Iterates over all operations in this circuit."""
    return iter(self._operation_sequence)

  def __getitem__(self, keys):
    """Index operator.

    Can be used to extract an operation or a subcircuit. Examples for use cases:

        circ[3]           --> the Operation at index 3
        circ[5:8]         --> a Circuit containing the operations in the index
                              slice 5:8
        circ[[1, 3, 6]]   --> a Circuit containing the operations at indices
                              1, 3, 6

    Supports also multiple of these keys. If so, the specified parts are
    concatenated (the result is always a Circuit then). For example,
    `circ[:5, 10:20]` is equivalent to `circ[:5] + circ[10:20]`.

    Args:
        keys: indices, slices, or iterables of indices to indicate which parts
            of the circuit to extract.

    Returns:
        the selected operation if keys is a single index, or otherwise a Circuit
        containing the selected operations.

    Raises:
      TypeError: if the type of key is not supported. Allowed types are
          integer-like objects, slices and iterables of int, as well as tuples
          of those.
    """

    # The structure of this method with its many branches is mainly due to
    # runtime constraints.

    if isinstance(keys, tuple):
      operations = []

      for key in keys:
        try:
          # EAFP principle: easier to ask for forgiveness than permission
          key = key.__index__()
        except (AttributeError, TypeError):
          if isinstance(key, slice):
            operations.extend(self._operation_sequence[key])
          elif hasattr(key, '__getitem__'):
            try:
              operations.extend(self._operation_sequence[k.__index__()]
                                for k in key)
            except (AttributeError, TypeError):
              raise TypeError('unsupported key type: %s'%type(key).__name__)
          else:
            raise TypeError('unsupported key type: %s'%type(key).__name__)
        else:
          operations.append(self._operation_sequence[key])
      return Circuit(self._num_qubits, operations, skip_checks=True)
    else:
      try:
        # EAFP principle: easier to ask for forgiveness than permission
        key = keys.__index__()
      except (AttributeError, TypeError):
        if isinstance(keys, slice):
          return self.extract_slice(keys)
        elif hasattr(keys, '__getitem__'):
          try:
            return self.subcircuit(keys)
          except TypeError:
            raise TypeError('unsupported key type: %s'%type(keys).__name__)
        else:
          raise TypeError('unsupported key type: %s'%type(keys).__name__)
      else:
        return self.operation(key)

  def extract_slice(self, key):
    """Constructs a new Circuit by extracting a slice from this circuit.

    For a slice named key, `circ.extract_slice(key)` behaves equivalent to
    `circ[key]`. The latter is more "user-friendly" and also more powerful, but
    it is slower. Calling extract_slice is recommended if (and only if) runtime
    is crucial.

    Args:
        key: a slice specifying the operations to extract from this circuit.

    Returns:
        a new instance of Circuit containing the selected operations.

    Raises:
        TypeError: if key is not a slice.
    """
    if not isinstance(key, slice):
      raise TypeError('key is not a slice (found type: %s)'%type(key).__name__)
    return Circuit(
        self._num_qubits,
        self._operation_sequence[key],
        skip_checks=True
    )

  def subcircuit(self, key):
    """Constructs a new Circuit by extracting selected operations from this circuit.

    For an iterable of indices named key, `circ.subcircuit(key)` behaves
    equivalent to `circ[key]`. The latter is more "user-friendly" and also more
    powerful, but it is slower. Calling subcircuit is recommended if (and only
    if) runtime is crucial.

    Args:
        key: an iterable of indices specifying the operations to extract from
        this circuit.

    Returns:
        a new instance of Circuit containing the selected operations.

    Raises:
        TypeError: if key is not an iterable of int.
    """
    try:
      operations = [self._operation_sequence[k.__index__()] for k in key]
    except (AttributeError, TypeError):
      raise TypeError('key is not an iterable of int (found type: %s)'
                      %type(key).__name__)
    return Circuit(self._num_qubits, operations, skip_checks=True)


class Operation:
  """A quantum operation.

  An Operation is a gate applied on concrete qubits.

  The Operation class is immutable.
  """

  def __init__(self, gate, qubits):
    """Initializes a new Operation.

    Args:
        gate: the gate to be performed.
        qubits: the qubit indices which the gate acts on.

    Raises:
        TypeError: if gate is not a Gate or qubits not a sequence of int.
        ValueError: if qubits contains negative or duplicate values, or if the
            length of qubits does not match the number of qubits for the gate.
    """
    qubits = tuple(_cast_to_int(qubit, 'qubit') for qubit in qubits)

    if not isinstance(gate, Gate):
      raise TypeError('gate is not a Gate (found type: %s)'%type(gate).__name__)
    if len(qubits) != len(set(qubits)):
      raise ValueError('qubits %s contain duplicate values'%str(qubits))
    if any(qubit < 0 for qubit in qubits):
      illegal_qubits = sorted(set(qubit for qubit in qubits if qubit < 0))
      raise ValueError('illegal qubit indices: %s (must be non-negative)'
                       %', '.join(str(qubit) for qubit in illegal_qubits))
    if len(qubits) != gate.get_num_qubits():
      raise ValueError(
          'num_qubits of gate does not match len(qubits) [%d vs %d]'
          %(gate.get_num_qubits(), len(qubits))
      )

    self._gate = gate
    self._qubits = qubits

  def get_gate(self):
    return self._gate

  def get_qubits(self):
    return self._qubits

  def get_num_qubits(self):
    return len(self._qubits)

  def replace_gate(self, gate):
    """Replaces the gate for this Operation.

    This method is equivalent to `Operation(gate, operation.get_qubits())`, but
    it may be more convenient.

    Args:
        gate: the gate to be used instead.

    Returns:
        a new Operation instance with the specified gate which acts on the same
        qubits as self.

    Raises:
        TypeError: if gate is not a Gate.
        ValueError: if the number of qubits for the gate does not match the
            number of qubits for this Operation.
    """
    return Operation(gate, self.get_qubits())

  def replace_qubits(self, qubits):
    """Replaces the qubits for this Operation.

    This method is equivalent to `Operation(operation.get_gate(), qubits)`, but
    it may be more convenient.

    Args:
        qubits: the qubits to be used instead.

    Returns:
        a new Operation instance with the same gate as self which acts on the
        specified qubits.

    Raises:
        TypeError: if qubits not a sequence of int.
        ValueError: if qubits contains negative or duplicate values, or if the
            length of qubits does not match the number of qubits for this
            Operation.
    """
    return Operation(self.get_gate(), qubits)

  def permute_qubits(self,
                     permutation,
                     inverse=False):
    """Constructs an operation with permuted roles of the qubits.

    Args:
        permutation: the permutation specifying how the qubits change roles.
        inverse: whether permutation is to be interpreted inversely or not.

    Returns:
        an operation with permuted roles of the qubits.

    Raises:
        ValueError: if permutation does not have a valid shape or is not
            actually a permutation, or if its length does not match the number
            of qubits of this gate.
    """

    permutation = np.array(permutation).astype(int, casting='safe')
    num_qubits = self.get_num_qubits()

    if np.array_equal(permutation, np.arange(num_qubits)):  # trivial perm.
      return self  # spares creation of a new object

    _check_permutation(permutation, num_qubits)

    if inverse:
      permutation = np.argsort(permutation)

    return self.replace_qubits(np.array(self.get_qubits())[permutation])

  @functools.lru_cache(maxsize=10000)
  def commutes_trivially_with(self, other):
    """Checks whether this operation trivially commutes with another operation.

    To operations are defined to commute trivially if they act on a disjoint set
    of qubits.

    Args:
        other: the other operation.

    Returns:
        a bool indicating whether this operation trivially commutes with the
        other one.

    Raises:
        TypeError: if other is not an Operation.
    """
    if not isinstance(other, Operation):
      raise TypeError('unexpected type for other: %s (expected an Operation)'
                      %type(other).__name__)

    return set(self.get_qubits()).isdisjoint(other.get_qubits())

  @functools.lru_cache(maxsize=10000)
  def cancels_with(self,
                   other,
                   phase_invariant=False,
                   **kwargs):
    """Checks whether two operations cancel each other.

    Args:
        other: the other operation.
        phase_invariant: specifies whether the complex phase should be taken
            into account. This makes a difference if this gate equals
            exp(i*phi)*other^dagger where exp(i*phi) is a non-trivial complex
            phase. For such a pair of gates, cancels_with(...) will return False
            (True) if phase_invariant==False (phase_invariant==True).
        **kwargs: keyword arguments passed to Gate.cancels_with(...).

    Returns:
        a bool indicating whether this operation cancels the other one.

    Raises:
        TypeError: if other is not an Operation.
    """
    if not isinstance(other, Operation):
      raise TypeError('unexpected type for other: %s (expected an Operation)'
                      %type(other).__name__)

    if self.commutes_trivially_with(other):
      return (self.get_gate().is_identity(phase_invariant=phase_invariant)
              and other.get_gate().is_identity(phase_invariant=phase_invariant))
    else:
      self_gate, other_gate = self._fit_together(self, other)
      return self_gate.cancels_with(
          other_gate,
          phase_invariant=phase_invariant,
          **kwargs
      )

  @functools.lru_cache(maxsize=10000)
  def commutes_with(self,
                    other,
                    *args, **kwargs):
    """Checks whether two gates commute with each other.

    Args:
        other: the other operation.
        *args: positional arguments passed to Gate.cancels_with(...).
        **kwargs: keyword arguments passed to Gate.cancels_with(...).

    Returns:
        a bool indicating whether this gate commutes with the other one.

    Raises:
        TypeError: if other is not an Operation.
    """
    if not isinstance(other, Operation):
      raise TypeError('unexpected type for other: %s (expected an Operation)'
                      %type(other).__name__)

    if self.commutes_trivially_with(other):
      return True
    else:
      self_gate, other_gate = self._fit_together(self, other)
      return self_gate.commutes_with(other_gate, *args, **kwargs)

  @staticmethod
  def _fit_together(*operations):
    """Brings the gates for a group of operations into a common format.

    To "bring into a common format" means to achieve conformity for the qubit
    indices on the level of the class Gate.

    For example,

        _fit_together(
            Operation(single_qubit_gate, [1]),
            Operation(three_qubit_gate, [0, 1, 2]),
            Operation(two_qubit_gate, [0, 1])
        )

    would return a list of three gates equivalent to

        [
            eye ⊗ single_qubit_gate ⊗ eye,
            three_qubit_gate,
            two_qubit_gate ⊗ eye,
        ]

    where ⊗ is the Kronecker product. In this example, the qubits of the
    operations (2nd argument in the initializer) were chosen such that they are
    sorted and do not skip any intermediate index. The only reason behind this
    is to keep the example simple; _fit_together(...) works correctly without
    these assumptions.

    Args:
        *operations: the group of operations.

    Returns:
        a list of gates.
    """

    # extract the essential information from the operations
    operations = [
        (operation.get_qubits(), operation.get_gate())
        for operation in operations
    ]

    # construct the union set of the qubits for all operations (and sort it)
    old_qubits = sorted(set(itertools.chain.from_iterable(
        qubits for qubits, _ in operations
    )))
    num_old_qubits = len(old_qubits)

    # map the old qubit indices (which might be scattered anywhere between 0 and
    # the total number of qubits) to new qubit indices which are packed
    # compactly between 0 and num_old_qubits; so, the computational space into
    # which the gates are extended is restricted to the active qubits, which
    # avoids to unnecessarily blow up this computational space to potentially
    # the total number of qubits in the circuit
    reassign_qubits = {
        old_qubit: new_qubit
        for new_qubit, old_qubit in enumerate(old_qubits)
    }.get

    # bring the gates into a common format
    return [
        gate.apply_on(tuple(map(reassign_qubits, qubits)), num_old_qubits)
        for qubits, gate in operations
    ]


class Gate(abc.ABC):
  """Representation of a unitary quantum gate.

  A unitary quantum gate is a unitary operation applied to a group of qubits.

  This class is the base class for all gates. Subclasses should implement the
  following method:
    * get_operator

  Gate and all its subclasses are supposed to be immutable.
  """

  def __init__(self, num_qubits):
    """Initializes a new Gate.

    Args:
        num_qubits: the number of qubits this gate acts on. This number must
            correspond to the dimension of the get_operator() and
            get_pauli_transform() methods.

    Raises:
        TypeError: if num_qubits is not an int.
    """

    if type(num_qubits) != int:  # want only int and not any possible subtype, so pylint: disable=unidiomatic-typecheck
      raise TypeError('illegal type for num_qubits: %s (must be int)'
                      %type(num_qubits).__name__)

    self._num_qubits = num_qubits

  def get_num_qubits(self):
    """Returns the number of qubits this gate acts on.

    Returns:
        the number of qubits this gate acts on.
    """
    return self._num_qubits

  @abc.abstractmethod
  def get_operator(self):
    """Returns the unitary operator represented by this gate.

    Returns:
        the unitary operator represented by this gate.
    """
    pass

  def get_pauli_transform(self):
    """Returns the Pauli transform matrix for this gate.

    See also: compute_pauli_transform()

    Returns:
        a matrix encoding the action of this gate on the Pauli operators.
    """
    return compute_pauli_transform(self.get_operator())

  def is_identity(self, phase_invariant=False, **kwargs):
    """Checks whether this gate is identity.

    Args:
        phase_invariant: specifies whether the complex phase should be taken
            into account. This makes a difference for operations of the form
            exp(i*phi)*identity where exp(i*phi) is a non-trivial complex phase.
            is_identity(...) for such a gate will return False (True) if
            phase_invariant==False (phase_invariant==True).
        **kwargs: keyword arguments passed to np.isclose(...) or
            np.allclose(...).

    Returns:
        a bool indicating whether this gate is identity.
    """
    if phase_invariant:
      return np.allclose(
          self.get_pauli_transform(),
          np.eye(4 ** self.get_num_qubits() - 1),
          **kwargs
      )
    else:
      return np.allclose(
          self.get_operator(),
          np.eye(2 ** self.get_num_qubits()),
          **kwargs
      )

  def cancels_with(self,
                   other,
                   phase_invariant=False,
                   **kwargs):
    """Checks whether two gates cancel each other.

    Args:
        other: the other gate.
        phase_invariant: specifies whether the complex phase should be taken
            into account. This makes a difference if this gate equals
            exp(i*phi)*other^dagger where exp(i*phi) is a non-trivial complex
            phase. For such a pair of gates, cancels_with(...) will return False
            (True) if phase_invariant==False (phase_invariant==True).
        **kwargs: keyword arguments passed to np.isclose(...) or
            np.allclose(...).

    Returns:
        a bool indicating whether this gate cancels the other one.

    Raises:
        TypeError: if other is not a Gate.
        ValueError: if the number of qubits does not match.
    """
    self._check_compatible_gate(other, 'cancellation')

    if phase_invariant:
      # pauli_transforms are orthogonal, so the transposed matrix is the inverse
      return np.allclose(
          self.get_pauli_transform(),
          other.get_pauli_transform().T,
          **kwargs
      )
    else:
      # operators are unitary, so the adjoint matrix is the inverse
      return np.allclose(
          self.get_operator(),
          other.get_operator().T.conj(),
          **kwargs
      )

  def commutes_with(self,
                    other,
                    phase_invariant=False,
                    **kwargs):
    """Checks whether two gates commute with each other.

    Args:
        other: the other gate.
        phase_invariant: specifies whether to check a more general commutation
            rule which considers two gates as commuting if exchanging them
            induces just a complex phase. This makes a difference if

                self * other == exp(i * phi) * other * self

            where exp(i*phi) is a non-trivial complex phase; in particular, this
            includes the case of anti-commutation [exp(i * phi) == -1]. For such
            a pair of gates, commutes_with(...) will return False (True) if
            phase_invariant==False (phase_invariant==True).
        **kwargs: keyword arguments passed to np.isclose(...) or
            np.allclose(...).

    Returns:
        a bool indicating whether this gate commutes with the other one.

    Raises:
        TypeError: if other is not a Gate.
        ValueError: if the number of qubits does not match.
    """
    self._check_compatible_gate(other, 'commutation')

    if phase_invariant:
      self_mat = self.get_pauli_transform()
      other_mat = other.get_pauli_transform()
    else:
      self_mat = self.get_operator()
      other_mat = other.get_operator()

    return np.allclose(
        np.dot(self_mat, other_mat),
        np.dot(other_mat, self_mat),
        **kwargs
    )

  def permute_qubits(self, permutation, inverse=False):
    """Constructs a gate with permuted roles of the qubits.

    See also: permute_qubits()

    Args:
        permutation: the permutation specifying how the qubits change roles.
        inverse: whether permutation is to be interpreted inversely or not.

    Returns:
        a gate with permuted roles of the qubits. This method can return any
        subtype of Gate which can be different from the input type.

    Raises:
        ValueError: if permutation does not have a valid shape or is not
            actually a permutation, or if its length does not match the number
            of qubits for this gate.
    """
    if np.array_equal(permutation, np.arange(self.get_num_qubits())):
      # permutation is trivial
      return self  # conserves type, and spares creation of a new object
    else:
      return MatrixGate(permute_qubits(
          self.get_operator(),
          permutation,
          inverse=inverse
      ))

  def apply_on(self,
               active_qubits,
               total_num_qubits):
    """Applies this gate on a new set of qubits.

    This can mean that the roles of the qubits get permuted, that extra qubits
    are added, or a combination of both.

    What this method does can be understood easiest by considering a gate
    representing a product operator (note that this method works correctly
    without this assumption!)

        U = kron(U_0, U_1, ..., U_{num_active_qubits - 1})

    where num_active_qubits == self.get_num_qubits(). In this case, it will
    return the operator

        U' = kron(U'_0, U'_1, ..., U'_{total_num_qubits - 1})

    such that

        U'_{active_qubits[n]} == U_n
        U'_m                  == eye

    for all m in set(range(total_num_qubits)).difference(active_qubits).

    Explicitly, for num_active_qubits == 3, i.e. U = kron(U_0, U_1, U_2), we get
    the following results:

    * if active_qubits == [0, 1, 2] and total_num_qubits == 3:
          U' = kron(U_0, U_1, U_2)
    * if active_qubits == [1, 2, 0] and total_num_qubits == 3:
          U' = kron(U_2, U_0, U_1)
    * if active_qubits == [0, 1, 2] and total_num_qubits == 4:
          U' = kron(U_0, U_1, U_2, eye)
    * if active_qubits == [1, 2, 0] and total_num_qubits == 4:
          U' = kron(U_2, U_0, U_1, eye)
    * if active_qubits == [0, 3, 2] and total_num_qubits == 4:
          U' = kron(U_0, eye, U_2, U_1)
    * if active_qubits == [3, 0, 2] and total_num_qubits == 4:
          U' = kron(U_1, eye, U_2, U_0)

    Args:
        active_qubits: specifies the new index roles for the active qubits.
        total_num_qubits: the total number of qubits to be considered.

    Returns:
        a Gate instance which represents the desired operation. This method can
        return any subtype of Gate which can be different from the input type.

    Raises:
        TypeError: if active_qubits is not a sequence of int or total_num_qubits
            is not an integer type.
        ValueError: if active_qubits is invalid or not consistent with the
            number of qubits for this gate, or if total_num_qubits is too small.
    """
    active_qubits = np.array(active_qubits).astype(int, casting='safe')
    if active_qubits.ndim != 1:
      raise TypeError('active_qubits is not a sequence of int [shape: %s]'
                      %str(active_qubits.shape))

    self_num_qubits = self.get_num_qubits()
    total_num_qubits = _cast_to_int(total_num_qubits, 'total_num_qubits')

    if total_num_qubits < self_num_qubits:
      raise ValueError('number of qubits cannot be reduced (from %d to %d)'
                       %(self_num_qubits, total_num_qubits))
    if active_qubits.size != self_num_qubits:
      raise ValueError('illegal length for active_qubits: %d (expected: %d)'
                       %(active_qubits.size, self_num_qubits))
    if not (np.min(active_qubits) >= 0
            and np.max(active_qubits) < total_num_qubits
            and len(set(active_qubits)) == self_num_qubits):
      raise ValueError('illegal value for active_qubits: %s'
                       ' (expected a %d-length permutation of range(%d))'
                       %(active_qubits, self_num_qubits, total_num_qubits))

    # In the following, we will actually construct the gate to be returned
    # according to the specification in the docstring. This process is performed
    # in two steps:
    #
    # 1. If active_qubits is not already in normal (ascending) order, construct
    #    a new gate with permuted indices such that this new gate acting on
    #    sorted(active_qubits) is equivalent to self acting on active_qubits.
    # 2. If there are additional qubits, we now construct a new gate with its
    #    operation being extended on these qubits such that their quantum state
    #    remains untouched.
    #
    # Example: Let's consider (again for simplicity) for the input the product
    # operator
    #
    #     kron(A, B, C).
    #          0  1  2    <-- qubit indices
    #
    # For active_qubits == [3, 0, 2] and total_num_qubits == 5, the end result
    # will be
    #
    #     kron(B, eye, C, A, eye).
    #          0   1   2  3   4    <-- qubit indices
    #
    # Between step 1 and 2, we have -- as an intermediate value -- a gate which
    # still represents the operation on the active qubits only, but with these
    # active qubits sorted in the same way as in the final version of the gate:
    #
    #     kron(B, C, A).
    #          0  1  2    <-- qubit indices

    # step 1: permute the qubits if necessary
    #
    # Note: The permute_qubits(...) method is itself responsible for detecting
    # cases in which self can be returned. This helps to prevent the creation of
    # new objects if avoidable. In particular, gate can be self if active_qubits
    # is already sorted, but there are more difficult cases and it is much
    # easier to consistently treat them all in the permute_qubits(...) method.
    gate = self.permute_qubits(np.argsort(active_qubits))

    # step 2: extend the operation on additional qubits if necessary
    #
    # again, create a new object only if necessary
    if self_num_qubits == total_num_qubits:
      return gate
    else:
      return MatrixGate(extend_operator(
          gate.get_operator(),
          np.isin(np.arange(total_num_qubits), active_qubits)
      ))

  def __eq__(self, other, **kwargs):
    if not isinstance(other, Gate):
      return False
    if self.get_num_qubits() != other.get_num_qubits():
      return False
    return np.allclose(
        self.get_pauli_transform(),
        other.get_pauli_transform(),
        **kwargs
    )

  def _check_compatible_gate(self, other, relation_name):
    """Checks whether other is a Gate with the same number of qubits.

    Args:
        other: the other gate.
        relation_name: the name for the relation to be display in an error
            message.

    Raises:
        TypeError: if other is not a Gate.
        ValueError: if the number of qubits does not match.
    """

    if not isinstance(other, Gate):
      raise TypeError('unexpected type for other: %s (expected a Gate)'
                      %type(other).__name__)
    if self.get_num_qubits() != other.get_num_qubits():
      raise ValueError(
          '%s relation not well-defined because the number of qubits does not'
          ' match (%d vs %d)'
          %(relation_name, self.get_num_qubits(), other.get_num_qubits())
      )


class MatrixGate(Gate):
  """A unitary gate defined directly by its operator.

  As a subclass of Gate, MatrixGate is immutable.
  """

  def __init__(self, operator):
    """Initializes a new MatrixGate.

    Args:
        operator: np.ndarray with shape (2 ** num_qubits, 2 ** num_qubits) and
            dtype complex.
            The unitary operator represented by this gate.

    Raises:
        ValueError: if operator does not have a valid shape or if it is not
            unitary.
    """

    operator = np.array(operator, dtype=complex)
    num_qubits = _analyse_operator(operator)
    _check_unitary(operator, num_qubits)

    super().__init__(num_qubits)
    self._operator = operator
    self._pauli_transform = compute_pauli_transform(operator)

  def get_operator(self):
    """Returns the unitary operator represented by this gate.

    Implements abstract method from parent class Gate. This implementation
    returns a copy of the array that has been specified when initializing this
    MatrixGate.

    See also: Gate.get_operator()

    Returns:
        the unitary operator represented by this gate.
    """
    return self._operator.copy()

  def get_pauli_transform(self):
    """Returns the Pauli transform matrix for this gate.

    Overrides method from parent class Gate. This implementation looks up the
    value that has been computed during initialization of this MatrixGate.

    See also: Gate.get_pauli_transform()

    Returns:
        a matrix encoding the action of this gate on the Pauli operators.
    """
    return self._pauli_transform.copy()


class PhasedXGate(Gate):
  """A single-qubit gate that performs a rotation around a logical axis on the equator.

  The gate is represented by the unitary operator

      U = exp(i*pa*pauli_z) * exp(i*ra*pauli_x) * exp(-i*pa*pauli_z)

  where ra is the rotation_angle and pa is the phase_angle. Here, * for matrices
  is the matrix product and exp() is the matrix exponential.

  The circuit which describes this gate therefore reads:

      ───exp(-i*pa*pauli_z)───exp(i*ra*pauli_x)───exp(i*pa*pauli_z)───

  Note that the order in the circuit is reversed w.r.t. the formula above. This
  can be motivated as follows: the leftmost operator in the circuit is the first
  one to be applied to the state (which is a vector multiplied from the right),
  so it must be the rightmost factor in the operator product.

  As a subclass of Gate, PhasedXGate is immutable.
  """

  def __init__(self, rotation_angle, phase_angle):
    """Initializes a new PhasedXGate.

    Args:
        rotation_angle: the rotation angle (in radians).
        phase_angle: the angle between the rotation axis and the x axis (in
            radians).

    Raises:
        TypeError: if rotation_angle or phase_angle is not a float and cannot be
            casted to a float.
    """
    super().__init__(num_qubits=1)

    self._rotation_angle = float(rotation_angle)
    self._phase_angle = float(phase_angle)

  def get_rotation_angle(self):
    """Returns the rotation angle.

    Returns:
        the rotation angle (in radians).
    """
    return self._rotation_angle

  def get_phase_angle(self):
    """Returns the angle between the rotation axis and the x axis.

    Returns:
        the angle between the rotation axis and the x axis (in radians).
    """
    return self._phase_angle

  def shift_rotation_angle(self, added_angle):
    """Constructs a PhasedXGate with a shifted rotation angle.

    `phased_x_gate.shift_rotation_angle(added_angle)` is a convenience shortcut
    for `circuit.PhasedXGate(phased_x_gate.get_rotation_angle() + added_angle,
    phased_x_gate.get_phase_angle())`.

    Args:
        added_angle: the angle to be added to the rotation angle.

    Returns:
        a PhasedXGate whose rotation angle equals `self.get_rotation_angle() +
        added_angle`, with the same phase angle.

    Raises:
        TypeError: if added_angle is not a float and cannot be casted to a
            float.
    """
    return PhasedXGate(
        self.get_rotation_angle() + added_angle,
        self.get_phase_angle()
    )

  def shift_phase_angle(self, added_angle):
    """Constructs a PhasedXGate with a shifted phase angle.

    `phased_x_gate.shift_phase_angle(added_angle)` is a convenience shortcut for
    `circuit.PhasedXGate(phased_x_gate.get_rotation_angle(),
    phased_x_gate.get_phase_angle() + added_angle)`.

    Args:
        added_angle: the angle to be added to the phase angle.

    Returns:
        a PhasedXGate whose phase angle equals `self.get_phase_angle() +
        added_angle`, with the same rotation angle.

    Raises:
        TypeError: if added_angle is not a float and cannot be casted to a
            float.
    """
    return PhasedXGate(
        self.get_rotation_angle(),
        self.get_phase_angle() + added_angle
    )

  @classmethod
  def flip_x(cls):
    """Constructs a PhasedXGate representing a Pauli-X gate.

    Returns:
        a new PhasedXGate instance representing a Pauli-X gate.
    """
    return cls(np.pi, 0.0)

  @classmethod
  def flip_y(cls):
    """Constructs a PhasedXGate representing a Pauli-Y gate.

    Returns:
        a new PhasedXGate instance representing a Pauli-Y gate.
    """
    return cls(np.pi, 0.5 * np.pi)

  @classmethod
  def rot_x(cls, rotation_angle):
    """Constructs a PhasedXGate representing a rotation around the logical x axis.

    Args:
        rotation_angle: the rotation angle around the x axis (in radians).

    Returns:
        a new PhasedXGate instance representing a rotation around the logical x
        axis.

    Raises:
        TypeError: if added_angle is not a float and cannot be casted to a
            float.
    """
    return cls(rotation_angle, 0.0)

  @classmethod
  def rot_y(cls, rotation_angle):
    """Constructs a PhasedXGate representing a rotation around the logical y axis.

    Args:
        rotation_angle: the rotation angle around the y axis (in radians).

    Returns:
        a new PhasedXGate instance representing a rotation around the logical y
        axis.

    Raises:
        TypeError: if added_angle is not a float and cannot be casted to a
            float.
    """
    return cls(rotation_angle, 0.5 * np.pi)

  def get_operator(self):
    # implements method from parent class Gate
    rotation_angle = self.get_rotation_angle()
    phase_angle = self.get_phase_angle()

    return np.array([
        [
            np.cos(0.5 * rotation_angle),
            -1.0j * np.exp(-1.0j * phase_angle) * np.sin(0.5 * rotation_angle)
        ],
        [
            -1.0j * np.exp(1.0j * phase_angle) * np.sin(0.5 * rotation_angle),
            np.cos(0.5 * rotation_angle)
        ]
    ])

  def get_pauli_transform(self):
    # overrides method from parent class Gate
    #
    # This implementation avoids to construct the operator which might be a
    # little more efficient.
    rotation_angle = self.get_rotation_angle()
    phase_angle = self.get_phase_angle()

    return transform.Rotation.from_euler(
        'zxz',
        [-phase_angle, rotation_angle, phase_angle]
    ).as_dcm()

  def is_identity(self, phase_invariant=False, **kwargs):
    # overrides method from parent class Gate
    #
    # This implementation avoids to construct the operator/pauli_transform which
    # might be a little more efficient.
    if phase_invariant:
      criterion = np.exp(1.0j * self.get_rotation_angle())
    else:
      criterion = np.cos(0.5 * self.get_rotation_angle())
    return bool(np.isclose(criterion, 1.0, **kwargs))


class RotZGate(Gate):
  """A single-qubit gate that performs a rotation around the logical z axis.

  As a subclass of Gate, RotZGate is immutable.
  """

  def __init__(self, rotation_angle):
    """Initializes a new RotZGate.

    Args:
        rotation_angle: the rotation angle around the z axis (in radians).

    Raises:
        TypeError: if rotation_angle is not a float and cannot be casted to a
            float.
    """
    super().__init__(num_qubits=1)

    self._rotation_angle = float(rotation_angle)

  def get_rotation_angle(self):
    """Returns the rotation angle around the z axis.

    Returns:
        the rotation angle around the z axis (in radians).
    """
    return self._rotation_angle

  def shift_rotation_angle(self, added_angle):
    """Constructs a RotZGate with a shifted rotation angle.

    `rot_z_gate.shift_rotation_angle(added_angle)` is a convenience shortcut for
    `circuit.RotZGate(rot_z_gate.get_rotation_angle() + added_angle)`.

    Args:
        added_angle: the angle to be added to the rotation angle.

    Returns:
        a RotZGate whose rotation angle equals `self.get_rotation_angle() +
        added_angle`.

    Raises:
        TypeError: if added_angle is not a float and cannot be casted to a
            float.
    """
    return RotZGate(self.get_rotation_angle() + added_angle)

  def get_operator(self):
    # implements method from parent class Gate
    return np.diag([1.0, np.exp(1.0j * self.get_rotation_angle())])

  def get_pauli_transform(self):
    # overrides method from parent class Gate
    #
    # This implementation avoids to construct the operator which might be a
    # little more efficient.
    rotation_angle = self.get_rotation_angle()
    return transform.Rotation.from_euler('z', rotation_angle).as_dcm()

  def is_identity(self, phase_invariant=False, **kwargs):
    # overrides method from parent class Gate
    #
    # This implementation avoids to construct the operator which might be a
    # little more efficient.
    return bool(np.isclose(
        np.exp(1.0j * self.get_rotation_angle()),
        1.0,
        **kwargs
    ))


class ControlledZGate(Gate):
  """A Controlled-Z gate (or CZ gate for short).

  The Controlled-Z gate induces a phase flip iff both qubits are in state 1:

      |00>  ─>   |00>
      |01>  ─>   |01>
      |10>  ─>   |10>
      |11>  ─>  -|11>

  As a subclass of Gate, ControlledZGate is immutable.
  """

  def __init__(self):
    super().__init__(num_qubits=2)

  def get_operator(self):
    # implements method from parent class Gate
    return np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)

  def is_identity(self, phase_invariant=False, **kwargs):
    # overrides method from parent class Gate
    return False

  def permute_qubits(self, permutation, inverse=False):
    # overrides method from parent class Gate
    _check_permutation(np.array(permutation).astype(int, casting='safe'), 2)
    # Controlled-Z is symmetric.
    return self


def compute_pauli_transform(operator):
  """Compute the action of a unitary operator on the Pauli operators.

  We consider a density matrix rho under a unitary transformation U (* denotes
  the matrix product):

      rho -> rho' = U * rho * U^dagger

  Since rho and rho' are Hermitian, they can be written as a linear combination
  of the Pauli operators. For example, in the single-qubit case, we have:

      rho  = 1/2 * (identity + x  * pauli_x + y  * pauli_y + z  * pauli_z)
      rho' = 1/2 * (identity + x' * pauli_x + y' * pauli_y + z' * pauli_z)

  The return value of compute_pauli_transform(U) is the matrix which transforms
  the coefficients (x, y, z) into (x', y', z'):

      [x']   [                 ]   [x]
      [y'] = [ pauli_transform ] * [y]
      [z']   [                 ]   [z]

  This also works for more than one qubit. The only aspect that changes is the
  number of Pauli operators and therefore also the number of coefficients.

  This Pauli transform matrix defines a gate in a phase-invariant way, which can
  simplify some analysis tasks for a gate, and makes it easier to compare
  different gates.

  Args:
      operator: np.ndarray with shape (2 ** num_qubits, 2 ** num_qubits) and
          dtype complex.
          The unitary operator acting on a group of qubits.

  Returns:
      a matrix encoding the transformation of Pauli operators under the
      specified operation.

  Raises:
      ValueError: if operator does not have a valid shape or if it is not
          unitary.
  """

  operator = np.array(operator)
  num_qubits = _analyse_operator(operator)
  _check_unitary(operator, num_qubits)

  # define 1-qubit Pauli group
  pauli_1 = np.array([
      [[1.0, 0.0], [0.0, 1.0]],     # identity
      [[0.0, 1.0], [1.0, 0.0]],     # pauli_x
      [[0.0, -1.0j], [1.0j, 0.0]],  # pauli_y
      [[1.0, 0.0], [0.0, -1.0]]     # pauli_z
  ])

  # construct multi-qubit Pauli group
  #
  # e.g. if num_qubits == 2, will be sorted as
  #
  #     pauli_n[0] = kron(identity, identity)
  #     pauli_n[1] = kron(identity, pauli_x)
  #     pauli_n[2] = kron(identity, pauli_y)
  #     pauli_n[3] = kron(identity, pauli_z)
  #     pauli_n[4] = kron(pauli_x, identity)
  #     pauli_n[5] = kron(pauli_x, pauli_x)
  #        ...               ...
  pauli_n = pauli_1
  for _ in range(num_qubits-1):
    pauli_n = np.kron(pauli_n, pauli_1)

  # Sets pauli_transform[j,k] to
  #
  #     <dot(pauli_n[j+1], operator), dot(operator, pauli_n[k+1])>
  #     ----------------------------------------------------------
  #                            2 ** num_qubits
  #
  # where <.,.> is the Hilbert-Schmidt product. The first element of pauli_n
  # which is the n-qubit identity does not need to be included because it is
  # always mapped to identity.
  pauli_transform = 0.5 ** num_qubits * np.tensordot(
      np.matmul(pauli_n[1:], operator),
      np.matmul(operator, pauli_n[1:]).conj(),
      axes=[(1, 2), (1, 2)]
  )

  assert np.allclose(np.imag(pauli_transform), 0.0)

  pauli_transform = np.real(pauli_transform)

  return pauli_transform


def permute_qubits(operator,
                   permutation,
                   inverse=False):
  """Permutes the roles of the qubits.

  Transforms the operator U as:

      U -> U' = Perm * U * Perm^dagger

  Perm is a unitary operator that encodes the permutation of the qubits.
  Depending on the inverse argument of this method, Perm has one of the
  following forms:

      normal:  Perm = sum_{x in {0, 1}^n} |x[permutation]><x| =
                    = sum_{x in {0, 1}^n} |x><x[inv_permutation]|
      inverse: Perm = sum_{x in {0, 1}^n} |x[inv_permutation]><x| =
                    = sum_{x in {0, 1}^n} |x><x[permutation]|

  with `inv_permutation = np.argsort(permutation)`.

  example: U = kron(U_0, U_1, U_2, U_3, U_4), permutation=[3, 4, 1, 2, 0]
    +> normal:  U' = kron(U_3, U_4, U_1, U_2, U_0)
    +> inverse: U' = kron(U_4, U_2, U_3, U_0, U_1)

  Args:
      operator: np.ndarray with shape (2 ** num_qubits, 2 ** num_qubits) and
          dtype complex.
          The unitary operator.
      permutation: the permutation specifying how the qubits change roles.
      inverse: whether permutation is to be interpreted inversely or not.

  Returns:
      the operator with permuted roles of the qubits.

  Raises:
      ValueError: if operator does not have a valid shape or if it is not
          unitary, if permutation does not have a valid shape or is not actually
          a permutation, or if the length of permutation does not match the
          number of qubits for the operator.
  """

  permutation = np.array(permutation).astype(int, casting='safe')
  operator = np.array(operator, dtype=complex)

  num_qubits = _analyse_operator(operator)
  _check_unitary(operator, num_qubits)

  if np.array_equal(permutation, np.arange(num_qubits)):  # trivial permutation
    return operator

  _check_permutation(permutation, num_qubits)

  if inverse:
    permutation = np.argsort(permutation)

  # Given a unitary operator U and a permutation perm, we want to construct the
  # operator U' which satisfies
  #
  #     <perm(x)|U'|perm(y)> = <x|U|y>                                       (*)
  #
  # for all bitstrings x=(x_0, ..., x_{N-1}) and y=(y_0, ..., y_{N-1}) where N
  # is the number of qubits; perm(x) is to be read as
  #
  #     perm(x) = (x_{perm[0]}, ..., x_{perm[N-1]}).
  #
  # E.g., for perm == [1, 0], this would mean:
  #
  #     <00|U'|00> = <00|U|00>     <01|U'|01> = <10|U|10>
  #     <10|U'|10> = <01|U|01>     <11|U'|11> = <11|U|11>
  #
  # (*) is equivalent to:
  #
  #     <x|U'|y> = <inv_perm(x)|U|inv_perm(y)>
  #
  # According to the indexing convention for multi-qubit states, the bitstring x
  # corresponds to the index
  #
  #     ind(x) = sum_{n=0}^{N-1} 2^(N-n-1) x_n,
  #
  # and then analogically
  #
  #     ind(inv_perm(x)) = sum_{n=0}^{N-1} 2^(N-n-1) x_{inv_perm[n]} =
  #                      = sum_{n=0}^{N-1} 2^(N-perm[n]-1) x_n
  #
  # Hence, in conclusion, the index map
  #
  #     s: sum_{n=0}^{N-1} 2^(N-perm[n]-1) x_n -> sum_{n=0}^{N-1} 2^(N-n-1) x_n
  #
  # shuffles the matrix elements of U such that we obtain U'. This map is
  # computed in the following: the entries of shuffling satisfy the condition
  #
  #     shuffling[ind(x)] == ind(inv_perm(x)),
  #
  # and so indexing the array operator with shuffling rearranges the entries of
  # operator according to the index map s.
  #
  # Note that the itertools.product call produces all bitstrings in ascending
  # order whose length matches num_qubits; e.g., for num_qubits == 2, we would
  # get (as a list): [(0, 0), (0, 1), (1, 0), (1, 1)].
  shuffling = np.dot(
      list(itertools.product(range(2), repeat=num_qubits)),
      np.left_shift(1, num_qubits - permutation - 1)  # 2^(N-perm[n]-1)
  )

  return operator[tuple(np.meshgrid(shuffling, shuffling, indexing='ij'))]


def extend_operator(operator,
                    is_qubit_active):
  """Extends an operator on additional qubits.

  Args:
      operator: np.ndarray with shape (2 ** num_qubits_in, 2 ** num_qubits_in)
          and dtype complex.
          The operator acting on the original set of qubits.
      is_qubit_active: a sequence of bool with length num_qubits_out. The n-th
          element specifies whether the n-th qubit is active (True) or not
          (False). Exactly num_qubits_in elements must be set to True.

  Returns:
      the operator acting on the extended set of qubits.

  Raises:
      TypeError: if is_qubit_active is not a sequence of bool.
      ValueError: if operator does not have a valid shape or if is_qubit_active
          does not match the number of qubits for the operator.
  """

  operator = np.array(operator, dtype=complex)

  is_qubit_active = np.array(is_qubit_active)
  try:
    is_qubit_active = is_qubit_active.astype(bool, casting='safe')
  except TypeError:
    raise TypeError('is_qubit_active is not a sequence of bool [%s cannot be'
                    r' casted safely to bool]'%is_qubit_active.dtype)
  if is_qubit_active.ndim != 1:
    raise TypeError('is_qubit_active is not a sequence of bool (found: ndim=%d)'
                    %is_qubit_active.ndim)

  num_qubits_in = _analyse_operator(operator)
  num_qubits_out = is_qubit_active.size

  if np.sum(is_qubit_active) != num_qubits_in:
    raise ValueError(
        '%d active qubit(s) not compatible with operator dimension %d [i.e. %d '
        'qubit(s)]'%(np.sum(is_qubit_active), 2 ** num_qubits_in, num_qubits_in)
    )

  if num_qubits_in == num_qubits_out:
    return operator

  active_qubits, = is_qubit_active.nonzero()
  start_active = active_qubits[0]
  end_active = active_qubits[-1] + 1

  if start_active + num_qubits_in == end_active:
    # The active qubits are not separated. This case is important because it is
    # automatically satisfied if num_qubits_in == 1.
    # Special treatment for (mediocre critical) speed-up.

    if start_active > 0:
      operator = np.kron(np.eye(2 ** start_active), operator)
    if end_active < num_qubits_out:
      operator = np.kron(operator, np.eye(2 ** (num_qubits_out - end_active)))

    return operator
  else:
    # Using the Kronecker product, we extend the operator by a passive component
    # that performs the identity operation on the remaining qubits. To be able
    # to combine these two components into one operator, we have to bring them
    # into a common format where every qubit corresponds to an own array
    # dimension, or, to be more precise, a pair of array dimensions to represent
    # also the rows and columns of the operator.
    #
    # The arrangement of these dimensions is
    #
    #     row_0, row_1, ..., row_{N-1}, col_0, col_1, ..., col_{N-1}
    #
    # where N == num_qubits_out. The index n in row_n and col_n corresponds to
    # the qubit indices. This is this array format on which we perform the
    # Kronecker product, and then we can just reshape the result into the common
    # matrix format, i.e. (row, col).
    #
    # The proper way to bring the active component into this format is to
    # reshape it such that shape[n] == shape[n+N] == 2 if the n-th qubit is
    # active, otherwise shape[n] == shape[n+N] == 1. We can obtain each half of
    # shape from is_qubit_active + 1, and then use np.tile(...) to duplicate
    # this sequence. For the passive component, the proceeding is similar: we
    # reshape np.eye(...) such that shape[n] ==shape[n+N] == 1 if the n-th qubit
    # is active, otherwise shape[n] == shape[n+N] == 2. We can obtain each half
    # of shape from 2 - is_qubit_active, and then again duplicate this using
    # np.tile(...).
    #
    # Example: Let's assume that is_qubit_active == [True, False, False, True].
    # Then, we first set
    #
    #    active_qubit_operator = operator.reshape([2, 1, 1, 2, 2, 1, 1, 2])
    #    passive_qubit_operator = np.eye(4).reshape([1, 2, 2, 1, 1, 2, 2, 1])
    #
    # np.kron(active_qubit_operator, passive_qubit_operator) yields an array of
    # shape (2, 2, 2, 2, 2, 2, 2, 2), and reshaping this to (16, 16) gives the
    # operator that we are looking for.

    active_qubit_operator = operator.reshape(np.tile(is_qubit_active + 1, 2))
    passive_qubit_operator = np.eye(2 ** (num_qubits_out - num_qubits_in)) \
                             .reshape(np.tile(2 - is_qubit_active, 2))

    return np.kron(active_qubit_operator, passive_qubit_operator) \
           .reshape(np.full(2, 2 ** num_qubits_out))


def _analyse_operator(operator, param_name='operator'):
  """Checks the properties of an operator and extracts the number of qubits.

  Args:
    operator: the operator to be analysed.
    param_name: the parameter name as displayed in potential error messages.

  Returns:
    returns: number of qubits.

  Raises:
      ValueError: if operator does not have a valid shape or if it is not
          unitary.
  """

  if operator.ndim != 2:
    raise ValueError(
        '%s must be a 2D array (found: ndim=%d)'
        %(param_name, operator.ndim)
    )

  rows, cols = operator.shape

  if rows != cols:
    raise ValueError(
        '%s must be a square matrix [found: shape=(%d, %d)]'
        %(param_name, rows, cols)
    )

  num_qubits = rows.bit_length()-1

  if rows != 2 ** num_qubits:
    raise ValueError(
        'dimension of %s must be a power of 2 (found: dim=%d)'
        %(param_name, rows)
    )

  return num_qubits


def _check_unitary(operator, num_qubits, param_name='operator'):
  """Checks whether an operator is unitary.

  Args:
      operator: the operator to be checked.
      num_qubits: the number of qubits for the operator.
      param_name: the parameter name as displayed in potential error messages.

  Raises:
      ValueError: if operator is not unitary.
  """
  if not np.allclose(np.dot(operator, operator.T.conj()),
                     np.eye(2 ** num_qubits)):
    raise ValueError('%s is not unitary'%param_name)


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


def _check_permutation(permutation, length):
  """Checks the properties of a permutation.

  Args:
    permutation: the permutation (candidate).
    length: the length of the permutation.

  Raises:
      ValueError: permutation does not have a valid shape or is not a actually
          a permutation.
  """
  if not np.array_equal(permutation.shape, [length]):
    raise ValueError('illegal shape for permutation: %s [expected: (%d,)]'
                     %(permutation.shape, length))
  if not np.array_equal(np.sort(permutation), np.arange(length)):
    raise ValueError('not a valid permutation: %s'%permutation)
