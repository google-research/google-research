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

# Lint as: python3
"""Tests for transform."""

import itertools
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
from scipy import stats

from rl4circopt import circuit
from rl4circopt import transform


def _random_operation(*qubits):
  return circuit.Operation(
      circuit.MatrixGate(stats.unitary_group.rvs(2 ** len(qubits))),
      qubits
  )


def _elementwise_is(sequence_a, sequence_b):
  sequence_a = tuple(sequence_a)
  sequence_b = tuple(sequence_b)

  if len(sequence_a) == len(sequence_b):
    return all(
        elem_a is elem_b
        for elem_a, elem_b in zip(sequence_a, sequence_b)
    )
  else:
    return False


class PointTransformationTest(parameterized.TestCase):

  def test_initializer_and_getters(self):
    # preparation work
    focus_in = [
        _random_operation(1, 2)
    ]
    context_in = transform.TransformationContext(
        circuit.Circuit(5, None),
        circuit.Circuit(5, None),
        circuit.Circuit(5, None)
    )
    locations_in = (0,)
    perform_rule = lambda operation_in: [operation_in]
    rule_id = object()

    # initialize the PointTransformation
    transformation = transform.PointTransformation(
        attention_circ=transform.AttentionCircuit(
            focus=focus_in,
            context=context_in,
            locations=locations_in
        ),
        perform_rule=perform_rule,
        rule_id=rule_id
    )

    # check type and value transformation.focus()
    focus_out = transformation.focus()
    self.assertIs(type(focus_out), tuple)
    self.assertTrue(_elementwise_is(focus_out, focus_in))

    # check transformation.context()
    self.assertIs(transformation.context(), context_in)

    # check type and value transformation.locations()
    locations_out = transformation.locations()
    self.assertIs(type(locations_out), tuple)
    self.assertTrue(_elementwise_is(locations_out, locations_in))

    # check transformation.context()
    self.assertIs(transformation.rule_id(), rule_id)

  def test_initializer_circ_type_error(self):
    circ = circuit.Circuit(5, None)
    perform_rule = lambda operation_in: [operation_in]

    with self.assertRaisesRegex(
        TypeError,
        r'attention_circ is not an AttentionCircuit \(found type: Circuit\)'):
      transform.PointTransformation(circ, perform_rule, None)

  def test_initializer_focus_size_error(self):
    attention_circ = transform.AttentionCircuit(
        focus=[_random_operation(1), _random_operation(3)],
        context=transform.TransformationContext(
            circuit.Circuit(5, None),
            circuit.Circuit(5, None),
            circuit.Circuit(5, None)
        )
    )
    perform_rule = lambda operation_in: [operation_in]

    with self.assertRaisesRegex(
        ValueError,
        r'focus of attention_circ for PointTransformation must have size 1'
        r' \(found: 2\)'):
      transform.PointTransformation(attention_circ, perform_rule, None)

  def test_perform(self):
    # preparation work
    num_qubits = 5

    operation_before1 = _random_operation(0)
    operation_before2 = _random_operation(1)
    operation_before3 = _random_operation(2)
    operation_between1 = _random_operation(0)
    operation_between2 = _random_operation(1)
    operation_after = _random_operation(1)

    operation_in = _random_operation(1)
    operation_out1 = _random_operation(1)
    operation_out2 = _random_operation(1)

    # define the transformation rule
    def perform_rule(operation):
      # check operation
      self.assertIs(operation, operation_in)

      # return the modified operations
      return [operation_out1, operation_out2]

    # initialize the PointTransformation
    transformation = transform.PointTransformation(
        attention_circ=transform.AttentionCircuit(
            focus=[operation_in],
            context=transform.TransformationContext(
                circuit.Circuit(num_qubits, [
                    operation_before1,
                    operation_before2,
                    operation_before3
                ]),
                circuit.Circuit(num_qubits, [
                    operation_between1,
                    operation_between2
                ]),
                circuit.Circuit(num_qubits, [
                    operation_after
                ])
            )
        ),
        perform_rule=perform_rule,
        rule_id=None
    )

    # call the method to be tested
    circ_out = transformation.perform()

    # check type for circ_out
    self.assertIs(type(circ_out), circuit.Circuit)

    # check the value for circ_out
    self.assertTrue(_elementwise_is(
        circ_out.get_operation_sequence(),
        [
            operation_before1,
            operation_before2,
            operation_before3,
            operation_out1,
            operation_out2,
            operation_between1,
            operation_between2,
            operation_after
        ]
    ))


class PairTransformationTest(parameterized.TestCase):

  def test_initializer_and_getters(self):
    # preparation work
    focus_in = [
        _random_operation(1, 2),
        _random_operation(2, 3)
    ]
    context_in = transform.TransformationContext(
        circuit.Circuit(5, None),
        circuit.Circuit(5, None),
        circuit.Circuit(5, None)
    )
    locations_in = (0, 1)
    perform_rule = lambda operation_in: [operation_in]
    rule_id = object()

    # initialize the PairTransformation
    transformation = transform.PairTransformation(
        attention_circ=transform.AttentionCircuit(
            focus=focus_in,
            context=context_in,
            locations=locations_in
        ),
        perform_rule=perform_rule,
        rule_id=rule_id
    )

    # check type and value transformation.focus()
    focus_out = transformation.focus()
    self.assertIs(type(focus_out), tuple)
    self.assertTrue(_elementwise_is(focus_out, focus_in))

    # check transformation.context()
    self.assertIs(transformation.context(), context_in)

    # check type and value transformation.locations()
    locations_out = transformation.locations()
    self.assertIs(type(locations_out), tuple)
    self.assertTrue(_elementwise_is(locations_out, locations_in))

    # check transformation.context()
    self.assertIs(transformation.rule_id(), rule_id)

  def test_initializer_circ_type_error(self):
    circ = circuit.Circuit(5, None)
    perform_rule = lambda operation_in: [operation_in]

    with self.assertRaisesRegex(
        TypeError,
        r'attention_circ is not an AttentionCircuit \(found type: Circuit\)'):
      transform.PairTransformation(circ, perform_rule, None)

  def test_initializer_focus_size_error(self):
    attention_circ = transform.AttentionCircuit(
        focus=[_random_operation(2)],
        context=transform.TransformationContext(
            circuit.Circuit(5, None),
            circuit.Circuit(5, None),
            circuit.Circuit(5, None)
        )
    )
    perform_rule = lambda operation_in: [operation_in]

    with self.assertRaisesRegex(
        ValueError,
        r'focus of attention_circ for PairTransformation must have size 2'
        r' \(found: 1\)'):
      transform.PairTransformation(attention_circ, perform_rule, None)

  def test_initializer_redundant_transformation_error(self):
    attention_circ = transform.AttentionCircuit(
        focus=[_random_operation(2, 3), _random_operation(5)],
        context=transform.TransformationContext(
            circuit.Circuit(5, None),
            circuit.Circuit(5, None),
            circuit.Circuit(5, None)
        )
    )
    perform_rule = lambda operation_in: [operation_in]

    with self.assertRaisesRegex(
        ValueError,
        r'transformation redundant because operations trivially commute'):
      transform.PairTransformation(attention_circ, perform_rule, None)

  def test_perform(self):
    # preparation work
    num_qubits = 5

    operation_before1 = _random_operation(0)
    operation_before2 = _random_operation(1)
    operation_before3 = _random_operation(2)
    operation_between1 = _random_operation(0)
    operation_between2 = _random_operation(1)
    operation_after = _random_operation(1)

    operation_in1 = _random_operation(1)
    operation_in2 = _random_operation(1)
    operations_out_first1 = _random_operation(1)
    operations_out_first2 = _random_operation(1)
    operations_out_second1 = _random_operation(1)
    operations_out_second2 = _random_operation(1)
    operations_out_second3 = _random_operation(1)

    # define the transformation rule
    def perform_rule(operation_first, operation_second):
      # check operation_first and operation_second
      self.assertIs(operation_first, operation_in1)
      self.assertIs(operation_second, operation_in2)

      # return the modified operations
      operations_out_first = [
          operations_out_first1,
          operations_out_first2
      ]
      operations_out_second = [
          operations_out_second1,
          operations_out_second2,
          operations_out_second3
      ]
      return operations_out_first, operations_out_second

    # initialize the PointTransformation
    transformation = transform.PairTransformation(
        attention_circ=transform.AttentionCircuit(
            focus=[operation_in1, operation_in2],
            context=transform.TransformationContext(
                circuit.Circuit(num_qubits, [
                    operation_before1,
                    operation_before2,
                    operation_before3
                ]),
                circuit.Circuit(num_qubits, [
                    operation_between1,
                    operation_between2
                ]),
                circuit.Circuit(num_qubits, [
                    operation_after
                ])
            )
        ),
        perform_rule=perform_rule,
        rule_id=None
    )

    # call the method to be tested
    circ_out = transformation.perform()

    # check type for circ_out
    self.assertIs(type(circ_out), circuit.Circuit)

    # check the value for circ_out
    self.assertTrue(_elementwise_is(
        circ_out.get_operation_sequence(),
        [
            operation_before1,
            operation_before2,
            operation_before3,
            operations_out_first1,
            operations_out_first2,
            operation_between1,
            operation_between2,
            operations_out_second1,
            operations_out_second2,
            operations_out_second3,
            operation_after
        ]
    ))


class GroupTransformationTest(parameterized.TestCase):

  def test_initializer_and_getters(self):
    # preparation work
    focus_in = [
        _random_operation(1, 2),
        _random_operation(2, 3),
        _random_operation(3, 4)
    ]
    context_in = transform.TransformationContext(
        circuit.Circuit(5, None),
        circuit.Circuit(5, None),
        circuit.Circuit(5, None)
    )
    locations_in = (0, 1, 2)
    perform_rule = lambda operation_in: [operation_in]
    rule_id = object()

    # initialize the PairTransformation
    transformation = transform.GroupTransformation(
        attention_circ=transform.AttentionCircuit(
            focus=focus_in,
            context=context_in,
            locations=locations_in
        ),
        perform_rule=perform_rule,
        rule_id=rule_id
    )

    # check type and value transformation.focus()
    focus_out = transformation.focus()
    self.assertIs(type(focus_out), tuple)
    self.assertTrue(_elementwise_is(focus_out, focus_in))

    # check transformation.context()
    self.assertIs(transformation.context(), context_in)

    # check type and value transformation.locations()
    locations_out = transformation.locations()
    self.assertIs(type(locations_out), tuple)
    self.assertTrue(_elementwise_is(locations_out, locations_in))

    # check transformation.context()
    self.assertIs(transformation.rule_id(), rule_id)

  def test_initializer_circ_type_error(self):
    circ = circuit.Circuit(5, None)
    perform_rule = lambda operation_in: [operation_in]

    with self.assertRaisesRegex(
        TypeError,
        r'attention_circ is not an AttentionCircuit \(found type: Circuit\)'):
      transform.GroupTransformation(circ, perform_rule, None)

  def test_perform(self):
    # preparation work
    num_qubits = 5

    operation_before1 = _random_operation(0)
    operation_before2 = _random_operation(1)
    operation_before3 = _random_operation(2)
    operation_between1 = _random_operation(0)
    operation_between2 = _random_operation(1)
    operation_after = _random_operation(1)

    operation_in1 = _random_operation(1)
    operation_in2 = _random_operation(1)
    operation_in3 = _random_operation(1)
    operation_in4 = _random_operation(1)
    operation_out1 = _random_operation(1)
    operation_out2 = _random_operation(1)
    operation_out3 = _random_operation(1)

    # define the transformation rule
    def perform_rule(operations_in):
      # check type and value for operations_in
      self.assertIs(type(operations_in), tuple)
      self.assertTrue(_elementwise_is(
          operations_in,
          [operation_in1, operation_in2, operation_in3, operation_in4]
      ))

      # return the modified operations
      return [operation_out1, operation_out2, operation_out3]

    # initialize the PointTransformation
    transformation = transform.GroupTransformation(
        attention_circ=transform.AttentionCircuit(
            focus=[operation_in1, operation_in2, operation_in3, operation_in4],
            context=transform.TransformationContext(
                circuit.Circuit(num_qubits, [
                    operation_before1,
                    operation_before2,
                    operation_before3
                ]),
                circuit.Circuit(num_qubits, [
                    operation_between1,
                    operation_between2
                ]),
                circuit.Circuit(num_qubits, [
                    operation_after
                ])
            )
        ),
        perform_rule=perform_rule,
        rule_id=None
    )

    # call the method to be tested
    circ_out = transformation.perform()

    # check type for circ_out
    self.assertIs(type(circ_out), circuit.Circuit)

    # check the value for circ_out
    self.assertTrue(_elementwise_is(
        circ_out.get_operation_sequence(),
        [
            operation_before1,
            operation_before2,
            operation_before3,
            operation_out1,
            operation_out2,
            operation_out3,
            operation_between1,
            operation_between2,
            operation_after
        ]
    ))


class AttentionCircuitTest(parameterized.TestCase):

  def test_initializer_with_locations_and_getters(self):
    # preparation work
    focus_in = (
        _random_operation(1, 2),
        _random_operation(1)
    )
    focus_length = len(focus_in)
    context = transform.TransformationContext(
        circuit.Circuit(5, [
            _random_operation(2),
            _random_operation(3),
            _random_operation(4)
        ]),
        circuit.Circuit(5, [
            _random_operation(0)
        ]),
        circuit.Circuit(5, [
            _random_operation(0),
            _random_operation(1, 2)
        ])
    )
    locations_in = (3, 5)

    # construct the AttentionCircuit
    att_circ = transform.AttentionCircuit(
        focus_in,
        context,
        locations=locations_in
    )

    # check type and value for att_circ.focus()
    focus_out = att_circ.focus()
    self.assertIs(type(focus_out), tuple)
    self.assertTrue(_elementwise_is(focus_out, focus_in))

    # check att_circ.context()
    self.assertIs(att_circ.context(), context)

    # check type and value for att_circ.locations()
    locations_out = att_circ.locations()
    self.assertIs(type(locations_out), tuple)
    self.assertTrue(_elementwise_is(locations_out, locations_in))

    # check type and value for len(att_circ)
    length = len(att_circ)
    self.assertIs(type(length), int)
    self.assertEqual(length, focus_length)

  def test_initializer_without_locations_and_getters(self):
    # preparation work
    focus_in = (
        _random_operation(1, 2),
        _random_operation(1)
    )
    focus_length = len(focus_in)
    context = transform.TransformationContext(
        circuit.Circuit(5, [
            _random_operation(2),
            _random_operation(3),
            _random_operation(4)
        ]),
        circuit.Circuit(5, [
            _random_operation(0)
        ]),
        circuit.Circuit(5, [
            _random_operation(0),
            _random_operation(1, 2)
        ])
    )

    # construct the AttentionCircuit
    att_circ = transform.AttentionCircuit(
        focus_in,
        context,
        locations=None
    )

    # check type and value for att_circ.focus()
    focus_out = att_circ.focus()
    self.assertIs(type(focus_out), tuple)
    self.assertTrue(_elementwise_is(focus_out, focus_in))

    # check att_circ.context()
    self.assertIs(att_circ.context(), context)

    # check that att_circ.locations() is None
    self.assertIsNone(att_circ.locations())

    # check type and value for len(att_circ)
    length = len(att_circ)
    self.assertIs(type(length), int)
    self.assertEqual(length, focus_length)

  @parameterized.parameters([
      [42, r'\'int\' object is not iterable'],
      [[42], r'only Operation objects allowed in focus \(found types: int\)']
  ])
  def test_initializer_focus_type_error(self, focus, message):
    context = transform.TransformationContext(
        circuit.Circuit(5, None),
        circuit.Circuit(5, None),
        circuit.Circuit(5, None)
    )

    with self.assertRaisesRegex(TypeError, message):
      transform.AttentionCircuit(focus, context)

  def test_initializer_empty_focus_error(self):
    context = transform.TransformationContext(
        circuit.Circuit(5, None),
        circuit.Circuit(5, None),
        circuit.Circuit(5, None)
    )

    with self.assertRaisesRegex(ValueError, r'focus must not be empty'):
      transform.AttentionCircuit((), context)

  def test_initializer_context_type_error(self):
    focus = (
        _random_operation(1, 2),
        _random_operation(1)
    )

    with self.assertRaisesRegex(
        TypeError,
        r'context is not a TransformationContext \(found type: int\)'):
      transform.AttentionCircuit(focus, 42)

  @parameterized.parameters([
      [42, r'\'int\' object is not iterable'],
      [[47.11], r'location is not integer-like \(found type: float\)']
  ])
  def test_initializer_locations_type_error(self, locations, message):
    focus = (
        _random_operation(1, 2),
        _random_operation(1)
    )
    context = transform.TransformationContext(
        circuit.Circuit(5, None),
        circuit.Circuit(5, None),
        circuit.Circuit(5, None)
    )

    with self.assertRaisesRegex(TypeError, message):
      transform.AttentionCircuit(focus, context, locations=locations)

  def test_initializer_locations_length_error(self):
    focus = (
        _random_operation(1, 2),
        _random_operation(1)
    )
    context = transform.TransformationContext(
        circuit.Circuit(5, [
            _random_operation(2),
            _random_operation(3),
            _random_operation(4)
        ]),
        circuit.Circuit(5, [
            _random_operation(0)
        ]),
        circuit.Circuit(5, [
            _random_operation(0),
            _random_operation(1, 2)
        ])
    )

    with self.assertRaisesRegex(
        ValueError,
        r'inconsistent lengths for focus and locations: 2 vs. 1'):
      transform.AttentionCircuit(focus, context, locations=(3,))


class TransformationContextTest(parameterized.TestCase):

  def test_initializer_and_getters(self):
    # preparation work: create three circuits before, between and after
    num_qubits = 5
    before = circuit.Circuit(num_qubits, [
        _random_operation(0, 2),
        _random_operation(4),
        _random_operation(1)
    ])
    between = circuit.Circuit(num_qubits, [
        _random_operation(0),
        _random_operation(4)
    ])
    after = circuit.Circuit(num_qubits, [
        _random_operation(0, 1),
        _random_operation(1, 2),
        _random_operation(2, 3, 4)
    ])

    # construct the TransformationContext
    context = transform.TransformationContext(before, between, after)

    # check before, between and after
    self.assertIs(context.before(), before)
    self.assertIs(context.between(), between)
    self.assertIs(context.after(), after)

  @parameterized.parameters([
      [
          42,
          circuit.Circuit(5, None),
          circuit.Circuit(5, None),
          'int, Circuit, Circuit'
      ],
      [
          circuit.Circuit(6, None),
          47.11,
          circuit.Circuit(6, None),
          'Circuit, float, Circuit'
      ],
      [
          circuit.Circuit(7, None),
          circuit.Circuit(7, None),
          'hello',
          'Circuit, Circuit, str'
      ],
  ])
  def test_initializer_type_error(self, before, between, after, type_string):
    with self.assertRaisesRegex(
        TypeError,
        r'before, between and after must be Circuits \(found types: %s\)'
        %type_string):
      transform.TransformationContext(before, between, after)

  @parameterized.parameters([
      [7, 5, 5],
      [8, 4, 8],
      [3, 3, 6],
      [2, 3, 4]
  ])
  def test_initializer_inconsistent_num_qubits_error(self,
                                                     num_before,
                                                     num_between,
                                                     num_after):
    before = circuit.Circuit(num_before, None)
    between = circuit.Circuit(num_between, None)
    after = circuit.Circuit(num_after, None)

    with self.assertRaisesRegex(
        ValueError,
        r'inconsistent number of qubits for before, between and after:'
        r' \(%d, %d, %d\)'%(num_before, num_between, num_after)):
      transform.TransformationContext(before, between, after)

  def test_inject(self):
    # preparation work: create operations
    num_qubits = 5
    operation_a = _random_operation(0)
    operation_b = _random_operation(0, 1)
    operation_c1 = _random_operation(1)
    operation_c2 = _random_operation(1, 2)
    operation_c3 = _random_operation(2)
    operation_d1 = _random_operation(2, 3)
    operation_d2 = _random_operation(3)
    operation_e1 = _random_operation(3, 4)
    operation_e2 = _random_operation(4)

    # preparation work: construct the TransformationContext
    context = transform.TransformationContext(
        circuit.Circuit(num_qubits, [operation_a]),
        circuit.Circuit(num_qubits, [operation_c1, operation_c2, operation_c3]),
        circuit.Circuit(num_qubits, [operation_e1, operation_e2])
    )

    # call the method to be tested
    circ_full = context.inject([operation_b], [operation_d1, operation_d2])

    # check type for circ_full
    self.assertIs(type(circ_full), circuit.Circuit)

    # check value for circ_full
    self.assertTrue(_elementwise_is(
        circ_full.get_operation_sequence(),
        [
            operation_a,
            operation_b,
            operation_c1,
            operation_c2,
            operation_c3,
            operation_d1,
            operation_d2,
            operation_e1,
            operation_e2
        ]
    ))

  @parameterized.parameters([
      [[42], [_random_operation(1, 2)]],
      [[_random_operation(1, 2)], [42]]
  ])
  def test_inject_type_error(self, operations_first, operations_second):
    num_qubits = 4
    context = transform.TransformationContext(
        circuit.Circuit(num_qubits, None),
        circuit.Circuit(num_qubits, None),
        circuit.Circuit(num_qubits, None)
    )

    with self.assertRaisesRegex(
        TypeError,
        r'found illegal type\(s\) in operation_sequence: int \(expected:'
        r' Operation\)'):
      context.inject(operations_first, operations_second)


class FocusSingleOperationTest(parameterized.TestCase):

  @parameterized.parameters([3, -2])  # both locations are equivalent
  def test_successful(self, location):
    # preparation work
    operation0 = _random_operation(0)
    operation1 = _random_operation(0, 1)
    operation2 = _random_operation(1)
    operation3 = _random_operation(1, 2)
    operation4 = _random_operation(2)

    circ = circuit.Circuit(5, [
        operation0,
        operation1,
        operation2,
        operation3,
        operation4
    ])

    # call the function to be tested
    attention_circ = transform.focus_single_operation(circ, location)

    # check type of attention_circ
    self.assertIs(type(attention_circ), transform.AttentionCircuit)

    # check the focus of attention_circ
    self.assertLen(attention_circ, 1)
    self.assertTrue(_elementwise_is(attention_circ.focus(), [operation3]))

    # check the context of attention_circ
    context = attention_circ.context()

    self.assertTrue(_elementwise_is(
        context.before().get_operation_sequence(),
        [operation0, operation1, operation2]
    ))
    self.assertEmpty(context.between())
    self.assertTrue(_elementwise_is(
        context.after().get_operation_sequence(),
        [operation4]
    ))

    # check the locations of attention_circ
    self.assertTupleEqual(attention_circ.locations(), (3,))

  def test_circ_type_error(self):
    with self.assertRaisesRegex(
        TypeError,
        r'circ is not a Circuit \(found type: range\)'):
      transform.focus_single_operation(range(10), 3)

  def test_location_type_error(self):
    circ = circuit.Circuit(5, None)

    with self.assertRaisesRegex(
        TypeError,
        r'location is not integer-like \(found type: float\)'):
      transform.focus_single_operation(circ, 47.11)

  @parameterized.parameters([5, -6])
  def test_location_out_of_bounds_error(self, location):
    circ = circuit.Circuit(3, [
        _random_operation(0),
        _random_operation(0, 1),
        _random_operation(1),
        _random_operation(1, 2),
        _random_operation(2),
    ])

    with self.assertRaisesRegex(
        IndexError,
        r'location %d out of bounds for a Circuit of length 5'%location):
      transform.focus_single_operation(circ, location)


def _positive_example_circuit(*segments_and_operations):
  operations = []
  segments = {
      'focus': [],
      'before': [],
      'between': [],
      'after': []
  }
  max_qubit = 0

  for location, (segment_tag, operation) in enumerate(segments_and_operations):
    operations.append(operation)
    segments[segment_tag].append(location)
    max_qubit = np.maximum(max_qubit, max(operation.get_qubits()))

  circ = circuit.Circuit(max_qubit + 1, operations)

  # <checking that the example circuit makes sense>
  assert len(segments['focus']) == 2
  location_first, location_second = segments['focus']  # length checked in previous line, so pylint: disable=unbalanced-tuple-unpacking

  assert all(
      location_before < location_second
      for location_before in segments['before']
  )
  assert all(
      location_first < location_between < location_second
      for location_between in segments['between']
  )
  assert all(
      location_after > location_first
      for location_after in segments['after']
  )

  pool_to_the_left = [
      location_before
      for location_before in segments['before']
      if location_before > location_first
  ]
  pool_to_the_right = [
      location_after
      for location_after in segments['after']
      if location_after < location_second
  ]

  assert all(
      circ[location_second].commutes_trivially_with(circ[location])
      for location in segments['between'] + pool_to_the_right
  )
  assert all(
      circ[location_first].commutes_trivially_with(circ[location])
      for location in pool_to_the_left + segments['between']
  )
  assert all(
      loc0 < loc1 or circ[loc0].commutes_trivially_with(circ[loc1])
      for loc0, loc1 in itertools.product(pool_to_the_left, segments['between'])
  )
  assert all(
      loc0 < loc1 or circ[loc0].commutes_trivially_with(circ[loc1])
      for loc0, loc1 in itertools.product(pool_to_the_left, pool_to_the_right)
  )
  assert all(
      loc0 < loc1 or circ[loc0].commutes_trivially_with(circ[loc1])
      for loc0, loc1 in itertools.product(segments['between'],
                                          pool_to_the_right)
  )
  # </checking that the example circuit makes sense>

  return circ, transform.AttentionCircuit(
      focus=circ[segments['focus']].get_operation_sequence(),
      context=transform.TransformationContext(
          before=circ[segments['before']],
          between=circ[segments['between']],
          after=circ[segments['after']]
      ),
      locations=segments['focus']
  )


def _positive_focus_operation_pair_examples():
  yield _positive_example_circuit(
      ['focus', _random_operation(0, 1)],
      ['focus', _random_operation(0, 1)]
  )

  yield _positive_example_circuit(
      ['focus', _random_operation(0, 1)],
      ['focus', _random_operation(1, 0)]
  )

  yield _positive_example_circuit(
      ['before', _random_operation(0, 1)],
      ['before', _random_operation(0)],
      ['focus', _random_operation(0, 1)],
      ['focus', _random_operation(0, 1)],
      ['after', _random_operation(1)],
      ['after', _random_operation(0)]
  )

  yield _positive_example_circuit(
      ['focus', _random_operation(1, 2)],
      ['between', _random_operation(0)],
      ['between', _random_operation(3)],
      ['focus', _random_operation(1, 2)]
  )

  yield _positive_example_circuit(
      ['before', _random_operation(0, 1)],
      ['before', _random_operation(1, 2)],
      ['before', _random_operation(0)],
      ['focus', _random_operation(1, 2)],
      ['between', _random_operation(0)],
      ['between', _random_operation(3)],
      ['focus', _random_operation(1, 2)],
      ['after', _random_operation(1)],
      ['after', _random_operation(2)]
  )

  yield _positive_example_circuit(
      ['focus', _random_operation(0, 1)],
      ['before', _random_operation(2, 3)],
      ['between', _random_operation(3, 4)],
      ['focus', _random_operation(1, 2)]
  )

  yield _positive_example_circuit(
      ['focus', _random_operation(2, 3)],
      ['between', _random_operation(0, 1)],
      ['after', _random_operation(1, 2)],
      ['focus', _random_operation(3, 4)]
  )

  yield _positive_example_circuit(
      ['focus', _random_operation(0, 1)],
      ['before', _random_operation(3, 4)],
      ['before', _random_operation(2, 3)],
      ['focus', _random_operation(1, 2)]
  )

  yield _positive_example_circuit(
      ['focus', _random_operation(2, 3)],
      ['after', _random_operation(1, 2)],
      ['after', _random_operation(0, 1)],
      ['focus', _random_operation(3, 4)]
  )

  yield _positive_example_circuit(
      ['focus', _random_operation(0, 1)],
      ['before', _random_operation(2, 3)],
      ['after', _random_operation(0, 3)],
      ['focus', _random_operation(1, 2)]
  )

  for enclosed_operations in itertools.permutations([
      ['before', _random_operation(2)],
      ['between', _random_operation(3)],
      ['after', _random_operation(0)]]):
    yield _positive_example_circuit(*(
        [['focus', _random_operation(0, 1)]] +
        list(enclosed_operations) +
        [['focus', _random_operation(1, 2)]]
    ))

  for enclosed_operations in itertools.permutations([
      ['before', _random_operation(3)],
      ['between', _random_operation(4)],
      ['after', _random_operation(0, 1)]]):
    yield _positive_example_circuit(*(
        [['focus', _random_operation(1, 2)]] +
        list(enclosed_operations) +
        [['focus', _random_operation(2, 3)]]
    ))

  for enclosed_operations in itertools.permutations([
      ['before', _random_operation(2, 3)],
      ['between', _random_operation(4)],
      ['after', _random_operation(0)]]):
    yield _positive_example_circuit(*(
        [['focus', _random_operation(0, 1)]] +
        list(enclosed_operations) +
        [['focus', _random_operation(1, 2)]]
    ))

  for enclosed_operations in itertools.permutations([
      ['before', _random_operation(3, 4)],
      ['between', _random_operation(5)],
      ['after', _random_operation(0, 1)]]):
    yield _positive_example_circuit(*(
        [['focus', _random_operation(1, 2)]] +
        list(enclosed_operations) +
        [['focus', _random_operation(2, 3)]]
    ))


class FocusOperationPairTest(parameterized.TestCase):

  @parameterized.parameters(_positive_focus_operation_pair_examples())
  def test_positive(self, circ, att_circ_expected):
    assert len(att_circ_expected) == 2
    location_first, location_second = att_circ_expected.locations()

    # call the function to be tested
    att_circ = transform.focus_operation_pair(
        circ,
        location_first,
        location_second
    )

    # check the type for att_circ
    self.assertIsInstance(att_circ, transform.AttentionCircuit)

    # check the focus for att_circ
    self.assertLen(att_circ, 2)
    self.assertTrue(_elementwise_is(
        att_circ.focus(),
        att_circ_expected.focus()
    ))

    # check the locations for att_circ
    self.assertTupleEqual(
        att_circ.locations(),
        (location_first, location_second)
    )

    # check the context for att_circ
    self.assertTrue(_elementwise_is(
        att_circ.context().before().get_operation_sequence(),
        att_circ_expected.context().before().get_operation_sequence()
    ))
    self.assertTrue(_elementwise_is(
        att_circ.context().between().get_operation_sequence(),
        att_circ_expected.context().between().get_operation_sequence()
    ))
    self.assertTrue(_elementwise_is(
        att_circ.context().after().get_operation_sequence(),
        att_circ_expected.context().after().get_operation_sequence()
    ))

  @parameterized.parameters([
      [
          circuit.Circuit(1, [
              _random_operation(0),
              _random_operation(0),
              _random_operation(0)
          ]),
          0, 2
      ],
      [
          circuit.Circuit(2, [
              _random_operation(0, 1),
              _random_operation(0),
              _random_operation(0, 1)
          ]),
          0, 2
      ],
      [
          circuit.Circuit(3, [
              _random_operation(0, 1),
              _random_operation(1),
              _random_operation(1, 2)
          ]),
          0, 2
      ],
      [
          circuit.Circuit(3, [
              _random_operation(0, 1),
              _random_operation(0, 2),
              _random_operation(1, 2)
          ]),
          0, 2
      ],
      [
          circuit.Circuit(3, [
              _random_operation(0, 1),
              _random_operation(0, 2),
              _random_operation(1),
              _random_operation(1, 2)
          ]),
          0, 3
      ],
      [
          circuit.Circuit(3, [
              _random_operation(0, 1),
              _random_operation(2),
              _random_operation(0, 2),
              _random_operation(1, 2)
          ]),
          0, 3
      ],
      [
          circuit.Circuit(4, [
              _random_operation(0, 1),
              _random_operation(0, 3),
              _random_operation(2, 3),
              _random_operation(1, 2)
          ]),
          0, 3
      ]
  ])
  def test_negative(self, circ, location_first, location_second):
    with self.assertRaises(transform.OperationsNotAlignedError):
      transform.focus_operation_pair(circ, location_first, location_second)

  def test_circ_type_error(self):
    with self.assertRaisesRegex(
        TypeError,
        r'circ is not a Circuit \(found type: range\)'):
      transform.focus_operation_pair(range(10), 3, 5)

  def test_location_first_type_error(self):
    circ = circuit.Circuit(3, [
        _random_operation(0),
        _random_operation(0, 1),
        _random_operation(1),
        _random_operation(1, 2),
        _random_operation(2)
    ])

    with self.assertRaisesRegex(
        TypeError,
        r'location_first is not integer-like \(found type: float\)'):
      transform.focus_operation_pair(circ, 47.11, 3)

  def test_location_second_type_error(self):
    circ = circuit.Circuit(3, [
        _random_operation(0),
        _random_operation(0, 1),
        _random_operation(1),
        _random_operation(1, 2),
        _random_operation(2)
    ])

    with self.assertRaisesRegex(
        TypeError,
        r'location_second is not integer-like \(found type: float\)'):
      transform.focus_operation_pair(circ, 3, 47.11)

  @parameterized.parameters([5, -6])
  def test_location_first_out_of_bounds_error(self, location_first):
    circ = circuit.Circuit(3, [
        _random_operation(0),
        _random_operation(0, 1),
        _random_operation(1),
        _random_operation(1, 2),
        _random_operation(2),
    ])

    with self.assertRaisesRegex(
        IndexError,
        r'location_first %d out of bounds for a Circuit of length 5'
        %location_first):
      transform.focus_operation_pair(circ, location_first, 3)

  @parameterized.parameters([5, -6])
  def test_location_second_out_of_bounds_error(self, location_second):
    circ = circuit.Circuit(3, [
        _random_operation(0),
        _random_operation(0, 1),
        _random_operation(1),
        _random_operation(1, 2),
        _random_operation(2),
    ])

    with self.assertRaisesRegex(
        IndexError,
        r'location_second %d out of bounds for a Circuit of length 5'
        %location_second):
      transform.focus_operation_pair(circ, 3, location_second)

  @parameterized.parameters([
      [4, 3],
      [-1, 3],
      [4, -2],
      [-1, -2]
  ])
  def test_locations_not_sorted_error(self, location_first, location_second):
    circ = circuit.Circuit(3, [
        _random_operation(0),
        _random_operation(0, 1),
        _random_operation(1),
        _random_operation(1, 2),
        _random_operation(2),
    ])

    with self.assertRaisesRegex(
        ValueError,
        r'location_first not smaller than location_second:'
        r' 4 \(or -1\) vs 3 \(or -2\)'):
      transform.focus_operation_pair(circ, location_first, location_second)


class FocusLocalGroupTest(parameterized.TestCase):

  @parameterized.parameters([
      # all locations are equivalent
      [(2, 3, 5)],
      [(2, -5, 5)],
      [(-6, -5, -3)],
  ])
  def test_successful(self, locations):
    # preparation work: create the operations and the circuit
    operation0 = _random_operation(1)
    operation1 = _random_operation(0, 1)
    operation2 = _random_operation(1)
    operation3 = _random_operation(1)
    operation4 = _random_operation(2, 3)
    operation5 = _random_operation(1)
    operation6 = _random_operation(0)
    operation7 = _random_operation(0, 1)

    circ = circuit.Circuit(4, [
        operation0,
        operation1,
        operation2,
        operation3,
        operation4,
        operation5,
        operation6,
        operation7
    ])

    # call the function to be tested
    attention_circ = transform.focus_local_group(circ, locations)

    # check type of attention_circ
    self.assertIs(type(attention_circ), transform.AttentionCircuit)

    # check the focus of attention_circ
    self.assertLen(attention_circ, 3)
    self.assertTrue(_elementwise_is(
        attention_circ.focus(),
        [operation2, operation3, operation5]
    ))

    # check the context of attention_circ
    context = attention_circ.context()
    self.assertTrue(_elementwise_is(
        context.before().get_operation_sequence(),
        [operation0, operation1]
    ))
    self.assertTrue(_elementwise_is(
        context.between().get_operation_sequence(),
        [operation4]
    ))
    self.assertTrue(_elementwise_is(
        context.after().get_operation_sequence(),
        [operation6, operation7]
    ))

    # check the locations of attention_circ
    self.assertTupleEqual(attention_circ.locations(), (2, 3, 5))

  def test_circ_type_error(self):
    with self.assertRaisesRegex(
        TypeError,
        r'circ is not a Circuit \(found type: range\)'):
      transform.focus_local_group(range(10), [3, 4])

  def test_location_type_error(self):
    circ = circuit.Circuit(5, [
        _random_operation(0),
        _random_operation(0, 1),
        _random_operation(1),
        _random_operation(1, 2)
    ])

    with self.assertRaisesRegex(
        TypeError,
        r'location is not integer-like \(found type: float\)'):
      transform.focus_local_group(circ, [2, 47.11])

  def test_locations_empty_error(self):
    circ = circuit.Circuit(3, [
        _random_operation(0),
        _random_operation(0, 1),
        _random_operation(1),
        _random_operation(1, 2),
        _random_operation(2)
    ])

    with self.assertRaisesRegex(ValueError, r'locations must not be empty'):
      transform.focus_local_group(circ, [])

  def test_duplicate_locations_error(self):
    circ = circuit.Circuit(3, [
        _random_operation(0),
        _random_operation(0, 1),
        _random_operation(1),
        _random_operation(1),
        _random_operation(2)
    ])

    with self.assertRaisesRegex(
        ValueError,
        r'locations contains duplicate elements'):
      transform.focus_local_group(circ, [2, 2, 3])

  def test_nonlocal_operations_error(self):
    circ = circuit.Circuit(3, [
        _random_operation(0),
        _random_operation(1),
        _random_operation(1),
        _random_operation(1, 2),
        _random_operation(2)
    ])

    with self.assertRaisesRegex(
        ValueError,
        r'focus contains non-local operations'):
      transform.focus_local_group(circ, [1, 2, 3])

  def test_not_the_same_qubit_error(self):
    circ = circuit.Circuit(3, [
        _random_operation(0),
        _random_operation(1),
        _random_operation(2),
        _random_operation(1),
        _random_operation(2)
    ])

    with self.assertRaisesRegex(
        ValueError,
        r'operations in the focus act on different qubits'):
      transform.focus_local_group(circ, [1, 2, 3])

  @parameterized.parameters([
      [(1, 2, 5), 5],
      [(1, -3, 5), 5],
      [(-5, 2, 5), 5],
      [(-6, 2, 3), -6],
      [(-6, -3, 3), -6],
      [(-6, -3, -2), -6]
  ])
  def test_location_out_of_bounds_error(self, locations, illegal_location):
    circ = circuit.Circuit(3, [
        _random_operation(0),
        _random_operation(1),
        _random_operation(1),
        _random_operation(1),
        _random_operation(2)
    ])

    with self.assertRaisesRegex(
        IndexError,
        r'location %d out of bounds for a Circuit of length 5'
        %illegal_location):
      transform.focus_local_group(circ, locations)

  def test_operations_not_aligned_error(self):
    circ = circuit.Circuit(4, [
        _random_operation(0),
        _random_operation(1),
        _random_operation(1),
        _random_operation(1, 3),
        _random_operation(1),
        _random_operation(2)
    ])

    with self.assertRaises(transform.OperationsNotAlignedError):
      transform.focus_local_group(circ, [1, 2, 4])


if __name__ == '__main__':
  absltest.main()
