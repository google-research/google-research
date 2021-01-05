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

"""Tests for schema_io."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy

from absl.testing import absltest
from absl.testing import parameterized
import six

from tunas import schema
from tunas import schema_io


# First style of decorator use: decorating a namedtuple subclass.
@schema_io.register_namedtuple(
    'schema_io_test.NamedTuple1',
    deprecated_names=['schema_io_test.DeprecatedNamedTuple1'])
class NamedTuple1(collections.namedtuple('NamedTuple1', ['foo'])):
  pass


# Second style of decorator use: registering a raw namedtuple.
NamedTuple2 = collections.namedtuple('NamedTuple2', ['bar'])
schema_io.register_namedtuple('schema_io_test.NamedTuple2')(NamedTuple2)


# Decorator with default argument values.
@schema_io.register_namedtuple(
    'schema_io_test.NamedTuple3',
    deprecated_names=['schema_io_test.DeprecatedNamedTuple3'],
    defaults={'foo': 3, 'bar': 'hi'})
class NamedTuple3(collections.namedtuple('NamedTuple3', ['foo', 'bar'])):
  pass


class SchemaIoTest(parameterized.TestCase):

  def test_register_namedtuple_exceptions(self):
    with self.assertRaisesRegex(ValueError, 'Duplicate name'):
      cls = collections.namedtuple('namedtuple3', ['baz'])
      schema_io.register_namedtuple('schema_io_test.NamedTuple1')(cls)

    with self.assertRaisesRegex(ValueError, 'Duplicate class'):
      schema_io.register_namedtuple('NewNameForTheSameClass')(NamedTuple1)

    with self.assertRaisesRegex(ValueError, 'not a namedtuple'):
      schema_io.register_namedtuple('NotANamedTuple')(dict)

  def test_namedtuple_class_to_name(self):
    self.assertEqual(
        schema_io.namedtuple_class_to_name(NamedTuple1),
        'schema_io_test.NamedTuple1')
    self.assertEqual(
        schema_io.namedtuple_class_to_name(NamedTuple2),
        'schema_io_test.NamedTuple2')

  def test_namedtuple_class_to_name_not_registered(self):
    cls = collections.namedtuple('cls', ['x'])
    with self.assertRaisesRegex(
        KeyError, 'Namedtuple class .* is not registered'):
      schema_io.namedtuple_class_to_name(cls)

  def test_namedtuple_name_to_class_not_registered(self):
    with self.assertRaisesRegex(
        KeyError, 'Namedtuple name \'blahblah\' is not registered'):
      schema_io.namedtuple_name_to_class('blahblah')

  def test_namedtuple_name_to_class(self):
    self.assertEqual(
        schema_io.namedtuple_name_to_class('schema_io_test.NamedTuple1'),
        NamedTuple1)
    self.assertEqual(
        schema_io.namedtuple_name_to_class('schema_io_test.NamedTuple2'),
        NamedTuple2)

  def test_namedtuple_deprecated_name_to_class(self):
    self.assertEqual(
        schema_io.namedtuple_name_to_class(
            'schema_io_test.DeprecatedNamedTuple1'),
        NamedTuple1)

  def _run_serialization_test(self,
                              structure,
                              expected_type=None):
    """Convert the structure to serialized JSON, then back to a string."""
    expected_value = copy.deepcopy(structure)

    serialized = schema_io.serialize(structure)
    self.assertIsInstance(serialized, six.string_types)

    restored = schema_io.deserialize(serialized)
    self.assertEqual(restored, expected_value)

    if expected_type is not None:
      self.assertIsInstance(restored, expected_type)

  def test_serialization_with_simple_structures(self):
    # Primitives.
    self._run_serialization_test(None)
    self._run_serialization_test(1)
    self._run_serialization_test(0.5)
    self._run_serialization_test(1.0)
    self._run_serialization_test('foo')

    # Lists and tuples.
    self._run_serialization_test([1, 2, 3])
    self._run_serialization_test((1, 2, 3))

    # Dictionaries.
    self._run_serialization_test({'a': 3, 'b': 4})
    self._run_serialization_test({10: 'x', 20: 'y'})
    self._run_serialization_test({(1, 2): 'x', (3, 4): 'y'})

    # Namedtuples
    self._run_serialization_test(NamedTuple1(42), expected_type=NamedTuple1)
    self._run_serialization_test(NamedTuple2(12345), expected_type=NamedTuple2)

    # OneOf nodes.
    self._run_serialization_test(schema.OneOf((1, 2, 3), 'tag'))

  def test_namedtuple_deserialization_with_deprecated_names(self):
    restored = schema_io.deserialize(
        '["namedtuple:schema_io_test.DeprecatedNamedTuple1",["foo",51]]')
    self.assertEqual(restored, NamedTuple1(51))
    self.assertIsInstance(restored, NamedTuple1)

  def test_serialization_with_nested_structures(self):
    """Verify that to_json and from_json are recursively called on children."""
    # Lists and tuples
    self._run_serialization_test((((1,),),))
    self._run_serialization_test([[[1]]])

    # Dictionaries.
    self._run_serialization_test({'a': {'b': {'c': {'d': 'e'}}}})

    # Namedtuples
    self._run_serialization_test(NamedTuple1(NamedTuple2(NamedTuple1(42))))

    # OneOf nodes
    self._run_serialization_test(
        schema.OneOf((
            schema.OneOf((
                schema.OneOf((1, 2, 3), 'innermost'),
            ), 'inner'),
        ), 'outer'))

    # Composite data structure containing many different types.
    self._run_serialization_test(
        {'a': NamedTuple1([(schema.OneOf([{'b': 3}], 't'),)])})

  def test_serialization_with_bad_type(self):
    with self.assertRaisesRegex(ValueError, 'Unrecognized type'):
      schema_io.serialize(object())

  def test_deserialization_defaults(self):
    # NamedTuple1 accepts one argument: foo. It has not default value.
    # NamedTuple3 accepts two arguments: foo and bar. Both have default values.

    # Use default arguments for both foo and bar.
    value = schema_io.deserialize(
        """["namedtuple:schema_io_test.NamedTuple3"]""")
    self.assertEqual(value, NamedTuple3(foo=3, bar='hi'))

    # Use default argument for bar only.
    value = schema_io.deserialize(
        """["namedtuple:schema_io_test.NamedTuple3", ["foo", 42]]""")
    self.assertEqual(value, NamedTuple3(foo=42, bar='hi'))

    # Use default argument for foo only.
    value = schema_io.deserialize(
        """["namedtuple:schema_io_test.NamedTuple3", ["bar", "bye"]]""")
    self.assertEqual(value, NamedTuple3(foo=3, bar='bye'))

    # Don't use any default arguments.
    value = schema_io.deserialize(
        """["namedtuple:schema_io_test.NamedTuple3",
            ["foo", 9], ["bar", "x"]]""")
    self.assertEqual(value, NamedTuple3(foo=9, bar='x'))

    # Default values should also work when we refer to a namedtuple by a
    # deprecated name.
    value = schema_io.deserialize(
        """["namedtuple:schema_io_test.DeprecatedNamedTuple3"]""")
    self.assertEqual(value, NamedTuple3(foo=3, bar='hi'))

    # Serialized value references a field that doesn't exist in the namedtuple.
    with self.assertRaisesRegex(ValueError, 'Invalid field: baz'):
      schema_io.deserialize(
          """["namedtuple:schema_io_test.NamedTuple3", ["baz", 10]]""")

    # Serialized value is missing a field that should exist in the namedtuple.
    with self.assertRaisesRegex(ValueError, 'Missing field: foo'):
      schema_io.deserialize("""["namedtuple:schema_io_test.NamedTuple1"]""")


if __name__ == '__main__':
  absltest.main()
