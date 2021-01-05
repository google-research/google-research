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

"""Tests for supcon.hparam."""

import enum

from absl.testing import absltest
import attr
from supcon import hparam


class ValidNumericEnum(enum.Enum):
  FIRST = 1
  SECOND = 2


class ValidStringEnum(enum.Enum):
  FIRST = 'first'
  SECOND = 'second'


class InvalidEnum(enum.Enum):
  FIRST = ValidNumericEnum
  SECOND = ValidStringEnum


@attr.s
class NonHParams:
  dummy_param = True


@hparam.s
class ValidBoolParams:
  true_param = hparam.field('tbp', default=True)
  false_param = hparam.field('fbp', default=False)


@hparam.s
class ValidNumericParams:
  int_param = hparam.field('inp', default=123)
  float_param = hparam.field('fnp', default=12.3)


@hparam.s
class ValidStringParams:
  string_param = hparam.field('sp', default='foo')


@hparam.s
class ValidEnumParams:
  numeric_enum_param = hparam.field('nep', default=ValidNumericEnum.FIRST)
  string_enum_param = hparam.field('sep', default=ValidStringEnum.FIRST)


@hparam.s
class ValidListParams:
  bool_list = hparam.field('blp', default=[True, False])
  int_list = hparam.field('ilp', default=[1])
  float_tuple = hparam.field('flp', default=(1.23,))
  str_list = hparam.field('slp', default=['foo', 'bar'])


@hparam.s
class ValidNestedParams2:
  numeric_params = hparam.nest(ValidNumericParams)
  string_params1 = hparam.nest(ValidStringParams, prefix='s1')
  string_params2 = hparam.nest(ValidStringParams, prefix='s2')


@hparam.s
class ValidNestedParams1:
  nested_params = hparam.nest(ValidNestedParams2)
  bool_params = hparam.nest(ValidBoolParams)
  enum_params = hparam.nest(ValidEnumParams)


@hparam.s
class ValidHParams():
  nested_params = hparam.nest(ValidNestedParams1)
  list_params = hparam.nest(ValidListParams)


@hparam.s
class InvalidDuplicateParams():
  bool_params = hparam.nest(ValidBoolParams)
  duplicate_param = hparam.field('tbp', default='Im a duplicate')


@hparam.s
class InvalidNonHParamFieldParams():
  non_hparam = attr.ib(default=True)


class HparamTest(absltest.TestCase):

  def assert_serialization_works(self, hparams):
    serialized = hparams.serialize()
    new_hparams = type(hparams)(serialized)
    self.assertEqual(new_hparams, hparams)

    new_hparams = type(hparams)()
    new_hparams.parse(serialized)
    self.assertEqual(new_hparams, hparams)

    compact_serialized = hparams.serialize(omit_defaults=True)
    new_hparams = type(hparams)(compact_serialized)
    self.assertEqual(new_hparams, hparams)

    new_hparams = type(hparams)()
    new_hparams.parse(compact_serialized)
    self.assertEqual(new_hparams, hparams)

    return new_hparams

  def test_valid_hparams(self):
    hparams = ValidHParams()
    self.assertEqual(
        'inp=123,fnp=12.3,s1sp=foo,s2sp=foo,tbp=1,fbp=0,nep=1,sep=first,'
        'blp=[1,0],ilp=[1],flp=[1.23],slp=[foo,bar]', hparams.serialize())
    self.assertEqual('', hparams.serialize(omit_defaults=True))
    self.assert_serialization_works(hparams)
    hparams.list_params.bool_list = [True, True, True]
    self.assert_serialization_works(hparams)
    hparams.list_params.int_list = [1, 2, 3]
    self.assert_serialization_works(hparams)
    hparams.list_params.float_tuple = [1.23, 4.56]
    self.assert_serialization_works(hparams)
    hparams.list_params.str_list.append('baz')
    self.assert_serialization_works(hparams)
    hparams.nested_params.bool_params.true_param = False
    self.assert_serialization_works(hparams)
    hparams.nested_params.bool_params.false_param = True
    self.assert_serialization_works(hparams)
    hparams.nested_params.enum_params.numeric_enum_param = (
        ValidNumericEnum.SECOND)
    self.assert_serialization_works(hparams)
    hparams.nested_params.enum_params.string_enum_param = ValidStringEnum.SECOND
    self.assert_serialization_works(hparams)
    hparams.nested_params.nested_params.numeric_params.int_param = 456
    self.assert_serialization_works(hparams)
    hparams.nested_params.nested_params.numeric_params.float_param = 4.56
    self.assert_serialization_works(hparams)
    hparams.nested_params.nested_params.string_params1.string_param = 'bar'
    self.assert_serialization_works(hparams)
    hparams.nested_params.nested_params.string_params2.string_param = 'baz'
    self.assert_serialization_works(hparams)
    self.assertEqual(
        'inp=456,fnp=4.56,s1sp=bar,s2sp=baz,tbp=0,fbp=1,nep=2,sep=second,'
        'blp=[1,1,1],ilp=[1,2,3],flp=[1.23,4.56],slp=[foo,bar,baz]',
        hparams.serialize())
    self.assertEqual(
        'inp=456,fnp=4.56,s1sp=bar,s2sp=baz,tbp=0,fbp=1,nep=2,sep=second,'
        'blp=[1,1,1],ilp=[1,2,3],flp=[1.23,4.56],slp=[foo,bar,baz]',
        hparams.serialize(omit_defaults=True))
    self.assertEqual(
        '{"nested_params": {"nested_params": {"numeric_params": '
        '{"int_param": 456, "float_param": 4.56}, "string_params1": '
        '{"string_param": "bar"}, "string_params2": {"string_param": "baz"}}, '
        '"bool_params": {"true_param": false, '
        '"false_param": true}, "enum_params": {"numeric_enum_param": '
        '"ValidNumericEnum.SECOND", "string_enum_param": '
        '"ValidStringEnum.SECOND"}}, "list_params": {"bool_list": '
        '[true, true, true], "int_list": [1, 2, 3], "float_tuple": '
        '[1.23, 4.56], "str_list": ["foo", "bar", "baz"]}}',
        hparams.serialize(readable=True))

  def test_initialize_in_constructor(self):
    hparams = ValidHParams(
        list_params=ValidListParams(
            bool_list=[True, True, True],
            int_list=[1, 2, 3],
            float_tuple=[1.23, 4.56],
            str_list=['foo', 'bar', 'baz']),
        nested_params=ValidNestedParams1(
            bool_params=ValidBoolParams(true_param=False, false_param=True),
            enum_params=ValidEnumParams(
                numeric_enum_param=ValidNumericEnum.SECOND,
                string_enum_param=ValidStringEnum.SECOND),
            nested_params=ValidNestedParams2(
                numeric_params=ValidNumericParams(
                    int_param=456, float_param=4.56),
                string_params1=ValidStringParams(string_param='bar'),
                string_params2=ValidStringParams(string_param='baz'))))
    self.assertEqual(hparams.list_params.bool_list, [True, True, True])
    self.assertEqual(hparams.list_params.int_list, [1, 2, 3])
    self.assertEqual(hparams.list_params.float_tuple, [1.23, 4.56])
    self.assertEqual(hparams.list_params.str_list, ['foo', 'bar', 'baz'])
    self.assertEqual(hparams.nested_params.bool_params.true_param, False)
    self.assertEqual(hparams.nested_params.bool_params.false_param, True)
    self.assertEqual(hparams.nested_params.enum_params.numeric_enum_param,
                     ValidNumericEnum.SECOND)
    self.assertEqual(hparams.nested_params.enum_params.string_enum_param,
                     ValidStringEnum.SECOND)
    self.assertEqual(
        hparams.nested_params.nested_params.numeric_params.int_param, 456)
    self.assertEqual(
        hparams.nested_params.nested_params.numeric_params.float_param, 4.56)
    self.assertEqual(
        hparams.nested_params.nested_params.string_params1.string_param, 'bar')
    self.assertEqual(
        hparams.nested_params.nested_params.string_params2.string_param, 'baz')
    self.assertEqual(
        'inp=456,fnp=4.56,s1sp=bar,s2sp=baz,tbp=0,fbp=1,nep=2,sep=second,'
        'blp=[1,1,1],ilp=[1,2,3],flp=[1.23,4.56],slp=[foo,bar,baz]',
        hparams.serialize())

  def test_assign_int_to_float_param(self):
    hparams = ValidHParams()
    hparams.nested_params.nested_params.numeric_params.float_param = 4
    self.assertEqual(
        'inp=123,fnp=4.0,s1sp=foo,s2sp=foo,tbp=1,fbp=0,nep=1,sep=first,'
        'blp=[1,0],ilp=[1],flp=[1.23],slp=[foo,bar]', hparams.serialize())
    self.assertIsInstance(
        hparams.nested_params.nested_params.numeric_params.float_param, float)

  def test_assign_string_to_float_param(self):
    hparams = ValidHParams()
    with self.assertRaisesRegex(
        TypeError,
        'Expected a numeric type, but found 4.0 with type <class \'str\'>.'):
      hparams.nested_params.nested_params.numeric_params.float_param = '4.0'

  def test_assign_tuple_to_list_param(self):
    hparams = ValidHParams()
    hparams.list_params.float_tuple = (4, 5)
    self.assertEqual(
        'inp=123,fnp=12.3,s1sp=foo,s2sp=foo,tbp=1,fbp=0,nep=1,sep=first,'
        'blp=[1,0],ilp=[1],flp=[4.0,5.0],slp=[foo,bar]', hparams.serialize())
    self.assertIsInstance(hparams.list_params.float_tuple, list)

  def test_assign_scalar_to_list_param(self):
    hparams = ValidHParams()
    hparams.list_params.float_tuple = 4
    self.assertEqual(
        'inp=123,fnp=12.3,s1sp=foo,s2sp=foo,tbp=1,fbp=0,nep=1,sep=first,'
        'blp=[1,0],ilp=[1],flp=[4.0],slp=[foo,bar]', hparams.serialize())
    self.assertIsInstance(hparams.list_params.float_tuple, list)

  def test_assign_empty_list_to_list_param(self):
    hparams = ValidHParams()
    hparams.list_params.float_tuple = []
    self.assertEqual(
        'inp=123,fnp=12.3,s1sp=foo,s2sp=foo,tbp=1,fbp=0,nep=1,sep=first,'
        'blp=[1,0],ilp=[1],flp=[],slp=[foo,bar]', hparams.serialize())

  def test_assign_string_to_list_param(self):
    hparams = ValidHParams()
    hparams.list_params.str_list = '[baz,foo]'
    self.assertEqual(
        'inp=123,fnp=12.3,s1sp=foo,s2sp=foo,tbp=1,fbp=0,nep=1,sep=first,'
        'blp=[1,0],ilp=[1],flp=[1.23],slp=["[baz,foo]"]', hparams.serialize())
    self.assertIsInstance(hparams.list_params.str_list, list)

  def test_assign_float_to_int_param(self):
    hparams = ValidHParams()
    with self.assertRaisesRegex(
        TypeError,
        'Expected an integer value, but found 4.0 with type <class \'float\'>.'
    ):
      hparams.nested_params.nested_params.numeric_params.int_param = 4.0

  def test_assign_string_to_int_param(self):
    hparams = ValidHParams()
    with self.assertRaisesRegex(
        TypeError,
        'Expected an integer value, but found 4 with type <class \'str\'>.'):
      hparams.nested_params.nested_params.numeric_params.int_param = '4'

  def test_invalid_none_param(self):
    with self.assertRaisesRegex(TypeError,
                                'Fields cannot have a default value of None.'):
      hparam.field('np', default=None)

  def test_invalid_empty_list_param(self):
    with self.assertRaisesRegex(
        TypeError, 'Empty iterables cannot be used as default values.'):
      hparam.field('elp', default=[])

  def test_invalid_nested_list_param(self):
    with self.assertRaisesRegex(
        ValueError, 'Nested iterables and dictionaries are not supported.'):
      hparam.field('nlp', default=[['foo']])

  def test_invalid_dict_param(self):
    with self.assertRaisesRegex(
        TypeError, 'Only numbers, strings, and lists are supported. '
        'Found <class \'dict\'>.'):
      hparam.field('dp', default={'key': 'value'})

  def test_invalid_mixed_list_param(self):
    with self.assertRaisesRegex(TypeError,
                                'Iterables of mixed type are not supported.'):
      hparam.field('mlp', default=[1, 'foo'])

  def test_invalid_nest_non_hparam(self):
    with self.assertRaisesRegex(
        TypeError, 'Nested hparams classes must use the @hparam.s decorator'):
      hparam.nest(NonHParams)

  def test_invalid_nest_instance(self):
    nested = ValidNestedParams1()
    with self.assertRaisesRegex(
        TypeError, r'nest\(\) must be passed a class, not an instance.'):
      hparam.nest(nested)

  def test_invalid_nest_class_with_field(self):
    with self.assertRaisesRegex(
        TypeError,
        'Supported types include: number, string, Enum, and lists of those '
        'types.'):
      hparam.field('nhp', default=ValidBoolParams)

  def test_invalid_nest_instance_with_field(self):
    with self.assertRaisesRegex(
        TypeError,
        'Supported types include: number, string, Enum, and lists of those '
        'types.'):
      hparam.field('nhp', default=ValidBoolParams())

  def test_invalid_duplicate_param(self):
    with self.assertRaisesRegex(KeyError, 'Abbrev tbp is duplicated.'):
      _ = InvalidDuplicateParams()

  def test_invalid_nonhparam_field_param(self):
    with self.assertRaisesRegex(
        AssertionError,
        'Could not find hparam metadata for field non_hparam. Did you create '
        'a field without using hparam.field()?'):
      _ = InvalidNonHParamFieldParams()

  def test_deserialize_partial(self):
    hparams = ValidHParams('sep=second,blp=[1,1,1]')
    self.assertEqual(hparams.nested_params.enum_params.string_enum_param,
                     ValidStringEnum.SECOND)
    self.assertEqual(hparams.list_params.bool_list, [True, True, True])

  def test_deserialize_other_bool_formats(self):
    hparams = ValidHParams('blp=[True, true,False,false,1,0]')
    self.assertEqual(hparams.list_params.bool_list,
                     [True, True, False, False, True, False])

  def test_deserialize_other_float_formats(self):
    hparams = ValidHParams('flp=[1e-3, -2.6E4, 7.000, 3.14_15_93]')
    self.assertEqual(hparams.list_params.float_tuple,
                     [0.001, -26000., 7., 3.141593])

  def test_deserialize_other_int_formats(self):
    # TODO(sarna): Maybe add support for ints in other bases, like 0xdeadbeef.
    hparams = ValidHParams('ilp=[100_000]')
    self.assertEqual(hparams.list_params.int_list, [100000])

  def test_deserialize_invalid_int_forma(self):
    with self.assertRaisesRegex(ValueError,
                                r'invalid literal for int\(\) with base 10'):
      ValidHParams('ilp=[10*5]')

  def test_confusing_string(self):
    hparams = ValidHParams()
    hparams.nested_params.nested_params.string_params1.string_param = (
        'foo,tbp=evil')
    reconstructed = self.assert_serialization_works(hparams)
    self.assertEqual(
        reconstructed.nested_params.nested_params.string_params1.string_param,
        'foo,tbp=evil')

  def test_confusing_string_list(self):
    hparams = ValidHParams()
    hparams.list_params.str_list = ['foo,bar', 'baz']
    reconstructed = self.assert_serialization_works(hparams)
    self.assertEqual(reconstructed.list_params.str_list, ['foo,bar', 'baz'])

  def test_extra_quotes(self):
    hparams = ValidHParams()
    hparams.nested_params.nested_params.string_params1.string_param = (
        '"quoted,tbp=evil"')
    reconstructed = self.assert_serialization_works(hparams)
    self.assertEqual(
        reconstructed.nested_params.nested_params.string_params1.string_param,
        '"quoted,tbp=evil"')

  def test_deserialize_list_without_brackets(self):
    with self.assertRaisesRegex(ValueError, 'Malformed hyperparameter value'):
      ValidHParams('flp=0.2,-3.6,4e2')

  def test_deserialize_duplicate_value(self):
    with self.assertRaisesRegex(
        ValueError, 'Duplicate assignment to hyperparameter \'sep\''):
      ValidHParams('sep=second,sep=second')


if __name__ == '__main__':
  absltest.main()
