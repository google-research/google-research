# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

# Forked with minor changes from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/training/python/training/hparam_test.py pylint: disable=line-too-long
"""Tests for hparam."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from model_pruning.python import hparam


class HParamsTest(tf.test.TestCase):

  def testEmpty(self):
    hparams = hparam.HParams()
    self.assertDictEqual({}, hparams.values())
    hparams.parse('')
    self.assertDictEqual({}, hparams.values())
    with self.assertRaisesRegexp(ValueError, 'Unknown hyperparameter'):
      hparams.parse('xyz=123')

  def testContains(self):
    hparams = hparam.HParams(foo=1)
    self.assertTrue('foo' in hparams)
    self.assertFalse('bar' in hparams)

  def testSomeValues(self):
    hparams = hparam.HParams(aaa=1, b=2.0, c_c='relu6', d='/a/b=c/d')
    self.assertDictEqual(
        {'aaa': 1, 'b': 2.0, 'c_c': 'relu6', 'd': '/a/b=c/d'},
        hparams.values())
    expected_str = ('[(\'aaa\', 1), (\'b\', 2.0), (\'c_c\', \'relu6\'), '
                    '(\'d\', \'/a/b=c/d\')]')
    self.assertEqual(expected_str, str(hparams.__str__()))
    self.assertEqual(expected_str, str(hparams))
    self.assertEqual(1, hparams.aaa)
    self.assertEqual(2.0, hparams.b)
    self.assertEqual('relu6', hparams.c_c)
    self.assertEqual('/a/b=c/d', hparams.d)
    hparams.parse('aaa=12')
    self.assertDictEqual({
        'aaa': 12,
        'b': 2.0,
        'c_c': 'relu6',
        'd': '/a/b=c/d'
    }, hparams.values())
    self.assertEqual(12, hparams.aaa)
    self.assertEqual(2.0, hparams.b)
    self.assertEqual('relu6', hparams.c_c)
    self.assertEqual('/a/b=c/d', hparams.d)
    hparams.parse('c_c=relu4, b=-2.0e10')
    self.assertDictEqual({
        'aaa': 12,
        'b': -2.0e10,
        'c_c': 'relu4',
        'd': '/a/b=c/d'
    }, hparams.values())
    self.assertEqual(12, hparams.aaa)
    self.assertEqual(-2.0e10, hparams.b)
    self.assertEqual('relu4', hparams.c_c)
    self.assertEqual('/a/b=c/d', hparams.d)
    hparams.parse('c_c=,b=0,')
    self.assertDictEqual({'aaa': 12, 'b': 0, 'c_c': '', 'd': '/a/b=c/d'},
                         hparams.values())
    self.assertEqual(12, hparams.aaa)
    self.assertEqual(0.0, hparams.b)
    self.assertEqual('', hparams.c_c)
    self.assertEqual('/a/b=c/d', hparams.d)
    hparams.parse('c_c=2.3",b=+2,')
    self.assertEqual(2.0, hparams.b)
    self.assertEqual('2.3"', hparams.c_c)
    hparams.parse('d=/a/b/c/d,aaa=11,')
    self.assertEqual(11, hparams.aaa)
    self.assertEqual(2.0, hparams.b)
    self.assertEqual('2.3"', hparams.c_c)
    self.assertEqual('/a/b/c/d', hparams.d)
    hparams.parse('b=1.5,d=/a=b/c/d,aaa=10,')
    self.assertEqual(10, hparams.aaa)
    self.assertEqual(1.5, hparams.b)
    self.assertEqual('2.3"', hparams.c_c)
    self.assertEqual('/a=b/c/d', hparams.d)
    with self.assertRaisesRegexp(ValueError, 'Unknown hyperparameter'):
      hparams.parse('x=123')
    with self.assertRaisesRegexp(ValueError, 'Could not parse'):
      hparams.parse('aaa=poipoi')
    with self.assertRaisesRegexp(ValueError, 'Could not parse'):
      hparams.parse('aaa=1.0')
    with self.assertRaisesRegexp(ValueError, 'Could not parse'):
      hparams.parse('b=12x')
    with self.assertRaisesRegexp(ValueError, 'Could not parse'):
      hparams.parse('b=relu')
    with self.assertRaisesRegexp(ValueError, 'Must not pass a list'):
      hparams.parse('aaa=[123]')
    self.assertEqual(10, hparams.aaa)
    self.assertEqual(1.5, hparams.b)
    self.assertEqual('2.3"', hparams.c_c)
    self.assertEqual('/a=b/c/d', hparams.d)

  def testWithPeriodInVariableName(self):
    hparams = hparam.HParams()
    hparams.add_hparam(name='a.b', value=0.0)
    hparams.parse('a.b=1.0')
    self.assertEqual(1.0, getattr(hparams, 'a.b'))
    hparams.add_hparam(name='c.d', value=0.0)
    with self.assertRaisesRegexp(ValueError, 'Could not parse'):
      hparams.parse('c.d=abc')
    hparams.add_hparam(name='e.f', value='')
    hparams.parse('e.f=abc')
    self.assertEqual('abc', getattr(hparams, 'e.f'))
    hparams.add_hparam(name='d..', value=0.0)
    hparams.parse('d..=10.0')
    self.assertEqual(10.0, getattr(hparams, 'd..'))

  def testSetFromMap(self):
    hparams = hparam.HParams(a=1, b=2.0, c='tanh')
    hparams.override_from_dict({'a': -2, 'c': 'identity'})
    self.assertDictEqual({'a': -2, 'c': 'identity', 'b': 2.0}, hparams.values())

    hparams = hparam.HParams(x=1, b=2.0, d=[0.5])
    hparams.override_from_dict({'d': [0.1, 0.2, 0.3]})
    self.assertDictEqual({'d': [0.1, 0.2, 0.3], 'x': 1, 'b': 2.0},
                         hparams.values())

  def testFunction(self):
    def f(x):
      return x
    hparams = hparam.HParams(function=f)
    self.assertEqual(hparams.function, f)

    json_str = hparams.to_json()
    self.assertEqual(json_str, '{}')

  def testBoolParsing(self):
    for value in 'true', 'false', 'True', 'False', '1', '0':
      for initial in False, True:
        hparams = hparam.HParams(use_gpu=initial)
        hparams.parse('use_gpu=' + value)
        self.assertEqual(hparams.use_gpu, value in ['True', 'true', '1'])

  def testBoolParsingFail(self):
    hparams = hparam.HParams(use_gpu=True)
    with self.assertRaisesRegexp(ValueError, r'Could not parse.*use_gpu'):
      hparams.parse('use_gpu=yep')

  def testLists(self):
    hparams = hparam.HParams(aaa=[1], b=[2.0, 3.0], c_c=['relu6'])
    self.assertDictEqual({
        'aaa': [1],
        'b': [2.0, 3.0],
        'c_c': ['relu6']
    }, hparams.values())
    self.assertEqual([1], hparams.aaa)
    self.assertEqual([2.0, 3.0], hparams.b)
    self.assertEqual(['relu6'], hparams.c_c)
    hparams.parse('aaa=[12]')
    self.assertEqual([12], hparams.aaa)
    hparams.parse('aaa=[12,34,56]')
    self.assertEqual([12, 34, 56], hparams.aaa)
    hparams.parse('c_c=[relu4,relu12],b=[1.0]')
    self.assertEqual(['relu4', 'relu12'], hparams.c_c)
    self.assertEqual([1.0], hparams.b)
    hparams.parse('c_c=[],aaa=[-34]')
    self.assertEqual([-34], hparams.aaa)
    self.assertEqual([], hparams.c_c)
    hparams.parse('c_c=[_12,3\'4"],aaa=[+3]')
    self.assertEqual([3], hparams.aaa)
    self.assertEqual(['_12', '3\'4"'], hparams.c_c)
    with self.assertRaisesRegexp(ValueError, 'Unknown hyperparameter'):
      hparams.parse('x=[123]')
    with self.assertRaisesRegexp(ValueError, 'Could not parse'):
      hparams.parse('aaa=[poipoi]')
    with self.assertRaisesRegexp(ValueError, 'Could not parse'):
      hparams.parse('aaa=[1.0]')
    with self.assertRaisesRegexp(ValueError, 'Could not parse'):
      hparams.parse('b=[12x]')
    with self.assertRaisesRegexp(ValueError, 'Could not parse'):
      hparams.parse('b=[relu]')
    with self.assertRaisesRegexp(ValueError, 'Must pass a list'):
      hparams.parse('aaa=123')

  def testParseValuesWithIndexAssigment1(self):
    """Assignment to an index position."""
    parse_dict = hparam.parse_values('arr[1]=10', {'arr': int})
    self.assertEqual(len(parse_dict), 1)
    self.assertIsInstance(parse_dict['arr'], dict)
    self.assertDictEqual(parse_dict['arr'], {1: 10})

  def testParseValuesWithIndexAssigment1_IgnoreUnknown(self):
    """Assignment to an index position."""
    parse_dict = hparam.parse_values(
        'arr[1]=10,b=5', {'arr': int}, ignore_unknown=True)
    self.assertEqual(len(parse_dict), 1)
    self.assertIsInstance(parse_dict['arr'], dict)
    self.assertDictEqual(parse_dict['arr'], {1: 10})

  def testParseValuesWithIndexAssigment2(self):
    """Assignment to multiple index positions."""
    parse_dict = hparam.parse_values('arr[0]=10,arr[5]=20', {'arr': int})
    self.assertEqual(len(parse_dict), 1)
    self.assertIsInstance(parse_dict['arr'], dict)
    self.assertDictEqual(parse_dict['arr'], {0: 10, 5: 20})

  def testParseValuesWithIndexAssigment2_IgnoreUnknown(self):
    """Assignment to multiple index positions."""
    parse_dict = hparam.parse_values(
        'arr[0]=10,arr[5]=20,foo=bar', {'arr': int}, ignore_unknown=True)
    self.assertEqual(len(parse_dict), 1)
    self.assertIsInstance(parse_dict['arr'], dict)
    self.assertDictEqual(parse_dict['arr'], {0: 10, 5: 20})

  def testParseValuesWithIndexAssigment3(self):
    """Assignment to index positions in multiple names."""
    parse_dict = hparam.parse_values('arr[0]=10,arr[1]=20,L[5]=100,L[10]=200',
                                     {'arr': int,
                                      'L': int})
    self.assertEqual(len(parse_dict), 2)
    self.assertIsInstance(parse_dict['arr'], dict)
    self.assertDictEqual(parse_dict['arr'], {0: 10, 1: 20})
    self.assertIsInstance(parse_dict['L'], dict)
    self.assertDictEqual(parse_dict['L'], {5: 100, 10: 200})

  def testParseValuesWithIndexAssigment3_IgnoreUnknown(self):
    """Assignment to index positions in multiple names."""
    parse_dict = hparam.parse_values(
        'arr[0]=10,C=5,arr[1]=20,B[0]=kkk,L[5]=100,L[10]=200',
        {'arr': int, 'L': int}, ignore_unknown=True)
    self.assertEqual(len(parse_dict), 2)
    self.assertIsInstance(parse_dict['arr'], dict)
    self.assertDictEqual(parse_dict['arr'], {0: 10, 1: 20})
    self.assertIsInstance(parse_dict['L'], dict)
    self.assertDictEqual(parse_dict['L'], {5: 100, 10: 200})

  def testParseValuesWithIndexAssigment4(self):
    """Assignment of index positions and scalars."""
    parse_dict = hparam.parse_values('x=10,arr[1]=20,y=30',
                                     {'x': int,
                                      'y': int,
                                      'arr': int})
    self.assertEqual(len(parse_dict), 3)
    self.assertIsInstance(parse_dict['arr'], dict)
    self.assertDictEqual(parse_dict['arr'], {1: 20})
    self.assertEqual(parse_dict['x'], 10)
    self.assertEqual(parse_dict['y'], 30)

  def testParseValuesWithIndexAssigment4_IgnoreUnknown(self):
    """Assignment of index positions and scalars."""
    parse_dict = hparam.parse_values(
        'x=10,foo[0]=bar,arr[1]=20,zzz=78,y=30',
        {'x': int, 'y': int, 'arr': int}, ignore_unknown=True)
    self.assertEqual(len(parse_dict), 3)
    self.assertIsInstance(parse_dict['arr'], dict)
    self.assertDictEqual(parse_dict['arr'], {1: 20})
    self.assertEqual(parse_dict['x'], 10)
    self.assertEqual(parse_dict['y'], 30)

  def testParseValuesWithIndexAssigment5(self):
    """Different variable types."""
    parse_dict = hparam.parse_values('a[0]=5,b[1]=true,c[2]=abc,d[3]=3.14', {
        'a': int,
        'b': bool,
        'c': str,
        'd': float
    })
    self.assertEqual(set(parse_dict.keys()), {'a', 'b', 'c', 'd'})
    self.assertIsInstance(parse_dict['a'], dict)
    self.assertDictEqual(parse_dict['a'], {0: 5})
    self.assertIsInstance(parse_dict['b'], dict)
    self.assertDictEqual(parse_dict['b'], {1: True})
    self.assertIsInstance(parse_dict['c'], dict)
    self.assertDictEqual(parse_dict['c'], {2: 'abc'})
    self.assertIsInstance(parse_dict['d'], dict)
    self.assertDictEqual(parse_dict['d'], {3: 3.14})

  def testParseValuesWithIndexAssigment5_IgnoreUnknown(self):
    """Different variable types."""
    parse_dict = hparam.parse_values(
        'a[0]=5,cc=4,b[1]=true,c[2]=abc,mm=2,d[3]=3.14',
        {'a': int, 'b': bool, 'c': str, 'd': float},
        ignore_unknown=True)
    self.assertEqual(set(parse_dict.keys()), {'a', 'b', 'c', 'd'})
    self.assertIsInstance(parse_dict['a'], dict)
    self.assertDictEqual(parse_dict['a'], {0: 5})
    self.assertIsInstance(parse_dict['b'], dict)
    self.assertDictEqual(parse_dict['b'], {1: True})
    self.assertIsInstance(parse_dict['c'], dict)
    self.assertDictEqual(parse_dict['c'], {2: 'abc'})
    self.assertIsInstance(parse_dict['d'], dict)
    self.assertDictEqual(parse_dict['d'], {3: 3.14})

  def testParseValuesWithBadIndexAssigment1(self):
    """Reject assignment of list to variable type."""
    with self.assertRaisesRegexp(ValueError,
                                 r'Assignment of a list to a list index.'):
      hparam.parse_values('arr[1]=[1,2,3]', {'arr': int})

  def testParseValuesWithBadIndexAssigment1_IgnoreUnknown(self):
    """Reject assignment of list to variable type."""
    with self.assertRaisesRegexp(ValueError,
                                 r'Assignment of a list to a list index.'):
      hparam.parse_values(
          'arr[1]=[1,2,3],c=8', {'arr': int}, ignore_unknown=True)

  def testParseValuesWithBadIndexAssigment2(self):
    """Reject if type missing."""
    with self.assertRaisesRegexp(ValueError,
                                 r'Unknown hyperparameter type for arr'):
      hparam.parse_values('arr[1]=5', {})

  def testParseValuesWithBadIndexAssigment2_IgnoreUnknown(self):
    """Ignore missing type."""
    hparam.parse_values('arr[1]=5', {}, ignore_unknown=True)

  def testParseValuesWithBadIndexAssigment3(self):
    """Reject type of the form name[index]."""
    with self.assertRaisesRegexp(ValueError,
                                 'Unknown hyperparameter type for arr'):
      hparam.parse_values('arr[1]=1', {'arr[1]': int})

  def testParseValuesWithBadIndexAssigment3_IgnoreUnknown(self):
    """Ignore type of the form name[index]."""
    hparam.parse_values('arr[1]=1', {'arr[1]': int}, ignore_unknown=True)

  def testWithReusedVariables(self):
    with self.assertRaisesRegexp(ValueError,
                                 'Multiple assignments to variable \'x\''):
      hparam.parse_values('x=1,x=1', {'x': int})

    with self.assertRaisesRegexp(ValueError,
                                 'Multiple assignments to variable \'arr\''):
      hparam.parse_values('arr=[100,200],arr[0]=10', {'arr': int})

    with self.assertRaisesRegexp(
        ValueError, r'Multiple assignments to variable \'arr\[0\]\''):
      hparam.parse_values('arr[0]=10,arr[0]=20', {'arr': int})

    with self.assertRaisesRegexp(ValueError,
                                 'Multiple assignments to variable \'arr\''):
      hparam.parse_values('arr[0]=10,arr=[100]', {'arr': int})

  def testJson(self):
    hparams = hparam.HParams(aaa=1, b=2.0, c_c='relu6', d=True)
    self.assertDictEqual({
        'aaa': 1,
        'b': 2.0,
        'c_c': 'relu6',
        'd': True
    }, hparams.values())
    self.assertEqual(1, hparams.aaa)
    self.assertEqual(2.0, hparams.b)
    self.assertEqual('relu6', hparams.c_c)
    hparams.parse_json('{"aaa": 12, "b": 3.0, "c_c": "relu4", "d": false}')
    self.assertDictEqual({
        'aaa': 12,
        'b': 3.0,
        'c_c': 'relu4',
        'd': False
    }, hparams.values())
    self.assertEqual(12, hparams.aaa)
    self.assertEqual(3.0, hparams.b)
    self.assertEqual('relu4', hparams.c_c)

    json_str = hparams.to_json()
    hparams2 = hparam.HParams(aaa=10, b=20.0, c_c='hello', d=False)
    hparams2.parse_json(json_str)
    self.assertEqual(12, hparams2.aaa)
    self.assertEqual(3.0, hparams2.b)
    self.assertEqual('relu4', hparams2.c_c)
    self.assertEqual(False, hparams2.d)

    hparams3 = hparam.HParams(aaa=123)
    self.assertEqual('{"aaa": 123}', hparams3.to_json())
    self.assertEqual('{\n  "aaa": 123\n}', hparams3.to_json(indent=2))
    self.assertEqual('{"aaa"=123}', hparams3.to_json(separators=(';', '=')))

    hparams4 = hparam.HParams(aaa=123, b='hello', c_c=False)
    self.assertEqual(
        '{"aaa": 123, "b": "hello", "c_c": false}',
        hparams4.to_json(sort_keys=True))

  def testSetHParam(self):
    hparams = hparam.HParams(aaa=1, b=2.0, c_c='relu6', d=True)
    self.assertDictEqual({
        'aaa': 1,
        'b': 2.0,
        'c_c': 'relu6',
        'd': True
    }, hparams.values())
    self.assertEqual(1, hparams.aaa)
    self.assertEqual(2.0, hparams.b)
    self.assertEqual('relu6', hparams.c_c)

    hparams.set_hparam('aaa', 12)
    hparams.set_hparam('b', 3.0)
    hparams.set_hparam('c_c', 'relu4')
    hparams.set_hparam('d', False)
    self.assertDictEqual({
        'aaa': 12,
        'b': 3.0,
        'c_c': 'relu4',
        'd': False
    }, hparams.values())
    self.assertEqual(12, hparams.aaa)
    self.assertEqual(3.0, hparams.b)
    self.assertEqual('relu4', hparams.c_c)

  def testSetHParamListNonListMismatch(self):
    hparams = hparam.HParams(a=1, b=[2.0, 3.0])
    with self.assertRaisesRegexp(ValueError, r'Must not pass a list'):
      hparams.set_hparam('a', [1.0])
    with self.assertRaisesRegexp(ValueError, r'Must pass a list'):
      hparams.set_hparam('b', 1.0)

  def testSetHParamTypeMismatch(self):
    hparams = hparam.HParams(
        int_=1, str_='str', bool_=True, float_=1.1, list_int=[1, 2], none=None)

    with self.assertRaises(ValueError):
      hparams.set_hparam('str_', 2.2)

    with self.assertRaises(ValueError):
      hparams.set_hparam('int_', False)

    with self.assertRaises(ValueError):
      hparams.set_hparam('bool_', 1)

    with self.assertRaises(ValueError):
      hparams.set_hparam('int_', 2.2)

    with self.assertRaises(ValueError):
      hparams.set_hparam('list_int', [2, 3.3])

    with self.assertRaises(ValueError):
      hparams.set_hparam('int_', '2')

    # Casting int to float is OK
    hparams.set_hparam('float_', 1)

    # Getting stuck with NoneType :(
    hparams.set_hparam('none', '1')
    self.assertEqual('1', hparams.none)

  def testGet(self):
    hparams = hparam.HParams(aaa=1, b=2.0, c_c='relu6', d=True, e=[5.0, 6.0])

    # Existing parameters with default=None.
    self.assertEqual(1, hparams.get('aaa'))
    self.assertEqual(2.0, hparams.get('b'))
    self.assertEqual('relu6', hparams.get('c_c'))
    self.assertEqual(True, hparams.get('d'))
    self.assertEqual([5.0, 6.0], hparams.get('e', None))

    # Existing parameters with compatible defaults.
    self.assertEqual(1, hparams.get('aaa', 2))
    self.assertEqual(2.0, hparams.get('b', 3.0))
    self.assertEqual(2.0, hparams.get('b', 3))
    self.assertEqual('relu6', hparams.get('c_c', 'default'))
    self.assertEqual(True, hparams.get('d', True))
    self.assertEqual([5.0, 6.0], hparams.get('e', [1.0, 2.0, 3.0]))
    self.assertEqual([5.0, 6.0], hparams.get('e', [1, 2, 3]))

    # Existing parameters with incompatible defaults.
    with self.assertRaises(ValueError):
      hparams.get('aaa', 2.0)

    with self.assertRaises(ValueError):
      hparams.get('b', False)

    with self.assertRaises(ValueError):
      hparams.get('c_c', [1, 2, 3])

    with self.assertRaises(ValueError):
      hparams.get('d', 'relu')

    with self.assertRaises(ValueError):
      hparams.get('e', 123.0)

    with self.assertRaises(ValueError):
      hparams.get('e', ['a', 'b', 'c'])

    # Nonexistent parameters.
    self.assertEqual(None, hparams.get('unknown'))
    self.assertEqual(123, hparams.get('unknown', 123))
    self.assertEqual([1, 2, 3], hparams.get('unknown', [1, 2, 3]))

  def testDel(self):
    hparams = hparam.HParams(aaa=1, b=2.0)

    with self.assertRaises(ValueError):
      hparams.set_hparam('aaa', 'will fail')

    with self.assertRaises(ValueError):
      hparams.add_hparam('aaa', 'will fail')

    hparams.del_hparam('aaa')
    hparams.add_hparam('aaa', 'will work')
    self.assertEqual('will work', hparams.get('aaa'))

    hparams.set_hparam('aaa', 'still works')
    self.assertEqual('still works', hparams.get('aaa'))


if __name__ == '__main__':
  tf.test.main()
