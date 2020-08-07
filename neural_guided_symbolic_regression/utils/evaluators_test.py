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

"""Tests for evaluators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from neural_guided_symbolic_regression.utils import evaluators


class OperatorTest(tf.test.TestCase):

  def test_divide_with_zero_divisor(self):
    # array, scalar.
    np.testing.assert_allclose(
        evaluators.divide_with_zero_divisor([0., 1., 2.], 0.), [0., 0., 0.])
    # array, array.
    np.testing.assert_allclose(
        evaluators.divide_with_zero_divisor([0., 1., 2.], [1., 2., 0]),
        [0., 0.5, 0.])
    # scalar, array.
    np.testing.assert_allclose(
        evaluators.divide_with_zero_divisor(2., [1., 2., 0.]), [2., 1., 0.])

  def test_power_with_zero_base(self):
    # array, scalar.
    np.testing.assert_allclose(
        evaluators.power_with_zero_base([0., 0., 2.], -1.), [0., 0., 0.5])
    # array, array.
    np.testing.assert_allclose(
        evaluators.power_with_zero_base([0., 0., 2.], [-3., -2., -1.]),
        [0., 0., 0.5])
    # scalar, array.
    np.testing.assert_allclose(
        # power(0, 0) should be 1.
        evaluators.power_with_zero_base(0., [1., -2., 0.]), [0., 0., 1.])


class NumpyArrayEvalTest(parameterized.TestCase):

  def test_arguments_not_dict(self):
    with self.assertRaisesRegex(ValueError,
                                'Input arguments expected to be a dict'):
      evaluators.numpy_array_eval('x', arguments=[42])

  def test_callables_not_dict(self):
    with self.assertRaisesRegex(ValueError,
                                'Input callables expected to be a dict'):
      evaluators.numpy_array_eval('sqrt( 4 )', callables=[np.sqrt])

  def test_unknown_argument(self):
    with self.assertRaisesRegex(SyntaxError, 'Unknown argument: \'x\''):
      evaluators.numpy_array_eval('x', arguments={'y': 2})

  def test_unknown_callable(self):
    with self.assertRaisesRegex(SyntaxError, 'Unknown callable: \'foo\''):
      evaluators.numpy_array_eval('foo( 4 )', callables={'bar': np.sin})

  @parameterized.parameters([
      # String of string is not a valid expression.
      '"hello world"',
      # List is not a valid expression.
      '[]',
      # Dictionary is not a valid expression.
      '{}',
      # Boolean statement is not a valid expression.
      '1==1'
  ])
  def test_malformed_string(self, string):
    with self.assertRaisesRegex(SyntaxError, 'Malformed string'):
      evaluators.numpy_array_eval(string)

  def test_number(self):
    # Integers.
    self.assertEqual(evaluators.numpy_array_eval('42'), 42)
    self.assertEqual(evaluators.numpy_array_eval('-42'), -42)
    # Floats.
    self.assertAlmostEqual(evaluators.numpy_array_eval('4.2'), 4.2)
    self.assertAlmostEqual(evaluators.numpy_array_eval('-4.2'), -4.2)

  @parameterized.parameters([
      # Test argument as a number.
      ({'x': 1.}, -1.),
      # Test argument as a numpy array.
      ({'x': np.array([1., 2., 3.])}, np.array([-1., -2., -3.])),
  ])
  def test_unary_operator(self, arguments, expected):
    np.testing.assert_allclose(
        evaluators.numpy_array_eval('-x', arguments=arguments), expected)

  @parameterized.parameters([
      ('x + y', np.array([2., 6., 15.])),
      ('x - y', np.array([0., -2., -9.])),
      ('x * y', np.array([1., 8., 36.])),
      ('x / y', np.array([1., 0.5, 0.25])),
      ('x ** y', np.array([1., 16., 531441.])),
      ('x + y / x - x * y', np.array([1., -4., -29.])),
  ])
  def test_numpy_array_binary_operators(self, string, expected):
    arguments = {'x': np.array([1., 2., 3.]), 'y': np.array([1., 4., 12.])}
    np.testing.assert_allclose(
        evaluators.numpy_array_eval(string, arguments=arguments), expected)

  @parameterized.parameters([
      ('x + y', 3.),
      ('x - y', -1.),
      ('x * y', 2.),
      ('x / y', 0.5),
      ('x ** y', 1.),
      ('y * x / (y - 1) + y', 4.),
  ])
  def test_number_binary_operators(self, string, expected):
    arguments = {'x': 1., 'y': 2.}
    self.assertAlmostEqual(
        evaluators.numpy_array_eval(string, arguments=arguments), expected)

  @parameterized.parameters([
      ({'x': 3., 'y': 2.}, 9.),
      ({'x': 2., 'y': 0.5}, 1.4142135624),
  ])
  def test_number_binary_operators_power(self, arguments, expected):
    self.assertAlmostEqual(
        evaluators.numpy_array_eval('x ** y', arguments=arguments), expected)

  @parameterized.parameters([
      ('x + y', np.array([11., 12., 13.])),
      ('x - y', np.array([-9., -8., -7.])),
      ('x * y', np.array([10., 20., 30.])),
      ('x / y', np.array([0.1, 0.2, 0.3])),
      ('x ** y', np.array([1., 1024., 59049.])),
      ('y * x / (y - 20) + y', np.array([9., 8., 7.])),
  ])
  def test_numpy_array_and_number_binary_operators(self, string, expected):
    arguments = {'x': np.array([1., 2., 3.]), 'y': 10.}
    np.testing.assert_allclose(
        evaluators.numpy_array_eval(string, arguments=arguments), expected)

  @parameterized.parameters([
      # Unary callables.
      ('sqrt( 4 )', {}, 2.),
      ('sqrt( x )', {'x': 4}, 2.),
      ('sqrt( x )', {'x': np.array([1., 4., 9.])}, np.array([1., 2., 3.])),
      # Binary callables.
      ('add( 4 , 6 )', {}, 10.),
      ('add( x , 6 )', {'x': 4}, 10.),
      ('add( x , 6 )', {'x': np.array([1., 4., 9.])}, np.array([7., 10., 15.])),
  ])
  def test_callables(self, string, arguments, expected):
    np.testing.assert_allclose(
        evaluators.numpy_array_eval(string, arguments=arguments), expected)

  def test_one_zero_element_in_divisor_array_expression(self):
    arguments = {'x': 1., 'y': np.array([2., 0., 0.5])}
    np.testing.assert_allclose(
        evaluators.numpy_array_eval('x / y', arguments=arguments),
        np.array([0.5, 0., 2.])
    )


class EvaluateExpressionStrings1dGridTest(tf.test.TestCase):

  def test_incorrect_numpy_array_shape(self):
    with self.assertRaisesRegexp(ValueError,
                                 r'The shape of n is expected to be \(1, 3\) '
                                 r'but got \(1, 5\)'):
      evaluators.evaluate_expression_strings_1d_grid(
          ['n + 1'],
          num_samples=1,
          num_grids=3,
          arguments={'n': np.array([[-1., 0., 1., 2., 3.]])}
      )

  def test_incorrect_argument_type(self):
    with self.assertRaisesRegexp(ValueError,
                                 r'Argument should be np.ndarray, int, or '
                                 r'float. but got n, <(class|type) \'list\'>'):
      evaluators.evaluate_expression_strings_1d_grid(
          ['n + 1'],
          num_samples=1,
          num_grids=3,
          # Invalid argument type: list.
          arguments={'n': [[-1., 0., 1.]]}
      )

  def test_argument_int(self):
    np.testing.assert_allclose(
        evaluators.evaluate_expression_strings_1d_grid(
            ['a + 1', 'b + a'],
            num_samples=1,
            num_grids=3,
            arguments={'a': 1, 'b': 3}
        ),
        [[[2, 2, 2]], [[4, 4, 4]]]
    )

  def test_arguments_numpy_array_and_int(self):
    np.testing.assert_allclose(
        evaluators.evaluate_expression_strings_1d_grid(
            ['n + 1', 'n + a'],
            num_samples=1,
            num_grids=3,
            arguments={'a': 2, 'n': np.array([[-1., 0., 1.]])}
        ),
        [[[0, 1, 2]], [[1, 2, 3]]]
    )

  def test_arguments_numpy_array_and_int_with_callable(self):
    np.testing.assert_allclose(
        evaluators.evaluate_expression_strings_1d_grid(
            ['sin(n + a)'],
            num_samples=1,
            num_grids=3,
            callables={'sin': np.sin},
            arguments={'a': 2, 'n': np.array([[-1., 0., 1.]])}
        ),
        [[[np.sin(1), np.sin(2), np.sin(3)]]]
    )


if __name__ == '__main__':
  tf.test.main()
