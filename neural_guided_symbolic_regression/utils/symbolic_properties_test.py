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

"""Tests for symbolic_properties."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import six
import sympy
import tensorflow.compat.v1 as tf
from neural_guided_symbolic_regression.utils import symbolic_properties


class SymbolicPropertiesTest(parameterized.TestCase, tf.test.TestCase):

  def test_catch_exception(self):
    @symbolic_properties.catch_exception
    def mock_symbolic_property(expression_string, foo, bar):
      del expression_string, foo, bar
      raise ValueError('error message abc')

    with self.assertRaisesRegex(
        ValueError,
        r'Fail to get symbolic property of expression 1 \+ x '
        r'from mock_symbolic_property\: error message abc'):
      mock_symbolic_property('1 + x', 42, bar=99)

  @parameterized.parameters([
      # Expression.
      '1 + 2 * x', 'x / ( 1 + x )', 'x / ( 1 - x)', '1 + 2 * sqrt( x )',
      # Single number.
      '1', '-1', '2.', '-2.',
  ])
  def test_check_is_finite_true(self, expression):
    self.assertTrue(
        symbolic_properties.check_is_finite(sympy.sympify(expression)))

  @parameterized.parameters([
      # Only one term.
      '1 / 0', 'x / 0', 'x / ( x - x)',
      # More than one term.
      '-1 / 0 + x', '1 / 0 - x / 0', 'x / 0 + 1',
  ])
  def test_check_is_finite_false(self, expression):
    self.assertFalse(
        symbolic_properties.check_is_finite(sympy.sympify(expression)))

  @parameterized.parameters([
      sympy.sympify('1. + 2.'),
      sympy.sympify('1.'),
      sympy.sympify('-2.'),
      sympy.sympify('x / x'),
      sympy.sympify('c', locals={'c': 1.}),
  ])
  def test_check_is_number_true(self, expression):
    self.assertTrue(symbolic_properties.check_is_number(expression))

  @parameterized.parameters([
      sympy.sympify('x + 2.'),
      sympy.sympify('x'),
      sympy.sympify('-x'),
      sympy.sympify('x / 0'),
      sympy.sympify('x', locals={'c': 1.}),
  ])
  def test_check_is_number_false(self, expression):
    self.assertFalse(symbolic_properties.check_is_number(expression))

  @parameterized.parameters([
      ('-1', 0.),
      ('0', 0.),
      ('1', 0.),
      ('x * 5', 1.),
      ('5 * x', 1.),
      ('-x * 5', 1.),
      ('x * 0', 0.),
      ('5 * x * x', 2.),
      ('0 * x * x', 0.),
      ('5 / x / x', -2.),
      ('0 / x / x', 0.),
      ('5 / x', -1.),
      ('0 / x', 0.),
      ('5 * x ** 0.5', 0.5),
      ('5 * x ** ( 1 / 2 )', 0.5),
      ('5 * sqrt( x )', 0.5),
      ('5 * x ** -0.5', -0.5),
      ('5 * x ** ( -1 / 2 )', -0.5),
      ('5 / sqrt( x )', -0.5),
  ])
  def test_get_power(self, term_expression, expected_power):
    np.testing.assert_almost_equal(
        symbolic_properties.get_power(sympy.sympify(term_expression)),
        expected_power)

  @parameterized.parameters([
      '1 / 0',
      'x / 0',
      '1 / x / 0',
      '-1 / 0',
      '-x / 0',
      '-1 / x / 0',
  ])
  def test_get_power_nan(self, term_expression):
    np.testing.assert_equal(
        symbolic_properties.get_power(sympy.sympify(term_expression)), np.nan)

  def test_get_leading_power_invalid_x0(self):
    with self.assertRaisesRegex(
        ValueError, 'x0 is expected to be 0 or inf, got 5'):
      symbolic_properties.get_leading_power('x ** 2 + x + 1', x0='5')

  @parameterized.parameters([
      ('1', '0', 0.),
      ('1', 'inf', 0.),
      ('0', '0', 0.),
      ('0', 'inf', 0.),
      ('x - 1', '0', 0.),
      ('x - 1', 'inf', 1.),
      ('x - x', '0', 0.),
      ('x - x', 'inf', 0.),
      ('1 / x', '0', -1.),
      ('1 / x', 'inf', -1.),
      ('1 / x + 1', '0', -1.),
      ('1 / x + 1', 'inf', 0.),
      ('1 / x ** 2 + 1 / x', '0', -2.),
      ('1 / x ** 2 + 1 / x', 'inf', -1.),
      ('1 / x + x * x', '0', -1.),
      ('1 / x + x * x', 'inf', 2.),
      ('1 / x + 2 + x * x', '0', -1.),
      ('1 / x + 2 + x * x', 'inf', 2.),
      ('1 / ( 1 + x )', '0', 0.),
      ('1 / ( 1 + x )', 'inf', -1.),
      # Soft coulomb potential.
      ('1 / sqrt( 1 + x ** 2 )', '0', 0.),
      ('1 / sqrt( 1 + x ** 2 )', 'inf', -1.),
      # Complicated expression.
      ('1 / ( x - ( ( 1 / ( 1 - x ) ) - x ) )', '0', 0.),
      ('1 / ( x - ( ( 1 / ( 1 - x ) ) - x ) )', 'inf', -1.),
  ])
  def test_get_leading_power(
      self, expression_string, x0, expected_leading_power):
    np.testing.assert_almost_equal(
        symbolic_properties.get_leading_power(expression_string, x0),
        expected_leading_power)

  @parameterized.parameters([
      ('c', '0', {'c': 2.}, 0.),
      ('c', 'inf', {'c': 2.}, 0.),
      # Harmonic oscillator potential
      # x ** 2 cancelled out, leading order x
      ('( x - c1 ) ** 2 - ( x - c2 ) ** 2', 'inf', {'c1': -2., 'c2': 2.}, 1.),
      ('( x - c1 ) ** 2 + ( x - c2 ) ** 2', 'inf', {'c1': -2., 'c2': 2.}, 2.),
      ('-( x - c1 ) ** 2 - ( x - c2 ) ** 2', 'inf', {'c1': -2., 'c2': 2.}, 2.),
      # Coulomb potential.
      ('1 / sqrt( ( x - c1 ) ** 2 ) - 1 / sqrt( ( x - c2 ) ** 2 )',
       # 1 / x cancelled out, leading order 1 / x ** 2
       'inf', {'c1': -2., 'c2': 2.}, -2.),
      ('1 / sqrt( ( x - c1 ) ** 2 ) + 1 / sqrt( ( x - c2 ) ** 2 )',
       'inf', {'c1': -2., 'c2': 2.}, -1.),
      ('-1 / sqrt( ( x - c1 ) ** 2 ) - 1 / sqrt( ( x - c2 ) ** 2 )',
       'inf', {'c1': -2., 'c2': 2.}, -1.),
      # Soft coulomb potential.
      ('1 / sqrt( 1 + ( x - c1 ) ** 2 ) - 1 / sqrt( 1 + ( x - c2 ) ** 2 )',
       # 1 / x cancelled out, leading order 1 / x ** 2
       'inf', {'c1': -2., 'c2': 2.}, -2.),
      ('1 / sqrt( 1 + ( x - c1 ) ** 2 ) + 1 / sqrt( 1 + ( x - c2 ) ** 2 )',
       'inf', {'c1': -2., 'c2': 2.}, -1.),
      ('-1 / sqrt( 1 + ( x - c1 ) ** 2 ) - 1 / sqrt( 1 + ( x - c2 ) ** 2 )',
       'inf', {'c1': -2., 'c2': 2.}, -1.),
  ])
  def test_get_leading_power_coefficients(
      self, expression_string, x0, coefficients, expected_leading_power):
    np.testing.assert_almost_equal(
        symbolic_properties.get_leading_power(
            expression_string, x0, coefficients=coefficients),
        expected_leading_power)

  @parameterized.parameters([
      ('1 / 0', '0'),
      ('1 / 0', 'inf'),
      ('x * x + 1 / ( x - x )', '0'),
      ('x * x + 1 / ( x - x )', 'inf'),
      ('1 / ( x - x )', '0'),
      ('1 / ( x - x )', 'inf'),
      ('x / ( x - x )', '0'),
      ('x / ( x - x )', 'inf'),
      ('x / ( 1 - 1 )', '0'),
      ('x / ( 1 - 1 )', 'inf'),
      ('( x - x ) / ( x - x )', '0'),
      ('( x - x ) / ( x - x )', 'inf'),
  ])
  def test_get_leading_power_nan(self, expression_string, x0):
    np.testing.assert_equal(
        symbolic_properties.get_leading_power(expression_string, x0), np.nan)

  def test_get_symbolic_property_functions(self):
    symbolic_property_functions = (
        symbolic_properties.get_symbolic_property_functions())
    self.assertSetEqual(
        set(symbolic_property_functions.keys()),
        set(['leading_at_0', 'leading_at_inf']))
    for symbolic_property_function in six.itervalues(
        symbolic_property_functions):
      self.assertTrue(callable(symbolic_property_function))

  def test_assert_property_names_valid(self):
    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        ValueError,
        'unknown property is not in the allowed symbolic properties'):
      symbolic_properties.assert_property_names_valid(
          symbolic_property_names=['leading_at_0', 'unknown property'],
          allowed_symbolic_properties=['leading_at_0', 'leading_at_inf'])


if __name__ == '__main__':
  tf.test.main()
