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

"""Functions compute symbolic properties of expression."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools

from absl import logging
import numpy as np
import sympy


def catch_exception(function):
  """Catches exception in symbolic property function.

  Sympy may fail in unexpected way. If it happens, we want to know the
  expression string and symbolic property function. This decorate catches the
  exception and shows those informations with the original error message.

  Args:
    function: Callable, symbolic property function. This first argument is
        expression_string.

  Returns:
    function_wrapper catch the exception.
  """
  def function_wrapper(expression_string, *args, **kwargs):
    try:
      return function(expression_string, *args, **kwargs)
    except Exception as error:
      raise ValueError(
          'Fail to get symbolic property of expression %s from %s: %s'
          % (expression_string, function.__name__, error))
  return function_wrapper


def check_is_finite(expression):
  """Checks whether all terms are finite (not inf or nan) in expression.

  Args:
    expression: Sympy expression.

  Returns:
    Boolean. True if this expression does not contain inf or nan. Otherwise,
        False.
  """
  # NOTE(leeley): If sympy expression is a single numerical value, it may be
  # represented as float or int instead of sympy expression.
  # For example,
  # sympy.sympify('1.') is sympy.core.numbers.Float
  # sympy.sympify('c', locals={'c': 1.}) is float.
  if isinstance(expression, (int, float)):
    return np.isfinite(expression)

  for check_symbol in [sympy.oo, sympy.zoo, sympy.nan]:
    if expression.has(check_symbol):
      return False
  return True


def check_is_number(expression):
  """Checks whether expression is a number.

  Args:
    expression: Sympy expression.

  Returns:
    Boolean.
  """
  # NOTE(leeley): If sympy expression is a single numerical value, it may be
  # represented as float or int instead of sympy expression.
  # For example,
  # sympy.sympify('1.') is sympy.core.numbers.Float
  # sympy.sympify('c', locals={'c': 1.}) is float.
  return isinstance(expression, (int, float)) or expression.is_number


def get_power(term_expression, symbol='x'):
  """Gets power p of term c * x ** p.

  For f(x) = c * x ** p, evaluate f at x1 and x2. The power p is computed by
  log(f(x1) / f(x2)) / log(x1 / x2).

  Note the way this function get the power of monomial term only works for
  c * x ** p. Other forms like c * (x - 1) ** p do not work.
  If the coefficient c is infinite, return nan.
  p is a float number and can be positive, zero or negative.
  The term expression can be 5 * x * x, 1 / x, 3 * sqrt(x) and so on.

  Args:
    term_expression: Sympy expression. In the form of c * x ** p.
    symbol: String. Default variable symbol in term_expression.

  Returns:
    Float. Power p of input term expression c * x ** p.
  """
  if not check_is_finite(term_expression):
    return np.nan

  if term_expression.is_number:
    return 0.

  f = sympy.lambdify(sympy.Symbol(symbol), term_expression)
  x1, x2 = 5., 10.
  y1, y2 = f(x1), f(x2)
  return np.log(y1 / y2) / np.log(x1 / x2)


@catch_exception
def get_leading_power(expression_string, x0, symbol='x', coefficients=None):
  """Gets the leading power of expression at given integer point.

  Args:
    expression_string: String. The expression to expand.
    x0: String. The point to expand the expression.
        Only '0', 'inf' are allowed.
    symbol: String. Symbol of variable in expression.
    coefficients: Dict {coefficient_name: coefficient_value}.
        * coefficient_name: String, the coefficient symbol in expression string.
        * coefficient_value: Float, the value of coefficient in expression
              string.
        The symbols of coefficients in the expression_string will be replaced
        by the values given in this dict when computing the symbolic property.

  Returns:
    Float. The power of the leading term of input expression. Nan will be
    returned for infinite expression or sympy fail to parse the expression.

  Raises:
    ValueError: If x0 is not 0 or inf.
  """
  if isinstance(expression_string, bytes):
    expression_string = expression_string.decode('utf-8')
  expression = sympy.simplify(
      sympy.sympify(expression_string, locals=coefficients))

  # NOTE(leeley): sympy < 1.0 has issues to expand constant expression. This
  # issue is fixed in sympy > 1.0.
  if check_is_number(expression) and check_is_finite(expression):
    return 0

  if x0 == 'inf':
    x0 = sympy.oo
  elif x0 == '0':
    x0 = 0
  else:
    raise ValueError('x0 is expected to be 0 or inf, got %s.' % x0)

  try:
    # NOTE(leeley): for 'x ** 2 + 1 / 0', the first term of the series will be
    # x ** 2. However, this expression is invalid since 1 / 0 is infinite.
    # There is no good way to check this in sympy, so we will check whether
    # infinite exists in the first 6 terms.
    terms = itertools.islice(
        # n=None will create a generator of terms.
        expression.series(x=sympy.sympify(symbol), x0=x0, n=None), 6)
    for i, term_expression in enumerate(terms):
      if i == 0:
        leading_power = get_power(term_expression, symbol=symbol)
      if np.isnan(get_power(term_expression, symbol=symbol)):
        logging.info(
            '%s in %s expansion at %s is infnite, return leading power nan.',
            str(term_expression), expression_string, str(x0))
        return np.nan
    return leading_power
  except (AssertionError, KeyError) as error:
    # If sympy crash, return nan.
    logging.warning(
        'Fail to get the leading power of %s at %s, %s',
        expression_string, str(x0), error)
    return np.nan


def get_symbolic_property_functions(symbol='x', coefficients=None):
  """Gets a dict of symbolic property functions.

  The keys in the dict are:
    * 'leading_at_0'
    * 'leading_at_inf'

  Args:
    symbol: String. Symbol of variable in expression.
    coefficients: Dict {coefficient_name: coefficient_value}.
        * coefficient_name: String, the coefficient symbol in expression string.
        * coefficient_value: Float, the value of coefficient in expression
              string.
        The symbols of coefficients in the expression_string will be replaced
        by the values given in this dict when computing the symbolic property.

  Returns:
    Dict. {symbolic_property_name: symbolic_property_function}
      * symbolic_property_name: String, the name of a symbolic property.
      * symbolic_property_function: Callable, its takes only one input argument
            expression_string.
  """
  return {
      'leading_at_0': functools.partial(
          get_leading_power,
          x0='0',
          symbol=symbol,
          coefficients=coefficients),
      'leading_at_inf': functools.partial(
          get_leading_power,
          x0='inf',
          symbol=symbol,
          coefficients=coefficients),
  }


def assert_property_names_valid(
    symbolic_property_names, allowed_symbolic_properties):
  """Converts the symbolic_properties string to list of symbolic property names.

  Args:
    symbolic_property_names: List of strings, name of symbolic properties.
    allowed_symbolic_properties: List of strings, allowed symbolic properties.

  Raises:
    ValueError: If symbolic property name in symbolic_properties_string is not
        allowed in allowed_symbolic_properties.
  """
  for symbolic_property_name in symbolic_property_names:
    if symbolic_property_name not in allowed_symbolic_properties:
      raise ValueError(
          '%s is not in the allowed symbolic properties %s'
          % (symbolic_property_name, allowed_symbolic_properties))
