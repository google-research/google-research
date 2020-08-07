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

"""States record the grammar parsing tree information."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import nltk
import numpy as np

from neural_guided_symbolic_regression.utils import postprocessor


class StateBase(object):
  """State object for Monte Carlo Tree Search.

  Subclasses should define the following methods:
    * is_terminal
    * copy
    * _equal
    * _info
  """

  def is_terminal(self):
    """Whether the current state is a terminal state.

    Returns:
      Boolean.
    """
    raise NotImplementedError('Must be implemented by subclass.')

  def copy(self):
    """Gets a copy of current state.

    Returns:
      State object.
    """
    raise NotImplementedError('Must be implemented by subclass.')

  def __eq__(self, other):
    """Defines the equality operator.

    Args:
      other: Another State object.

    Returns:
      Boolean whether two states equal.
    """
    return isinstance(other, type(self)) and self._equal(other)

  def _equal(self, other):
    """Defines the equality operator for the subclass.

    This private method will be called in __eq__().

    Args:
      other: Another State object.

    Returns:
      Boolean whether two states equal.
    """
    raise NotImplementedError('Must be implemented by subclass.')

  def __repr__(self):
    """Defines behavior for when repr() is called on an instance of this class.

    Returns:
      String.
    """
    return '%s [%s]' % (self.__class__.__name__, self._info())

  def _info(self):
    """Defines the information to display in __repr__().

    Returns:
      String.
    """
    raise NotImplementedError('Must be implemented by subclass.')


class ExpressionStateBase(StateBase):
  """State object of expression generation.

  Subclasses should define the following methods:
    * get_expression
  """

  def get_expression(self):
    """Gets the expression of current state.

    Returns:
      String.
    """
    raise NotImplementedError('Must be implemented by subclass.')


class ProductionRulesState(ExpressionStateBase):
  """Records the grammar parsing tree by grammar production rules sequence."""

  def __init__(self, production_rules_sequence, stack=None):
    """Initializer.

    If this state is the initial state with no production rules sequence, pass
    a list of one symbol string to stack argument. This will enforce the next
    production rule to append starting with this symbol.

    Args:
      production_rules_sequence: List of nltk.grammar.Production objects. This
          sequence is obtained by a preorder traversal of the context-free
          grammar parsing tree.
      stack: GrammarLhsStack object or list, the stack to store the string of
          left hand side symbol. The left hand side symbol of valid production
          rule to append must match the top element in the stack. If the input
          is a list, the last element in the list is the top element in the
          stack.

    Raises:
      ValueError: If stack is not list, GrammarLhsStack or None.
    """
    self._production_rules_sequence = production_rules_sequence
    if stack is None:
      self._stack = postprocessor.production_rules_sequence_to_stack(
          production_rules_sequence)
    elif isinstance(stack, list):
      self._stack = postprocessor.GrammarLhsStack(stack)
    elif isinstance(stack, postprocessor.GrammarLhsStack):
      self._stack = stack.copy()
    else:
      raise ValueError('stack is expected to be list, GrammarLhsStack or '
                       'None, but got %s.' % type(stack))
    # Log the state information defined in __repr__.
    logging.info('Create %s', self)

  @property
  def production_rules_sequence(self):
    """Gets the production rules sequence.

    Returns:
      List of nltk.grammar.Production objects.
    """
    return self._production_rules_sequence[:]

  def generate_history(self):
    """Generates the history of the expression generation.

    For example, if the current production rules in production_rules_sequence
    is ['S -> S "+" T', 'S -> T', 'T -> "y"', 'T -> "x"']

    The expression generation history when each production rule is appended is
    ['S + T', 'T + T', 'y + T', 'y + x'].

    Returns:
      List of expression strings.
    """
    production_rules_sequence = self.production_rules_sequence
    history = []
    for partial_sequence_length in range(1, len(production_rules_sequence) + 1):
      history.append(
          postprocessor.production_rules_sequence_to_expression_string(
              prod_rules_sequence=production_rules_sequence[
                  :partial_sequence_length],
              delimiter=' ',
              check_all_terminal=False))
    return history

  def is_valid_to_append(self, production_rule):
    """Whether a production rule is valid to append.

    The left hand side symbol of production rule need to match the top symbol
    in the grammar left hand side symbol stack.

    Args:
      production_rule: nltk.grammar.Production object. The production rule to
          append on the production rule sequence in the current state.

    Returns:
      Boolean.
    """
    return self.stack_peek() == production_rule.lhs().symbol()

  def stack_peek(self):
    """Gets the top symbol in stack.

    The next non terminal symbol to expand.

    Returns:
      String of symbol.
    """
    return self._stack.peek()

  def append_production_rule(self, production_rule):
    """Appends a production rule on the sequence and returns a new state.

    Args:
      production_rule: nltk.grammar.Production object. The production rule to
          append on the production rule sequence in the current state.

    Returns:
      A ProductionRulesState object.

    Raises:
      ValueError: If the left hand side symbol of production rule does not
          match the top symbol in the grammar left hand side stack.
    """
    if not self.is_valid_to_append(production_rule):
      raise ValueError('The left hand side symbol of production rule %s does '
                       'not match the top symbol in the grammar left hand side '
                       'stack (%s)' % (production_rule, self.stack_peek()))

    self._stack.pop()
    self._stack.push_reversed_list(
        postprocessor.get_non_terminal_rhs(production_rule))
    self._production_rules_sequence.append(production_rule)
    logging.info('Append production rule: %s, %s', production_rule, self)

  def is_terminal(self):
    """Whether the last production rule in the sequence is a terminal rule.

    If the last production rule in the production_rules_sequence has left hand
    side symbol of terminal rule defined in constants.DUMMY_LHS_SYMBOL.

    Returns:
      Boolean whether current state is terminal.
    """
    return self._stack.is_empty()

  def copy(self):
    """Gets a copy of current state.

    Returns:
      ProductionRulesState object.
    """
    logging.info('Create a copy of ProductionRulesState.')
    return ProductionRulesState(
        production_rules_sequence=self.production_rules_sequence,
        stack=self._stack.copy())

  def _equal(self, other):
    """Defines the equality operator for ProductionRulesState.

    This private method will be called in __eq__().

    Args:
      other: Another State object.

    Returns:
      Boolean whether two states equal.
    """
    if len(self.production_rules_sequence) != len(
        other.production_rules_sequence):
      return False
    else:
      return all(
          rule1 == rule2 for rule1, rule2 in zip(
              self.production_rules_sequence, other.production_rules_sequence))

  def get_expression(self, coefficients=None):
    """Gets the expression of current state.

    Args:
      coefficients: Dict of coefficients values in expression string.
          {coefficient_symbol: value}. If not None, the values of the
          coefficients will replace the symbols of coefficients in the
          expression string.

    Returns:
      String.
    """
    return _numericalize_coefficients(self._get_expression()[1], coefficients)

  def _get_expression(self):
    """Gets the expression and symbols of current state.

    Returns:
      expression: String.
      symbols: List of symbols.
    """
    symbols = postprocessor.production_rules_sequence_to_symbols(
        prod_rules_sequence=self.production_rules_sequence)
    return ' '.join([str(symbol) for symbol in symbols]), symbols

  def _info(self):
    """Defines information to display when __repr__() is called.

    Returns:
      String.
    """
    expression, symbols = self._get_expression()
    num_terminals = sum(
        nltk.grammar.is_terminal(symbol) for symbol in symbols)
    num_symbols = len(symbols)
    if num_symbols:
      terminal_ratio = float(num_terminals) / num_symbols
    else:
      terminal_ratio = np.nan
    return ('symbols: %s, '
            'length_production_rules_sequence: %d, '
            'stack top: %s, '
            'num_terminals / num_symbols: %d / %d, '
            'terminal_ratio: %4.2f'
            % (expression,
               len(self.production_rules_sequence),
               self.stack_peek(),
               num_terminals,
               num_symbols,
               terminal_ratio))


def _numericalize_coefficients(raw_symbols, coefficients):
  """Replaces the symbols of coefficients in the expression string with values.

  If there is coefficient symbol in raw_symbols which is not in coefficients
  dict, it will remain symbolic in the expression string.

  Args:
    raw_symbols: List of context-free grammar symbols or strings.
    coefficients: Dict of coefficients values in expression string.
        {coefficient_symbol: value}. If not None, the values of the
        coefficients will replace the symbols of coefficients in the
        expression string.

  Returns:
    Expression string.
  """
  if coefficients is None:
    coefficients = {}
  symbols = []
  for symbol in map(str, raw_symbols):
    if symbol in coefficients:
      symbols.append(str(coefficients[symbol]))
    else:
      symbols.append(symbol)
  return ' '.join(symbols)
