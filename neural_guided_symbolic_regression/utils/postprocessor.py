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

"""Postprocessor functions convert expression array to expression string.

The autoencoder model uses feature_preprocessors API from model_ops.ModelBase to
convert expression string input to expression tensor. The decoder model will
output the reconstructed expression tensor. However, the model_ops.ModelBase
does not have postprocessor API. This module includes postprocessor functions
and can be used in the model_fn of the autoencoder model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nltk

from neural_guided_symbolic_regression.utils import arithmetic_grammar
from neural_guided_symbolic_regression.utils import constants


class GrammarLhsStack(object):
  """Stack stores the string of left hand side symbol of the production rule.

  This stack behaves like a python list when it is not empty. When it is empty
  and the pop method is called, instead of raising an error, the string of the
  left hand side symbol of the dummy production rule defined in the constants
  will be returned.
  """

  def __init__(self, initialize_list=None):
    """Initializer.

    Args:
      initialize_list: A list of strings to initialize the stack. Default None,
          will initialize the stack as empty.

    Attributes:
      _stack: A list to store elements of string in stack.
    """
    if initialize_list is None:
      self._stack = []
    else:
      # Use slice to copy the list.
      self._stack = initialize_list[:]

  def copy(self):
    """Gets a copy of GrammarLhsStack object.

    Returns:
      A GrammarLhsStack object.
    """
    return GrammarLhsStack(self._stack)

  def to_list(self):
    """Gets the non-dummy symbols in stack as a list.

    Returns:
      A list of strings.
    """
    # Use slice to copy the list.
    return self._stack[:]

  def is_empty(self):
    """Whether there is no non-dummy symbols in stack.

    Returns:
      Boolean.
    """
    return not self._stack

  def peek(self):
    """Looks the top element of this stack without removing it from the stack.

    If stack is empty, the string of the dummy symbol defined in the constants
    will be returned.

    Returns:
      A string of the left hand side symbol of the context-free grammar
          production rule.
    """
    return self._stack[-1] if self._stack else constants.DUMMY_LHS_SYMBOL

  def pop(self):
    """Pops an element from the stack.

    This method pops the top element in the stack when it is not empty.
    Otherwise, the string of the dummy symbol defined in the constants will be
    returned.

    Returns:
      A string of the left hand side symbol of the context-free grammar
          production rule.
    """
    return self._stack.pop() if self._stack else constants.DUMMY_LHS_SYMBOL

  def push(self, symbol):
    """Pushes a new string of symbol into the stack.

    Args:
      symbol: String of symbol.
    """
    self._stack.append(symbol)

  def push_reversed_list(self, symbol_list):
    """Pushes a list of strings of symbol into the stack in reversed order.

    Note the last element in the symbol_list will be pushed into the stack
    first. For example, for a stack with elements [a, b] and a symbol_list
    [c, d, e]. After calling the push_list method, the new stack will have
    elements [a, b, e, d, c]. When calling pop method, c will be the first
    element popping out from the stack.

    This method is used when performing a preorder traversal of the context-free
    grammar parsing tree.

    For example, if a production rule is a -> b c d
    The right hand side symbols will be [b, c, d]. In a preorder traversal, we
    want symbol b to be processed first, so b should be on the top of the stack.
    Thus, the list of right hand side symbols should be reversed before extend
    to the stack.

    Args:
      symbol_list: List of strings of symbol.
    """
    self._stack.extend(reversed(symbol_list))


def production_rules_sequence_to_stack(prod_rules_sequence):
  """Pushs a sequence of production rules to a GrammarLhsStack instance.

  Args:
    prod_rules_sequence: List of nltk.grammar.Production objects.

  Returns:
    A GrammarLhsStack object.

  Raises:
    ValueError: If left hand side symbol of the production rule to push into the
        stack does not match the top symbol in the stack.
  """
  stack = GrammarLhsStack()
  if not prod_rules_sequence:
    return stack
  stack.push_reversed_list(get_non_terminal_rhs(prod_rules_sequence[0]))
  for prod_rule in prod_rules_sequence[1:]:
    lhs = stack.pop()
    if lhs != prod_rule.lhs().symbol():
      raise ValueError('Left hand side symbol of production rule %s does not '
                       'match the symbol in the stack (%s)' % (prod_rule, lhs))
    stack.push_reversed_list(get_non_terminal_rhs(prod_rule))
  return stack


def production_rules_sequence_to_symbols(prod_rules_sequence):
  """Gets list of symbols from production rules sequence.

  Args:
    prod_rules_sequence: List of nltk.grammar.Production objects. This sequence
        is obtained by a preorder traversal of the context-free grammar parsing
        tree.

  Returns:
    A list of strings of symbol.
  """
  if not prod_rules_sequence:
    return []
  sequence = [prod_rules_sequence[0].lhs()]
  for prod_rule in prod_rules_sequence:
    if prod_rule.lhs().symbol() == constants.DUMMY_LHS_SYMBOL:
      break
    for i, symbol in enumerate(sequence):
      if symbol == prod_rule.lhs():
        sequence = sequence[:i] + list(prod_rule.rhs()) + sequence[i + 1:]
        break
  return sequence


def production_rules_sequence_to_expression_string(prod_rules_sequence,
                                                   delimiter,
                                                   check_all_terminal=False):
  """Generates expression from production rules sequence.

  Args:
    prod_rules_sequence: List of nltk.grammar.Production objects. This sequence
        is obtained by a preorder traversal of the context-free grammar parsing
        tree.
    delimiter: String, the delimiter of tokens in expression string.
    check_all_terminal: Boolean, raise ValueError if not all the symbols are
        terminal.

  Returns:
    A string of the expression.

  Raises:
    ValueError: If not all the symbols generated by production rules sequence
        are terminal and check_all_terminal flag is true.
  """
  sequence = production_rules_sequence_to_symbols(prod_rules_sequence)
  if check_all_terminal and not arithmetic_grammar.all_terminal(sequence):
    raise ValueError('Not all the symbols are terminal.')
  return delimiter.join([str(symbol) for symbol in sequence])


def get_non_terminal_rhs(prod_rule):
  """Identifies non-terminal in the right hand side symbols in production rule.

  Args:
    prod_rule: nltk.grammar.Production objects.

  Returns:
    A list of nltk.grammar.Nonterminal symbols.
  """
  non_terminal_rhs = []
  for rhs in prod_rule.rhs():
    if (isinstance(rhs, nltk.grammar.Nonterminal)
        and rhs.symbol() != constants.DUMMY_RHS_SYMBOL):
      non_terminal_rhs.append(rhs.symbol())
  return non_terminal_rhs
