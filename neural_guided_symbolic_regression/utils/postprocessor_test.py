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

"""Tests for postprocessor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nltk
import tensorflow.compat.v1 as tf

from neural_guided_symbolic_regression.utils import arithmetic_grammar
from neural_guided_symbolic_regression.utils import constants
from neural_guided_symbolic_regression.utils import postprocessor


class ExpressionContextFreeGrammarPostprocessorTest(tf.test.TestCase):

  def setUp(self):
    super(ExpressionContextFreeGrammarPostprocessorTest, self).setUp()
    grammar_rules = [
        'S -> S "+" T',
        'S -> T',
        'T -> "(" S ")"',
        'T -> "x"',
    ]
    # Get list of nltk.grammar.Production objects.
    self.grammar = arithmetic_grammar.Grammar(grammar_rules)
    self.prod_rules_dict = {
        k: v
        for k, v in zip(grammar_rules + [constants.DUMMY_PRODUCTION_RULE],
                        self.grammar.prod_rules)
    }
    self.delimiter = ' '

  def test_production_rules_sequence_to_symbols_not_terminal(self):
    prod_rule_strings = [
        'S -> S "+" T',
        'S -> T',
    ]
    prod_rules_sequence = [
        self.prod_rules_dict[prod_rule_string]
        for prod_rule_string in prod_rule_strings
    ]
    # Parsing tree:
    #  S
    #  |
    #  S "+" T
    #  |
    #  T
    # Expression (non-terminal):
    # T + T
    t, = nltk.grammar.nonterminals('T')
    self.assertEqual(
        postprocessor.production_rules_sequence_to_symbols(prod_rules_sequence),
        [t, '+', t])

  def test_production_rules_sequence_to_symbols_terminal(self):
    prod_rule_strings = [
        'S -> S "+" T',
        'S -> T',
        'T -> "x"',
        'T -> "x"',
        # The generation of symbols by grammar production rules sequence will
        # stop if all the symbols are terminal. For the grammar rules in this
        # unittest, the last two dummy rules are actually not used.
        constants.DUMMY_PRODUCTION_RULE,
        constants.DUMMY_PRODUCTION_RULE,
    ]
    prod_rules_sequence = [
        self.prod_rules_dict[prod_rule_string]
        for prod_rule_string in prod_rule_strings
    ]

    # Expression:
    # x + x
    # Parsing tree:
    #  S
    #  |
    #  S "+" T
    #  |     |
    #  T    "x"
    #  |
    # "x"
    # Production rules sequence (preorder tree traversal)
    # 'S -> S "+" T'
    # 'S -> T'
    # 'T -> "x"
    # 'T -> "x"

    self.assertEqual(
        postprocessor.production_rules_sequence_to_symbols(prod_rules_sequence),
        ['x', '+', 'x'])

  def test_production_rules_sequence_to_symbols_empty(self):
    # Empty production rule sequence should have empty symbols.
    self.assertEqual(postprocessor.production_rules_sequence_to_symbols([]), [])

  def test_production_rules_sequence_to_expression_string_not_terminal(self):
    prod_rule_strings = [
        'S -> S "+" T',
        'S -> T',
    ]
    prod_rules_sequence = [
        self.prod_rules_dict[prod_rule_string]
        for prod_rule_string in prod_rule_strings
    ]
    # Parsing tree:
    #  S
    #  |
    #  S "+" T
    #  |
    #  T
    # Expression (non-terminal):
    # T + T
    self.assertEqual(
        postprocessor.production_rules_sequence_to_expression_string(
            prod_rules_sequence, self.delimiter), 'T + T')

    with self.assertRaisesRegexp(ValueError,
                                 'Not all the symbols are terminal.'):
      postprocessor.production_rules_sequence_to_expression_string(
          prod_rules_sequence, self.delimiter, check_all_terminal=True)

  def test_production_rules_sequence_to_expression_string(self):
    prod_rule_strings = [
        'S -> S "+" T',
        'S -> T',
        'T -> "x"',
        'T -> "x"',
        constants.DUMMY_PRODUCTION_RULE,
        constants.DUMMY_PRODUCTION_RULE,
    ]
    prod_rules_sequence = [
        self.prod_rules_dict[prod_rule_string]
        for prod_rule_string in prod_rule_strings
    ]

    # Expression:
    # x + x
    # Parsing tree:
    #  S
    #  |
    #  S "+" T
    #  |     |
    #  T    "x"
    #  |
    # "x"
    # Production rules sequence (preorder tree traversal)
    # 'S -> S "+" T'
    # 'S -> T'
    # 'T -> "x"
    # 'T -> "x"

    self.assertEqual(
        postprocessor.production_rules_sequence_to_expression_string(
            prod_rules_sequence, self.delimiter), 'x + x')

  def test_production_rules_sequence_to_expression_string_empty(self):
    self.assertEqual(
        postprocessor.production_rules_sequence_to_expression_string(
            [], self.delimiter), '')

  def test_get_non_terminal_rhs(self):
    prod_rule = nltk.CFG.fromstring(
        ['S -> S "+" "(" T "*" "a" ")"']).productions()[0]
    self.assertEqual(postprocessor.get_non_terminal_rhs(prod_rule), ['S', 'T'])

  def test_production_rules_sequence_to_stack(self):
    prod_rule_strings = [
        'S -> S "+" T',
        'S -> T',
    ]
    prod_rules_sequence = [
        self.prod_rules_dict[prod_rule_string]
        for prod_rule_string in prod_rule_strings
    ]
    stack = postprocessor.production_rules_sequence_to_stack(
        prod_rules_sequence)
    self.assertEqual(stack._stack, ['T', 'T'])

  def test_production_rules_sequence_to_stack_empty_sequence(self):
    stack = postprocessor.production_rules_sequence_to_stack([])
    self.assertEqual(stack._stack, [])

  def test_production_rules_sequence_to_stack_invalid(self):
    prod_rule_strings = [
        'S -> S "+" T',
        'T -> "x"',
    ]
    prod_rules_sequence = [
        self.prod_rules_dict[prod_rule_string]
        for prod_rule_string in prod_rule_strings
    ]
    with self.assertRaisesRegexp(
        ValueError,
        'Left hand side symbol of production rule T -> \'x\' does not match '
        r'the symbol in the stack \(S\)'):
      postprocessor.production_rules_sequence_to_stack(prod_rules_sequence)


class GrammarLhsStackTest(tf.test.TestCase):

  def test_initialize_list_copy(self):
    initialize_list = ['x']
    stack = postprocessor.GrammarLhsStack(initialize_list)
    # The elements in stack should not change with initialize_list after
    # stack initialization.
    initialize_list[0] = ['y']
    self.assertEqual(stack.pop(), 'x')

  def test_copy(self):
    stack = postprocessor.GrammarLhsStack(['a', 'b', 'c'])
    copy_stack = stack.copy()
    # Use assertIs to check exact type rather than assertIsInstance.
    # https://docs.python.org/2/library/unittest.html#unittest.TestCase.assertIsInstance
    self.assertIs(type(copy_stack), postprocessor.GrammarLhsStack)
    self.assertEqual(copy_stack._stack, ['a', 'b', 'c'])
    # copy_stack is a copy of stack. The change in stack after copy will not
    # affect copy_stack.
    stack.pop()
    self.assertEqual(stack._stack, ['a', 'b'])
    self.assertEqual(copy_stack._stack, ['a', 'b', 'c'])

  def test_to_list(self):
    stack = postprocessor.GrammarLhsStack(['a', 'b', 'c'])
    stack_list = stack.to_list()
    self.assertEqual(stack_list, ['a', 'b', 'c'])
    stack.pop()
    # stack_list is a copy.
    self.assertEqual(stack_list, ['a', 'b', 'c'])
    # to_list() will always get the current symbols in stack.
    self.assertEqual(stack.to_list(), ['a', 'b'])

  def test_is_empty(self):
    self.assertFalse(postprocessor.GrammarLhsStack(['a', 'b', 'c']).is_empty())
    self.assertTrue(postprocessor.GrammarLhsStack([]).is_empty())

  def test_peek(self):
    stack = postprocessor.GrammarLhsStack(['x'])
    self.assertEqual(stack.peek(), 'x')
    # Peek does not change the stack.
    self.assertEqual(stack._stack, ['x'])

  def test_peek_empty(self):
    stack = postprocessor.GrammarLhsStack()
    self.assertEqual(stack.peek(), constants.DUMMY_LHS_SYMBOL)
    # Peek does not change the stack.
    self.assertEqual(stack._stack, [])

  def test_pop(self):
    stack = postprocessor.GrammarLhsStack(['x'])
    # Non-empty stack.
    self.assertEqual(stack.pop(), 'x')
    # Empty stack.
    self.assertEqual(stack.pop(), constants.DUMMY_LHS_SYMBOL)

  def test_push(self):
    stack = postprocessor.GrammarLhsStack()
    stack.push('x')
    self.assertEqual(stack.pop(), 'x')

  def test_push_reversed_list(self):
    stack = postprocessor.GrammarLhsStack(['a'])
    stack.push_reversed_list(['b', 'c'])
    self.assertEqual(stack.pop(), 'b')
    self.assertEqual(stack.pop(), 'c')
    self.assertEqual(stack.pop(), 'a')


if __name__ == '__main__':
  tf.test.main()
