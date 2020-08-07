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

"""Tests for states."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import nltk
import tensorflow.compat.v1 as tf

from neural_guided_symbolic_regression.mcts import states
from neural_guided_symbolic_regression.utils import arithmetic_grammar
from neural_guided_symbolic_regression.utils import constants
from neural_guided_symbolic_regression.utils import postprocessor


class StateBaseTest(tf.test.TestCase):

  def setUp(self):
    super(StateBaseTest, self).setUp()
    self.state = states.StateBase()

  def test_is_terminal(self):
    with self.assertRaisesRegexp(NotImplementedError,
                                 'Must be implemented by subclass'):
      self.state.is_terminal()

  def test_copy(self):
    with self.assertRaisesRegexp(NotImplementedError,
                                 'Must be implemented by subclass'):
      self.state.copy()

  def test_equal(self):
    with self.assertRaisesRegexp(NotImplementedError,
                                 'Must be implemented by subclass'):
      self.state._equal(None)

  def test_info(self):
    with self.assertRaisesRegexp(NotImplementedError,
                                 'Must be implemented by subclass'):
      self.state._info()


class ExpressionStateBaseTest(tf.test.TestCase):

  def test_get_expression(self):
    state = states.ExpressionStateBase()
    with self.assertRaisesRegexp(NotImplementedError,
                                 'Must be implemented by subclass'):
      state.get_expression()


class ProductionRulesStateTest(tf.test.TestCase):

  def setUp(self):
    super(ProductionRulesStateTest, self).setUp()
    grammar_rules = [
        'S -> S "+" T',
        'S -> T',
        'T -> "(" S ")"',
        'T -> "x"',
        'S -> R',
        'R -> "y"',
    ]
    # Get list of nltk.grammar.Production objects.
    self.grammar = arithmetic_grammar.Grammar(grammar_rules)
    self.production_rules_dict = {
        k: v
        for k, v in zip(grammar_rules + [constants.DUMMY_PRODUCTION_RULE],
                        self.grammar.prod_rules)
    }

  def _strings_to_production_rules(self, production_rules_strings):
    return [
        self.production_rules_dict[production_rules_string]
        for production_rules_string in production_rules_strings]

  def test_eq(self):
    production_rules_sequence = self._strings_to_production_rules([
        'S -> S "+" T',
        'S -> T',
        'T -> "x"',
        'T -> "x"',
        constants.DUMMY_PRODUCTION_RULE,
    ])
    state1 = states.ProductionRulesState(production_rules_sequence)
    state2 = states.ProductionRulesState(production_rules_sequence)
    self.assertEqual(state1, state2)

  def test_eq_length_not_equal(self):
    production_rules_sequence = self._strings_to_production_rules([
        'S -> S "+" T',
        'S -> T',
        'T -> "x"',
        'T -> "x"',
        constants.DUMMY_PRODUCTION_RULE,
    ])
    state1 = states.ProductionRulesState(production_rules_sequence)
    state2 = states.ProductionRulesState(production_rules_sequence[:-1])
    self.assertNotEqual(state1, state2)

  def test_is_terminal_end_with_terminal_rule(self):
    production_rules_sequence = self._strings_to_production_rules([
        'S -> S "+" T',
        'S -> T',
        'T -> "x"',
        'T -> "x"',
        # NOTE(leeley): I want to mimic the procedure in the grammar variational
        # autoencoder to use DUMMY_PRODUCTION_RULE as the padding rule.
        # The generation of symbols by grammar production rules sequence will
        # stop if all the symbols are terminal. For the grammar rules in this
        # unittest, the last one dummy rules are actually not used.
        constants.DUMMY_PRODUCTION_RULE,
    ])

    state = states.ProductionRulesState(
        production_rules_sequence=production_rules_sequence)
    self.assertTrue(state.is_terminal())

  def test_is_terminal_end_without_terminal_rule(self):
    production_rules_sequence = self._strings_to_production_rules([
        'S -> S "+" T',
        'S -> T',
        'T -> "x"',
        'T -> "x"',
    ])

    state = states.ProductionRulesState(
        production_rules_sequence=production_rules_sequence)
    self.assertTrue(state.is_terminal())

  def test_is_terminal_empty(self):
    self.assertTrue(
        states.ProductionRulesState(production_rules_sequence=[]).is_terminal())
    self.assertFalse(
        states.ProductionRulesState(production_rules_sequence=[], stack=['S'])
        .is_terminal())

  def test_get_expression_not_terminal(self):
    production_rules_sequence = self._strings_to_production_rules([
        'S -> S "+" T',
        'S -> T',
        'T -> "x"',
    ])
    # Parsing tree:
    #  S
    #  |
    #  S "+" T
    #  |
    #  T
    #  |
    # "x"
    # Expression (non-terminal):
    # x + T
    state = states.ProductionRulesState(
        production_rules_sequence=production_rules_sequence)
    self.assertEqual(state.get_expression(), 'x + T')
    self.assertEqual(state.get_expression(coefficients={'x': 42}), '42 + T')

  def test_repr_empty(self):
    state = states.ProductionRulesState(
        production_rules_sequence=[], stack=nltk.grammar.nonterminals('S'))
    self.assertEqual(
        str(state),
        'ProductionRulesState [symbols: , '
        'length_production_rules_sequence: 0, '
        'stack top: S, '
        'num_terminals / num_symbols: 0 / 0, '
        'terminal_ratio:  nan]')

  def test_repr_with_terminal_rule(self):
    production_rules_sequence = self._strings_to_production_rules([
        'S -> S "+" T',
        'S -> T',
        'T -> "x"',
        'T -> "x"',
        constants.DUMMY_PRODUCTION_RULE,
    ])
    state = states.ProductionRulesState(
        production_rules_sequence=production_rules_sequence)
    self.assertEqual(
        str(state),
        'ProductionRulesState [symbols: x + x, '
        'length_production_rules_sequence: 5, '
        'stack top: Nothing, '
        'num_terminals / num_symbols: 3 / 3, '
        'terminal_ratio: 1.00]')

  def test_repr_without_terminal_rule(self):
    production_rules_sequence = self._strings_to_production_rules([
        'S -> S "+" T',
        'S -> T',
        'T -> "x"',
        'T -> "x"',
    ])
    state = states.ProductionRulesState(
        production_rules_sequence=production_rules_sequence)
    self.assertEqual(
        str(state),
        'ProductionRulesState [symbols: x + x, '
        'length_production_rules_sequence: 4, '
        'stack top: Nothing, '
        'num_terminals / num_symbols: 3 / 3, '
        'terminal_ratio: 1.00]')

  def test_repr_expression_not_terminal(self):
    production_rules_sequence = self._strings_to_production_rules([
        'S -> S "+" T',
        'S -> T',
        'T -> "x"',
    ])
    state = states.ProductionRulesState(
        production_rules_sequence=production_rules_sequence)
    self.assertEqual(
        str(state),
        'ProductionRulesState [symbols: x + T, '
        'length_production_rules_sequence: 3, '
        'stack top: T, '
        'num_terminals / num_symbols: 2 / 3, '
        'terminal_ratio: 0.67]')

  def test_init_stack_none(self):
    # _stack attribute should be created from the input
    # production_rules_sequence.
    production_rules_sequence = self._strings_to_production_rules([
        'S -> S "+" T',
        'S -> R',
    ])
    state = states.ProductionRulesState(
        production_rules_sequence=production_rules_sequence, stack=None)
    # Use assertIs to check exact type rather than assertIsInstance.
    # https://docs.python.org/2/library/unittest.html#unittest.TestCase.assertIsInstance
    self.assertIs(type(state._stack), postprocessor.GrammarLhsStack)
    # Add 'S -> S "+" T': first push 'T', then push 'S' to the stack.
    # Stack ['T', 'S']
    # Add 'S -> R': pop 'S', then push 'R' to the stack.
    # Stack ['T', R']
    self.assertEqual(state._stack.to_list(), ['T', 'R'])

  def test_init_stack_list(self):
    state = states.ProductionRulesState(
        production_rules_sequence=[], stack=['T', 'R'])
    # Use assertIs to check exact type rather than assertIsInstance.
    # https://docs.python.org/2/library/unittest.html#unittest.TestCase.assertIsInstance
    self.assertIs(type(state._stack), postprocessor.GrammarLhsStack)
    self.assertEqual(state._stack.to_list(), ['T', 'R'])

  def test_init_stack_grammar_lhs_stack(self):
    state = states.ProductionRulesState(
        production_rules_sequence=[],
        stack=postprocessor.GrammarLhsStack(['T', 'R']))
    # Use assertIs to check exact type rather than assertIsInstance.
    # https://docs.python.org/2/library/unittest.html#unittest.TestCase.assertIsInstance
    self.assertIs(type(state._stack), postprocessor.GrammarLhsStack)
    self.assertEqual(state._stack.to_list(), ['T', 'R'])

  def test_init_stack_invalid(self):
    with self.assertRaisesRegexp(
        ValueError, 'stack is expected to be list, '
        'GrammarLhsStack or None, but got '
        '<class \'str\'>'):
      states.ProductionRulesState(production_rules_sequence=[], stack='foo')

  def test_generate_history(self):
    production_rules_sequence = self._strings_to_production_rules(
        ['S -> S "+" T', 'S -> T', 'T -> "x"'])
    state = states.ProductionRulesState(
        production_rules_sequence=production_rules_sequence)
    self.assertListEqual(state.generate_history(), ['S + T', 'T + T', 'x + T'])

  def test_is_valid_to_append(self):
    production_rules_sequence = self._strings_to_production_rules(
        ['S -> S "+" T'])
    state = states.ProductionRulesState(
        production_rules_sequence=production_rules_sequence)
    # The current stack is [T, S], the next production rule should start with S.
    self.assertTrue(state.is_valid_to_append(
        self.production_rules_dict['S -> T']))
    self.assertFalse(state.is_valid_to_append(
        self.production_rules_dict['T -> "x"']))

  def test_is_valid_to_append_init_stack(self):
    state = states.ProductionRulesState(
        production_rules_sequence=[], stack=['S'])
    # The current stack is [S], the next production rule should start with S.
    self.assertTrue(state.is_valid_to_append(
        self.production_rules_dict['S -> T']))
    self.assertFalse(state.is_valid_to_append(
        self.production_rules_dict['T -> "x"']))

  def test_stack_peek(self):
    production_rules_sequence = self._strings_to_production_rules(
        ['S -> S "+" T'])
    state = states.ProductionRulesState(
        production_rules_sequence=production_rules_sequence)
    self.assertEqual(state.stack_peek(), 'S')

  def test_stack_peek_init_stack(self):
    state = states.ProductionRulesState(
        production_rules_sequence=[], stack=['S'])
    self.assertEqual(state.stack_peek(), 'S')

  def test_append_production_rule_invalid(self):
    production_rules_sequence = self._strings_to_production_rules([
        'S -> S "+" T',
        'S -> T',
    ])
    state = states.ProductionRulesState(
        production_rules_sequence=production_rules_sequence)
    # The current stack is [T, T], the next production rule should start with T.
    # A ValueError will be raised if the production rule to append does not have
    # left hand side symbol T.
    with self.assertRaisesRegexp(
        ValueError,
        r'The left hand side symbol of production rule S -> T does not match '
        r'the top symbol in the grammar left hand side stack \(T\)'):
      state.append_production_rule(self.production_rules_dict['S -> T'])

  def test_append_production_rule(self):
    production_rules_sequence = self._strings_to_production_rules([
        'S -> S "+" T',
        'S -> T',
    ])
    state = states.ProductionRulesState(
        production_rules_sequence=production_rules_sequence)
    self.assertLen(state.production_rules_sequence, 2)
    # The grammar left hand side symbol stack is [T, T], the next production
    # rule should start with T.
    state.append_production_rule(self.production_rules_dict['T -> "x"'])
    self.assertLen(state.production_rules_sequence, 3)
    # The grammar left hand side symbol stack is [T], the next production rule
    # should start with T.
    state.append_production_rule(self.production_rules_dict['T -> "x"'])
    self.assertLen(state.production_rules_sequence, 4)
    # The grammar left hand side symbol stack is empty, the next production rule
    # can only be the dummy production rule.
    state.append_production_rule(
        self.production_rules_dict[constants.DUMMY_PRODUCTION_RULE])
    self.assertLen(state.production_rules_sequence, 5)

  def test_copy(self):
    production_rules_sequence = self._strings_to_production_rules([
        'S -> S "+" T',
        'S -> T',
    ])
    state = states.ProductionRulesState(
        production_rules_sequence=production_rules_sequence)
    new_state = state.copy()
    self.assertEqual(state, new_state)
    # Change in state will not affect new_state.
    state.append_production_rule(self.production_rules_dict['T -> "x"'])
    self.assertLen(state.production_rules_sequence, 3)
    self.assertLen(new_state.production_rules_sequence, 2)


class NumericalizeCoefficientsTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(NumericalizeCoefficientsTest, self).setUp()
    self.s = nltk.grammar.nonterminals('S')[0]

  @parameterized.parameters([
      ([nltk.grammar.nonterminals('S')[0], '+', 'a'], None, 'S + a'),
      ([nltk.grammar.nonterminals('S')[0], '+', 'a'], {}, 'S + a'),
      ([nltk.grammar.nonterminals('S')[0], '+', 'a'], {'b': 42}, 'S + a'),
      ([nltk.grammar.nonterminals('S')[0], '+', 'a'], {'a': 0.5}, 'S + 0.5'),
      (['a', '+', 'b'], {'a': 0.5, 'b': 1}, '0.5 + 1'),
  ])
  def test_numericalize_coefficients(self,
                                     raw_symbols,
                                     coefficients,
                                     expected_expression_string):
    self.assertEqual(
        states._numericalize_coefficients(
            raw_symbols=raw_symbols, coefficients=coefficients),
        expected_expression_string)


if __name__ == '__main__':
  tf.test.main()
