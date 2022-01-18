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

"""Tests for arithmetic_grammar."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from six.moves import map
import tensorflow.compat.v1 as tf

from neural_guided_symbolic_regression.utils import arithmetic_grammar


class ReadGrammarFromFileTest(tf.test.TestCase):

  def setUp(self):
    super(ReadGrammarFromFileTest, self).setUp()
    # NLTK grammar use double quotes for production rules.
    # pylint: disable=g-inconsistent-quotes
    self.expected_set = set([
        "S -> S '+' T",
        "S -> S '-' T",
        "S -> S '*' T",
        "S -> S '/' T",
        "S -> T",
        "T -> '(' S ')'",
        "T -> 'x'",
        "T -> '1'",
    ])
    # pylint: enable=g-inconsistent-quotes

  def test_read_grammar_return_grammar(self):
    grammar = arithmetic_grammar.read_grammar_from_file(
        'third_party/google_research/google_research/'
        'neural_guided_symbolic_regression/grammar/'
        'univariate_one_constant_grammar.txt',
        return_list=False)
    production_rules_set = set(map(str, grammar.productions()))
    self.assertEqual(production_rules_set, self.expected_set)

  def test_read_grammar_return_list(self):
    grammar = arithmetic_grammar.read_grammar_from_file(
        'third_party/google_research/google_research/'
        'neural_guided_symbolic_regression/grammar/'
        'univariate_one_constant_grammar.txt',
        return_list=True)
    production_rules_set = set(map(str, grammar))
    self.assertEqual(production_rules_set, self.expected_set)


class ArithmeticGrammarTest(parameterized.TestCase, tf.test.TestCase):

  def test_input_grammar_rules_not_list(self):
    with self.assertRaisesRegex(ValueError,
                                'The input grammar_rules should be list.'):
      arithmetic_grammar.Grammar('foo')

  def test_input_grammar_rules_not_unique(self):
    with self.assertRaisesRegex(ValueError,
                                'The grammar production rules are not unique.'):
      arithmetic_grammar.Grammar(['foo', 'foo'])

  def test_input_grammar_rules_contain_padding_dummy_production_rule(self):
    # If dummy production rule exists in the input grammar rules, it will be
    # duplicated with the dummy production rule appended in the
    # arithmetic_grammar.
    with self.assertRaisesRegex(ValueError,
                                'The grammar production rules are not unique.'):
      arithmetic_grammar.Grammar(['foo', 'Nothing -> None'])

  def test_input_grammar_rules_not_change(self):
    grammar_rules = ['S -> T', 'T -> "x"']
    arithmetic_grammar.Grammar(grammar_rules)
    self.assertListEqual(grammar_rules, ['S -> T', 'T -> "x"'])

  def test_basic_production_rules(self):
    grammar_rules = [
        'S -> S "+" T',
        'S -> T',
        'T -> "(" S ")"',
        'T -> "x"',
    ]

    grammar = arithmetic_grammar.Grammar(grammar_rules)

    self.assertLen(grammar.prod_rules, 5)
    self.assertEqual(grammar.num_production_rules, 5)
    self.assertEqual(grammar.padding_rule_index, 4)
    self.assertEqual(grammar.start_index.symbol(), 'S')
    self.assertEqual(str(grammar.start_rule), "S -> S '+' T")
    self.assertEqual(grammar.unique_lhs, ['Nothing', 'S', 'T'])
    self.assertEqual(grammar.num_unique_lhs, 3)
    np.testing.assert_allclose(
        grammar.masks,
        [[0., 0., 0., 0., 1.], [1., 1., 0., 0., 0.], [0., 0., 1., 1., 0.]])
    np.testing.assert_allclose(grammar.prod_rule_index_to_lhs_index,
                               [1, 1, 2, 2, 0])
    self.assertEqual(grammar.prod_rule_rhs_indices, [[1, 2], [2], [1], [], []])
    self.assertEqual(grammar.max_rhs_indices_size, 2)

  def test_basic_production_rules_add_unique_production_rule_to_start(self):
    grammar_rules = [
        'S -> S "+" T',
        'S -> T',
        'T -> "(" S ")"',
        'T -> "x"',
    ]

    grammar = arithmetic_grammar.Grammar(
        grammar_rules, add_unique_production_rule_to_start=True)

    self.assertLen(grammar.prod_rules, 6)
    self.assertEqual(grammar.num_production_rules, 6)
    self.assertEqual(grammar.padding_rule_index, 5)
    self.assertEqual(grammar.start_index.symbol(), 'O')
    self.assertEqual(str(grammar.start_rule), 'O -> S')
    self.assertEqual(grammar.unique_lhs, ['Nothing', 'O', 'S', 'T'])
    self.assertEqual(grammar.num_unique_lhs, 4)
    np.testing.assert_allclose(
        grammar.masks,
        [[0., 0., 0., 0., 0., 1.],
         [1., 0., 0., 0., 0., 0.],
         [0., 1., 1., 0., 0., 0.],
         [0., 0., 0., 1., 1., 0.]])
    np.testing.assert_allclose(grammar.prod_rule_index_to_lhs_index,
                               [1, 2, 2, 3, 3, 0])
    self.assertEqual(grammar.prod_rule_rhs_indices,
                     [[2], [2, 3], [3], [2], [], []])
    self.assertEqual(grammar.max_rhs_indices_size, 2)

  def test_basic_production_rules_padding_at_end_false(self):
    grammar_rules = [
        'S -> S "+" T',
        'S -> T',
        'T -> "(" S ")"',
        'T -> "x"',
    ]

    grammar = arithmetic_grammar.Grammar(grammar_rules, padding_at_end=False)

    self.assertLen(grammar.prod_rules, 5)
    self.assertEqual(grammar.num_production_rules, 5)
    self.assertEqual(grammar.padding_rule_index, 0)
    self.assertEqual(grammar.start_index.symbol(), 'S')
    self.assertEqual(str(grammar.start_rule), "S -> S '+' T")
    self.assertEqual(grammar.unique_lhs, ['Nothing', 'S', 'T'])
    self.assertEqual(grammar.num_unique_lhs, 3)
    np.testing.assert_allclose(
        grammar.masks,
        [[1., 0., 0., 0., 0.], [0., 1., 1., 0., 0.], [0., 0., 0., 1., 1.]])
    np.testing.assert_allclose(grammar.prod_rule_index_to_lhs_index,
                               [0, 1, 1, 2, 2])
    self.assertEqual(grammar.prod_rule_rhs_indices, [[], [1, 2], [2], [1], []])
    self.assertEqual(grammar.max_rhs_indices_size, 2)

  @parameterized.parameters([
      (True, True, "\t0: S -> T\n\t1: T -> 'x'\n\t2: Nothing -> None\n"),
      (True, False, "0: S -> T\n1: T -> 'x'\n2: Nothing -> None\n"),
      (False, True, "\t0: Nothing -> None\n\t1: S -> T\n\t2: T -> 'x'\n"),
      (False, False, "0: Nothing -> None\n1: S -> T\n2: T -> 'x'\n"),
  ])
  def test_grammar_to_string(self, padding_at_end, indent, expected_string):
    grammar_rules = [
        'S -> T',
        'T -> "x"',
    ]

    grammar = arithmetic_grammar.Grammar(
        grammar_rules, padding_at_end=padding_at_end)

    self.assertEqual(grammar.grammar_to_string(indent=indent), expected_string)

  def test_invalid_grammar_string_no_space_before_arrow(self):
    with self.assertRaisesRegex(ValueError, 'Unable to parse'):
      # No space between arrow and left hand side symbol.
      arithmetic_grammar.Grammar(['a-> b'])

  def test_invalid_grammar_string_no_space_after_arrow(self):
    # No space between arrow and right hand side symbol.
    # This is a valid input and should not raise error.
    arithmetic_grammar.Grammar(['a ->b'])

  def test_invalid_grammar_string_no_arrow(self):
    with self.assertRaisesRegex(ValueError, 'Unable to parse'):
      # Invalid input with no arrow.
      arithmetic_grammar.Grammar(['a b'])

  def test_invalid_grammar_string_two_left_hand_side_symbols(self):
    with self.assertRaisesRegex(ValueError, 'Unable to parse'):
      # Invalid input with more than one left hand side symbol.
      arithmetic_grammar.Grammar(['a b -> c'])

  def test_invalid_grammar_string_no_left_hand_side_symbol(self):
    with self.assertRaisesRegex(ValueError, 'Unable to parse'):
      # Invalid input with no left hand side symbol.
      arithmetic_grammar.Grammar([' -> c'])

  def test_invalid_grammar_string_empty_right_hand_side_symbol(self):
    # No right hand side symbol.
    # This is a valid input and should not raise error.
    arithmetic_grammar.Grammar(['a -> '])

  def test_parse_expressions_to_indices_sequences_input_not_list(self):
    grammar_rules = [
        'S -> S "+" T',
        'S -> T',
        'T -> "(" S ")"',
        'T -> "x"',
    ]

    grammar = arithmetic_grammar.Grammar(grammar_rules)

    with self.assertRaisesRegex(
        ValueError, 'expression_strings is expected to be list, but got'):
      grammar.parse_expressions_to_indices_sequences(
          # Note the input expression_strings is a string not a list of strings.
          expression_strings='x + ( x )',
          max_length=8
      )

  def test_parse_expressions_to_indices_sequences_short_max_length(self):
    grammar_rules = [
        'S -> S "+" T',
        'S -> T',
        'T -> "(" S ")"',
        'T -> "x"',
    ]

    grammar = arithmetic_grammar.Grammar(grammar_rules)

    with self.assertRaisesRegex(
        ValueError,
        r'The number of production rules to parse expression .* '
        'can not be greater than max_length'):
      grammar.parse_expressions_to_indices_sequences(
          expression_strings=['x + ( x )'],
          max_length=2
      )

  def test_parse_expressions_to_indices_sequences_invalid_expression_string(
      self):
    grammar_rules = [
        'S -> S "+" T',
        'S -> T',
        'T -> "(" S ")"',
        'T -> "x"',
    ]

    grammar = arithmetic_grammar.Grammar(grammar_rules)
    with self.assertRaisesRegex(
        ValueError, 'cannot be parsed to production rules'):
      grammar.parse_expressions_to_indices_sequences(
          expression_strings=['x x'],
          max_length=8
      )

  def test_grammar_with_callables(self):
    grammar_rules = [
        'S -> S "+" S',  # index 0
        'S -> S "-" S',  # index 1
        'S -> "FUNCTION1(" P ")"',  # index 2
        'P -> T',  # index 3
        'P -> "1" "+" T',  # index 4
        'S -> T',  # index 5
        'T -> "FUNCTION2(" "x" "," "c" ")"',  # index 6
    ]  # padding rule index 7
    grammar = arithmetic_grammar.Grammar(grammar_rules)
    indices_sequences = grammar.parse_expressions_to_indices_sequences(
        expression_strings=[
            'FUNCTION1( FUNCTION2( x , c ) ) - '
            'FUNCTION2( x , c ) + FUNCTION2( x , c )'],
        max_length=10
    )
    np.testing.assert_equal(
        indices_sequences,
        [
            # Preorder traversal of parsing tree.
            # S
            # |
            # S                        '+'             S
            # |                                        |
            # S         '-'             S              T
            # |                         |              |
            # 'FUNCTION1(' P ')'        T      'FUNCTION2( x , c )'
            #              |            |
            #              T     'FUNCTION2( x , c )'
            #              |
            # 'FUNCTION2( x , c )'
            [
                0,  # 'S -> S "+" S'
                1,  # 'S -> S "-" S'
                2,  # 'S -> "FUNCTION1(" P ")"'
                3,  # 'P -> T'
                6,  # 'T -> "FUNCTION2(" "x" "," "c" ")"'
                5,  # 'S -> T'
                6,  # 'T -> "FUNCTION2(" "x" "," "c" ")"'
                5,  # 'S -> T'
                6,  # 'T -> "FUNCTION2(" "x" "," "c" ")"'
                7,  # Padding dummy production rule.
            ]
        ]
    )

  def test_parse_expressions_to_indices_sequences(self):
    grammar_rules = [
        'S -> S "+" T',  # index 0
        'S -> T',    # index 1
        'T -> "(" S ")"',  # index 2
        'T -> "x"',  # index 3
    ]  # padding rule index 4

    grammar = arithmetic_grammar.Grammar(grammar_rules)
    indices_sequences = grammar.parse_expressions_to_indices_sequences(
        expression_strings=['x + ( x )'],
        max_length=8
    )

    np.testing.assert_equal(
        indices_sequences,
        [
            # Expression string: 'x + ( x )'
            # Preorder traversal of parsing tree.
            # S
            # |
            # S '+' T
            # |     |
            # T    '(' S ')'
            # |        |
            # 'x'     'x'
            [
                0,  # 'S -> S "+" T'
                1,  # 'S -> T'
                3,  # 'T -> "x"'
                2,  # 'T -> "(" S ")"'
                1,  # 'S -> T'
                3,  # 'T -> "x"'
                4,  # Padding dummy production rule.
                4,  # Padding dummy production rule.
            ]
        ]
    )

  def test_parse_expressions_to_indices_sequences_padding_at_end_false(self):
    grammar_rules = [
        'S -> S "+" T',  # index 1
        'S -> T',    # index 2
        'T -> "(" S ")"',  # index 3
        'T -> "x"',  # index 4
    ]  # padding rule index 0

    grammar = arithmetic_grammar.Grammar(grammar_rules, padding_at_end=False)
    indices_sequences = grammar.parse_expressions_to_indices_sequences(
        expression_strings=['x + ( x )'],
        max_length=8
    )

    np.testing.assert_equal(
        indices_sequences,
        [
            # Expression string: 'x + ( x )'
            # Preorder traversal of parsing tree.
            # S
            # |
            # S '+' T
            # |     |
            # T    '(' S ')'
            # |        |
            # 'x'     'x'
            [
                1,  # 'S -> S "+" T'
                2,  # 'S -> T'
                4,  # 'T -> "x"'
                3,  # 'T -> "(" S ")"'
                2,  # 'S -> T'
                4,  # 'T -> "x"'
                0,  # Padding dummy production rule.
                0,  # Padding dummy production rule.
            ]
        ]
    )

  def test_parse_expressions_to_indices_sequences_pad_front_unique_start(self):
    grammar_rules = [
        'S -> S "+" T',  # index 2
        'S -> T',    # index 3
        'T -> "(" S ")"',  # index 4
        'T -> "x"',  # index 5
    ]  # padding rule index 0
    # 'O -> S' will be added with index 1.

    grammar = arithmetic_grammar.Grammar(
        grammar_rules,
        padding_at_end=False,
        add_unique_production_rule_to_start=True)
    indices_sequences = grammar.parse_expressions_to_indices_sequences(
        expression_strings=['x + ( x )'],
        max_length=8
    )

    np.testing.assert_equal(
        indices_sequences,
        [
            # Expression string: 'x + ( x )'
            # Preorder traversal of parsing tree.
            # O
            # |
            # S
            # |
            # S '+' T
            # |     |
            # T    '(' S ')'
            # |        |
            # 'x'     'x'
            [
                1,  # 'O -> S'
                2,  # 'S -> S "+" T'
                3,  # 'S -> T'
                5,  # 'T -> "x"'
                4,  # 'T -> "(" S ")"'
                3,  # 'S -> T'
                5,  # 'T -> "x"'
                0,  # Padding dummy production rule.
            ]
        ]
    )

  def test_parse_expressions_to_tensor(self):
    grammar_rules = [
        'S -> S "+" T',
        'S -> T',
        'T -> "(" S ")"',
        'T -> "x"',
    ]

    grammar = arithmetic_grammar.Grammar(grammar_rules)

    expression_tensor = grammar.parse_expressions_to_tensor(
        expression_strings=['x + ( x )'],
        max_length=8
    )

    np.testing.assert_allclose(
        expression_tensor,
        [
            # Expression string: 'x + ( x )'
            # Preorder traversal of parsing tree.
            # S
            # |
            # S '+' T
            # |     |
            # T    '(' S ')'
            # |        |
            # 'x'     'x'
            [
                [1., 0., 0., 0., 0.],  # 'S -> S "+" T'
                [0., 1., 0., 0., 0.],  # 'S -> T'
                [0., 0., 0., 1., 0.],  # 'T -> "x"'
                [0., 0., 1., 0., 0.],  # 'T -> "(" S ")"'
                [0., 1., 0., 0., 0.],  # 'S -> T'
                [0., 0., 0., 1., 0.],  # 'T -> "x"'
                [0., 0., 0., 0., 1.],  # Padding dummy production rule.
                [0., 0., 0., 0., 1.],  # Padding dummy production rule.
            ]
        ]
    )

  def test_parse_expressions_to_tensor_padding_at_end_false(self):
    grammar_rules = [
        'S -> S "+" T',
        'S -> T',
        'T -> "(" S ")"',
        'T -> "x"',
    ]

    grammar = arithmetic_grammar.Grammar(grammar_rules, padding_at_end=False)

    expression_tensor = grammar.parse_expressions_to_tensor(
        expression_strings=['x + ( x )'],
        max_length=8
    )

    np.testing.assert_allclose(
        expression_tensor,
        [
            # Expression string: 'x + ( x )'
            # Preorder traversal of parsing tree.
            # S
            # |
            # S '+' T
            # |     |
            # T    '(' S ')'
            # |        |
            # 'x'     'x'
            [
                [0., 1., 0., 0., 0.],  # 'S -> S "+" T'
                [0., 0., 1., 0., 0.],  # 'S -> T'
                [0., 0., 0., 0., 1.],  # 'T -> "x"'
                [0., 0., 0., 1., 0.],  # 'T -> "(" S ")"'
                [0., 0., 1., 0., 0.],  # 'S -> T'
                [0., 0., 0., 0., 1.],  # 'T -> "x"'
                [1., 0., 0., 0., 0.],  # Padding dummy production rule.
                [1., 0., 0., 0., 0.],  # Padding dummy production rule.
            ]
        ]
    )


if __name__ == '__main__':
  tf.test.main()
