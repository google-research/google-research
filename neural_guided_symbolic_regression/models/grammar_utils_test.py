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

"""Tests for grammar_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from neural_guided_symbolic_regression.models import grammar_utils
from neural_guided_symbolic_regression.utils import arithmetic_grammar


class GrammarUtilsTest(tf.test.TestCase):

  def test_load_grammar(self):
    grammar = grammar_utils.load_grammar(
        grammar_path='third_party/google_research/google_research/'
        'neural_guided_symbolic_regression/grammar/'
        'univariate_one_constant_grammar.txt')
    grammar_production_rule_strings = [str(rule) for rule in grammar.prod_rules]
    self.assertIsInstance(grammar, arithmetic_grammar.Grammar)
    self.assertListEqual(
        grammar_production_rule_strings,
        [
            # pylint: disable=g-inconsistent-quotes
            "Nothing -> None",  # Padding at 0-th index.
            "O -> S",  # Unique starting production rule at 1-st index.
            "S -> S '+' T",
            "S -> S '-' T",
            "S -> S '*' T",
            "S -> S '/' T",
            "S -> T",
            "T -> '(' S ')'",
            "T -> 'x'",
            "T -> '1'",
            # pylint: enable=g-inconsistent-quotes
        ])


class GetNumExpressionsTest(parameterized.TestCase):
  """Tests get_num_expressions().

  In this test, we use this grammar.
  S -> S '+' T
  S -> S '-' T
  S -> S '*' T
  S -> S '/' T
  S -> T
  T -> '(' S ')'
  T -> 'x'
  T -> '1'

  It contains:
  s_to_s_t = 4
  s_to_t = 1
  t_to_s = 1
  t_to_terminal = 2
  """

  def test_get_num_expressions_entire_arrays(self):
    n_s, n_t = grammar_utils._get_num_expressions(
        s_to_s_t=4, s_to_t=1, t_to_s=1, t_to_terminal=2,
        max_num_production_rules=2)
    np.testing.assert_allclose(n_s, [1, 5, 35])
    np.testing.assert_allclose(n_t, [1, 3, 7])

  @parameterized.parameters(
      # A sequence of 0 production rule, 1 choice (empty).
      (0, 1),
      # A sequence of 1 production rule, the number of production rules
      # starting with S.
      (1, 5),
      # A sequence of 2 production rules.
      # (5 * 1 + 1 * 3) * 4 + 3 = 35
      (2, 35))
  def test_get_num_expressions(
      self, max_num_production_rules, expected_num_expressions):
    self.assertEqual(
        grammar_utils.get_num_expressions(
            s_to_s_t=4, s_to_t=1, t_to_s=1, t_to_terminal=2,
            max_num_production_rules=max_num_production_rules),
        expected_num_expressions)


if __name__ == '__main__':
  tf.test.main()
