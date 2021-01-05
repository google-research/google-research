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

"""Tests for expression_generalization_metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

from neural_guided_symbolic_regression.utils import expression_generalization_metrics


class ExpressionGeneralizationMetricsTest(parameterized.TestCase):

  def test_combine_list_values_in_dict(self):
    expression_dict = {'x + 1': ['x + 1'],
                       '1': ['1', '(1)', '((1))']}
    all_expressions = (
        expression_generalization_metrics.combine_list_values_in_dict(
            expression_dict))
    self.assertCountEqual(all_expressions, ['x + 1', '1', '(1)', '((1))'])

  @parameterized.parameters([
      # 'x + 1' and '(x + 1)' should be semantically equivalent.
      ({'1': ['1'], 'x': ['x'], 'x + 1': ['x + 1'], 'x - 1': ['x - 1']},
       {'1': ['1'], 'x': ['x'], 'x + 1': ['(x + 1)'], 'x - 1': ['(x - 1)']},
       False,
       ['1', 'x'], ['x + 1', 'x - 1'],
       ['1', 'x', 'x + 1', 'x - 1'], []),
      # All expressions are unseen.
      ({'2*x + 1': ['2*x + 1'], '3*x + 1': ['3*x + 1']},
       {'1': ['1'], 'x': ['x']},
       False,
       [], ['2*x + 1', '3*x + 1'],
       [], ['2*x + 1', '3*x + 1']),
      # One additional expression in training_expressions should not affect the
      # result.
      ({'1': ['1'], 'x': ['x']},
       {'1': ['1'], 'x': ['x'], 'x + 1': ['(x + 1)']},
       False,
       ['1', 'x'], [],
       ['1', 'x'], []),
      # When training_expressions is empty.
      ({'1': ['1'], 'x': ['x']},
       {},
       False,
       [], ['1', 'x'],
       [], ['1', 'x']),
      # When one simplified expression has multiple expression equivalences.
      ({'1': ['1', '1', '(1)', '((1))']},
       {'x': ['x']},
       False,
       [], ['1', '1', '(1)', '((1))'],
       [], ['1', '1', '(1)', '((1))']),
      # When generated_expressions contains duplicates.
      ({'x': ['x', 'x', 'x']},
       {'x': ['x']},
       False,
       ['x', 'x', 'x'], [],
       ['x', 'x', 'x'], []),
      # When all generated_expressions are syntactic novelty but not semantic
      # novelty.
      ({'1': ['1', '(1)']},
       {'1': ['((1))']},
       False,
       [], ['1', '(1)'],
       ['1', '(1)'], []),
      # When generated_expressions and training_expressions are the same.
      ({'x': ['((x))']},
       {'x': ['((x))']},
       False,
       ['((x))'], [],
       ['((x))'], []),
      # When sympy.simplify makes a mistake in computing simplified expressions
      # for generated_expressions.
      ({'(x)': ['((x))']},
       {'x': ['((x))']},
       False,
       ['((x))'], [],
       ['((x))'], []),
      # Test whether deduplicate works.
      ({'1': ['1', '1', '(1)', '(1)']},
       {'x': ['x']},
       True,
       [], ['1', '(1)'],
       [], ['1']),
      # Test whether deduplicate works.
      ({'1': ['1', '1', '(1)', '(1)']},
       {'1': ['1']},
       True,
       ['1', '1'], ['(1)'],
       ['1', '1', '(1)', '(1)'], []),

  ])
  def test_get_seen_and_unseen_expressions(
      self,
      generated_expressions,
      training_expressions,
      deduplicate_unseen,
      expected_syntactic_seen_expressions,
      expected_syntactic_unseen_expressions,
      expected_semantic_seen_expressions,
      expected_semantic_unseen_expressions):
    seen_and_unseen_expressions = (
        expression_generalization_metrics.get_seen_and_unseen_expressions(
            generated_expressions, training_expressions, deduplicate_unseen))
    # The ordering of the expressions does not matter.
    self.assertCountEqual(
        seen_and_unseen_expressions.syntactic_novelty[0],
        expected_syntactic_seen_expressions)
    self.assertCountEqual(
        seen_and_unseen_expressions.syntactic_novelty[1],
        expected_syntactic_unseen_expressions)
    self.assertCountEqual(
        seen_and_unseen_expressions.semantic_novelty[0],
        expected_semantic_seen_expressions)
    self.assertCountEqual(
        seen_and_unseen_expressions.semantic_novelty[1],
        expected_semantic_unseen_expressions)

  @parameterized.parameters([
      (['x + 1', 'x - 1'],
       ['1', 'x'],
       expression_generalization_metrics.NoveltySummary(
           num_seen=2,
           num_unseen=2,
           novelty_rate=0.5)),
      (['x + 1', 'x - 1'],
       [],
       expression_generalization_metrics.NoveltySummary(
           num_seen=2,
           num_unseen=0,
           novelty_rate=0)),
      ([],
       ['1', 'x'],
       expression_generalization_metrics.NoveltySummary(
           num_seen=0,
           num_unseen=2,
           novelty_rate=1)),
      # With replicates.
      (['x + 1', 'x - 1'],
       ['1', '1', 'x'],
       expression_generalization_metrics.NoveltySummary(
           num_seen=2,
           num_unseen=3,
           novelty_rate=0.6)),
  ])
  def test_get_novelty_rate(
      self,
      seen_expressions,
      unseen_expressions,
      expected):
    result = expression_generalization_metrics.get_novelty_rate(
        seen_expressions, unseen_expressions)
    self.assertEqual(result, expected)

  def test_get_novelty_rate_raises(self):
    with self.assertRaisesRegex(ValueError,
                                'Total number of expressions cannot be zero.'):
      expression_generalization_metrics.get_novelty_rate([], [])

  def test_get_distance_from_expected_condition_ignore_sympy_failure(self):
    expression_df = pd.DataFrame({
        'expression_string': [
            'good string 1', 'good string 2', 'sympy failed string 1',
            'non-terminated string 1'
        ],
        'true_leading_at_0': [-1, -1, np.nan, np.nan],
        'true_leading_at_inf': [0, -2, np.nan, np.nan],
        'expected_leading_at_0': [-1, -1, -1, 9],
        'expected_leading_at_inf': [-1, -1, -1, 9],
        'is_terminal': [1, 1, 1, 0]
    })
    result = (
        expression_generalization_metrics.get_distance_from_expected_condition(
            expression_df,
            distance_for_nonterminal=99,
            distance_for_sympy_failure=None))
    expected = pd.DataFrame({
        'expected_leading_at_0': [-1, 9],
        'expected_leading_at_inf': [-1, 9],
        'distance_from_expected_condition': [1, 99]
    })
    pd.testing.assert_frame_equal(
        result, expected, check_dtype=False, check_like=True)

  def test_get_distance_from_expected_condition_consider_sympy_failure(self):
    expression_df = pd.DataFrame({
        'expression_string': ['sympy failed string 1', 'sympy failed string 2'],
        'true_leading_at_0': [np.nan, np.nan],
        'true_leading_at_inf': [np.nan, np.nan],
        'expected_leading_at_0': [9, 9],
        'expected_leading_at_inf': [9, 9],
        'is_terminal': [1, 1]
    })
    result = (
        expression_generalization_metrics.get_distance_from_expected_condition(
            expression_df,
            distance_for_nonterminal=99,
            distance_for_sympy_failure=99))
    expected = pd.DataFrame({
        'expected_leading_at_0': [9],
        'expected_leading_at_inf': [9],
        'distance_from_expected_condition': [99]
    })
    pd.testing.assert_frame_equal(
        result, expected, check_dtype=False, check_like=True)


if __name__ == '__main__':
  tf.test.main()
