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

"""Tests for generate_empirical_distribution_df."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

from neural_guided_symbolic_regression.models import grammar_utils
from neural_guided_symbolic_regression.utils import generate_empirical_distribution_df


class GenerateEmpiricalDistributionDfHelperTest(parameterized.TestCase):

  @parameterized.parameters([
      ([1, 4], 5),
      ([1, 4, 3], 5),
      ([1, 4, 3, 5, 2, 6], 3),
      ([1, 4, 3, 5, 2, 6, 8], 3)
  ])
  def test_get_number_valid_next_step(
      self,
      prod_rules_sequence_indices,
      expected):
    grammar = grammar_utils.load_grammar(
        grammar_path='third_party/google_research/google_research/'
        'neural_guided_symbolic_regression/grammar/'
        'univariate_one_constant_grammar.txt')
    number_valid_next_step = (
        generate_empirical_distribution_df.get_number_valid_next_step(
            prod_rules_sequence_indices, grammar))
    self.assertEqual(number_valid_next_step, expected)


class GenerateEmpiricalDistributionDfMainTest(parameterized.TestCase):

  def setUp(self):
    super(GenerateEmpiricalDistributionDfMainTest, self).setUp()
    # Production rule sequence of ( 1 ) is 1,6,7,6,9.
    # Production rule sequence of ( x ) is 1,6,7,6,8.
    self.expression_df = pd.DataFrame(
        {'expression_string': ['( 1 )', '( x )'],
         'leading_at_0': [0, 1],
         'leading_at_inf': [0, 1]})
    self.grammar = grammar_utils.load_grammar(
        grammar_path='third_party/google_research/google_research/'
        'neural_guided_symbolic_regression/grammar/'
        'univariate_one_constant_grammar.txt')
    self.max_length = 11

  def test_get_partial_sequence_df(self):
    partial_sequence_df = (
        generate_empirical_distribution_df.get_partial_sequence_df(
            self.expression_df, self.grammar, self.max_length))
    expected_partial_sequence_indices = ['1', '1_6', '1_6_7', '1_6_7_6',
                                         '1', '1_6', '1_6_7', '1_6_7_6']
    self.assertListEqual(
        list(partial_sequence_df['partial_sequence_indices'].values),
        expected_partial_sequence_indices)

  @parameterized.parameters([
      (None,
       'partial_sequence_indices',
       ['1', 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
      (None,
       'partial_sequence_indices',
       ['1_6', 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
      (None,
       'partial_sequence_indices',
       ['1_6_7', 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
      (None,
       'partial_sequence_indices',
       ['1_6_7_6', 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
      (2,
       'tail_partial_sequence_indices',
       ['1', 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
      (2,
       'tail_partial_sequence_indices',
       ['1_6', 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
      (2,
       'tail_partial_sequence_indices',
       ['6_7', 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
      (2,
       'tail_partial_sequence_indices',
       ['7_6', 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
  ])
  def test_get_empirical_distribution_df(self,
                                         tail_length,
                                         level_name,
                                         multi_index_to_check,
                                         expected_probabilities):
    properties = ['leading_at_0', 'leading_at_inf']
    num_production_rules = len(self.grammar.prod_rules)
    partial_sequence_df = (
        generate_empirical_distribution_df.get_partial_sequence_df(
            self.expression_df, self.grammar, self.max_length))
    empirical_distribution_df = (
        generate_empirical_distribution_df.get_empirical_distribution_df(
            partial_sequence_df, properties, num_production_rules, tail_length))
    levels = [level_name] + properties
    np.testing.assert_array_almost_equal(
        empirical_distribution_df.xs(multi_index_to_check,
                                     level=levels).values[0],
        expected_probabilities)

  def test_get_empirical_distribution_df_without_condition(self):
    num_production_rules = len(self.grammar.prod_rules)
    partial_sequence_df = (
        generate_empirical_distribution_df.get_partial_sequence_df(
            self.expression_df, self.grammar, self.max_length))
    empirical_distribution_df = (
        generate_empirical_distribution_df.get_empirical_distribution_df(
            partial_sequence_df, [], num_production_rules, None))
    expected = pd.DataFrame(
        np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5]]),
        columns=range(10))
    expected['partial_sequence_indices'] = ['1', '1_6', '1_6_7', '1_6_7_6']
    expected.set_index('partial_sequence_indices', inplace=True)
    pd.testing.assert_frame_equal(empirical_distribution_df,
                                  expected,
                                  check_dtype=False,
                                  check_index_type=False,
                                  check_column_type=False,
                                  check_names=False)

  @parameterized.parameters([
      ('1_6_7_6', 1, '6'),
      ('1_6_7_6', 2, '7_6'),
      ('1_6', 3, '1_6'),
  ])
  def test_extract_tail_partial_sequence(self,
                                         partial_sequence_string,
                                         tail_length,
                                         expected):
    tail_partial_sequence_string = (
        generate_empirical_distribution_df.extract_tail_partial_sequence(
            partial_sequence_string, tail_length))
    self.assertEqual(tail_partial_sequence_string, expected)


if __name__ == '__main__':
  tf.test.main()
