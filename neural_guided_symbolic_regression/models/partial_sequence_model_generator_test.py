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

"""Tests for partial_sequence_model_generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import mock
import numpy as np
import numpy.testing as numpy_testing
import pandas as pd
import tensorflow.compat.v1 as tf

from neural_guided_symbolic_regression.models import grammar_utils
from neural_guided_symbolic_regression.models import partial_sequence_model_generator


class GenerateNextProductionRuleRandomlyTest(parameterized.TestCase):

  @parameterized.parameters([
      (1, 1), (2, 1), (3, 2), (4, 2), (5, 1), (6, 2), (7, 1), (8, 2)])
  def test_generate_next_production_rule_randomly(
      self, seed, expected_next_production_rule):
    self.assertEqual(
        partial_sequence_model_generator.generate_next_production_rule_randomly(
            num_production_rules=4,
            # Indices 0 and 3 will never be selected.
            next_production_rule_distribution=np.array([0., 1., 1., 0.]),
            random_state=np.random.RandomState(seed)),
        expected_next_production_rule)


class GenerateNextProductionRuleFromModelTest(parameterized.TestCase):

  def setUp(self):
    super(GenerateNextProductionRuleFromModelTest, self).setUp()

    self.max_length = 6
    self.partial_sequence = np.array([1, 2])
    self.partial_sequence_length = 2
    self.next_production_rule_mask = np.array([0., 1., 1., 0.])

    self.padded_partial_sequence = np.array([1, 2, 0, 0, 0, 0])
    self.returned_masked_probabilities = [[0., 0.25, 0.75, 0.]]

  @parameterized.parameters([
      # No sampling. The result is deterministic and random_state is not used.
      (False, 1, 2), (False, 2, 2), (False, 3, 2), (False, 4, 2), (False, 5, 2),
      # Sampling the masked_probabilities.
      (True, 1, 2), (True, 2, 2), (True, 3, 2), (True, 4, 2), (True, 5, 1),
  ])
  def test_generate_next_production_rule_from_model(
      self, sampling, seed, expected_next_production_rule):
    sess = mock.MagicMock()
    sess.run = mock.MagicMock(return_value=self.returned_masked_probabilities)

    self.assertEqual(
        partial_sequence_model_generator
        .generate_next_production_rule_from_model(
            sess,
            max_length=self.max_length,
            partial_sequence=self.partial_sequence,
            next_production_rule_mask=self.next_production_rule_mask,
            sampling=sampling,
            random_state=np.random.RandomState(seed)),
        expected_next_production_rule)
    sess.run.assert_called_once()
    # Since assert_called_once_with() cannot compare numpy array, use call_args
    # to get the input arguments.
    # http://www.voidspace.org.uk/python/mock/mock.html#mock.Mock.call_args
    self.assertTupleEqual(
        sess.run.call_args[0], ('predictions/masked_probabilities:0',))

    feed_dict = sess.run.call_args[1]['feed_dict']
    self.assertLen(feed_dict, 3)
    np.testing.assert_allclose(
        feed_dict['serving_input/partial_sequence:0'],
        np.array([self.padded_partial_sequence], dtype=np.int32))
    np.testing.assert_allclose(
        feed_dict['serving_input/partial_sequence_length:0'],
        np.array([self.partial_sequence_length], dtype=np.int32))
    np.testing.assert_allclose(
        feed_dict['serving_input/next_production_rule_mask:0'],
        np.array([self.next_production_rule_mask]))

  def test_generate_next_production_rule_from_model_with_conditions(self):
    sess = mock.MagicMock()
    sess.run = mock.MagicMock(return_value=self.returned_masked_probabilities)

    self.assertEqual(
        partial_sequence_model_generator
        .generate_next_production_rule_from_model(
            sess,
            max_length=self.max_length,
            partial_sequence=self.partial_sequence,
            next_production_rule_mask=self.next_production_rule_mask,
            sampling=False,
            conditions={
                'numerical_values': np.array([[1., 2., 3.]]),
                'symbolic_property_0': np.array([0.]),
                'symbolic_property_1': np.array([1.])}),
        2)
    sess.run.assert_called_once()
    # Since assert_called_once_with() cannot compare numpy array, use call_args
    # to get the input arguments.
    # http://www.voidspace.org.uk/python/mock/mock.html#mock.Mock.call_args
    self.assertTupleEqual(
        sess.run.call_args[0], ('predictions/masked_probabilities:0',))

    feed_dict = sess.run.call_args[1]['feed_dict']
    self.assertLen(feed_dict, 6)
    # Check conditions.
    np.testing.assert_allclose(
        feed_dict['serving_input/numerical_values:0'], np.array([[1., 2., 3.]]))
    np.testing.assert_allclose(
        feed_dict['serving_input/symbolic_property_0:0'], np.array([0]))
    np.testing.assert_allclose(
        feed_dict['serving_input/symbolic_property_1:0'], np.array([1]))


class GenerateExpressionTest(parameterized.TestCase):

  def setUp(self):
    super(GenerateExpressionTest, self).setUp()

    # The grammar contains:
    # 0: Nothing -> None
    # 1: O -> S
    # 2: S -> S '+' T
    # 3: S -> S '-' T
    # 4: S -> S '*' T
    # 5: S -> S '/' T
    # 6: S -> T
    # 7: T -> '(' S ')'
    # 8: T -> 'x'
    # 9: T -> '1'
    self.grammar = grammar_utils.load_grammar(
        grammar_path='third_party/google_research/google_research/'
        'neural_guided_symbolic_regression/grammar/'
        'univariate_one_constant_grammar.txt')

    self.empirical_distribution_df = pd.DataFrame(
        np.array([[-4, -4], [-3, -3], [-3, 0], [-3, 1]]),
        columns=['leading_at_0', 'leading_at_inf'])
    self.empirical_distribution_df['partial_sequence_indices'] = '1'
    self.empirical_distribution_df.set_index(
        ['partial_sequence_indices', 'leading_at_0', 'leading_at_inf'],
        inplace=True)
    self.empirical_distribution_df = pd.DataFrame(
        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0.05, 0.9, 0.05, 0, 0, 0],
         [0, 0, 0.25, 0.25, 0, 0.5, 0, 0, 0, 0],
         [0, 0, 0.25, 0.25, 0, 0.5, 0, 0, 0, 0]],
        index=self.empirical_distribution_df.index)
    self.limited_history_empirical_distribution_df = (
        self.empirical_distribution_df.copy())
    self.limited_history_empirical_distribution_df.index.names = (
        ['tail_partial_sequence_indices'] +
        self.limited_history_empirical_distribution_df.index.names[1:])
    self.empirical_distribution_df_without_condition = (
        self.empirical_distribution_df.iloc[[0], :].reset_index(
            level=['leading_at_0', 'leading_at_inf'], drop=True))
    self.limited_history_empirical_distribution_df_without_condition = (
        self.limited_history_empirical_distribution_df.iloc[[0], :].reset_index(
            level=['leading_at_0', 'leading_at_inf'], drop=True))
    self.next_production_rule_mask = [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]

  def test_get_masked_probabilities_from_model_partial_sequence_too_long(self):
    with self.assertRaisesRegex(
        ValueError,
        r'The length of partial_sequence \(3\) cannot be greater than '
        r'max_length \(2\)'):
      partial_sequence_model_generator.get_masked_probabilities_from_model(
          sess=None,  # Not used in this test.
          max_length=2,
          partial_sequence=np.array([5, 6, 7]),
          next_production_rule_mask=None,  # Not used in this test.
          )

  def test_get_starting_partial_sequence_partial_sequence_not_none(self):
    input_partial_sequence = [1, 2, 3]

    partial_sequence = (
        partial_sequence_model_generator._get_starting_partial_sequence(
            partial_sequence=input_partial_sequence,
            grammar=self.grammar,
            random_state=np.random.RandomState()))

    # Modify partial_sequence in place shouldn't affect input_partial_sequence.
    partial_sequence.append(42)
    self.assertListEqual(partial_sequence, [1, 2, 3, 42])
    self.assertListEqual(input_partial_sequence, [1, 2, 3])

  def test_get_starting_partial_sequence_partial_sequence_none(self):
    self.assertListEqual(
        partial_sequence_model_generator._get_starting_partial_sequence(
            partial_sequence=None,
            grammar=self.grammar,
            random_state=np.random.RandomState()),
        [1])

  @parameterized.parameters([
      # The max length is reached before expression is terminal.
      ([2, 3, 4, 5, 2, 3, 4, 5],
       None,
       {'expression_string': 'S / T * T - T + T',
        'is_terminal': False,
        'production_rule_sequence': [1, 2, 3, 4, 5],
        'history': [
            'S', 'S + T', 'S - T + T', 'S * T - T + T', 'S / T * T - T + T']}),
      ([2, 3, 4, 5, 2, 3, 4, 5],
       [1],
       {'expression_string': 'S / T * T - T + T',
        'is_terminal': False,
        'production_rule_sequence': [1, 2, 3, 4, 5],
        'history': [
            'S', 'S + T', 'S - T + T', 'S * T - T + T', 'S / T * T - T + T']}),
      ([3, 4, 5, 2, 3, 4, 5],
       [1, 2],
       {'expression_string': 'S / T * T - T + T',
        'is_terminal': False,
        'production_rule_sequence': [1, 2, 3, 4, 5],
        'history': [
            'S', 'S + T', 'S - T + T', 'S * T - T + T', 'S / T * T - T + T']}),
      ([4, 5, 2, 3, 4, 5],
       [1, 2, 3],
       {'expression_string': 'S / T * T - T + T',
        'is_terminal': False,
        'production_rule_sequence': [1, 2, 3, 4, 5],
        'history': [
            'S', 'S + T', 'S - T + T', 'S * T - T + T', 'S / T * T - T + T']}),
      # The expression is terminal before max length is reached.
      ([6, 9],
       None,
       {'expression_string': '1',
        'is_terminal': True,
        'production_rule_sequence': [1, 6, 9],
        'history': ['S', 'T', '1']}),
      ([6, 9],
       [1],
       {'expression_string': '1',
        'is_terminal': True,
        'production_rule_sequence': [1, 6, 9],
        'history': ['S', 'T', '1']}),
      ([9],
       [1, 6],
       {'expression_string': '1',
        'is_terminal': True,
        'production_rule_sequence': [1, 6, 9],
        'history': ['S', 'T', '1']}),
  ])
  def test_generate_expression_sess_none(
      self, mock_next_production_rules, partial_sequence, expected_result):
    with mock.patch.object(
        partial_sequence_model_generator,
        'generate_next_production_rule_randomly',
        side_effect=mock_next_production_rules):
      self.assertDictEqual(
          partial_sequence_model_generator.generate_expression(
              sess=None,
              grammar=self.grammar,
              max_length=5,
              partial_sequence=partial_sequence),
          expected_result)

  @parameterized.parameters([
      # The max length is reached before expression is terminal.
      ([2, 3, 4, 5, 2, 3, 4, 5],
       None,
       {'expression_string': 'S / T * T - T + T',
        'is_terminal': False,
        'production_rule_sequence': [1, 2, 3, 4, 5],
        'history': [
            'S', 'S + T', 'S - T + T', 'S * T - T + T', 'S / T * T - T + T']}),
      ([2, 3, 4, 5, 2, 3, 4, 5],
       [1],
       {'expression_string': 'S / T * T - T + T',
        'is_terminal': False,
        'production_rule_sequence': [1, 2, 3, 4, 5],
        'history': [
            'S', 'S + T', 'S - T + T', 'S * T - T + T', 'S / T * T - T + T']}),
      ([3, 4, 5, 2, 3, 4, 5],
       [1, 2],
       {'expression_string': 'S / T * T - T + T',
        'is_terminal': False,
        'production_rule_sequence': [1, 2, 3, 4, 5],
        'history': [
            'S', 'S + T', 'S - T + T', 'S * T - T + T', 'S / T * T - T + T']}),
      ([4, 5, 2, 3, 4, 5],
       [1, 2, 3],
       {'expression_string': 'S / T * T - T + T',
        'is_terminal': False,
        'production_rule_sequence': [1, 2, 3, 4, 5],
        'history': [
            'S', 'S + T', 'S - T + T', 'S * T - T + T', 'S / T * T - T + T']}),
      # The expression is terminal before max length is reached.
      ([6, 9],
       None,
       {'expression_string': '1',
        'is_terminal': True,
        'production_rule_sequence': [1, 6, 9],
        'history': ['S', 'T', '1']}),
      ([6, 9],
       [1],
       {'expression_string': '1',
        'is_terminal': True,
        'production_rule_sequence': [1, 6, 9],
        'history': ['S', 'T', '1']}),
      ([9],
       [1, 6],
       {'expression_string': '1',
        'is_terminal': True,
        'production_rule_sequence': [1, 6, 9],
        'history': ['S', 'T', '1']}),
  ])
  def test_generate_expression_sess_not_none(
      self, mock_next_production_rules, partial_sequence, expected_result):
    with mock.patch.object(
        partial_sequence_model_generator,
        'generate_next_production_rule_from_model',
        side_effect=mock_next_production_rules):
      self.assertDictEqual(
          partial_sequence_model_generator.generate_expression(
              sess=mock.MagicMock(),
              grammar=self.grammar,
              max_length=5,
              partial_sequence=partial_sequence),
          expected_result)

  @parameterized.parameters(
      [('1', {
          'leading_at_0': -4,
          'leading_at_inf': -4
      }, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
       ('1', {
           'leading_at_0': -3
       }, [0, 0, 0.5 / 3, 0.5 / 3, 0.05 / 3, 1.9 / 3, 0.05 / 3, 0, 0, 0]),
       ('1', {
           'leading_at_0': -4,
           'leading_at_inf': -4,
           'monotonicity_at_inf': 0
       }, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])])
  def test_get_next_production_rule_distribution(
      self, current_partial_sequence_indices, symbolic_properties_dict,
      expected_result):
    next_production_rule_distribution = (
        partial_sequence_model_generator.get_next_production_rule_distribution(
            self.empirical_distribution_df,
            None, current_partial_sequence_indices,
            symbolic_properties_dict,
            self.next_production_rule_mask))
    numpy_testing.assert_almost_equal(next_production_rule_distribution,
                                      expected_result)

  def test_get_next_production_rule_distribution_without_condition(self):
    next_production_rule_distribution = (
        partial_sequence_model_generator.get_next_production_rule_distribution(
            self.empirical_distribution_df_without_condition,
            None, '1',
            {'leading_at_0': -4,
             'leading_at_inf': -4},
            self.next_production_rule_mask))
    expected_result = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    numpy_testing.assert_almost_equal(next_production_rule_distribution,
                                      expected_result)

  @parameterized.parameters(
      [('1', {
          'leading_at_0': -4,
          'leading_at_inf': -4
      }, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
       ('1', {
           'leading_at_0': -3
       }, [0, 0, 0.5 / 3, 0.5 / 3, 0.05 / 3, 1.9 / 3, 0.05 / 3, 0, 0, 0]),
       ('1', {
           'leading_at_0': -4,
           'leading_at_inf': -4,
           'monotonicity_at_inf': 0
       }, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])])
  def test_get_next_production_rule_distribution_limited_history(
      self, current_tail_partial_sequence_indices, symbolic_properties_dict,
      expected_result):
    next_production_rule_distribution = (
        partial_sequence_model_generator.get_next_production_rule_distribution(
            self.limited_history_empirical_distribution_df,
            1, current_tail_partial_sequence_indices,
            symbolic_properties_dict,
            self.next_production_rule_mask))
    numpy_testing.assert_almost_equal(next_production_rule_distribution,
                                      expected_result)

  def test_get_next_rule_distribution_limited_history_without_condition(self):
    next_production_rule_distribution = (
        partial_sequence_model_generator.get_next_production_rule_distribution(
            self.limited_history_empirical_distribution_df_without_condition,
            1, '1',
            {'leading_at_0': -4,
             'leading_at_inf': -4},
            self.next_production_rule_mask))
    expected_result = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    numpy_testing.assert_almost_equal(next_production_rule_distribution,
                                      expected_result)

  @parameterized.parameters(
      [('1', {
          'leading_at_0': -6,
          'leading_at_inf': -6
      }, None),
       ('2', {
           'leading_at_0': -4,
           'leading_at_inf': -4
       }, None)])
  def test_get_next_production_rule_distribution_not_found(
      self, current_partial_sequence_indices, symbolic_properties_dict,
      expected_result):
    next_production_rule_distribution = (
        partial_sequence_model_generator.get_next_production_rule_distribution(
            self.empirical_distribution_df,
            None, current_partial_sequence_indices,
            symbolic_properties_dict,
            self.next_production_rule_mask))
    self.assertEqual(next_production_rule_distribution, expected_result)


if __name__ == '__main__':
  tf.test.main()
