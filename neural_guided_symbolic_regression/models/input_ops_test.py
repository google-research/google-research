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

"""Tests for input_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf
from neural_guided_symbolic_regression.models import input_ops
from neural_guided_symbolic_regression.utils import arithmetic_grammar


class ParseProductionRuleSequenceBatchTest(tf.test.TestCase):

  def test_parse_production_rule_sequence_batch(self):
    grammar = arithmetic_grammar.Grammar(
        [
            'S -> S "+" T',  # index 1
            'S -> T',        # index 2
            'T -> "x"',      # index 3
            'T -> "1"',      # index 4
        ],
        padding_at_end=False)  # padding rule index 0
    input_features_tensor = {
        'expression_string': tf.constant([
            'x',      # Can be parsed into
                      # 'S -> T'        index 2
                      # 'T -> "x"'      index 3
            '1 + x',  # Can be parsed into
                      # 'S -> S "+" T'  index 1
                      # 'S -> T'        index 2
                      # 'T -> "1"'      index 4
                      # 'T -> "x"'      index 3
        ])}
    output_features_tensor = input_ops.parse_production_rule_sequence_batch(
        features=input_features_tensor,
        max_length=5,
        grammar=grammar)

    with self.test_session():
      self.assertAllEqual(
          output_features_tensor['expression_sequence'],
          [[2, 3, 0, 0, 0], [1, 2, 4, 3, 0]])
      self.assertAllEqual(
          output_features_tensor['expression_sequence_mask'],
          [[True, True, False, False, False], [True, True, True, True, False]])


class SamplePartialSequenceTest(parameterized.TestCase, tf.test.TestCase):

  def test_sample_partial_sequence_batch(self):
    features_tensor = input_ops.sample_partial_sequence_batch(
        features={
            'expression_sequence': tf.constant([
                [1, 2, 3, 0, 0], [5, 6, 0, 0, 0]]),
            'expression_sequence_mask': tf.constant([
                [True, True, True, False, False],
                [True, True, False, False, False]])})

    with self.test_session() as sess:
      features = sess.run(features_tensor)

    self.assertAllEqual(features['partial_sequence'].shape, (2, 5))
    self.assertAllEqual(features['partial_sequence_mask'].shape, (2, 5))
    self.assertAllEqual(features['partial_sequence_length'].shape, (2,))
    self.assertAllEqual(features['next_production_rule'].shape, (2,))


class GetNextProductionRuleMaskBatchTest(tf.test.TestCase):

  def setUp(self):
    super(GetNextProductionRuleMaskBatchTest, self).setUp()

    self.grammar = arithmetic_grammar.Grammar(
        [
            'S -> S "+" T',  # index 1
            'S -> T',        # index 2
            'T -> "x"',      # index 3
            'T -> "1"',      # index 4
        ],
        padding_at_end=False)  # padding rule index 0
    self.partial_sequences = np.array([
        [1, 0, 0, 0, 0, 0],  # expression 'S + T',
                             # the next production rule should start with S.
        [1, 2, 3, 0, 0, 0],  # expression 'x + T'
                             # the next production rule should start with T.
        [2, 0, 0, 0, 0, 0],  # expression 'T'
                             # the next production rule should start with T.
    ])
    self.partial_sequence_lengths = np.array([1, 3, 1])

    self.expected_next_production_rule_masks = np.array([
        [False, True, True, False, False],  # Only allow rules start with S.
        [False, False, False, True, True],  # Only allow rules start with T.
        [False, False, False, True, True],  # Only allow rules start with T.
    ])

  def test_get_next_production_rule_mask_batch_numpy(self):
    next_production_rule_masks = input_ops._get_next_production_rule_mask_batch(
        self.partial_sequences, self.partial_sequence_lengths, self.grammar)
    self.assertAllEqual(
        next_production_rule_masks, self.expected_next_production_rule_masks)

  def test_get_next_production_rule_mask_batch(self):
    features = input_ops.get_next_production_rule_mask_batch(
        features={
            'partial_sequence': tf.constant(self.partial_sequences),
            'partial_sequence_length':
                tf.constant(self.partial_sequence_lengths),
        },
        grammar=self.grammar)

    with self.test_session():
      self.assertAllEqual(
          features['next_production_rule_mask'],
          self.expected_next_production_rule_masks)


class EvaluateExpressionNumericallyBatchTest(tf.test.TestCase):

  def test_evaluate_expression_numerically_batch(self):
    features = input_ops.evaluate_expression_numerically_batch(
        features={
            'expression_string': tf.constant(
                ['1', 'x', '1 + x', 'x * x', '1 / ( x - 0.7 )'])},
        numerical_points=np.array([0.5, 1.5]),
        clip_value_min=-5.,
        clip_value_max=5.,
        symbol='x')

    with self.test_session():
      self.assertAllClose(
          features['numerical_values'],
          [[0.1, 0.1],
           [0.05, 0.15],
           [0.15, 0.25],
           [0.025, 0.225],
           [-0.5, 0.125]])


class SplitFeaturesLabelsTest(tf.test.TestCase):

  def test_split_features_labels(self):
    features, labels = input_ops.split_features_labels(
        features={
            'foo': tf.constant([[1, 2], [3, 4]]),
            'bar': tf.constant([9, 100])},
        label_key='bar')

    with self.test_session():
      self.assertListEqual(list(features.keys()), ['foo'])
      self.assertAllEqual(features['foo'], [[1, 2], [3, 4]])
      self.assertAllEqual(labels, [9, 100])


if __name__ == '__main__':
  tf.test.main()
