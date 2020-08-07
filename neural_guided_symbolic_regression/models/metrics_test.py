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

"""Tests for metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf
from neural_guided_symbolic_regression.models import metrics
from neural_guided_symbolic_regression.utils import arithmetic_grammar


# pylint: disable=g-inconsistent-quotes


class NextProductionRuleInfoBatchTextSummaryTest(tf.test.TestCase):

  def setUp(self):
    super(NextProductionRuleInfoBatchTextSummaryTest, self).setUp()
    self.grammar = arithmetic_grammar.Grammar(
        [
            'S -> S "+" T',  # index 1
            'S -> T',        # index 2
            'T -> "x"',      # index 3
            'T -> "1"',      # index 4
        ],
        padding_at_end=False)  # padding rule index 0

  def test_softmax_logits_info_string(self):
    output_info = metrics.probabilities_info_string(
        probabilities=np.asarray([0.1, 0.5, 0.3, 0.05, 0.05]),
        next_production_rule=3,
        grammar=self.grammar)

    self.assertListEqual(
        output_info,
        ["S -> S '+' T, probability: 0.50",
         " 0.10 |*0.50*| 0.30 |*0.05*| 0.05 "])

  def test_next_production_rule_info(self):
    info_string = metrics.next_production_rule_info(
        expression_string='x',
        partial_sequence=np.asarray([2, 0, 0, 0, 0]),
        partial_sequence_length=1,
        next_production_rule=3,
        unmasked_probabilities=np.asarray([0.05, 0.5, 0.3, 0.1, 0.05]),
        masked_probabilities=np.asarray([0., 0., 0., 0.7, 0.3]),
        grammar=self.grammar)

    self.assertEqual(
        info_string,
        "\texpression string:\n"
        "\tx\n"
        "\tpartial expression:\n"
        "\tT\n"
        "\ttrue next production rule:\n"
        "\tT -> 'x'\n"
        "\tunmasked prediction next production rule:\n"
        "\tS -> S '+' T, probability: 0.50\n"
        "\t 0.05 |*0.50*| 0.30 |*0.10*| 0.05 \n"
        "\tmasked prediction next production rule:\n"
        "\tT -> 'x', probability: 0.70\n"
        "\t 0.00 | 0.00 | 0.00 |*0.70*| 0.30 "
        )

  def test_next_production_rule_info_batch(self):
    output_info = metrics.next_production_rule_info_batch(
        expression_strings=['x', 'x + 1'],
        partial_sequences=np.asarray([[2, 0, 0, 0, 0], [1, 2, 3, 0, 0]]),
        partial_sequence_lengths=[1, 3],
        next_production_rules=[3, 4],
        unmasked_probabilities_batch=np.asarray([
            [0.05, 0.5, 0.3, 0.1, 0.05], [0.05, 0.5, 0.3, 0.1, 0.05]]),
        masked_probabilities_batch=np.asarray([
            [0., 0., 0., 0.7, 0.3], [0., 0., 0., 0.7, 0.3]]),
        grammar=self.grammar)

    np.testing.assert_equal(
        output_info,
        ["\texpression string:\n"
         "\tx\n"
         "\tpartial expression:\n"
         "\tT\n"
         "\ttrue next production rule:\n"
         "\tT -> 'x'\n"
         "\tunmasked prediction next production rule:\n"
         "\tS -> S '+' T, probability: 0.50\n"
         "\t 0.05 |*0.50*| 0.30 |*0.10*| 0.05 \n"
         "\tmasked prediction next production rule:\n"
         "\tT -> 'x', probability: 0.70\n"
         "\t 0.00 | 0.00 | 0.00 |*0.70*| 0.30 ",
         "\texpression string:\n"
         "\tx + 1\n"
         "\tpartial expression:\n"
         "\tx + T\n"
         "\ttrue next production rule:\n"
         "\tT -> '1'\n"
         "\tunmasked prediction next production rule:\n"
         "\tS -> S '+' T, probability: 0.50\n"
         "\t 0.05 |*0.50*| 0.30 | 0.10 |*0.05*\n"
         "\tmasked prediction next production rule:\n"
         "\tT -> 'x', probability: 0.70\n"
         "\t 0.00 | 0.00 | 0.00 |*0.70*|*0.30*"]
        )

  def test_next_production_rule_info_batch_empty(self):
    np.testing.assert_equal(
        metrics.next_production_rule_info_batch(
            expression_strings=[],
            partial_sequences=[],
            partial_sequence_lengths=[],
            next_production_rules=[],
            unmasked_probabilities_batch=[],
            masked_probabilities_batch=[],
            grammar=self.grammar),
        np.asarray([], dtype=np.unicode_))


class MaskByPartialSequenceLengthTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(MaskByPartialSequenceLengthTest, self).setUp()
    self.tensors = (
        tf.constant([[1, 2], [3, 4]]),
        tf.constant([[5, 6], [7, 8]]))

  def test_mask_by_partial_sequence_length_value_error(self):
    with self.assertRaisesRegex(
        ValueError,
        'partial_sequence_lengths is expected when target_length is not None'):
      metrics.mask_by_partial_sequence_length(
          tensors=self.tensors,
          partial_sequence_lengths=None,
          target_length=1)

  def test_mask_by_partial_sequence_length_no_change(self):
    masked_tensors = metrics.mask_by_partial_sequence_length(
        self.tensors,
        partial_sequence_lengths=None,
        target_length=None)

    with self.test_session() as sess:
      masked_tensors_values = sess.run(masked_tensors)
      self.assertAllEqual(
          masked_tensors_values, ([[1, 2], [3, 4]], [[5, 6], [7, 8]]))

  @parameterized.parameters([
      ([1, 1], ([[1, 2], [3, 4]], [[5, 6], [7, 8]])),
      ([1, 42], ([[1, 2]], [[5, 6]])),
  ])
  def test_mask_by_partial_sequence_length(
      self, partial_sequence_lengths, expected_values):
    masked_tensors = metrics.mask_by_partial_sequence_length(
        self.tensors,
        partial_sequence_lengths=tf.constant(partial_sequence_lengths),
        target_length=1)

    with self.test_session() as sess:
      masked_tensors_values = sess.run(masked_tensors)
      self.assertAllEqual(masked_tensors_values, expected_values)

  def test_mask_by_partial_sequence_length_empty_output_tensors(self):
    masked_tensors = metrics.mask_by_partial_sequence_length(
        self.tensors,
        partial_sequence_lengths=tf.constant([42, 42]),
        target_length=1)

    with self.test_session() as sess:
      masked_tensors_values = sess.run(masked_tensors)
      self.assertLen(masked_tensors_values, 2)
      self.assertAllEqual(masked_tensors_values[0].shape, (0, 2))
      self.assertAllEqual(masked_tensors_values[1].shape, (0, 2))


class NextProductionRuleValidRatioTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.unmasked_probabilities_batch_values_0 = np.asarray([
        [0.1, 0.1, 0.5, 0.3],
        [0.1, 0.1, 0.5, 0.3],
    ])
    self.next_production_rule_masks_values_0 = np.asarray([
        [True, True, True, True],  # The mask at maximum element is True.
        [True, True, True, True],  # The mask at maximum element is True.
    ])
    self.unmasked_probabilities_batch_values_1 = np.asarray([
        [0.1, 0.1, 0.5, 0.3],
        [0.1, 0.1, 0.5, 0.3],
    ])
    self.next_production_rule_masks_values_1 = np.asarray([
        [True, True, False, True],  # The mask at maximum element is False.
        [True, True, True, True],  # The mask at maximum element is True.
    ])

    self.unmasked_probabilities_batch = tf.placeholder(
        tf.float32, shape=[None, 4])
    self.next_production_rule_masks = tf.placeholder(
        tf.float32, shape=[None, 4])

  def test_next_production_rule_valid_ratio(self):
    value, update_op = metrics.next_production_rule_valid_ratio(
        unmasked_probabilities_batch=self.unmasked_probabilities_batch,
        next_production_rule_masks=self.next_production_rule_masks)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(
          update_op,
          feed_dict={
              self.unmasked_probabilities_batch:
                  self.unmasked_probabilities_batch_values_0,
              self.next_production_rule_masks:
                  self.next_production_rule_masks_values_0})
      self.assertAlmostEqual(value.eval(), 1.)
      sess.run(
          update_op,
          feed_dict={
              self.unmasked_probabilities_batch:
                  self.unmasked_probabilities_batch_values_1,
              self.next_production_rule_masks:
                  self.next_production_rule_masks_values_1})
      self.assertAlmostEqual(value.eval(), 0.75)

  def test_next_production_rule_valid_ratio_with_length(self):
    partial_sequence_lengths = tf.placeholder(tf.int32, shape=[None])
    value, update_op = metrics.next_production_rule_valid_ratio(
        unmasked_probabilities_batch=self.unmasked_probabilities_batch,
        next_production_rule_masks=self.next_production_rule_masks,
        partial_sequence_lengths=partial_sequence_lengths,
        target_length=1)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(
          update_op,
          feed_dict={
              self.unmasked_probabilities_batch:
                  self.unmasked_probabilities_batch_values_0,
              self.next_production_rule_masks:
                  self.next_production_rule_masks_values_0,
              partial_sequence_lengths: [42, 1]})
      # Only the second example has matched partial sequence length, thus will
      # be used to compute valid ratio. The next production rule in this example
      # is valid. So the mean ratio is 1.
      self.assertAlmostEqual(value.eval(), 1.)
      sess.run(
          update_op,
          feed_dict={
              self.unmasked_probabilities_batch:
                  self.unmasked_probabilities_batch_values_1,
              self.next_production_rule_masks:
                  self.next_production_rule_masks_values_1,
              partial_sequence_lengths: [1, 42]})
      # Only the first example has matched partial sequence length, thus will
      # be used to compute valid ratio. The next production rule in this example
      # is not valid. So the mean ratio is (1. + 0.) / 2 = 0.5
      self.assertAlmostEqual(value.eval(), 0.5)


class NextProductionRuleAccuracyTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.next_production_rules_values_0 = np.asarray(
        [1, 2, 3, 4])
    self.predict_next_production_rules_values_0 = np.asarray(
        [99, 99, 3, 99])
    self.next_production_rules_values_1 = np.asarray(
        [5, 6, 7, 8, 9, 10])
    self.predict_next_production_rules_values_1 = np.asarray(
        [5, 6, 7, 99, 99, 99])

    self.next_production_rules = tf.placeholder(
        tf.float32, shape=[None])
    self.predict_next_production_rules = tf.placeholder(
        tf.float32, shape=[None])

  def test_next_production_rule_accuracy(self):
    value, update_op = metrics.next_production_rule_accuracy(
        next_production_rules=self.next_production_rules,
        predict_next_production_rules=self.predict_next_production_rules)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(
          update_op,
          feed_dict={
              self.next_production_rules:
                  self.next_production_rules_values_0,
              self.predict_next_production_rules:
                  self.predict_next_production_rules_values_0})
      self.assertAlmostEqual(value.eval(), 0.25)
      sess.run(
          update_op,
          feed_dict={
              self.next_production_rules:
                  self.next_production_rules_values_1,
              self.predict_next_production_rules:
                  self.predict_next_production_rules_values_1})
      self.assertAlmostEqual(value.eval(), 0.4)

  def test_next_production_rule_accuracy_with_length(self):
    partial_sequence_lengths = tf.placeholder(tf.int32, shape=[None])
    value, update_op = metrics.next_production_rule_accuracy(
        next_production_rules=self.next_production_rules,
        predict_next_production_rules=self.predict_next_production_rules,
        partial_sequence_lengths=partial_sequence_lengths,
        target_length=1)

    with self.test_session() as sess:
      sess.run(tf.local_variables_initializer())
      sess.run(
          update_op,
          feed_dict={
              self.next_production_rules:
                  self.next_production_rules_values_0,
              self.predict_next_production_rules:
                  self.predict_next_production_rules_values_0,
              partial_sequence_lengths: [42, 42, 1, 42]})
      self.assertAlmostEqual(value.eval(), 1.)
      sess.run(
          update_op,
          feed_dict={
              self.next_production_rules:
                  self.next_production_rules_values_1,
              self.predict_next_production_rules:
                  self.predict_next_production_rules_values_1,
              partial_sequence_lengths: [42, 42, 42, 42, 42, 42]})
      self.assertAlmostEqual(value.eval(), 1.)


class GetLeadingPowersTest(parameterized.TestCase):

  @parameterized.parameters([
      (0, [(0, 0)]),
      (1, [(0, 1), (-1, 0), (0, -1), (1, 0)]),
      (2, [(0, 2), (-1, 1), (-2, 0), (-1, -1),
           (0, -2), (1, -1), (2, 0), (1, 1)]),
  ])
  def test_get_leading_powers(self, leading_powers_sum, expected):
    leading_powers = sorted(
        list(metrics.get_leading_powers(leading_powers_sum)))
    self.assertListEqual(leading_powers, sorted(expected))


if __name__ == '__main__':
  tf.test.main()
