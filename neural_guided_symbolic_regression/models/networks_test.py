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

"""Tests for networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow.compat.v1 as tf
from neural_guided_symbolic_regression.models import networks


class NetworksTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters([
      ([], [], (2, 4, 10)),
      ([], [0.1, 0.2, 0.3], (2, 4, 13)),
      (['leading_at_0'], [], (2, 4, 11)),
      (['leading_at_0', 'leading_at_inf'], [], (2, 4, 12)),
      (['leading_at_0', 'leading_at_inf'], [0.1, 0.2, 0.3], (2, 4, 15)),
  ])
  def test_partial_sequence_encoder(
      self, symbolic_properties, numerical_points, expected_shape):
    features = {
        'partial_sequence': tf.constant([[1, 2, 0, 0], [3, 0, 0, 0]]),
        'leading_at_0': tf.constant([0., 3.]),
        'leading_at_inf': tf.constant([1., -1.]),
        'numerical_values': tf.constant([[0.6, 0.5, 0.1], [1.6, 0.4, 0.3]]),
    }
    embedding_layer = networks.partial_sequence_encoder(
        features=features,
        symbolic_properties=symbolic_properties,
        numerical_points=numerical_points,
        num_production_rules=4,
        embedding_size=10)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      embedding_layer_value = sess.run(embedding_layer)
      self.assertEqual(embedding_layer_value.shape, expected_shape)

  @parameterized.parameters([False, True])
  def test_build_stacked_gru_model(self, bidirectional):
    logits = networks.build_stacked_gru_model(
        embedding_layer=tf.random_uniform([2, 4, 10], dtype=tf.float32),
        partial_sequence_length=tf.constant([2, 3]),
        gru_hidden_sizes=[10, 20],
        num_output_features=15,
        bidirectional=bidirectional)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      logits_value = sess.run(logits)
      self.assertEqual(logits_value.shape, (2, 15))


if __name__ == '__main__':
  tf.test.main()
