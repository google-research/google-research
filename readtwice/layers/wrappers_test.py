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

"""Tests for wrapper layers."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from readtwice import layers as readtwice_layers


class LayersTest(tf.test.TestCase, parameterized.TestCase):

  def test_residual_block_ordering(self):
    inputs = tf.constant([[1.0, -1.0], [0.5, -1.5]])

    inner_layer = tf.keras.layers.ReLU()
    normalization_layer = tf.keras.layers.Lambda(lambda x: 2 * x)

    residual_block_default_order = readtwice_layers.ResidualBlock(
        inner_layer=inner_layer,
        normalization_layer=normalization_layer,
        use_pre_activation_order=False)
    default_order_result = residual_block_default_order(inputs)

    residual_block_pre_act_order = readtwice_layers.ResidualBlock(
        inner_layer=inner_layer,
        normalization_layer=normalization_layer,
        use_pre_activation_order=True)
    pre_act_order_result = residual_block_pre_act_order(inputs)

    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.assertAllClose([[4.0, -2.0], [2.0, -3.0]], default_order_result)

    self.assertAllClose([[3.0, -1.0], [1.5, -1.5]], pre_act_order_result)

  @parameterized.named_parameters(
      ('default_order', False),
      ('pre_activation_order', True),
  )
  def test_residual_block_training_vs_inference_dropout(
      self, use_pre_activation_order):
    tf.compat.v1.random.set_random_seed(1234)
    np.random.seed(1234)

    batch_size = 3
    input_size = 10
    inputs = tf.constant(np.random.normal(size=[batch_size, input_size]))

    residual_block = readtwice_layers.ResidualBlock(
        dropout_probability=0.5,
        use_pre_activation_order=use_pre_activation_order)

    inference_output1 = residual_block(inputs, training=False)
    inference_output2 = residual_block(inputs, training=False)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(inference_output1, inference_output2)

    # Dropout makes this non-deterministic.
    training_output1 = residual_block(inputs, training=True)
    training_output2 = residual_block(inputs, training=True)
    self.assertNotAllClose(training_output1, training_output2)

  @parameterized.named_parameters(
      ('default_order', False),
      ('pre_activation_order', True),
  )
  def test_residual_block_training_vs_inference_normalization_layer(
      self, use_pre_activation_order):
    np.random.seed(1234)

    batch_size = 3
    input_size = 10
    inputs = tf.constant(np.random.normal(size=[batch_size, input_size]))

    residual_block = readtwice_layers.ResidualBlock(
        normalization_layer=tf.keras.layers.BatchNormalization(),
        dropout_probability=0.0,
        use_pre_activation_order=use_pre_activation_order)

    inference_output1 = residual_block(inputs, training=False)
    inference_output2 = residual_block(inputs, training=False)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(inference_output1, inference_output2)

    training_output1 = residual_block(inputs, training=True)
    training_output2 = residual_block(inputs, training=True)
    self.assertAllClose(training_output1, training_output2)

    # Batch normalization gives different results for training vs. inference.
    self.assertNotAllClose(inference_output1, training_output1)

#   @parameterized.named_parameters(
#       ('default_order', False),
#       ('pre_activation_order', True),
#   )
#   def test_residual_block_with_relative_attention(self,
#                                                   use_pre_activation_order):
#     np.random.seed(1234)

#     batch_size = 2
#     seq_len = 4
#     hidden_size = 10
#     inputs = tf.constant(
#         np.random.normal(size=[batch_size, seq_len, hidden_size]), tf.float32)

#     att_mask = tf.stack([
#         # Force each element in the first example to only attend to itself.
#         tf.eye(seq_len, dtype=tf.int32),
#         # The second example can attend everywhere.
#         tf.ones([seq_len, seq_len], dtype=tf.int32)
#     ])

#     inner_layer = readtwice_layers.RelativeAttention(
#         hidden_size=hidden_size,
#         num_heads=2,
#         relative_vocab_size=2,
#         initializer=tf.keras.initializers.Identity())

#     residual_block = readtwice_layers.ResidualBlock(
#         inner_layer=inner_layer,
#         normalization_layer=tf.keras.layers.Lambda(lambda x: x),
#         dropout_probability=0.0,
#         use_pre_activation_order=use_pre_activation_order)

#     relative_att_ids1 = tf.zeros(
#         [batch_size, seq_len, seq_len], dtype=tf.int32)
#     result1 = residual_block(
#         inputs, att_mask=att_mask, relative_att_ids=relative_att_ids1)

#     self.evaluate(tf.compat.v1.global_variables_initializer())
#     self.assertAllClose(2 * inputs[0], result1[0])

#     relative_att_ids2 = tf.tile([[[0, 1, 0, 1]]], [batch_size, seq_len, 1])
#     result2 = residual_block(
#         inputs, att_mask=att_mask, relative_att_ids=relative_att_ids2)

#     self.assertAllClose(result1[0], result2[0])
#     self.assertNotAllClose(result1[1], result2[1])


if __name__ == '__main__':
  tf.test.main()
