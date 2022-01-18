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

"""Tests for transformer layers."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from readtwice import layers as readtwice_layers


class TransformerLayersTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='no_dropout',
          attention_probs_dropout_prob=0.0,
          side_seq_len=0,
          share_kv_projections=False,
          num_cross_attention_heads=None,
          enable_default_side_input=False),
      dict(
          testcase_name='side_inputs',
          attention_probs_dropout_prob=0.0,
          side_seq_len=6,
          share_kv_projections=False,
          num_cross_attention_heads=None,
          enable_default_side_input=False),
      dict(
          testcase_name='side_inputs_with_dropout',
          attention_probs_dropout_prob=0.1,
          side_seq_len=6,
          share_kv_projections=False,
          num_cross_attention_heads=None,
          enable_default_side_input=False),
      dict(
          testcase_name='share_kv_projections',
          attention_probs_dropout_prob=0.0,
          side_seq_len=6,
          share_kv_projections=True,
          num_cross_attention_heads=None,
          enable_default_side_input=False),
      dict(
          testcase_name='cross_attention_0',
          attention_probs_dropout_prob=0.1,
          side_seq_len=6,
          share_kv_projections=False,
          num_cross_attention_heads=0,
          enable_default_side_input=False),
      dict(
          testcase_name='cross_attention_1',
          attention_probs_dropout_prob=0.1,
          side_seq_len=6,
          share_kv_projections=False,
          num_cross_attention_heads=1,
          enable_default_side_input=False),
      dict(
          testcase_name='cross_attention_3',
          attention_probs_dropout_prob=0.1,
          side_seq_len=6,
          share_kv_projections=False,
          num_cross_attention_heads=3,
          enable_default_side_input=False),
      dict(
          testcase_name='cross_attention_0_with_default_side_input',
          attention_probs_dropout_prob=0.1,
          side_seq_len=8,
          share_kv_projections=False,
          num_cross_attention_heads=0,
          enable_default_side_input=True),
      dict(
          testcase_name='cross_attention_1_with_default_side_input',
          attention_probs_dropout_prob=0.1,
          side_seq_len=8,
          share_kv_projections=False,
          num_cross_attention_heads=1,
          enable_default_side_input=True),
      dict(
          testcase_name='cross_attention_3_with_default_side_input',
          attention_probs_dropout_prob=0.1,
          side_seq_len=8,
          share_kv_projections=False,
          num_cross_attention_heads=3,
          enable_default_side_input=True),
  )
  def test_transformer_with_side_inputs_layers(
      self, attention_probs_dropout_prob, side_seq_len, share_kv_projections,
      num_cross_attention_heads, enable_default_side_input):
    tf.compat.v1.random.set_random_seed(1234)
    np.random.seed(1234)

    batch_size = 5
    main_seq_len = 11
    seq_len = main_seq_len
    num_heads = 7
    hidden_size = 21
    num_layers = 2

    # We use `placeholder_with_default` to simulate the TF v1 situation where
    # the static `batch_size` is unknown.
    inputs = tf.compat.v1.placeholder_with_default(
        np.random.normal(size=[batch_size, seq_len, hidden_size]).astype(
            np.float32),
        shape=[None, None, hidden_size])
    if side_seq_len > 0:
      seq_len += side_seq_len
      side_inputs = tf.compat.v1.placeholder_with_default(
          np.random.normal(size=[side_seq_len, hidden_size]).astype(np.float32),
          shape=[None, hidden_size])
    else:
      side_inputs = None

    att_mask = tf.compat.v1.placeholder_with_default(
        np.random.binomial(
            n=1, p=0.9, size=[batch_size, main_seq_len, seq_len]),
        shape=[None, None, None])

    layer = readtwice_layers.TransformerWithSideInputLayers(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        share_kv_projections=share_kv_projections,
        num_cross_attention_heads=num_cross_attention_heads,
        enable_default_side_input=enable_default_side_input)

    result = layer(inputs, side_input=side_inputs, att_mask=att_mask)

    static_batch_size = inputs.shape.as_list()[0]
    self.assertAllEqual([static_batch_size, main_seq_len, hidden_size],
                        result.shape.as_list())

    self.assertNotEmpty(layer.feed_forward_layers)

  def test_transformer_with_multiple_side_inputs_layers(self):
    tf.compat.v1.random.set_random_seed(1234)
    np.random.seed(1234)

    batch_size = 5
    main_seq_len = 11
    seq_len = main_seq_len
    num_heads = 7
    hidden_size = 21
    num_layers = 3

    # We use `placeholder_with_default` to simulate the TF v1 situation where
    # the static `batch_size` is unknown.
    inputs = tf.compat.v1.placeholder_with_default(
        np.random.normal(size=[batch_size, seq_len, hidden_size]).astype(
            np.float32),
        shape=[None, seq_len, hidden_size])

    side_inputs = []
    for _ in range(num_layers):
      side_seq_len = np.random.randint(1, 10)
      side_input = tf.compat.v1.placeholder_with_default(
          np.random.normal(size=[batch_size, side_seq_len, hidden_size]).astype(
              np.float32),
          shape=[None, side_seq_len, hidden_size])
      side_inputs.append(side_input)

    layer = readtwice_layers.TransformerWithSideInputLayers(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        attention_probs_dropout_prob=0.1,
        share_kv_projections=False)

    result = layer(inputs, side_input=side_inputs)

    static_batch_size = inputs.shape.as_list()[0]
    self.assertAllEqual([static_batch_size, main_seq_len, hidden_size],
                        result.shape.as_list())

    self.assertNotEmpty(layer.feed_forward_layers)


if __name__ == '__main__':
  tf.test.main()
