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

"""Tests for attention layers."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from readtwice import layers as readtwice_layers


class LayersTest(tf.test.TestCase, parameterized.TestCase):

  def assert_all_identical(self, *elements):
    if not elements:
      return
    first_element = elements[0]
    for element in elements[1:]:
      self.assertIs(element, first_element)

  @parameterized.named_parameters(
      ('without_side_input_no_dropout', 0, 0.0, False),
      ('without_side_input_no_dropout_shared_kv', 0, 0.0, True),
      ('without_side_input', 0, 0.1, False),
      ('without_side_input_shared_kv', 0, 0.1, True),
      ('with_side_input_no_dropout', 6, 0.0, False),
      ('with_side_input_no_dropout_shared_kv', 6, 0.0, True),
      ('with_side_input', 6, 0.1, False),
      ('with_side_input_shared_kv', 6, 0.1, True),
  )
  def test_fused_side_attention(self, side_seq_len, att_dropout_prob,
                                share_kv_projections):
    tf.compat.v1.random.set_random_seed(1234)
    np.random.seed(1234)

    batch_size = 5
    main_seq_len = 11
    seq_len = main_seq_len
    num_heads = 7
    hidden_size = 21

    # We use `placeholder_with_default` to simulate the TF v1 situation where
    # the static `batch_size` is unknown.
    main_input = tf.compat.v1.placeholder_with_default(
        np.random.normal(size=[batch_size, main_seq_len, hidden_size]).astype(
            np.float32),
        shape=[None, None, hidden_size])

    if side_seq_len > 0:
      seq_len += side_seq_len
      side_input = tf.compat.v1.placeholder_with_default(
          np.random.normal(size=[side_seq_len, hidden_size]).astype(np.float32),
          shape=[None, hidden_size])
    else:
      side_input = None

    att_mask = tf.compat.v1.placeholder_with_default(
        np.random.binomial(
            n=1, p=0.9, size=[batch_size, main_seq_len, seq_len]),
        shape=[None, None, None])

    layer = readtwice_layers.FusedSideAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        att_dropout_prob=att_dropout_prob,
        share_kv_projections=share_kv_projections)

    result = layer(
        main_input=main_input, side_input=side_input, att_mask=att_mask)
    self.assertAllEqual([batch_size, main_seq_len, hidden_size], result.shape)

  @parameterized.named_parameters(
      ('attn_heads_0_no_dropout', 0, 0.0),
      ('attn_heads_0', 0, 0.1),
      ('attn_heads_1_no_dropout', 1, 0.0),
      ('attn_heads_1', 1, 0.1),
      ('attn_heads_3_no_dropout', 3, 0.0),
      ('attn_heads_3', 3, 0.1),
      ('attn_heads_7_no_dropout', 7, 0.0),
      ('attn_heads_7', 7, 0.1),
  )
  def test_side_attention(
      self,
      num_heads,
      att_dropout_prob,
  ):
    tf.compat.v1.random.set_random_seed(1234)
    np.random.seed(1234)

    batch_size = 5
    main_seq_len = 11
    side_seq_len = 7
    hidden_size = 21
    max_block_index = 10

    main_input_np = np.random.normal(
        size=[batch_size, main_seq_len, hidden_size]).astype(np.float32)

    side_input_np = np.random.normal(size=[side_seq_len, hidden_size]).astype(
        np.float32)

    main_pos_np = np.random.randint(
        0, max_block_index, size=[batch_size]).astype(np.int32)

    side_pos_np = np.random.randint(
        0, max_block_index, size=[side_seq_len]).astype(np.int32)

    att_mask_np = np.random.binomial(
        n=1, p=0.9, size=[batch_size, main_seq_len, side_seq_len])

    att_value_mask_np = np.random.binomial(
        n=1, p=0.9, size=[batch_size, main_seq_len])

    for enable_default_side_input in [False, True]:
      for top_k_attention in [False, True]:
        for pos_embed_mode in [
            None, 'absolute', 'absolute_add_ln', 'simple_relative',
            'query_dot_relative'
        ]:
          for use_att_value_mask in [False, True]:
            # We use `placeholder_with_default` to simulate the TF v1 situation
            # where the static `batch_size` is unknown.
            main_input = tf.compat.v1.placeholder_with_default(
                main_input_np, shape=[None, None, hidden_size])

            side_input = tf.compat.v1.placeholder_with_default(
                side_input_np, shape=[None, hidden_size])

            att_mask = tf.compat.v1.placeholder_with_default(
                att_mask_np, shape=[None, None, None])

            if pos_embed_mode is not None:
              main_pos = tf.compat.v1.placeholder_with_default(
                  main_pos_np, shape=[None])
              side_pos = tf.compat.v1.placeholder_with_default(
                  side_pos_np, shape=[None])
              pos_embed_size = max(main_pos_np.max(), side_pos_np.max()) + 1
            else:
              main_pos = None
              side_pos = None
              pos_embed_size = None

            if use_att_value_mask:
              att_value_mask = tf.compat.v1.placeholder_with_default(
                  att_value_mask_np, shape=[None, None])
            else:
              att_value_mask = None

            layer = readtwice_layers.SideAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                att_dropout_prob=att_dropout_prob,
                enable_default_side_input=enable_default_side_input,
                top_k_attention=top_k_attention,
                pos_embed_mode=pos_embed_mode,
                pos_embed_size=pos_embed_size,
                use_one_hot_embeddings=bool(np.random.randint(2)))

            result = layer(
                main_input=main_input,
                side_input=side_input,
                main_pos=main_pos,
                side_pos=side_pos,
                att_value_mask=att_value_mask,
                att_mask=att_mask)
            self.assertAllEqual([batch_size, main_seq_len, hidden_size],
                                result.shape)

  def test_attention_head_projection(self):
    inputs = tf.ones([2, 3, 10])
    layer = readtwice_layers.ProjectAttentionHeads(num_heads=4, size_per_head=5)
    result = layer(inputs)
    self.assertAllEqual([2, 3, 4, 5], result.shape)

    inputs = tf.ones([2, 3, 4, 10])
    layer = readtwice_layers.ProjectAttentionHeads(num_heads=5, size_per_head=6)
    result = layer(inputs)
    self.assertAllEqual([2, 3, 4, 5, 6], result.shape)


if __name__ == '__main__':
  tf.test.main()
