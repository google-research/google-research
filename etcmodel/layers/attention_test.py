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

from etcmodel import layers as etc_layers
# `attention` module used for testing `_expand_local_ids_to_blocks` helper.
from etcmodel.layers import attention


class LayersTest(tf.test.TestCase, parameterized.TestCase):

  def assert_all_identical(self, *elements):
    if not elements:
      return
    first_element = elements[0]
    for element in elements[1:]:
      self.assertIs(element, first_element)

  @parameterized.named_parameters(
      ('using_gather', False),
      ('using_one_hot', True),
  )
  def test_relative_attention(self, use_one_hot_lookup):
    tf.compat.v1.random.set_random_seed(1234)
    np.random.seed(1234)

    batch_size = 3
    from_seq_len = 16
    to_seq_len = 17
    num_heads = 5
    from_hidden_size = 11
    to_hidden_size = 12
    output_hidden_size = 13
    total_key_size = 10
    total_value_size = 15
    relative_vocab_size = 21

    from_seq = tf.random.normal([batch_size, from_seq_len, from_hidden_size])
    to_seq = tf.random.normal([batch_size, to_seq_len, to_hidden_size])
    att_mask = tf.constant(
        np.random.binomial(
            n=1, p=0.9, size=[batch_size, from_seq_len, to_seq_len]))
    relative_att_ids = tf.random.uniform([batch_size, from_seq_len, to_seq_len],
                                         maxval=relative_vocab_size,
                                         dtype=tf.int32)

    layer = etc_layers.RelativeAttention(
        hidden_size=output_hidden_size,
        num_heads=num_heads,
        total_key_size=total_key_size,
        total_value_size=total_value_size,
        relative_vocab_size=relative_vocab_size,
        use_one_hot_lookup=use_one_hot_lookup)

    result = layer(
        from_seq=from_seq,
        to_seq=to_seq,
        att_mask=att_mask,
        relative_att_ids=relative_att_ids)
    self.assertAllEqual([batch_size, from_seq_len, output_hidden_size],
                        result.shape)

  @parameterized.named_parameters(
      ('using_gather', False),
      ('using_one_hot', True),
  )
  def test_relative_attention_self_attention(self, use_one_hot_lookup):
    tf.compat.v1.random.set_random_seed(1234)
    np.random.seed(1234)

    batch_size = 3
    seq_len = 16
    num_heads = 5
    input_hidden_size = 11
    output_hidden_size = 12
    total_key_size = 10
    total_value_size = 15
    relative_vocab_size = 21

    inputs = tf.constant(
        np.random.normal(size=[batch_size, seq_len, input_hidden_size]),
        tf.float32)
    att_mask = tf.constant(
        np.random.binomial(n=1, p=0.9, size=[batch_size, seq_len, seq_len]))
    relative_att_ids = tf.constant(
        np.random.randint(
            relative_vocab_size, size=[batch_size, seq_len, seq_len]))

    layer = etc_layers.RelativeAttention(
        hidden_size=output_hidden_size,
        num_heads=num_heads,
        total_key_size=total_key_size,
        total_value_size=total_value_size,
        relative_vocab_size=relative_vocab_size,
        use_one_hot_lookup=use_one_hot_lookup)

    result1 = layer(
        inputs, att_mask=att_mask, relative_att_ids=relative_att_ids)
    self.assertAllEqual([batch_size, seq_len, output_hidden_size],
                        result1.shape)

    result2 = layer(
        from_seq=inputs,
        to_seq=inputs,
        att_mask=att_mask,
        relative_att_ids=relative_att_ids)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(result1, result2)

  @parameterized.named_parameters(
      ('using_gather', False),
      ('using_one_hot', True),
  )
  def test_relative_attention_shared_sublayers(self, use_one_hot_lookup):
    tf.compat.v1.random.set_random_seed(1234)
    np.random.seed(1234)

    batch_size = 3
    from_seq_len = 16
    to_seq_len = 17
    num_heads = 5
    from_hidden_size = 11
    to_hidden_size = 12
    output_hidden_size = 13
    total_key_size = 10
    total_value_size = 15
    relative_vocab_size = 9

    from_seq = tf.constant(
        np.random.random(size=[batch_size, from_seq_len, from_hidden_size]))
    to_seq = tf.constant(
        np.random.random(size=[batch_size, to_seq_len, to_hidden_size]))
    att_mask = tf.constant(
        np.random.binomial(
            n=1, p=0.9, size=[batch_size, from_seq_len, to_seq_len]))

    layer = etc_layers.RelativeAttention(
        hidden_size=output_hidden_size,
        num_heads=num_heads,
        total_key_size=total_key_size,
        total_value_size=total_value_size,
        relative_vocab_size=relative_vocab_size,
        use_one_hot_lookup=use_one_hot_lookup)

    sharing_layer = etc_layers.RelativeAttention(
        hidden_size=output_hidden_size,
        num_heads=num_heads,
        total_key_size=total_key_size,
        total_value_size=total_value_size,
        query_projection=layer.query_projection,
        key_projection=layer.key_projection,
        value_projection=layer.value_projection,
        qkv_relative_attention=layer.qkv_relative_attention,
        output_projection=layer.output_projection)

    different_layer = etc_layers.RelativeAttention(
        hidden_size=output_hidden_size,
        num_heads=num_heads,
        total_key_size=total_key_size,
        total_value_size=total_value_size,
        relative_vocab_size=relative_vocab_size,
        use_one_hot_lookup=use_one_hot_lookup)

    result1 = layer(
        from_seq=from_seq,
        to_seq=to_seq,
        att_mask=att_mask,
        relative_att_ids=None)

    result2 = sharing_layer(
        from_seq=from_seq,
        to_seq=to_seq,
        att_mask=att_mask,
        relative_att_ids=None)

    result3 = different_layer(
        from_seq=from_seq,
        to_seq=to_seq,
        att_mask=att_mask,
        relative_att_ids=None)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(result1, result2)
    self.assertNotAllClose(result1, result3)

  def test_fused_global_local_attention_special_case_equivalence(self):
    # To test for correctness, we make sure the output is equivalent to
    # standard attention in the special case where `local_radius` covers the
    # entire long sequence length and projection weights are shared.
    # For simplicity, we don't use attention masks or relative attention ids
    # in this test.

    tf.compat.v1.random.set_random_seed(1234)
    np.random.seed(1234)

    batch_size = 3
    long_seq_len = 12
    global_seq_len = 6
    hidden_size = 10
    num_heads = 5
    local_radius = 15  # Must be >= `long_seq_len - 1` to remove sparsity.
    # relative_vocab_size = 9

    long_input = tf.constant(
        np.random.normal(size=[batch_size, long_seq_len, hidden_size]))
    global_input = tf.constant(
        np.random.normal(size=[batch_size, global_seq_len, hidden_size]))

    fused_att_layer = etc_layers.FusedGlobalLocalAttention(
        long_hidden_size=hidden_size,
        global_hidden_size=hidden_size,
        num_heads=num_heads,
        local_radius=local_radius,
        share_qkv_projections=True,
        share_att_output_projection=True)

    long_output, global_output = fused_att_layer(
        long_input,
        global_input,
        att_implementation='sparse')

    # [batch_size, long_seq_len + global_seq_len, hidden_size]
    fused_output = tf.concat([long_output, global_output], axis=1)

    # Create concatenated input for standard attention.

    # [batch_size, long_seq_len + global_seq_len, hidden_size]
    concat_input = tf.concat([long_input, global_input], axis=1)

    standard_att_layer = etc_layers.RelativeAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        query_projection=fused_att_layer.long_query_projection,
        key_projection=fused_att_layer.l2l_key_projection,
        value_projection=fused_att_layer.l2l_value_projection,
        output_projection=fused_att_layer.long_output_projection)

    expected_output = standard_att_layer(concat_input)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(expected_output, fused_output)

    # Make sure 'full' att_implementation gives the same output.
    long_output_full_att, global_output_full_att = fused_att_layer(
        long_input,
        global_input,
        att_implementation='full')

    self.assertAllClose(long_output, long_output_full_att)
    self.assertAllClose(global_output, global_output_full_att)

  @parameterized.named_parameters(
      dict(testcase_name='share_nothing'),
      dict(testcase_name='share_kv_projections', share_kv_projections=True),
      dict(testcase_name='share_qkv_projections', share_qkv_projections=True),
      dict(
          testcase_name='share_qkv_projections_supersedes_kv',
          share_kv_projections=True,
          share_qkv_projections=True),
      dict(
          testcase_name='share_att_output_projection',
          share_att_output_projection=True),
      dict(
          testcase_name='share_everything',
          share_qkv_projections=True,
          share_att_output_projection=True),
  )
  def test_fused_global_local_attention_shared_sublayers(
      self,
      share_kv_projections=False,
      share_qkv_projections=False,
      share_att_output_projection=False):

    hidden_size = 10

    layer = etc_layers.FusedGlobalLocalAttention(
        long_hidden_size=hidden_size,
        global_hidden_size=hidden_size,
        num_heads=5,
        local_radius=7,
        relative_vocab_size=9,
        share_kv_projections=share_kv_projections,
        share_qkv_projections=share_qkv_projections,
        share_att_output_projection=share_att_output_projection)
    # Run layer to make sure all variables are built.
    layer(
        long_input=tf.ones([1, 1, hidden_size]),
        global_input=tf.ones([1, 1, hidden_size]))

    if share_qkv_projections:
      self.assertIs(layer.long_query_projection, layer.global_query_projection)
      self.assert_all_identical(layer.l2l_key_projection,
                                layer.l2g_key_projection,
                                layer.g2g_key_projection,
                                layer.g2l_key_projection)
      self.assert_all_identical(layer.l2l_value_projection,
                                layer.l2g_value_projection,
                                layer.g2g_value_projection,
                                layer.g2l_value_projection)
    elif share_kv_projections:
      self.assertIsNot(layer.long_query_projection,
                       layer.global_query_projection)
      self.assertIs(layer.l2l_key_projection, layer.l2g_key_projection)
      self.assertIs(layer.g2g_key_projection, layer.g2l_key_projection)
      self.assertIsNot(layer.l2l_key_projection, layer.g2g_key_projection)
      self.assertIs(layer.l2l_value_projection, layer.l2g_value_projection)
      self.assertIs(layer.g2g_value_projection, layer.g2l_value_projection)
      self.assertIsNot(layer.l2l_value_projection, layer.g2g_value_projection)
    else:
      self.assertIsNot(layer.long_query_projection,
                       layer.global_query_projection)
      self.assertIsNot(layer.l2l_key_projection, layer.l2g_key_projection)
      self.assertIsNot(layer.l2l_key_projection, layer.g2g_key_projection)
      self.assertIsNot(layer.l2l_value_projection, layer.l2g_value_projection)
      self.assertIsNot(layer.l2l_value_projection, layer.g2g_value_projection)

    self.assertIsNot(layer.long_query_projection, layer.l2l_key_projection)
    self.assertIsNot(layer.long_query_projection, layer.l2l_value_projection)
    self.assertIsNot(layer.l2l_key_projection, layer.l2l_value_projection)

    if share_att_output_projection:
      self.assertIs(layer.long_output_projection,
                    layer.global_output_projection)
    else:
      self.assertIsNot(layer.long_output_projection,
                       layer.global_output_projection)

  def test_fused_global_local_attention_custom_total_att_size(self):
    tf.compat.v1.random.set_random_seed(1234)
    np.random.seed(1234)

    batch_size = 3
    long_seq_len = 12
    global_seq_len = 6
    hidden_size = 11
    num_heads = 5
    local_radius = 2
    total_att_size = 10
    relative_vocab_size = 9

    long_input = tf.constant(
        np.random.normal(size=[batch_size, long_seq_len, hidden_size]))
    global_input = tf.constant(
        np.random.normal(size=[batch_size, global_seq_len, hidden_size]))
    l2l_att_mask = tf.constant(
        np.random.binomial(
            n=1, p=0.9, size=[batch_size, long_seq_len, 2 * local_radius + 1]))
    g2g_att_mask = tf.constant(
        np.random.binomial(
            n=1, p=0.9, size=[batch_size, global_seq_len, global_seq_len]))
    l2g_att_mask = tf.constant(
        np.random.binomial(
            n=1, p=0.9, size=[batch_size, long_seq_len, global_seq_len]))
    g2l_att_mask = tf.constant(
        np.random.binomial(
            n=1, p=0.9, size=[batch_size, global_seq_len, long_seq_len]))
    l2l_relative_att_ids = tf.constant(
        np.random.randint(
            relative_vocab_size,
            size=[batch_size, long_seq_len, 2 * local_radius + 1]))
    g2g_relative_att_ids = tf.constant(
        np.random.randint(
            relative_vocab_size,
            size=[batch_size, global_seq_len, global_seq_len]))
    l2g_relative_att_ids = tf.constant(
        np.random.randint(
            relative_vocab_size,
            size=[batch_size, long_seq_len, global_seq_len]))
    g2l_relative_att_ids = tf.constant(
        np.random.randint(
            relative_vocab_size,
            size=[batch_size, global_seq_len, long_seq_len]))

    fused_att_layer = etc_layers.FusedGlobalLocalAttention(
        long_hidden_size=hidden_size,
        global_hidden_size=hidden_size,
        num_heads=num_heads,
        local_radius=local_radius,
        long_total_att_size=total_att_size,
        global_total_att_size=total_att_size,
        relative_vocab_size=relative_vocab_size,
        share_qkv_projections=True,
        share_att_output_projection=True)

    long_output, global_output = fused_att_layer(
        long_input,
        global_input,
        l2l_att_mask=l2l_att_mask,
        g2g_att_mask=g2g_att_mask,
        l2g_att_mask=l2g_att_mask,
        g2l_att_mask=g2l_att_mask,
        l2l_relative_att_ids=l2l_relative_att_ids,
        g2g_relative_att_ids=g2g_relative_att_ids,
        l2g_relative_att_ids=l2g_relative_att_ids,
        g2l_relative_att_ids=g2l_relative_att_ids)

    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.assertAllEqual([batch_size, long_seq_len, hidden_size],
                        long_output.shape)
    self.assertAllEqual([batch_size, global_seq_len, hidden_size],
                        global_output.shape)

  def test_attention_head_projection(self):
    inputs = tf.ones([2, 3, 10])
    layer = etc_layers.ProjectAttentionHeads(num_heads=4, size_per_head=5)
    result = layer(inputs)
    self.assertAllEqual([2, 3, 4, 5], result.shape)

    inputs = tf.ones([2, 3, 4, 10])
    layer = etc_layers.ProjectAttentionHeads(num_heads=5, size_per_head=6)
    result = layer(inputs)
    self.assertAllEqual([2, 3, 4, 5, 6], result.shape)

  @parameterized.named_parameters(
      ('using_gather', False),
      ('using_one_hot', True),
  )
  def test_qkv_relative_attention(self, use_one_hot_lookup):
    # batch_size: 2
    # query_len: 3
    # key_len: 4
    # num_heads: 2
    # key_size_per_head: 3
    # value_size_per_head: 5
    # relative_vocab_size: 6

    # [batch_size, query_len, num_heads, key_size_per_head]
    queries = tf.constant([
        [
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ],  #
        [
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    ])

    # [batch_size, key_len, num_heads, key_size_per_head]
    keys = tf.constant([
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[99.0, 0.0, 0.0], [-99.0, 0.0, 0.0]],
            [[0.0, 0.0, 99.0], [0.0, 0.0, -99.0]],
            [[0.0, 99.0, 0.0], [0.0, -99.0, 0.0]],
        ],  #
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[99.0, 0.0, 0.0], [-99.0, 0.0, 0.0]],
            [[0.0, 99.0, 0.0], [0.0, -99.0, 0.0]],
            [[0.0, 0.0, 99.0], [0.0, 0.0, -99.0]],
        ]
    ])

    # [batch_size, key_len, num_heads, value_size_per_head]
    values = tf.constant([
        [
            [[0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, -0.1]],
            [[0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, -0.2]],
            [[0.3, 0.3, 0.3, 0.3, 0.3], [0.3, 0.3, 0.3, 0.3, -0.3]],
            [[0.4, 0.4, 0.4, 0.4, 0.4], [0.4, 0.4, 0.4, 0.4, -0.4]],
        ],  #
        [
            [[-0.1, 0.1, 0.1, 0.1, 0.1], [-0.1, 0.1, 0.1, 0.1, -0.1]],
            [[-0.2, 0.2, 0.2, 0.2, 0.2], [-0.2, 0.2, 0.2, 0.2, -0.2]],
            [[-0.3, 0.3, 0.3, 0.3, 0.3], [-0.3, 0.3, 0.3, 0.3, -0.3]],
            [[-0.4, 0.4, 0.4, 0.4, 0.4], [-0.4, 0.4, 0.4, 0.4, -0.4]],
        ]
    ])

    # [batch_size, query_len, key_len]
    att_mask = tf.constant([
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        [
            [1, 0, 1, 1],
            [1, 0, 1, 1],
            [1, 0, 1, 1],
        ],
    ])

    # [batch_size, query_len, key_len]
    relative_att_ids = tf.constant([
        [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 5],
        ],
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 2, 2],
        ],
    ])

    # [relative_vocab_size, num_heads, key_size_per_head]
    relative_emb_table = [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [-99.0, 0.0, 0.0]],
        [[-99.0, 0.0, 0.0], [99.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, -99.0]],
    ]

    layer = etc_layers.QkvRelativeAttention(
        relative_vocab_size=6,
        use_one_hot_lookup=use_one_hot_lookup,
        initializer=tf.initializers.constant(relative_emb_table))

    result = layer(
        queries=queries,
        keys=keys,
        values=values,
        att_mask=att_mask,
        relative_att_ids=relative_att_ids)

    self.evaluate(tf.compat.v1.global_variables_initializer())

    expected = [
        [
            [[0.2, 0.2, 0.2, 0.2, 0.2], [0.35, 0.35, 0.35, 0.35, -0.35]],
            [[0.4, 0.4, 0.4, 0.4, 0.4], [0.2, 0.2, 0.2, 0.2, -0.2]],
            [[0.3, 0.3, 0.3, 0.3, 0.3], [0.15, 0.15, 0.15, 0.15, -0.15]],
        ],  #
        [
            [[-0.35, 0.35, 0.35, 0.35, 0.35], [-0.1, 0.1, 0.1, 0.1, -0.1]],
            [[-0.3, 0.3, 0.3, 0.3, 0.3], [-0.25, 0.25, 0.25, 0.25, -0.25]],
            [[-0.1, 0.1, 0.1, 0.1, 0.1], [-0.35, 0.35, 0.35, 0.35, -0.35]],
        ]
    ]
    self.assertAllEqual([2, 3, 2, 5], result.shape)
    self.assertAllClose(expected, result)

  @parameterized.named_parameters(
      dict(testcase_name='even_blocking_with_gather', local_radius=15),
      dict(testcase_name='uneven_blocking_with_gather', local_radius=16),
      dict(testcase_name='degenerate_blocking_with_gather', local_radius=35),
      dict(
          testcase_name='even_blocking_with_one_hot',
          local_radius=15,
          use_one_hot_lookup=True),
      dict(
          testcase_name='uneven_blocking_with_one_hot',
          local_radius=16,
          use_one_hot_lookup=True),
      dict(
          testcase_name='degenerate_blocking_with_one_hot',
          local_radius=35,
          use_one_hot_lookup=True),
      dict(
          testcase_name='even_blocking_with_gather_full_att',
          local_radius=15,
          att_implementation='full'),
      dict(
          testcase_name='uneven_blocking_with_gather_full_att',
          local_radius=16,
          att_implementation='full'),
      dict(
          testcase_name='degenerate_blocking_with_gather_full_att',
          local_radius=35,
          att_implementation='full'),
      dict(
          testcase_name='even_blocking_with_one_hot_full_att',
          local_radius=15,
          use_one_hot_lookup=True,
          att_implementation='full'),
      dict(
          testcase_name='uneven_blocking_with_one_hot_full_att',
          local_radius=16,
          use_one_hot_lookup=True,
          att_implementation='full'),
      dict(
          testcase_name='degenerate_blocking_with_one_hot_full_att',
          local_radius=35,
          use_one_hot_lookup=True,
          att_implementation='full'),
  )
  def test_qkv_relative_local_attention(self,
                                        local_radius,
                                        use_one_hot_lookup=False,
                                        att_implementation='sparse'):
    tf.compat.v1.random.set_random_seed(1234)
    np.random.seed(1234)

    batch_size = 2
    long_len = 64
    side_len = 6
    num_heads = 5
    key_size_per_head = 2
    value_size_per_head = 3
    relative_vocab_size = 7
    # Note: block_len = local_radius + 1

    queries = tf.constant(
        np.random.normal(
            size=[batch_size, long_len, num_heads, key_size_per_head]),
        tf.float32)
    keys = tf.constant(
        np.random.normal(
            size=[batch_size, long_len, num_heads, key_size_per_head]),
        tf.float32)
    values = tf.constant(
        np.random.normal(
            size=[batch_size, long_len, num_heads, value_size_per_head]),
        tf.float32)
    att_mask = tf.constant(
        np.random.binomial(
            n=1, p=0.9, size=[batch_size, long_len, 2 * local_radius + 1]))
    relative_att_ids = tf.constant(
        np.random.randint(
            relative_vocab_size,
            size=[batch_size, long_len, 2 * local_radius + 1]))

    side_keys = tf.constant(
        np.random.normal(
            size=[batch_size, side_len, num_heads, key_size_per_head]),
        tf.float32)
    side_values = tf.constant(
        np.random.normal(
            size=[batch_size, side_len, num_heads, value_size_per_head]),
        tf.float32)
    side_att_mask = tf.constant(
        np.random.binomial(n=1, p=0.9, size=[batch_size, long_len, side_len]))
    side_relative_att_ids = tf.constant(
        np.random.randint(
            relative_vocab_size, size=[batch_size, long_len, side_len]))

    layer = etc_layers.QkvRelativeLocalAttention(
        local_radius=local_radius,
        relative_vocab_size=relative_vocab_size,
        use_one_hot_lookup=use_one_hot_lookup)

    result1 = layer(
        queries,
        keys,
        values,
        att_mask=att_mask,
        relative_att_ids=relative_att_ids,
        side_keys=side_keys,
        side_values=side_values,
        side_att_mask=side_att_mask,
        side_relative_att_ids=side_relative_att_ids,
        att_implementation=att_implementation)
    self.assertAllEqual([batch_size, long_len, num_heads, value_size_per_head],
                        result1.shape)

    result2 = layer(
        queries,
        keys,
        values,
        att_mask=None,
        relative_att_ids=None,
        side_keys=side_keys,
        side_values=side_values,
        side_att_mask=None,
        side_relative_att_ids=None,
        att_implementation=att_implementation)
    self.assertAllEqual([batch_size, long_len, num_heads, value_size_per_head],
                        result2.shape)

    result3 = layer(
        queries,
        keys,
        values,
        att_mask=att_mask,
        relative_att_ids=relative_att_ids,
        side_keys=None,
        side_values=None,
        side_att_mask=None,
        side_relative_att_ids=None,
        att_implementation=att_implementation)
    self.assertAllEqual([batch_size, long_len, num_heads, value_size_per_head],
                        result3.shape)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertNotAllClose(result1, result2)
    self.assertNotAllClose(result2, result3)
    self.assertNotAllClose(result1, result3)

  @parameterized.named_parameters(
      ('even_blocking_with_gather', 15, False),
      ('uneven_blocking_with_gather', 16, False),
      ('degenerate_blocking_with_gather', 35, False),
      ('even_blocking_with_one_hot', 15, True),
      ('uneven_blocking_with_one_hot', 16, True),
      ('degenerate_blocking_with_one_hot', 35, True),
  )
  def test_qkv_relative_local_attention_full_att_implementation(
      self, local_radius, use_one_hot_lookup):
    # We check the validity of the `att_implementation` option
    # by confirming both internal implementations return the same output.

    tf.compat.v1.random.set_random_seed(1234)
    np.random.seed(1234)

    batch_size = 3
    long_len = 64
    side_len = 6
    num_heads = 5
    key_size_per_head = 2
    value_size_per_head = 3
    relative_vocab_size = 7
    # Note: block_len = local_radius + 1

    queries = tf.constant(
        np.random.normal(
            size=[batch_size, long_len, num_heads, key_size_per_head]),
        tf.float32)
    keys = tf.constant(
        np.random.normal(
            size=[batch_size, long_len, num_heads, key_size_per_head]),
        tf.float32)
    values = tf.constant(
        np.random.normal(
            size=[batch_size, long_len, num_heads, value_size_per_head]),
        tf.float32)
    att_mask = tf.constant(
        np.random.binomial(
            n=1, p=0.8, size=[batch_size, long_len, 2 * local_radius + 1]),
        dtype=tf.int32)
    relative_att_ids = tf.constant(
        np.random.randint(
            relative_vocab_size,
            size=[batch_size, long_len, 2 * local_radius + 1]),
        dtype=tf.int32)
    side_keys = tf.constant(
        np.random.normal(
            size=[batch_size, side_len, num_heads, key_size_per_head]),
        tf.float32)
    side_values = tf.constant(
        np.random.normal(
            size=[batch_size, side_len, num_heads, value_size_per_head]),
        tf.float32)
    side_att_mask = tf.constant(
        np.random.binomial(n=1, p=0.8, size=[batch_size, long_len, side_len]),
        dtype=tf.int32)
    side_relative_att_ids = tf.constant(
        np.random.randint(
            relative_vocab_size, size=[batch_size, long_len, side_len]),
        dtype=tf.int32)

    layer = etc_layers.QkvRelativeLocalAttention(
        local_radius=local_radius,
        relative_vocab_size=relative_vocab_size,
        use_one_hot_lookup=use_one_hot_lookup)

    sparse_implementation_result = layer(
        queries,
        keys,
        values,
        att_mask=att_mask,
        relative_att_ids=relative_att_ids,
        side_keys=side_keys,
        side_values=side_values,
        side_att_mask=side_att_mask,
        side_relative_att_ids=side_relative_att_ids,
        att_implementation='sparse')

    full_implementation_result = layer(
        queries,
        keys,
        values,
        att_mask=att_mask,
        relative_att_ids=relative_att_ids,
        side_keys=side_keys,
        side_values=side_values,
        side_att_mask=side_att_mask,
        side_relative_att_ids=side_relative_att_ids,
        att_implementation='full')

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(sparse_implementation_result,
                        full_implementation_result)


class HelpersTest(tf.test.TestCase):

  def test_expand_local_ids_to_blocks_with_even_blocking(self):
    # batch_size = 2
    # seq_len = 6
    # local_radius = 1
    # block_len = 2

    # [batch_size, seq_len, 2*local_radius + 1]
    local_ids = tf.constant([
        [
            [1, 2, 3],  #
            [4, 5, 6],  #
            [7, 8, 9],  #
            [10, 11, 12],  #
            [13, 14, 15],  #
            [16, 17, 18],  #
        ],  #
        [
            [-1, -2, -3],  #
            [-4, -5, -6],  #
            [-7, -8, -9],  #
            [-10, -11, -12],  #
            [-13, -14, -15],  #
            [-16, -17, -18],  #
        ],  #
    ])

    self.assertAllEqual(
        [
            [
                [
                    [0, 1, 2, 3, 0, 0],  #
                    [0, 0, 4, 5, 6, 0],  #
                ],  #
                [
                    [0, 7, 8, 9, 0, 0],  #
                    [0, 0, 10, 11, 12, 0],  #
                ],  #
                [
                    [0, 13, 14, 15, 0, 0],  #
                    [0, 0, 16, 17, 18, 0],  #
                ]
            ],  #
            [
                [
                    [0, -1, -2, -3, 0, 0],  #
                    [0, 0, -4, -5, -6, 0],  #
                ],  #
                [
                    [0, -7, -8, -9, 0, 0],  #
                    [0, 0, -10, -11, -12, 0],  #
                ],  #
                [
                    [0, -13, -14, -15, 0, 0],  #
                    [0, 0, -16, -17, -18, 0],  #
                ]
            ],  #
        ],
        attention._expand_local_ids_to_blocks(
            local_ids, mask_padding_ids=False))

    self.assertAllEqual(
        [
            [
                [
                    [0, 0, 2, 3, 0, 0],  #
                    [0, 0, 4, 5, 6, 0],  #
                ],  #
                [
                    [0, 7, 8, 9, 0, 0],  #
                    [0, 0, 10, 11, 12, 0],  #
                ],  #
                [
                    [0, 13, 14, 15, 0, 0],  #
                    [0, 0, 16, 17, 0, 0],  #
                ]
            ],  #
            [
                [
                    [0, 0, -2, -3, 0, 0],  #
                    [0, 0, -4, -5, -6, 0],  #
                ],  #
                [
                    [0, -7, -8, -9, 0, 0],  #
                    [0, 0, -10, -11, -12, 0],  #
                ],  #
                [
                    [0, -13, -14, -15, 0, 0],  #
                    [0, 0, -16, -17, 0, 0],  #
                ]
            ],  #
        ],
        attention._expand_local_ids_to_blocks(local_ids))

  def test_expand_local_ids_to_blocks_with_uneven_blocking(self):
    # batch_size = 2
    # seq_len = 5
    # local_radius = 2
    # block_len = 3

    # [batch_size, seq_len, 2*local_radius + 1]
    local_ids = tf.constant([
        [
            [1, 2, 3, 4, 5],  #
            [6, 7, 8, 9, 10],  #
            [11, 12, 13, 14, 15],  #
            [16, 17, 18, 19, 20],  #
            [21, 22, 23, 24, 25],  #
        ],  #
        [
            [-1, -2, -3, -4, -5],  #
            [-6, -7, -8, -9, -10],  #
            [-11, -12, -13, -14, -15],  #
            [-16, -17, -18, -19, -20],  #
            [-21, -22, -23, -24, -25],  #
        ],  #
    ])

    self.assertAllEqual(
        [
            [
                [
                    [0, 1, 2, 3, 4, 5, 0, 0, 0],  #
                    [0, 0, 6, 7, 8, 9, 10, 0, 0],  #
                    [0, 0, 0, 11, 12, 13, 14, 15, 0],  #
                ],  #
                [
                    [0, 16, 17, 18, 19, 20, 0, 0, 0],  #
                    [0, 0, 21, 22, 23, 24, 25, 0, 0],  #
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],  #
                ]
            ],  #
            [
                [
                    [0, -1, -2, -3, -4, -5, 0, 0, 0],  #
                    [0, 0, -6, -7, -8, -9, -10, 0, 0],  #
                    [0, 0, 0, -11, -12, -13, -14, -15, 0],  #
                ],  #
                [
                    [0, -16, -17, -18, -19, -20, 0, 0, 0],  #
                    [0, 0, -21, -22, -23, -24, -25, 0, 0],  #
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],  #
                ]
            ],  #
        ],
        attention._expand_local_ids_to_blocks(
            local_ids, mask_padding_ids=False))

    self.assertAllEqual(
        [
            [
                [
                    [0, 0, 0, 3, 4, 5, 0, 0, 0],  #
                    [0, 0, 0, 7, 8, 9, 10, 0, 0],  #
                    [0, 0, 0, 11, 12, 13, 14, 15, 0],  #
                ],  #
                [
                    [0, 16, 17, 18, 19, 0, 0, 0, 0],  #
                    [0, 0, 21, 22, 23, 0, 0, 0, 0],  #
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],  #
                ]
            ],  #
            [
                [
                    [0, 0, 0, -3, -4, -5, 0, 0, 0],  #
                    [0, 0, 0, -7, -8, -9, -10, 0, 0],  #
                    [0, 0, 0, -11, -12, -13, -14, -15, 0],  #
                ],  #
                [
                    [0, -16, -17, -18, -19, 0, 0, 0, 0],  #
                    [0, 0, -21, -22, -23, 0, 0, 0, 0],  #
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],  #
                ]
            ],  #
        ],
        attention._expand_local_ids_to_blocks(local_ids))

  def test_expand_local_ids_to_blocks_with_uneven_blocking_ones_mask(self):
    # batch_size = 1
    # seq_len = 7
    # local_radius = 2
    # block_len = 3

    # [batch_size, seq_len, 2*local_radius + 1]
    local_ids = tf.constant([
        [
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1],  #
        ],  #
    ])

    self.assertAllEqual(
        [
            [
                [
                    [0, 1, 1, 1, 1, 1, 0, 0, 0],  #
                    [0, 0, 1, 1, 1, 1, 1, 0, 0],  #
                    [0, 0, 0, 1, 1, 1, 1, 1, 0],  #
                ],  #
                [
                    [0, 1, 1, 1, 1, 1, 0, 0, 0],  #
                    [0, 0, 1, 1, 1, 1, 1, 0, 0],  #
                    [0, 0, 0, 1, 1, 1, 1, 1, 0],  #
                ],  #
                [
                    [0, 1, 1, 1, 1, 1, 0, 0, 0],  #
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],  #
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],  #
                ],  #
            ],  #
        ],
        attention._expand_local_ids_to_blocks(
            local_ids, mask_padding_ids=False))

    self.assertAllEqual(
        [
            [
                [
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],  #
                    [0, 0, 0, 1, 1, 1, 1, 0, 0],  #
                    [0, 0, 0, 1, 1, 1, 1, 1, 0],  #
                ],  #
                [
                    [0, 1, 1, 1, 1, 1, 0, 0, 0],  #
                    [0, 0, 1, 1, 1, 1, 1, 0, 0],  #
                    [0, 0, 0, 1, 1, 1, 1, 0, 0],  #
                ],  #
                [
                    [0, 1, 1, 1, 0, 0, 0, 0, 0],  #
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],  #
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],  #
                ],  #
            ],  #
        ],
        attention._expand_local_ids_to_blocks(local_ids))

  def test_expand_local_ids_to_blocks_with_degenerate_blocking(self):
    # batch_size = 2
    # seq_len = 2
    # local_radius = 2
    # block_len = 3

    # [batch_size, seq_len, 2*local_radius + 1]
    local_ids = tf.constant([
        [
            [1, 2, 3, 4, 5],  #
            [6, 7, 8, 9, 10],  #
        ],  #
        [
            [-1, -2, -3, -4, -5],  #
            [-6, -7, -8, -9, -10],  #
        ],  #
    ])

    self.assertAllEqual(
        [
            [  #
                [
                    [0, 1, 2, 3, 4, 5, 0, 0, 0],  #
                    [0, 0, 6, 7, 8, 9, 10, 0, 0],  #
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],  #
                ]  #
            ],  #
            [  #
                [
                    [0, -1, -2, -3, -4, -5, 0, 0, 0],  #
                    [0, 0, -6, -7, -8, -9, -10, 0, 0],  #
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],  #
                ]  #
            ],  #
        ],
        attention._expand_local_ids_to_blocks(
            local_ids, mask_padding_ids=False))

    self.assertAllEqual(
        [
            [  #
                [
                    [0, 0, 0, 3, 4, 0, 0, 0, 0],  #
                    [0, 0, 0, 7, 8, 0, 0, 0, 0],  #
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],  #
                ]  #
            ],  #
            [  #
                [
                    [0, 0, 0, -3, -4, 0, 0, 0, 0],  #
                    [0, 0, 0, -7, -8, 0, 0, 0, 0],  #
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],  #
                ]  #
            ],  #
        ],
        attention._expand_local_ids_to_blocks(local_ids))


if __name__ == '__main__':
  tf.test.main()
