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

"""Tests for coltran layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl.testing import parameterized
from ml_collections import ConfigDict
import numpy as np
import tensorflow as tf
from coltran.models import layers


layer_hparams = itertools.product(["mean", "learnable"],
                                  ["sc", "cs"])
layer_hparams = [(a+s, a, s) for a, s in layer_hparams]


class LayersTest(tf.test.TestCase, parameterized.TestCase):

  def test_cache_layer(self):
    cache = layers.Cache(canvas_shape=(2, 4))

    # update 1
    exp_first = tf.range(8, dtype=tf.float32)
    exp_first = tf.reshape(exp_first, (1, 2, 2, 2))
    index = tf.stack([0, 0])
    out = cache(inputs=(exp_first, index))
    out_slice = out.numpy()[:1, :2, :2, :2]
    self.assertTrue(np.allclose(out_slice, exp_first.numpy()))

    # update 2
    exp_second = tf.range(8, 16, dtype=tf.float32)
    exp_second = tf.reshape(exp_second, (1, 2, 2, 2))
    index = tf.stack([0, 2])
    out = cache(inputs=(exp_second, index))
    out_np = out.numpy()
    first, second = out_np[:1, :2, :2, :2], out_np[:1, :2, 2:, :2]
    self.assertTrue(np.allclose(second, exp_second.numpy()))
    self.assertTrue(np.allclose(first, exp_first.numpy()))

    # update 3 (special case)
    exp_third = tf.reshape([50.0, 100.0], (1, 1, 1, 2))
    index = tf.stack([0, 0])
    out = cache(inputs=(exp_third, index))
    out_np = out.numpy()
    self.assertTrue(np.allclose(out_np[0, 0, 0, :2], [50.0, 100.0]))

  def test_shift_layer(self):
    # shift down
    down_shift = layers.Shift(dimension=0, resolution=[3, 3])
    input_np = np.arange(9).reshape((1, 3, 3))
    input_t = tf.convert_to_tensor(input_np)
    input_down = down_shift(input_t).numpy()
    equality = input_np[:, :-1] == input_down[:, 1:]
    self.assertTrue(np.all(equality))

    # shift right.
    right_shift = layers.Shift(dimension=1, resolution=[3, 3])
    input_np = np.arange(9).reshape((1, 3, 3))
    input_t = tf.convert_to_tensor(input_np)
    input_right = right_shift(input_t).numpy()
    equality = input_np[:, :, :-1] == input_right[:, :, 1:]
    self.assertTrue(np.all(equality))

  def test_position_embed(self):
    pos_embed = layers.PositionEmbed(
        axes=[1, 2], max_lengths=[64, 32])
    inputs = tf.random.uniform(shape=(8, 64, 32, 256))
    embedded = pos_embed(inputs)
    for variable in pos_embed.variables:
      if len(variable.shape) == 3:
        self.assertEqual(variable.shape, (64, 1, 256))
      else:
        self.assertEqual(variable.shape, (32, 256))
    self.assertEqual(embedded.shape, (8, 64, 32, 256))

  @parameterized.named_parameters(*layer_hparams)
  def test_conditional_layer_norm(self, spatial_average, sequence):
    cond_layer_norm = layers.ConditionalLayerNorm(
        spatial_average=spatial_average, sequence=sequence)
    x = tf.random.uniform(shape=(8, 32, 32, 128))
    cond_inputs = tf.random.uniform(shape=(8, 32, 32, 128))
    out = cond_layer_norm(inputs=(x, cond_inputs))
    self.assertEqual(out.shape, (8, 32, 32, 128))

  def test_self_attention_nd_cond_scale(self):
    row_mask = layers.SelfAttentionND(
        hidden_size=256, num_heads=4, nd_block_size=[1, 32],
        resolution=[32, 32], cond_q=True, cond_k=True, cond_v=True,
        cond_scale=True)
    inputs = tf.random.uniform(shape=(1, 3, 32, 32, 3))
    cond_inputs = tf.random.uniform(shape=(1, 3, 32, 32, 3))
    output = row_mask(inputs=(inputs, cond_inputs))
    self.assertEqual(output.shape, (1, 3, 32, 32, 256))

  def test_self_attention_nd_cond_scale_false(self):
    row_mask = layers.SelfAttentionND(
        hidden_size=256, num_heads=4, nd_block_size=[1, 32],
        resolution=[32, 32], cond_q=True, cond_k=True, cond_v=True,
        cond_scale=False)
    inputs = tf.random.uniform(shape=(1, 3, 32, 32, 3))
    cond_inputs = tf.random.uniform(shape=(1, 3, 32, 32, 3))
    output = row_mask(inputs=(inputs, cond_inputs))
    self.assertEqual(output.shape, (1, 3, 32, 32, 256))

  def test_row_attention(self):
    # row with cache
    row = layers.SelfAttentionND(
        hidden_size=256, num_heads=4, nd_block_size=[1, 32],
        resolution=[32, 32])
    x = tf.random.uniform(shape=[4, 2, 32, 3])
    output = row(inputs=x)
    self.assertEqual(row.attention_dim_q, -3)
    self.assertEqual(row.attention_dim_k, -3)
    self.assertEqual(output.shape, (4, 2, 32, 256))

  def test_column_attention(self):
    # row with cache
    column = layers.SelfAttentionND(
        hidden_size=256, num_heads=4, nd_block_size=[32, 1],
        resolution=[32, 32])
    x = tf.random.uniform(shape=[4, 32, 2, 3])
    output = column(inputs=x)
    self.assertEqual(output.shape, (4, 32, 2, 256))

  def test_row_attention_mask(self):
    row_mask = layers.SelfAttentionND(
        hidden_size=256, num_heads=4, nd_block_size=[1, 32],
        resolution=[32, 32], mask="future")
    x = tf.random.uniform(shape=[4, 2, 32, 3])
    output = row_mask(inputs=x)
    self.assertEqual(row_mask.attention_dim_k, -3)
    self.assertEqual(row_mask.attention_dim_q, -3)
    self.assertEqual(output.shape, (4, 2, 32, 256))

  def test_col_attention_mask(self):
    col_mask = layers.SelfAttentionND(
        hidden_size=256, num_heads=8, nd_block_size=[4, 1],
        resolution=[4, 4], mask="future")
    x = tf.random.uniform(shape=[4, 4, 2, 3])
    output = col_mask(inputs=x)
    self.assertEqual(output.shape, (4, 4, 2, 256))
    self.assertEqual(col_mask.attention_dim_k, -4)
    self.assertEqual(col_mask.attention_dim_q, -4)

  def test_factorized_attention(self):
    config = ConfigDict()
    config.hidden_size = 256
    config.ff_size = 256
    config.num_encoder_layers = 2
    config.num_heads = 2
    fact = layers.FactorizedAttention(config)
    inputs = tf.random.uniform(shape=(8, 8, 8, 256))
    output = fact(inputs)
    self.assertEqual(output.shape, (8, 8, 8, 256))

if __name__ == "__main__":
  tf.test.main()
