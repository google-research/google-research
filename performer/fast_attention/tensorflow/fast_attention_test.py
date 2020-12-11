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

"""Tests for FAVOR attention."""
# pylint: disable=invalid-name, missing-function-docstring

import math
import tensorflow as tf
from performer.fast_attention.tensorflow import fast_attention


class TransformerLayersTest(tf.test.TestCase):

  def test_relu_noncausal_attention_block_output(self):
    batch_size = 1
    length = 10
    num_heads = 1
    dim = 4
    query = tf.ones([batch_size, length, num_heads, dim])
    key = tf.ones([batch_size, length, num_heads, dim])
    value = tf.ones([batch_size, length, num_heads, dim])
    kernel_transformation = fast_attention.relu_kernel_transformation
    attention_block_output = fast_attention.favor_attention(
        query, key, value, kernel_transformation, False)
    self.assertListEqual(attention_block_output.get_shape().as_list(),
                         [batch_size, length, num_heads, dim])

  def test_relu_causal_attention_block_output_shape(self):
    batch_size = 1
    length = 10
    num_heads = 1
    dim = 4
    query = tf.ones([batch_size, length, num_heads, dim])
    key = tf.ones([batch_size, length, num_heads, dim])
    value = tf.ones([batch_size, length, num_heads, dim])
    kernel_transformation = fast_attention.relu_kernel_transformation
    attention_block_output = fast_attention.favor_attention(
        query, key, value, kernel_transformation, True)
    self.assertListEqual(attention_block_output.get_shape().as_list(),
                         [batch_size, length, num_heads, dim])

  def test_softmax_noncausal_attention_block_output_shape(self):
    batch_size = 1
    length = 10
    num_heads = 1
    dim = 4
    num_random_features = 350
    query = tf.ones([batch_size, length, num_heads, dim])
    key = tf.ones([batch_size, length, num_heads, dim])
    value = tf.ones([batch_size, length, num_heads, dim])
    kernel_transformation = fast_attention.softmax_kernel_transformation
    projection_matrix = fast_attention.create_projection_matrix(
        num_random_features, dim)
    attention_block_output = fast_attention.favor_attention(
        query, key, value, kernel_transformation, False, projection_matrix)
    self.assertListEqual(attention_block_output.get_shape().as_list(),
                         [batch_size, length, num_heads, dim])

  def test_softmax_noncausal_attention_block_output(self):
    batch_size = 1
    length = 2
    num_heads = 1
    dim = 8
    num_random_features = 30000
    query = tf.random.normal([batch_size, length, num_heads, dim])
    key = tf.random.normal([batch_size, length, num_heads, dim])
    value = tf.random.normal([batch_size, length, num_heads, dim])
    kernel_transformation = fast_attention.softmax_kernel_transformation
    projection_matrix = fast_attention.create_projection_matrix(
        num_random_features, dim)
    attention_block_output = fast_attention.favor_attention(
        query, key, value, kernel_transformation, False, projection_matrix)

    query = tf.multiply(query, 1.0 / math.sqrt(float(dim)))
    attention_scores = tf.einsum("BXHD,BYHD->BXYH", query, key)
    attention_scores = tf.nn.softmax(attention_scores, axis=2)
    exact_attention_block_output = tf.einsum("BXYH,BYHD->BXHD",
                                             attention_scores, value)
    max_error = 2.0
    error = tf.math.abs(
        (exact_attention_block_output - attention_block_output) /
        exact_attention_block_output)
    self.assertLess(tf.math.reduce_max(tf.math.abs(error)), max_error)

  def test_fast_attention(self):
    hidden_size = 64
    num_heads = 4
    dropout = 0.5
    dim_per_head = hidden_size // num_heads
    layer = fast_attention.SelfAttention(hidden_size, num_heads, dropout)
    self.assertDictEqual(
        layer.get_config(), {
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "attention_dropout": dropout,
        })
    length = 2
    x = tf.ones([1, length, hidden_size])
    bias = tf.ones([1])
    cache = {
        "k": tf.zeros([1, 0, num_heads, dim_per_head]),
        "v": tf.zeros([1, 0, num_heads, dim_per_head]),
    }
    y = layer(x, bias, training=True, cache=cache)
    self.assertEqual(y.shape, (
        1,
        length,
        64,
    ))
    self.assertEqual(cache["k"].shape, (
        1,
        length,
        num_heads,
        dim_per_head,
    ))
    self.assertEqual(cache["v"].shape, (
        1,
        length,
        num_heads,
        dim_per_head,
    ))

  def test_custom_causal_gradients(self):
    L = 64
    B = 128
    H = 4
    D = 64
    M = 128
    qs = tf.random.normal([L, B, H, M])
    ks = tf.random.normal([L, B, H, M])
    vs = tf.random.normal([L, B, H, D])
    num_coefs = tf.random.normal(vs.shape)
    den_coefs = tf.random.normal([L, B, H])

    with tf.GradientTape() as tape:
      tape.watch([qs, ks, vs])
      num = fast_attention.causal_numerator(qs, ks, vs)
      den = fast_attention.causal_denominator(qs, ks)
      loss = tf.reduce_sum(num * num_coefs) + tf.reduce_sum(den * den_coefs) * 0

    grads1 = tape.gradient(loss, [qs, ks, vs])
    self.assertListEqual([L, B, H, M], grads1[0].get_shape().as_list())
    self.assertListEqual([L, B, H, M], grads1[1].get_shape().as_list())
    self.assertListEqual([L, B, H, D], grads1[2].get_shape().as_list())


if __name__ == "__main__":
  tf.test.main()
