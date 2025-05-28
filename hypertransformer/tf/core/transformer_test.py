# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Tests for `transformer.py`."""
import numpy as np
import tensorflow.compat.v1 as tf

from hypertransformer.tf.core import transformer


def _make_qkv_1():
  """Creates a specific query/key/value example."""
  v1 = np.array([1.0, 2.0, 3.0])
  v2 = np.array([5.0, 6.0, 7.0])
  q = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
  k = tf.constant([[0.0, 2.0], [15.0, 2.0]], dtype=tf.float32)
  # (1.0, 0.0) gives zero for k[0] and 15.0 for k[1] => choosing v2
  # (0.0, 1.0) gives 2.0 for both => choosing (v1 + v2) / 2
  v = tf.constant(np.stack([v1, v2]), dtype=tf.float32)
  return q, k, v, v1, v2


def _make_qkv_2():
  """Creates a larger query/key/value example for shape evaluation."""
  v1 = np.array([1.0, 2.0, 3.0])
  v2 = np.array([5.0, 6.0, 7.0])
  q = tf.constant([[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]],
                  dtype=tf.float32)
  k = tf.constant([[[0.0, 2.0, 0.0, 2.0], [15.0, 2.0, 15.0, 2.0]]],
                  dtype=tf.float32)
  v = tf.constant(np.stack([[v1, v2]]), dtype=tf.float32)
  return q, k, v, v1, v2


class TransformerTest(tf.test.TestCase):

  def test_attention(self):
    """Tests attention operation outputs on fixed examples."""
    q, k, v, v1, v2 = _make_qkv_1()
    result = transformer.attention(q, k, v, mask=None)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      output, _ = sess.run(result)
      self.assertAllClose(output[0], v2, rtol=1e-3)
      self.assertAllClose(output[1], (v1 + v2) / 2)

  def test_mha(self):
    """Verifies multi-headed attention layer output shapes."""
    params = transformer.TransformerParams(
        mha_output_dim=4, heads=2, num_layers=1, query_key_dim=2,
        internal_dim=2)
    mha = transformer.MultiHeadAttention(params)
    q, k, v, _, _ = _make_qkv_2()
    output, attn_weights = mha(q, k, v)
    self.assertEqual(output.shape, (1, 2, 4))
    self.assertEqual(attn_weights.shape, (1, 2, 2, 2))

  def _make_encoder(self):
    """Creates an encoder layer."""
    params = transformer.TransformerParams(
        mha_output_dim=512, heads=8, internal_dim=1024, num_layers=1,
        query_key_dim=128)
    sample_encoder_layer = transformer.EncoderLayer(params)
    return (sample_encoder_layer(tf.random.uniform((64, 43, 512)), False, None),
            params)

  def test_encoder(self):
    """Verifies encoder layer output shapes."""
    self.assertEqual(self._make_encoder()[0].shape, (64, 43, 512))

  def test_decoder(self):
    """Verifies decoder layer output shapes."""
    encoder_output, params = self._make_encoder()
    decoder_layer = transformer.DecoderLayer(params, name='decoder')
    decoder_output, _, _ = decoder_layer(
        tf.random.uniform((64, 50, 512)), encoder_output, False)
    self.assertEqual(decoder_output.shape, (64, 50, 512))


if __name__ == '__main__':
  tf.test.main()
