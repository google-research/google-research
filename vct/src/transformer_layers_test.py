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

"""Tests for transformer_layers."""

import tensorflow as tf

from vct.src import transformer_layers


class TransformerLayersTest(tf.test.TestCase):

  def test_make_transformer_block_encoder(self):
    for training in (True, False):
      with self.subTest(f"training={training}"):
        block = transformer_layers.TransformerBlock(
            d_model=8, seq_len=4, style="encoder")
        self.assertIsNone(block.look_ahead_mask)
        inp = tf.ones((16, 4, 8))
        otp = block(inp, enc_output=None, training=training)
        self.assertEqual(otp.shape, inp.shape)

  def test_make_transformer_block_decoder(self):
    for training in (True, False):
      with self.subTest(f"training={training}"):
        block = transformer_layers.TransformerBlock(
            d_model=8, seq_len=4, style="decoder")
        expected_look_ahead = tf.constant([
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        self.assertAllClose(block.look_ahead_mask, expected_look_ahead)
        inp = tf.ones((16, 4, 8))
        enc_output = tf.ones((16, 17, 8))
        otp = block(inp, enc_output=enc_output, training=training)
        self.assertEqual(otp.shape, inp.shape)

        with self.assertRaisesRegex(ValueError, "Need `enc_output`.+"):
          _ = block(inp, enc_output=None, training=training)

  def test_transformer(self):
    t = transformer_layers.Transformer(is_decoder=True,
                                       num_layers=2,
                                       d_model=8,
                                       num_head=2,
                                       seq_len=16)
    enc_output = tf.ones((3, 16, 8))
    otp = t(tf.ones((3, 16, 8)), enc_output=enc_output, training=True)
    self.assertEqual(otp.shape, (3, 16, 8))


if __name__ == "__main__":
  tf.test.main()
