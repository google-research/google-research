# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Tests for transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from summae import transformer as trf


class TransformerTest(parameterized.TestCase):

  def test_transformer_encoder_decoder(self):
    hidden_size = 128
    filter_size = 128
    num_encoder_layers = 2
    num_decoder_layers = 2
    num_encoder_heads = 2
    num_decoder_heads = 2
    attention_dropout = 0.1
    relu_dropout = 0.1
    postprocess_dropout = 0.1
    transformer_encoder_decoder = trf.TransformerEncoderDecoder(
        hidden_size, filter_size, num_encoder_layers, num_decoder_layers,
        num_encoder_heads, num_decoder_heads, attention_dropout, relu_dropout,
        postprocess_dropout)
    output = transformer_encoder_decoder(
        tf.ones((2, 10, 128)), tf.ones((2, 3, 128)), tf.zeros((2, 10)), False)
    self.assertEqual(output.shape, [2, 3, 128])

  def test_transformer_encoder(self):
    hidden_size = 128
    filter_size = 128
    num_encoder_layers = 3
    num_encoder_heads = 4
    attention_dropout = 0.1
    relu_dropout = 0.1
    postprocess_dropout = 0.1
    transformer_encoder = trf.TransformerEncoder(
        hidden_size, filter_size, num_encoder_layers, num_encoder_heads,
        attention_dropout, relu_dropout, postprocess_dropout)
    output = transformer_encoder(tf.ones((2, 10, 128)), tf.ones((2, 10)), False)
    self.assertEqual(output.shape, [2, 10, 128])

  def test_transformer_decoder(self):
    hidden_size = 128
    filter_size = 128
    num_decoder_layers = 2
    num_decoder_heads = 2
    attention_dropout = 0.1
    relu_dropout = 0.1
    postprocess_dropout = 0.1
    transformer_decoder = trf.TransformerDecoder(
        hidden_size, filter_size, num_decoder_layers, num_decoder_heads,
        attention_dropout, relu_dropout, postprocess_dropout)
    output = transformer_decoder(
        tf.ones((2, 10, 128)), tf.zeros((2, 10)), tf.ones((2, 5, 128)), False)
    self.assertEqual(output.shape, [2, 5, 128])

  def test_transformer_decoder_only(self):
    hidden_size = 128
    filter_size = 128
    num_decoder_layers = 3
    num_decoder_heads = 4
    attention_dropout = 0.1
    relu_dropout = 0.1
    postprocess_dropout = 0.1
    transformer_decoder_only = trf.TransformerDecoderOnly(
        hidden_size, filter_size, num_decoder_layers, num_decoder_heads,
        attention_dropout, relu_dropout, postprocess_dropout)
    output = transformer_decoder_only(tf.ones((2, 10, 128)), False)
    self.assertEqual(output.shape, [2, 10, 128])


if __name__ == '__main__':
  tf.enable_eager_execution()
  absltest.main()
