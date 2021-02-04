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

# Lint as: python3
"""Tests for upsampler."""

from ml_collections import ConfigDict
import tensorflow as tf
from coltran.models import upsampler
from coltran.utils import base_utils


class ColorUpsamplerTest(tf.test.TestCase):

  def get_config(self):
    config = ConfigDict()
    config.hidden_size = 128
    config.ff_size = 256
    config.num_heads = 2
    config.num_encoder_layers = 2
    config.num_symbols = 8
    return config

  def test_bit_upsampler_attention_num_channels_1(self):
    config = self.get_config()
    bit_upsampler = upsampler.ColorUpsampler(config=config)

    inputs = tf.random.uniform(shape=(8, 32, 32, 3), minval=0, maxval=256,
                               dtype=tf.int32)
    inputs_slice = tf.random.uniform(shape=(8, 32, 32, 1), minval=0, maxval=256,
                                     dtype=tf.int32)
    grayscale = tf.image.rgb_to_grayscale(inputs)
    channel_index = tf.random.uniform(
        shape=[8,], minval=0, maxval=3, dtype=tf.int32)

    logits = bit_upsampler(inputs=inputs,
                           inputs_slice=inputs_slice,
                           channel_index=channel_index)[0]
    self.assertEqual(logits.shape, (8, 32, 32, 1, 256))

    inputs = base_utils.convert_bits(inputs, n_bits_in=8, n_bits_out=3)
    output = bit_upsampler.sample(gray_cond=grayscale, bit_cond=inputs)
    self.assertEqual(output['bit_up_argmax'].shape, (8, 32, 32, 3))

  def test_bit_upsampler_attention_num_channels_3(self):
    config = self.get_config()
    bit_upsampler = upsampler.ColorUpsampler(config=config)

    inputs = tf.random.uniform(shape=(8, 32, 32, 3), minval=0, maxval=256,
                               dtype=tf.int32)
    grayscale = tf.image.rgb_to_grayscale(inputs)

    logits = bit_upsampler(inputs=inputs, inputs_slice=inputs)[0]
    self.assertEqual(logits.shape, (8, 32, 32, 3, 256))

    inputs = base_utils.convert_bits(inputs, n_bits_in=8, n_bits_out=3)
    output = bit_upsampler.sample(gray_cond=grayscale, bit_cond=inputs)
    self.assertEqual(output['bit_up_argmax'].shape, (8, 32, 32, 3))

  def test_color_upsampler_attention_num_channels_1(self):
    config = self.get_config()
    spatial_upsampler = upsampler.SpatialUpsampler(config=config)

    inputs = tf.random.uniform(shape=(8, 64, 64, 3), minval=0, maxval=256,
                               dtype=tf.int32)
    inputs_slice = tf.random.uniform(
        shape=(8, 64, 64, 1), minval=0, maxval=256, dtype=tf.int32)
    grayscale = tf.image.rgb_to_grayscale(inputs)
    channel_index = tf.random.uniform(
        shape=[8,], minval=0, maxval=3, dtype=tf.int32)

    logits = spatial_upsampler(
        inputs=inputs, inputs_slice=inputs_slice, channel_index=channel_index)
    logits = logits[0]
    self.assertEqual(logits.shape, (8, 64, 64, 1, 256))

    inputs = base_utils.convert_bits(inputs, n_bits_in=8, n_bits_out=3)
    output = spatial_upsampler.sample(gray_cond=grayscale, inputs=inputs)
    self.assertEqual(output['high_res_argmax'].shape, (8, 64, 64, 3))


if __name__ == '__main__':
  tf.test.main()
