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

"""Tests for lenses."""

from absl.testing import parameterized
import tensorflow as tf

from dedal.models import lenses


class MaskedGlobalMaxPooling1DTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('channels_last', 'channels_last'),
      ('channels_first', 'channels_first'),
  )
  def test_unmasked(self, data_format):
    n, l, c = 16, 8, 4
    inputs = (tf.random.normal([n, l, c]) if data_format == 'channels_last'
              else tf.random.normal([n, c, l]))
    layer = lenses.MaskedGlobalMaxPooling1D(data_format=data_format)
    layer_ref = tf.keras.layers.GlobalMaxPool1D(data_format=data_format)
    self.assertAllClose(layer(inputs), layer_ref(inputs))

  @parameterized.named_parameters(
      ('channels_last', 'channels_last'),
      ('channels_first', 'channels_first'),
  )
  def test_masked(self, data_format):
    n, l, c = 16, 8, 4
    inputs = (tf.random.normal([n, l, c]) if data_format == 'channels_last'
              else tf.random.normal([n, c, l]))
    mask = tf.concat([tf.ones([n, l // 2], tf.bool),
                      tf.zeros([n, l // 2], tf.bool)], 1)
    layer = lenses.MaskedGlobalMaxPooling1D(data_format=data_format)
    layer_ref = tf.keras.layers.GlobalMaxPool1D(data_format=data_format)
    if data_format == 'channels_last':
      self.assertAllClose(layer(inputs, mask=mask),
                          layer_ref(inputs[:, :l // 2]))
    else:
      self.assertAllClose(layer(inputs, mask=mask),
                          layer_ref(inputs[Ellipsis, :l // 2]))


class GlobalAttentionPooling1DTest(tf.test.TestCase):

  def test_shapes(self):
    n, l, c = 16, 8, 4
    inputs = tf.random.normal([n, l, c])
    layer = lenses.GlobalAttentionPooling1D()
    output = layer(inputs)
    self.assertAllEqual(output.shape, [n, c])

  def test_constant_input(self):
    n, l, c = 16, 8, 4
    inputs = tf.random.normal([n, 1, c])
    inputs = tf.tile(inputs, [1, l, 1])
    layer = lenses.GlobalAttentionPooling1D()
    output = layer(inputs)
    self.assertAllClose(output, inputs[:, 0, :])


if __name__ == '__main__':
  tf.test.main()
