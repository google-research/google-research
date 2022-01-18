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

"""Tests the `u_net` module."""
import tensorflow as tf

from flare_removal.python import u_net


class UNetTest(tf.test.TestCase):

  def test_zero_scale(self):
    model = u_net.get_model(
        input_shape=(128, 128, 1), scales=0, bottleneck_depth=32)
    model.summary()

    input_layer = model.get_layer('input')
    bottleneck_conv1 = model.get_layer('bottleneck_conv1')
    bottleneck_conv2 = model.get_layer('bottleneck_conv2')
    output_layer = model.get_layer('output')
    self.assertIs(input_layer.output, bottleneck_conv1.input)
    self.assertIs(bottleneck_conv1.output, bottleneck_conv2.input)
    self.assertIs(bottleneck_conv2.output, output_layer.input)
    self.assertAllEqual(model.input_shape, [None, 128, 128, 1])
    self.assertAllEqual(bottleneck_conv1.output_shape, [None, 128, 128, 32])
    self.assertAllEqual(bottleneck_conv2.output_shape, [None, 128, 128, 32])
    self.assertAllEqual(model.output_shape, [None, 128, 128, 1])

  def test_one_scale(self):
    model = u_net.get_model(
        input_shape=(64, 64, 3), scales=1, bottleneck_depth=128)
    model.summary()

    # Downscaling arm.
    input_layer = model.get_layer('input')
    down_conv1 = model.get_layer('down64_conv1')
    down_conv2 = model.get_layer('down64_conv2')
    down_pool = model.get_layer('down64_pool')
    bottleneck_conv1 = model.get_layer('bottleneck_conv1')
    self.assertIs(input_layer.output, down_conv1.input)
    self.assertIs(down_conv1.output, down_conv2.input)
    self.assertIs(down_conv2.output, down_pool.input)
    self.assertIs(down_pool.output, bottleneck_conv1.input)
    self.assertAllEqual(model.input_shape, [None, 64, 64, 3])
    self.assertAllEqual(down_conv1.output_shape, [None, 64, 64, 64])
    self.assertAllEqual(down_conv2.output_shape, [None, 64, 64, 64])
    self.assertAllEqual(down_pool.output_shape, [None, 32, 32, 64])
    self.assertAllEqual(bottleneck_conv1.output_shape, [None, 32, 32, 128])

    # Upscaling arm.
    bottleneck_conv2 = model.get_layer('bottleneck_conv2')
    up_2x = model.get_layer('up64_2x')
    up_2xconv = model.get_layer('up64_2xconv')
    up_concat = model.get_layer('up64_concat')
    up_conv1 = model.get_layer('up64_conv1')
    up_conv2 = model.get_layer('up64_conv2')
    output_layer = model.get_layer('output')
    self.assertIs(bottleneck_conv2.output, up_2x.input)
    self.assertIs(up_2x.output, up_2xconv.input)
    self.assertIs(up_2xconv.output, up_concat.input[0])
    self.assertIs(up_concat.output, up_conv1.input)
    self.assertIs(up_conv1.output, up_conv2.input)
    self.assertIs(up_conv2.output, output_layer.input)
    self.assertAllEqual(bottleneck_conv2.output_shape, [None, 32, 32, 128])
    self.assertAllEqual(up_2x.output_shape, [None, 64, 64, 128])
    self.assertAllEqual(up_2xconv.output_shape, [None, 64, 64, 64])
    self.assertAllEqual(up_concat.output_shape, [None, 64, 64, 128])
    self.assertAllEqual(up_conv1.output_shape, [None, 64, 64, 64])
    self.assertAllEqual(up_conv2.output_shape, [None, 64, 64, 64])
    self.assertAllEqual(output_layer.output_shape, [None, 64, 64, 3])

    # Skip connection.
    self.assertIs(down_conv2.output, up_concat.input[1])


if __name__ == '__main__':
  tf.test.main()
