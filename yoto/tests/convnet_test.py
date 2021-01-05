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

"""Tests for the conditional multi-layer perceptron in `mlp.py`."""

from absl.testing import parameterized
import tensorflow.compat.v1 as tf

from yoto.architectures import convnet


class ConvnetTest(parameterized.TestCase,
                  tf.test.TestCase):

  @parameterized.parameters(
      dict(num_blocks=3, layers_per_block=2, base_num_channels=16,
           upconv=False, conditioning_layer_sizes=None),
      dict(num_blocks=3, layers_per_block=2, base_num_channels=16,
           upconv=False, conditioning_layer_sizes=[128]),
      dict(num_blocks=3, layers_per_block=2, base_num_channels=16,
           upconv=True, conditioning_layer_sizes=[128]),)
  def test_sizes(self,
                 num_blocks,
                 layers_per_block,
                 base_num_channels,
                 upconv,
                 conditioning_layer_sizes,
                 input_shape=(12, 32, 32, 3),
                 conditioning_size=3,
                 channels_out=5):
    model = convnet.ConditionalConvnet(
        num_blocks=num_blocks, layers_per_block=layers_per_block,
        base_num_channels=base_num_channels, upconv=upconv,
        conditioning_layer_sizes=conditioning_layer_sizes,
        channels_out=channels_out)
    input_data = tf.ones(input_shape)
    batch_size = input_shape[0]
    if conditioning_layer_sizes:
      conditioning_data = tf.ones((batch_size, conditioning_size))
    else:
      conditioning_data = None

    output, endpoints = model(input_data, conditioning_data)
    for key, endpoint in endpoints.items():
      nblock = int(key.split("_")[-1])
      if upconv:
        output_spatial_size = (input_shape[1] * (2 ** num_blocks),
                               input_shape[2] * (2 ** num_blocks))
        expected_shape = [input_shape[0],
                          int(output_spatial_size[0] / (2 ** nblock)),
                          int(output_spatial_size[1] / (2 ** nblock)),
                          int(base_num_channels * 2 ** nblock),]
      else:
        expected_shape = [input_shape[0],
                          int(input_shape[1] / (2 ** (nblock + 1))),
                          int(input_shape[1] / (2 ** (nblock + 1))),
                          int(base_num_channels * 2 ** nblock),]

      print(key, endpoint.shape.as_list(), expected_shape)
      self.assertEqual(endpoint.shape.as_list(), expected_shape)
    if upconv:
      expected_shape = [input_shape[0],
                        int(input_shape[1] * (2 ** num_blocks)),
                        int(input_shape[2] * (2 ** num_blocks)),
                        channels_out,]
      print("output shape", output.shape.as_list(), expected_shape)
      self.assertEqual(output.shape.as_list(), expected_shape)

if __name__ == "__main__":
  tf.test.main()
