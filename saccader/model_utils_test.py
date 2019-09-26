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

"""Tests for model_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from saccader import model_utils


def _construct_images(batch_size):
  image_shape = (50, 50, 3)
  images = tf.convert_to_tensor(
      np.random.randn(*((batch_size,) + image_shape)), dtype=tf.float32)
  return images


def _construct_locations_list(batch_size, num_times):
  locations_list = [
      tf.convert_to_tensor(
          np.random.rand(batch_size, 2) * 2 - 1, dtype=tf.float32)
      for _ in range(num_times)
  ]
  return locations_list


def _count_parameters(vars_list):
  count = 0
  for v in vars_list:
    count += np.prod(v.get_shape().as_list())
  return count


class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("training_mode", True),
      ("inference_mode", False),
  )
  def test_build_conv_pool_layers(self, is_training):
    batch_size = 5
    filter_sizes_conv_layers = [(5, 16), (3, 32), (1, 64)]
    pool_params = {"type": "max", "stride": 2, "size": 2}
    images = _construct_images(batch_size)
    image_shape = tuple(images.get_shape().as_list()[1:])

    number_params = 0
    channels = image_shape[2]
    output_size = image_shape[0]
    for filter_size in filter_sizes_conv_layers:
      # Add filter parameters count.
      number_params += filter_size[0]**2 * channels * filter_size[1]
      output_size //= pool_params["size"]
      channels = filter_size[1]
      # Add batch norm mean and variance parameters count.
      number_params += channels * 4

    final_shape = (batch_size, output_size, output_size, channels)
    net, _ = model_utils.build_conv_pool_layers(
        images,
        filter_sizes_conv_layers=filter_sizes_conv_layers,
        activation=tf.nn.relu,
        pool_params=pool_params,
        batch_norm=True,
        regularizer=tf.nn.l2_loss,
        is_training=is_training)
    vars_list = tf.global_variables()

    self.assertEqual(_count_parameters(vars_list), number_params)

    init = tf.global_variables_initializer()
    self.evaluate(init)
    self.assertEqual(final_shape, self.evaluate(net).shape)

  @parameterized.named_parameters(
      ("training_mode", True),
      ("inference_mode", False),
  )
  def test_build_fc_layers(self, is_training):
    batch_size = 5
    num_units_fc_layers = [128, 64, 32]
    images = _construct_images(batch_size)
    image_shape = tuple(images.get_shape().as_list()[1:])

    net_input = tf.layers.flatten(images)

    number_params = 0
    num_units_prev = np.prod(image_shape)
    for num_units in num_units_fc_layers:
      filter_size = (num_units_prev, num_units)
      # Add filter parameters count.
      number_params += np.prod(filter_size)
      output_size = num_units
      num_units_prev = num_units
      # Add batch norm mean and variance parameters count.
      number_params += num_units * 4

    final_shape = (batch_size, output_size)
    net, _ = model_utils.build_fc_layers(
        net_input,
        num_units_fc_layers=num_units_fc_layers,
        activation=tf.nn.relu,
        batch_norm=True,
        regularizer=tf.nn.l2_loss,
        is_training=is_training)

    vars_list = tf.global_variables()

    self.assertEqual(_count_parameters(vars_list), number_params)

    init = tf.global_variables_initializer()
    self.evaluate(init)
    self.assertEqual(final_shape, self.evaluate(net).shape)

  @parameterized.named_parameters(
      ("training_mode", True),
      ("inference_mode", False),
  )
  def test_residual_layer(self, is_training):
    batch_size = 5
    conv_channels = 16
    conv_size = 3

    images = _construct_images(batch_size)
    image_shape = images.shape.as_list()[1:]
    net = model_utils.residual_layer(
        images,
        conv_size=conv_size,
        conv_channels=conv_channels,
        stride=1,
        dropout_rate=0,
        regularizer=tf.nn.l2_loss,
        activation=tf.nn.relu,
        normalization_type="batch",
        is_training=is_training)

    number_params = (
        3 * conv_channels + conv_size**2 * 3 * conv_channels + conv_channels * 4
        + conv_size**2 * conv_channels**2 + conv_channels * 4)

    vars_list = tf.global_variables()
    self.assertEqual(_count_parameters(vars_list), number_params)
    final_shape = (batch_size,) + tuple(image_shape[:-1]) + (conv_channels,)
    init = tf.global_variables_initializer()
    self.evaluate(init)
    self.assertEqual(final_shape, self.evaluate(net).shape)

  @parameterized.named_parameters(
      ("training_mode_zero_pad", True, True),
      ("inference_mode_zero_pad", False, True),
      ("training_mode_conv_pad", True, False),
      ("inference_mode_conv_pad", False, False),
  )
  def test_build_wide_residual_network(self, is_training, zero_pad):
    batch_size = 5
    num_classes = 10
    residual_blocks_per_group = 6
    number_groups = 3
    init_conv_channels = 16
    widening_factor = 2
    expand_rate = 2
    conv_size = 3

    images = _construct_images(batch_size)

    logits, endpoints = model_utils.build_wide_residual_network(
        images,
        num_classes,
        residual_blocks_per_group=residual_blocks_per_group,
        number_groups=number_groups,
        conv_size=conv_size,
        init_conv_channels=init_conv_channels,
        widening_factor=widening_factor,
        expand_rate=expand_rate,
        dropout_rate=0,
        regularizer=tf.nn.l2_loss,
        activation=tf.nn.relu,
        normalization_type="batch",
        is_training=is_training,
        zero_pad=zero_pad)

    # global average
    pre_out = endpoints["global_average_pool"]
    pre_out_shape = (batch_size,
                     init_conv_channels * widening_factor * expand_rate**
                     (number_groups - 1))
    init = tf.global_variables_initializer()
    self.evaluate(init)
    self.assertEqual(pre_out_shape, self.evaluate(pre_out).shape)
    self.assertEqual((batch_size, num_classes), self.evaluate(logits).shape)

  @parameterized.named_parameters(
      ("training_mode_zero_pad", True, True),
      ("inference_mode_zero_pad", False, True),
      ("training_mode_conv_pad", True, False),
      ("inference_mode_conv_pad", False, False),
  )
  def test_build_recurrent_wide_residual_network(self, is_training, zero_pad):
    batch_size = 5
    num_classes = 10
    num_times = 5
    number_groups = 3
    init_conv_channels = 16
    widening_factor = 2
    expand_rate = 2
    conv_size = 3

    images = _construct_images(batch_size)

    logits, endpoints = model_utils.build_recurrent_wide_residual_network(
        images,
        num_classes,
        num_times=num_times,
        number_groups=number_groups,
        conv_size=conv_size,
        init_conv_channels=init_conv_channels,
        widening_factor=widening_factor,
        expand_rate=expand_rate,
        dropout_rate=0,
        regularizer=tf.nn.l2_loss,
        activation=tf.nn.relu,
        normalization_type="batch",
        is_training=is_training,
        zero_pad=zero_pad)

    # global average
    pre_out = endpoints["global_average_pool"]
    pre_out_shape = (batch_size,
                     init_conv_channels * widening_factor * expand_rate**
                     (number_groups - 1))
    init = tf.global_variables_initializer()
    self.evaluate(init)
    self.assertEqual(pre_out_shape, self.evaluate(pre_out).shape)
    self.assertEqual((batch_size, num_classes), self.evaluate(logits).shape)

  @parameterized.named_parameters(
      ("training", True),
      ("inference", False),
  )
  def test_generator(self, is_training):
    batch_size = 5
    latent_dim = 256
    z = tf.constant(np.random.randn(batch_size, latent_dim), dtype=tf.float32)
    images = model_utils.cifar10_generator(z, is_training)
    init = tf.global_variables_initializer()
    self.evaluate(init)
    images = self.evaluate(images)
    self.assertEqual(images.shape, (batch_size, 32, 32, 3))


if __name__ == "__main__":
  tf.test.main()
