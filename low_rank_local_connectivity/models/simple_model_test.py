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

"""Tests for simple_model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from low_rank_local_connectivity.models import simple_model
from low_rank_local_connectivity.models import simple_model_config


def _construct_images(batch_size):
  image_shape = (28, 28, 3)
  images = tf.convert_to_tensor(
      np.random.randn(*((batch_size,) + image_shape)), dtype=tf.float32)
  return images


class SimpleModelTest(tf.test.TestCase, parameterized.TestCase):

  def test_import(self):
    self.assertIsNotNone(simple_model)

  @parameterized.parameters(
      itertools.product([1, 2, 3], [True, False], [
          'conv2d', 'wide_conv2d', 'low_rank_locally_connected2d',
          'locally_connected2d'])
  )
  def test_build_model(self, rank, is_training, special_type):
    batch_size = 5
    num_classes = 10
    num_channels = 3
    images = _construct_images(batch_size)
    config = simple_model_config.get_config()
    config.num_classes = num_classes
    config.num_channels = num_channels
    config.rank = rank
    config.batch_norm = True
    config.kernel_size_list = [3, 3, 3]
    config.num_filters_list = [64, 64, 64]
    config.strides_list = [2, 2, 1]
    config.layer_types = ['conv2d', special_type, 'conv2d']

    model = simple_model.SimpleNetwork(config)

    logits, _ = model(images, is_training)

    final_shape = (batch_size, num_classes)
    init = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init)
      self.assertEqual(final_shape, sess.run(logits).shape)

  @parameterized.parameters(
      itertools.product([1, 2, 3], [True, False], [
          'conv2d', 'wide_conv2d', 'low_rank_locally_connected2d',
          'locally_connected2d'])
  )
  def test_reuse_model(self, rank, is_training, special_type):
    batch_size = 5
    num_classes = 10
    num_channels = 3
    images = _construct_images(batch_size)
    config = simple_model_config.get_config()
    config.num_classes = num_classes
    config.num_channels = num_channels
    config.batch_norm = False
    config.kernel_size_list = [3, 3, 3]
    config.num_filters_list = [64, 64, 64]
    config.strides_list = [2, 2, 1]
    config.layer_types = ['conv2d', special_type, 'conv2d']
    config.rank = rank

    model = simple_model.SimpleNetwork(config)

    # Build once.
    logits1, _ = model(images, is_training)
    num_params = len(tf.all_variables())
    # Build twice.
    logits2, _ = model(images, is_training)
    # Ensure variables are reused.
    self.assertLen(tf.all_variables(), num_params)
    init = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init)
      # Ensure operations are the same after reuse.
      err_logits = (np.abs(sess.run(logits1 - logits2))).sum()
      self.assertAlmostEqual(err_logits, 0, 9)


if __name__ == '__main__':
  tf.test.main()
