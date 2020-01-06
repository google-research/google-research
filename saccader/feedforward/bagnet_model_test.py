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

"""Tests for bagnet_model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from saccader.feedforward import bagnet_config
from saccader.feedforward import bagnet_model


def _construct_images(batch_size):
  image_shape = (224, 224, 3)
  images = tf.convert_to_tensor(
      np.random.randn(batch_size, *image_shape), dtype=tf.float32)
  return images


class BAGNETModelTest(tf.test.TestCase, parameterized.TestCase):

  def test_import(self):
    self.assertIsNotNone(bagnet_model)

  @parameterized.parameters(
      itertools.product(
          [True, False],  # avg_pool
          [True, False],  # is_training
          )
      )
  def test_build_model(self, avg_pool, is_training):
    batch_size = 2
    num_classes = 10
    images = _construct_images(batch_size)
    config = bagnet_config.get_config()
    config.num_classes = num_classes
    config.avg_pool = avg_pool
    model = bagnet_model.BagNet(config)
    logits, _ = model(images, is_training)

    final_shape = (batch_size, num_classes)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init)
      self.assertEqual(final_shape, sess.run(logits).shape)

  @parameterized.parameters(
      itertools.product(
          [True, False],  # avg_pool
          [True, False],  # is_training
          )
      )
  def test_reuse_model(self, avg_pool, is_training):
    batch_size = 2
    num_classes = 10
    images = _construct_images(batch_size)
    config = bagnet_config.get_config()
    config.num_classes = num_classes
    config.avg_pool = avg_pool
    model = bagnet_model.BagNet(config)
    # Build once.
    logits1, _ = model(images, is_training)
    num_params = len(tf.all_variables())
    l2_loss1 = tf.losses.get_regularization_loss()
    # Build twice.
    logits2, _ = model(
        images, is_training)
    l2_loss2 = tf.losses.get_regularization_loss()

    # Ensure variables are reused.
    self.assertLen(tf.all_variables(), num_params)
    init = tf.global_variables_initializer()
    self.evaluate(init)
    logits1, logits2 = self.evaluate((logits1, logits2))
    l2_loss1, l2_loss2 = self.evaluate((l2_loss1, l2_loss2))
    np.testing.assert_almost_equal(logits1, logits2, decimal=9)
    np.testing.assert_almost_equal(l2_loss1, l2_loss2, decimal=9)

  @parameterized.named_parameters(
      ("case0", [0, 0, 0, 0]),
      ("case1", [1, 0, 0, 0]),
      ("case2", [1, 1, 0, 0]),
      ("case3", [1, 1, 1, 0]),
      ("case4", [1, 1, 1, 1]),
      ("case5", [2, 2, 2, 2]),
  )
  def test_receptive_field_size(self, kernel3):
    batch_size = 2
    num_classes = 10
    images = _construct_images(batch_size)
    config = bagnet_config.get_config()
    config.num_classes = num_classes
    config.kernel3 = kernel3
    config.batch_norm.enable = False  # Batch norm false to use random model.
    config.strides = [2, 2, 2, 1]
    config.blocks = [3, 4, 6, 3]
    model = bagnet_model.BagNet(config)
    _, endpoints = model(images, is_training=True)
    logits2d = endpoints["logits2d"]
    init = tf.global_variables_initializer()
    self.evaluate(init)

    # Compute gradient with respect to input of one of the outputs.
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=tf.random.uniform(
            shape=(batch_size,), minval=0, maxval=num_classes, dtype=tf.int32),
        logits=logits2d[:, 0, 0, :])
    grad = self.evaluate(
        tf.gradients(loss, images)[0])
    # Infer the receptive field as the number of non-zero gradients.
    receptive_field_size = 0
    for j in range(batch_size):
      for i in range(images.shape.as_list()[-1]):
        is_nonzero_grad = np.amax(grad[j, :, :, i], 1) > 1e-12
        receptive_field_size = max(np.sum(is_nonzero_grad),
                                   receptive_field_size)

    # Ensure receptive field is correct.
    self.assertEqual(model.receptive_field[0], receptive_field_size)

if __name__ == "__main__":
  tf.test.main()
