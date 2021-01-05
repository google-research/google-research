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

"""Tests for mnist_model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from large_margin.mnist import mnist_config
from large_margin.mnist import mnist_model
tf.disable_v2_behavior()


def _construct_images(batch_size):
  image_shape = (28, 28, 1)
  images = tf.convert_to_tensor(
      np.random.randn(*((batch_size,) + image_shape)), dtype=tf.float32)
  return images


class MNISTModelTest(tf.test.TestCase, parameterized.TestCase):

  def test_import(self):
    self.assertIsNotNone(mnist_model)

  @parameterized.named_parameters(
      ("training_mode", True),
      ("inference_mode", False),
  )
  def test_build_model(self, is_training):
    batch_size = 5
    num_classes = 10
    images = _construct_images(batch_size)
    config = mnist_config.ConfigDict()
    config.num_classes = num_classes

    model = mnist_model.MNISTNetwork(config)

    logits, _ = model(images, is_training)

    final_shape = (batch_size, num_classes)
    init = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init)
      self.assertEqual(final_shape, sess.run(logits).shape)

  @parameterized.named_parameters(
      ("training_mode", True),
      ("inference_mode", False),
  )
  def test_reuse_model(self, is_training):
    batch_size = 5
    images = _construct_images(batch_size)
    config = mnist_config.ConfigDict()

    model = mnist_model.MNISTNetwork(config)

    # Build once.
    logits1, _ = model(images, is_training)
    num_params = len(tf.all_variables())
    l2_loss1 = tf.losses.get_regularization_loss()
    # Build twice.
    logits2, _ = model(images, is_training)
    l2_loss2 = tf.losses.get_regularization_loss()

    # Ensure variables are reused.
    self.assertLen(tf.all_variables(), num_params)
    init = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init)
      # Ensure operations are the same after reuse.
      err_logits = (np.abs(sess.run(logits1 - logits2))).sum()
      self.assertAlmostEqual(err_logits, 0, 9)
      err_losses = (np.abs(sess.run(l2_loss1 - l2_loss2))).sum()
      self.assertAlmostEqual(err_losses, 0, 9)


if __name__ == "__main__":
  tf.test.main()
