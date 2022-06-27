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

"""Tests the `losses` module."""
import tensorflow as tf

from flare_removal.python import losses


class PerceptualLossTest(tf.test.TestCase):

  def test_identical_inputs(self):
    loss = losses.PerceptualLoss()
    images = tf.random.uniform((2, 192, 256, 3))
    self.assertAllClose(loss(images, images), 0.0)

  def test_different_inputs(self):
    loss = losses.PerceptualLoss()
    image_1 = tf.zeros((2, 192, 256, 3))
    image_2 = tf.random.uniform((2, 192, 256, 3))
    self.assertAllGreater(loss(image_1, image_2), 1.0)

  def test_similar_vs_different_inputs(self):
    loss = losses.PerceptualLoss()
    pure_bright = tf.ones((3, 256, 256, 3)) * tf.constant([0.9, 0.7, 0.7])
    pure_dark = tf.ones((3, 256, 256, 3)) * tf.constant([0.5, 0.2, 0.2])
    speckles = tf.random.uniform((3, 256, 256, 3))
    self.assertAllGreater(
        loss(pure_bright, speckles), loss(pure_bright, pure_dark))


class CompositeLossTest(tf.test.TestCase):

  def test_l1_with_weight(self):
    composite = losses.CompositeLoss()
    composite.add_loss('l1', 2.0)

    y_true = tf.constant(0.3, shape=(2, 192, 256, 3), dtype=tf.float32)
    y_pred = tf.constant(0.5, shape=(2, 192, 256, 3), dtype=tf.float32)
    self.assertAllClose(
        composite(y_true, y_pred),
        tf.reduce_mean(tf.abs(y_true - y_pred)) * 2.0)

  def test_l1_l2_different_weights(self):
    composite = losses.CompositeLoss()
    composite.add_loss('L1', 1.0)
    composite.add_loss('L2', 0.5)

    y_true = tf.constant(127, shape=(2, 192, 256, 3), dtype=tf.int32)
    y_pred = tf.constant(215, shape=(2, 192, 256, 3), dtype=tf.int32)
    l1 = tf.cast(tf.reduce_mean(tf.abs(y_true - y_pred)), tf.float32)
    l2 = tf.cast(tf.reduce_mean(tf.square(y_true - y_pred)), tf.float32)
    self.assertAllClose(composite(y_true, y_pred), l1 * 1.0 + l2 * 0.5)

  def test_composite_loss_equals_sum_of_components(self):
    composite = losses.CompositeLoss()
    mae = tf.keras.losses.MAE
    vgg = losses.PerceptualLoss()
    composite.add_loss(mae, 1.0)
    composite.add_loss(vgg, 2.0)

    y_true = tf.random.uniform((1, 192, 256, 3))
    y_pred = tf.random.uniform((1, 192, 256, 3))
    loss_value = composite(y_true, y_pred)
    mae_loss_value = tf.math.reduce_mean(mae(y_true, y_pred))
    vgg_loss_value = vgg(y_true, y_pred)
    self.assertAllClose(loss_value, mae_loss_value * 1.0 + vgg_loss_value * 2.0)

  def test_duplicate_component_raises_error(self):
    composite = losses.CompositeLoss()
    composite.add_loss('l1', 1.0)
    with self.assertRaisesRegex(ValueError, 'exist'):
      composite.add_loss('l1', 2.0)

  def test_call_before_adding_component_raises_error(self):
    composite = losses.CompositeLoss()
    y_true = tf.random.uniform((1, 192, 256, 3))
    y_pred = tf.random.uniform((1, 192, 256, 3))
    with self.assertRaises(AssertionError):
      composite(y_true, y_pred)

  def test_invalid_weight(self):
    composite = losses.CompositeLoss()
    with self.assertRaisesRegex(ValueError, r'-1\.0'):
      composite.add_loss('l2', -1.0)


if __name__ == '__main__':
  tf.test.main()
