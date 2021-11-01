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

"""Tests for contrastive_loss."""

import numpy as np
import tensorflow as tf

from supervised_pixel_contrastive_loss import contrastive_loss


class ContrastiveLossTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.batch_size = 2
    self.height = 4
    self.width = 4
    self.num_pixels = self.width * self.height
    self.channels = 128
    self.num_proj_layers = 3
    self.num_proj_channels = 64

    self.num_classes = 3
    self.num_pixels_per_class = 4
    self.num_ignore_pixels = (
        self.num_pixels - self.num_classes * self.num_pixels_per_class)

    self.ignore_labels = [-2, -1]
    labels = self.ignore_labels * (self.num_ignore_pixels // 2)
    for label in range(1, self.num_classes + 1):
      labels += [label] * self.num_pixels_per_class
    labels = np.sort(labels)
    labels = tf.reshape(tf.constant(labels), shape=(1, self.num_pixels, 1))
    self.labels = tf.concat([labels] * self.batch_size, axis=0)

  def test_generate_same_image_mask(self):
    mask = contrastive_loss.generate_same_image_mask(
        [self.num_pixels, self.num_pixels])

    expected_mask = np.zeros((2 * self.num_pixels, 2 * self.num_pixels))
    expected_mask[0:self.num_pixels, 0:self.num_pixels] = 1
    expected_mask[self.num_pixels:(2 * self.num_pixels),
                  self.num_pixels:(2 * self.num_pixels)] = 1
    expected_mask = tf.expand_dims(tf.constant(expected_mask), axis=0)
    self.assertAllClose(mask, expected_mask)

  def test_generate_ignore_mask(self):
    mask = contrastive_loss.generate_ignore_mask(
        self.labels, self.ignore_labels)
    expected_mask = np.zeros(
        shape=(self.batch_size, self.num_pixels, self.num_pixels))
    expected_mask[:, 0:self.num_ignore_pixels, :] = 1
    expected_mask[:, :, 0:self.num_ignore_pixels] = 1
    self.assertAllClose(expected_mask, mask)

  def test_generate_positive_and_negative_masks(self):
    positive_mask, negative_mask = (
        contrastive_loss.generate_positive_and_negative_masks(self.labels))

    expected_pos_mask = np.zeros(
        (self.batch_size, self.num_pixels, self.num_pixels))
    st = 0
    en = self.num_pixels_per_class // 2
    expected_pos_mask[:, st:en, st:en] = 1
    st += (self.num_pixels_per_class // 2)
    en += (self.num_pixels_per_class // 2)
    expected_pos_mask[:, st:en, st:en] = 1

    st = self.num_pixels_per_class
    en = 2 * self.num_pixels_per_class
    for _ in range(self.num_classes):
      expected_pos_mask[:, st:en, st:en] = 1
      st += self.num_pixels_per_class
      en += self.num_pixels_per_class
    expected_neg_mask = 1 - expected_pos_mask

    self.assertAllClose(positive_mask, expected_pos_mask)
    self.assertAllClose(negative_mask, expected_neg_mask)

  def test_collapse_spatial_dimensions(self):
    input_tensor = tf.random.uniform(
        shape=(self.batch_size, self.height, self.width, self.channels))
    expected_output = tf.reshape(
        input_tensor, shape=(self.batch_size, self.num_pixels, self.channels))

    self.assertAllClose(
        contrastive_loss.collapse_spatial_dimensions(input_tensor),
        expected_output)

  def test_projection_head(self):
    input_tensor = tf.random.uniform(
        shape=(self.batch_size, self.height, self.width, self.channels))

    output_tensor = contrastive_loss.projection_head(
        input_tensor, num_projection_layers=self.num_proj_layers,
        num_projection_channels=self.num_proj_channels)

    self.assertEqual(
        output_tensor.get_shape().as_list(),
        [self.batch_size, self.height, self.width, self.num_proj_channels])

    # Verify that the output is L2 normalized
    self.assertAllClose(
        tf.linalg.norm(output_tensor, axis=-1),
        tf.ones(shape=(self.batch_size, self.height, self.width)))

  def test_resize_and_project(self):
    input_height = 256
    input_width = 128
    input_tensor = tf.random.uniform(
        shape=(self.batch_size, input_height, input_width, self.channels))

    resize_height = input_height // 2
    resize_width = input_width // 2
    output_tensor = contrastive_loss.resize_and_project(
        input_tensor,
        resize_size=(resize_height, resize_width),
        num_projection_layers=self.num_proj_layers,
        num_projection_channels=self.num_proj_channels)

    self.assertEqual(
        output_tensor.get_shape().as_list(),
        [self.batch_size, resize_height, resize_width, self.num_proj_channels])

    # Verify that the output is L2 normalized
    self.assertAllClose(
        tf.linalg.norm(output_tensor, axis=-1),
        tf.ones(shape=(self.batch_size, resize_height, resize_width)))

  def test_contrastive_loss(self):
    logits = tf.cast(
        tf.matmul(self.labels, self.labels, transpose_b=True), dtype=tf.float32)

    positive_mask = tf.cast(
        tf.equal(self.labels, tf.transpose(self.labels, [0, 2, 1])), tf.float32)
    negative_mask = 1 - positive_mask

    ignore_mask = tf.math.reduce_any(
        tf.equal(self.labels,
                 tf.constant(self.ignore_labels, dtype=self.labels.dtype)),
        axis=2, keepdims=True)
    ignore_mask = tf.cast(
        tf.logical_or(ignore_mask, tf.transpose(ignore_mask, [0, 2, 1])),
        tf.float32)

    loss = contrastive_loss.compute_contrastive_loss(
        logits, positive_mask, negative_mask, ignore_mask)
    expected_loss = 2.4503216
    self.assertAllClose(loss, expected_loss)

  def test_within_image_supervised_pixel_contrastive_loss(self):
    loss = contrastive_loss.within_image_supervised_pixel_contrastive_loss(
        features=tf.cast(self.labels, tf.float32),
        labels=self.labels,
        ignore_labels=self.ignore_labels,
        temperature=1.0)
    expected_loss = 2.4503216
    self.assertAllClose(loss, expected_loss)

  def test_cross_image_supervised_pixel_contrastive_loss(self):
    loss = contrastive_loss.cross_image_supervised_pixel_contrastive_loss(
        features1=tf.cast(self.labels, tf.float32),
        features2=tf.cast(2 * self.labels, tf.float32),
        labels1=self.labels,
        labels2=self.labels,
        ignore_labels=self.ignore_labels,
        temperature=1.0)
    expected_loss = 5.46027374
    self.assertAllClose(loss, expected_loss)

  def test_symmetry_of_cross_image_supervised_pixel_contrastive_loss(self):
    features1 = tf.cast(self.labels, tf.float32)
    features2 = tf.cast(2 * self.labels, tf.float32)

    loss1 = contrastive_loss.cross_image_supervised_pixel_contrastive_loss(
        features1=features1,
        features2=features2,
        labels1=self.labels,
        labels2=self.labels,
        ignore_labels=self.ignore_labels,
        temperature=1.0)

    loss2 = contrastive_loss.cross_image_supervised_pixel_contrastive_loss(
        features1=features2,
        features2=features1,
        labels1=self.labels,
        labels2=self.labels,
        ignore_labels=self.ignore_labels,
        temperature=1.0)
    self.assertAllClose(loss1, loss2)


if __name__ == '__main__':
  tf.test.main()
