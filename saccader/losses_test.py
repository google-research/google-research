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

"""Tests for losses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from saccader import losses
from saccader.visual_attention import saccader
from saccader.visual_attention import saccader_config


def _construct_images(batch_size):
  image_shape = (50, 50, 3)
  images = tf.convert_to_tensor(
      np.random.randn(*((batch_size,) + image_shape)), dtype=tf.float32)
  return images


class LossesTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("case0", True),
      ("case1", False),
  )
  def test_reinforce_loss_continuous(self, use_punishment):
    num_times = 4
    num_classes = 10
    batch_size = 5
    sampling_stddev = 0.05
    mean_locations_t = [
        tf.convert_to_tensor(
            2 * np.random.rand(batch_size, 2) - 1, dtype=tf.float32)
        for _ in range(num_times)
    ]
    locations_t = [
        l + tf.convert_to_tensor(
            sampling_stddev * np.random.randn(batch_size, 2), dtype=tf.float32)
        for l in mean_locations_t
    ]

    logits_t = [
        tf.convert_to_tensor(
            np.random.randn(batch_size, num_classes), dtype=tf.float32)
        for _ in range(num_times)
    ]
    labels_t = [tf.convert_to_tensor(
        np.random.randint(low=0, high=num_classes, size=batch_size),
        dtype=tf.int32)] * num_times

    reinforce_loss = losses.reinforce_loss_continuous(
        logits_t,
        labels_t,
        locations_t,
        mean_locations_t,
        sampling_stddev,
        use_punishment=use_punishment)
    self.evaluate(reinforce_loss)

  @parameterized.named_parameters(
      ("case0", True),
      ("case1", False),
  )
  def test_reinforce_loss_discrete(self, use_punishment):
    num_times = 4
    num_classes = 10
    num_locations = 100
    batch_size = 5
    logits_t = [
        tf.convert_to_tensor(
            np.random.randn(batch_size, num_classes), dtype=tf.float32)
        for _ in range(num_times)
    ]
    labels_t = [tf.convert_to_tensor(
        np.random.randint(low=0, high=num_classes, size=batch_size),
        dtype=tf.int32)] * num_times

    loc_logits_t = [
        tf.convert_to_tensor(
            np.random.randn(batch_size, num_locations), dtype=tf.float32)
        for _ in range(num_times)
    ]
    loc_labels = [tf.convert_to_tensor(
        np.random.randint(low=0, high=num_locations, size=batch_size),
        dtype=tf.int32)] * num_times

    reinforce_loss = losses.reinforce_loss_discrete(
        logits_t,
        labels_t,
        loc_logits_t,
        loc_labels,
        use_punishment=use_punishment)
    self.evaluate(reinforce_loss)

  @parameterized.named_parameters(
      ("case0", "l1"),
      ("case1", "l2"),
  )
  def test_reconstruction_losses(self, norm):
    num_times = 4
    batch_size = 5
    sampling_stddev = 0.05
    images = _construct_images(batch_size)
    reconstructed_images = images + tf.random_uniform(
        images.shape.as_list(), minval=-1, maxval=1)
    mean_locations_t = [
        tf.convert_to_tensor(
            2 * np.random.rand(batch_size, 2) - 1, dtype=tf.float32)
        for _ in range(num_times)
    ]
    locations_t = [
        l + tf.convert_to_tensor(
            sampling_stddev * np.random.randn(batch_size, 2), dtype=tf.float32)
        for l in mean_locations_t
    ]

    loss, reinforce_loss = losses.reconstruction_losses(
        images,
        reconstructed_images,
        locations_t,
        mean_locations_t,
        sampling_stddev,
        norm=norm)
    total_loss = loss + reinforce_loss
    total_loss = self.evaluate(total_loss)
    self.assertDTypeEqual(total_loss, "float32")

  def test_saccader_pretrain_loss(self):
    batch_size = 2
    num_classes = 1001
    images = tf.random_uniform(
        shape=(batch_size, 224, 224, 3), minval=-1, maxval=1, dtype=tf.float32)
    config = saccader_config.get_config()
    config.num_classes = num_classes

    model = saccader.Saccader(config)

    model(
        images,
        num_times=6,
        is_training=True,
        policy="learned")

    num_params = len(tf.all_variables())

    pretrain_loss = losses.saccader_pretraining_loss(
        model, images, is_training=True)

    # Test loss does not introduce new variables.
    self.assertLen(tf.all_variables(), num_params)

    # Gradients with respect to location variables should exist.
    for v, g in zip(model.var_list_location, tf.gradients(
        pretrain_loss, model.var_list_location)):
      if v.trainable:
        self.assertIsNotNone(g)

    # Gradients with respect to classification variables should be None.
    for g in tf.gradients(
        pretrain_loss, model.var_list_classification):
      self.assertIsNone(g)

    # Test evaluating the loss
    self.evaluate(tf.global_variables_initializer())
    self.evaluate(pretrain_loss)


if __name__ == "__main__":
  tf.test.main()
