# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Tests for uflow_augmentation."""

# pylint:disable=g-importing-member
from functools import partial

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from uflow import uflow_augmentation
from uflow import uflow_utils


class UflowAugmentationTest(absltest.TestCase):

  def _check_images_and_flow(self, images, flow):
    # Check that the image2 warped by flow1 into image1 has lower pixelwise
    # error than the unwarped image
    image1, image2 = tf.unstack(images)
    image1 = tf.expand_dims(image1, axis=0)
    image2 = tf.expand_dims(image2, axis=0)
    flow = tf.expand_dims(flow, axis=0)
    mean_unwarped_diff = np.mean(np.abs(image1 - image2))
    warp = uflow_utils.flow_to_warp(flow)
    image2_to_image1 = uflow_utils.resample(image2, warp)
    mean_warped_diff = np.mean(np.abs(image2_to_image1 - image1))
    self.assertLess(mean_warped_diff, mean_unwarped_diff)

  def _create_images_and_flow(self):
    image1 = tf.constant(
        [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]],
        dtype=tf.float32)
    image2 = tf.constant(
        [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 0]],
         [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 0]],
         [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]],
        dtype=tf.float32)
    flow = tf.constant(
        [[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [2, 2], [2, 2], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [2, 2], [2, 2], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
         [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]],
        dtype=tf.float32)
    mask = tf.ones_like(flow[Ellipsis, :1], dtype=tf.float32)
    images = tf.stack([image1, image2])
    return images, flow, mask

  def _run_ten_checks(self, augmentation_function):
    # create test images and flow
    images, flow, mask = self._create_images_and_flow()
    # run several times to have different random parameters
    for _ in range(10):
      augmented_images, augmented_flow, _ = augmentation_function(
          images, flow, mask)
      # perform a simple check based on warping
      self._check_images_and_flow(augmented_images, augmented_flow)

  def test_random_flip_left_right(self):
    self._run_ten_checks(uflow_augmentation.random_flip_left_right)

  def test_random_flip_up_down(self):
    self._run_ten_checks(uflow_augmentation.random_flip_up_down)

  def test_random_scale(self):
    aug_func = partial(
        uflow_augmentation.random_scale, min_scale=0.5, max_scale=2.0)
    self._run_ten_checks(aug_func)

  def test_random_scale_second(self):
    aug_func = partial(
        uflow_augmentation.random_scale_second, min_scale=0.5, max_scale=2.0)
    self._run_ten_checks(aug_func)

  def test_random_rotation(self):
    aug_func = partial(
        uflow_augmentation.random_rotation, max_rotation=15)
    self._run_ten_checks(aug_func)

  def test_random_rotation_second(self):
    aug_func = partial(
        uflow_augmentation.random_rotation_second, max_rotation=15)
    self._run_ten_checks(aug_func)

  def test_random_crop(self):
    aug_func = partial(
        uflow_augmentation.random_crop, relative_offset=15)
    self._run_ten_checks(aug_func)

if __name__ == '__main__':
  absltest.main()
