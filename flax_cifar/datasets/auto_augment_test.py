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
"""Tests for flax_cifar.datasets.auto_augment."""

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from flax_cifar.datasets import auto_augment


class AutoAugmentTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('ShearX', 'ShearX'),
      ('ShearY', 'ShearY'),
      ('Cutout', 'Cutout'),
      ('TranslateX', 'TranslateX'),
      ('TranslateY', 'TranslateY'),
      ('Rotate', 'Rotate'),
      ('AutoContrast', 'AutoContrast'),
      ('Invert', 'Invert'),
      ('Equalize', 'Equalize'),
      ('Solarize', 'Solarize'),
      ('Posterize', 'Posterize'),
      ('Contrast', 'Contrast'),
      ('Color', 'Color'),
      ('Brightness', 'Brightness'),
      ('Sharpness', 'Sharpness'))
  def test_image_processing_function(self, name):
    function, min_strength, _ = auto_augment._available_augmentations()[name]
    cifar_image_shape = [32, 32, 3]
    image = tf.zeros(cifar_image_shape, tf.uint8)
    augmented_image = function(image, tf.cast(min_strength, tf.float32))
    self.assertEqual(augmented_image.shape, cifar_image_shape)
    self.assertEqual(augmented_image.dtype, tf.uint8)

  @parameterized.named_parameters(('cifar', 'cifar10'), ('svhn', 'svhn'))
  def test_autoaugment_function(self, dataset_name):
    autoaugment_fn = auto_augment.get_autoaugment_fn(dataset_name)
    image_shape = [32, 32, 3]  # Valid for cifar and svhn.
    image = tf.zeros(image_shape, tf.uint8)
    augmented_image = autoaugment_fn(image)
    self.assertEqual(augmented_image.shape, image_shape)
    self.assertEqual(augmented_image.dtype, tf.uint8)


if __name__ == '__main__':
  absltest.main()
