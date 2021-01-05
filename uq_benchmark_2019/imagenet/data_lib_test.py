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

# Lint as: python2, python3
"""Tests for imagenet.data_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import tensorflow.compat.v2 as tf
from uq_benchmark_2019 import image_data_utils
from uq_benchmark_2019.imagenet import data_lib


flags.DEFINE_bool('fake_data', True, 'Bypass tests that rely on real data and '
                  'use dummy random data for the remaining tests.')

tf.enable_v2_behavior()

BATCH_SIZE = 8
BATCHED_IMAGES_SHAPE = (BATCH_SIZE,) + data_lib.IMAGENET_SHAPE


class DataLibTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(['train', 'test', 'valid'])
  def test_fake_data(self, split):
    # config is ignored for fake data
    config = image_data_utils.DataConfig(split)
    dataset = data_lib.build_dataset(config, BATCH_SIZE, fake_data=True)
    image = next(iter(dataset))[0].numpy()
    self.assertEqual(image.shape, BATCHED_IMAGES_SHAPE)
    self.assertAllInRange(image, 0., 1.)
    self.assertTrue((image > 1./255).any())

  @parameterized.parameters(['train', 'test', 'valid'])
  def test_uncorrupted_data(self, split):
    config = image_data_utils.DataConfig(split)
    if not flags.FLAGS.fake_data:
      dataset = data_lib.build_dataset(config, BATCH_SIZE)
      image = next(iter(dataset))[0].numpy()
      self.assertEqual(image.shape, BATCHED_IMAGES_SHAPE)
      self.assertAllInRange(image, 0., 1.)
      self.assertTrue((image > 1./255).any())

  @parameterized.parameters(['train', 'test', 'valid'])
  def test_roll_pixels(self, split):
    config = image_data_utils.DataConfig(split, roll_pixels=5)
    if not flags.FLAGS.fake_data:
      dataset = data_lib.build_dataset(config, BATCH_SIZE)
      image = next(iter(dataset))[0].numpy()
      self.assertEqual(image.shape, BATCHED_IMAGES_SHAPE)
      self.assertAllInRange(image, 0., 1.)
      self.assertTrue((image > 1./255).any())

  @parameterized.parameters(['train', 'test', 'valid'])
  def test_static_imagenet_c(self, split):
    if not flags.FLAGS.fake_data:
      config = image_data_utils.DataConfig(
          split, corruption_static=True, corruption_level=3,
          corruption_type='pixelate')
      if split in ['train', 'valid']:
        with self.assertRaises(ValueError):
          data_lib.build_dataset(config, BATCH_SIZE)
      else:
        dataset = data_lib.build_dataset(config, BATCH_SIZE)
        image = next(iter(dataset))[0].numpy()
        self.assertEqual(image.shape, BATCHED_IMAGES_SHAPE)
        self.assertAllInRange(image, 0., 1.)
        self.assertTrue((image > 1./255).any())

  @parameterized.parameters(['train', 'test', 'valid'])
  def test_array_imagenet_c(self, split):
    if not flags.FLAGS.fake_data:
      config = image_data_utils.DataConfig(
          split, corruption_level=4, corruption_type='glass_blur')
      dataset = data_lib.build_dataset(config, BATCH_SIZE)
      image = next(iter(dataset))[0].numpy()
      self.assertEqual(image.shape, BATCHED_IMAGES_SHAPE)
      self.assertAllInRange(image, 0., 1.)
      self.assertTrue((image > 1./255).any())

  @parameterized.parameters(['train', 'test', 'valid'])
  def test_value_imagenet_c(self, split):
    if not flags.FLAGS.fake_data:
      config = image_data_utils.DataConfig(
          split, corruption_value=.25, corruption_type='brightness')
      dataset = data_lib.build_dataset(config, BATCH_SIZE)
      image = next(iter(dataset))[0].numpy()
      self.assertEqual(image.shape, BATCHED_IMAGES_SHAPE)
      self.assertAllInRange(image, 0., 1.)
      self.assertTrue((image > 1./255).any())

  def test_alt_dataset(self):
    if not flags.FLAGS.fake_data:
      config = image_data_utils.DataConfig('test', alt_dataset_name='celeb_a')
      dataset = data_lib.build_dataset(config, BATCH_SIZE)
      image = next(iter(dataset))[0].numpy()
      self.assertEqual(image.shape, BATCHED_IMAGES_SHAPE)
      self.assertAllInRange(image, 0., 1.)
      self.assertTrue((image > 1./255).any())

if __name__ == '__main__':
  absltest.main()
