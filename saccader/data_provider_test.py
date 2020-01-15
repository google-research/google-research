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

"""Tests data provider."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from saccader import data_provider

_IMAGE_SHAPE_DICT = {
    "imagenet331": (331, 331, 3),
    "imagenet224": (224, 224, 3),
}


def _get_test_cases():
  """Provides test cases."""
  is_training = [True, False]
  subset = ["train", "validation",]
  dataset_name = _IMAGE_SHAPE_DICT.keys()
  i = 0
  cases = []
  for d in dataset_name:
    for s in subset:
      for t in is_training:
        cases.append(("{}_{}_{}".format(d, s, t), d, s, t))
        i += 1
  return tuple(cases)


def _imagenet_standardization_zero_value():
  """The value to which 0 is mapped by ImageNet standardization."""
  return (-np.array(data_provider._MEAN_RGB_DICT["imagenet"]) /
          np.array(data_provider._STDDEV_RGB_DICT["imagenet"]))


class DataProviderTest(tf.test.TestCase, parameterized.TestCase):

  def test_import(self):
    self.assertIsNotNone(data_provider.get_data_provider)

  @parameterized.named_parameters(*_get_test_cases())
  def test_dataset(self, dataset_name, subset, is_training):
    batch_size = 1
    image_shape = _IMAGE_SHAPE_DICT[dataset_name]
    dataset = data_provider.get_data_provider(dataset_name)(
        subset=subset,
        batch_size=batch_size,
        data_dir=None,
        is_training=is_training)
    images, labels = dataset.images, dataset.labels
    self.assertEqual(
        images.shape.as_list(), [batch_size,] + list(image_shape))
    self.assertEqual(
        labels.shape.as_list(), [batch_size,])

    im, l = self.evaluate((images, labels))
    self.assertEqual(im.shape, (batch_size,) + image_shape)
    self.assertEqual(l.shape, (batch_size,))

  @parameterized.named_parameters(("3D", 3), ("4D", 4))
  def test_standardize_image(self, num_dims):
    image = tf.zeros((3,) * num_dims)
    result = self.evaluate(data_provider.standardize_image(image, "imagenet"))
    self.assertAllClose(
        result, np.broadcast_to(
            _imagenet_standardization_zero_value().reshape(
                (1,) * (num_dims - 1) + (3,)),
            (3,) * num_dims))

  def test_unstandardized_preprocessing_for_eval(self):
    image = tf.zeros((256, 256, 3))
    result = self.evaluate(
        data_provider.preprocess_imagenet_for_eval(
            image, 224, standardize=False))
    self.assertAllClose(result, np.zeros(result.shape))

  def test_standardized_preprocessing_for_eval(self):
    image = tf.zeros((256, 256, 3))
    result = self.evaluate(
        data_provider.preprocess_imagenet_for_eval(
            image, 224, standardize=True))
    self.assertAllClose(
        result,
        np.broadcast_to(
            _imagenet_standardization_zero_value().reshape((1, 1, 3)),
            (224, 224, 3)))

if __name__ == "__main__":
  tf.test.main()
