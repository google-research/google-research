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

"""Tests data provider."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf

from low_rank_local_connectivity.data_provider import get_data_provider

_IMAGE_SHAPE_DICT = {
    "mnist": (28, 28, 1),
    "cifar10": (32, 32, 3),
    "celeba32": (32, 32, 3),
}


def _get_test_cases():
  """Provides test cases."""
  is_training = [True, False]
  subset = ["train", "valid", "test"]
  dataset_name = _IMAGE_SHAPE_DICT.keys()
  i = 0
  cases = []
  for d in dataset_name:
    for s in subset:
      for t in is_training:
        cases.append(("case_%d" % i, d, s, t))
        i += 1
  return tuple(cases)


class DataProviderTest(tf.test.TestCase, parameterized.TestCase):

  def test_import(self):
    self.assertIsNotNone(get_data_provider)

  @parameterized.named_parameters(*_get_test_cases())
  def test_dataset(self, dataset_name, subset, is_training):
    batch_size = 1
    image_shape = _IMAGE_SHAPE_DICT[dataset_name]
    dataset = get_data_provider(dataset_name)(
        subset=subset,
        batch_size=batch_size,
        is_training=is_training)
    images, labels = dataset.images, dataset.labels

    im, l = self.evaluate((images, labels))
    self.assertEqual(im.shape, (batch_size,) + image_shape)

    self.assertEqual(l.shape, (batch_size,))


if __name__ == "__main__":
  tf.test.main()
