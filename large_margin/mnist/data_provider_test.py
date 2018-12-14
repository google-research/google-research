# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""Tests for MNIST data_provider."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import parameterized
import tensorflow as tf

from large_margin.mnist import data_provider

flags.DEFINE_string("data_dir", "", "Data directory.")

FLAGS = flags.FLAGS


class DataProviderTest(tf.test.TestCase, parameterized.TestCase):

  def test_import(self):
    self.assertIsNotNone(data_provider)

  @parameterized.named_parameters(
      ("train_subset", "train"),
      ("test_subset", "test")
      )
  def test_mnist(self, subset):
    if not FLAGS.data_dir:
      tf.logging.info("data_dir flag not provided. Quitting test")
      return
    batch_size = 10
    image_shape = (28, 28, 1)
    dataset = data_provider.MNIST(
        data_dir=FLAGS.data_dir, subset=subset, batch_size=batch_size)
    images, labels = dataset.images, dataset.labels

    with self.test_session() as sess:
      im, l = sess.run((images, labels))
      self.assertEqual(im.shape, (batch_size,) + image_shape)
      self.assertEqual(l.shape, (batch_size,))


if __name__ == "__main__":
  tf.test.main()
