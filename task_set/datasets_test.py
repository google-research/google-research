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

# Lint as: python3
"""Tests for task_set.datasets."""

import numpy as np

from task_set import datasets
import tensorflow.compat.v1 as tf


class DatasetsTest(tf.test.TestCase):

  def test_split_dataset(self):
    slices = tf.constant(np.arange(100))
    dataset = tf.data.Dataset.from_tensor_slices(slices)
    d1, d2, d3 = datasets.split_dataset(dataset, num_splits=3, num_per_split=10)

    d1_example = d1.make_one_shot_iterator().get_next()
    d2_example = d2.make_one_shot_iterator().get_next()
    d3_example = d3.make_one_shot_iterator().get_next()

    with self.test_session() as sess:
      d1s = [sess.run(d1_example) for _ in range(10)]
      d2s = [sess.run(d2_example) for _ in range(10)]
      d3s = [sess.run(d3_example) for _ in range(80)]
      np.testing.assert_equal(d1s, np.arange(10))
      np.testing.assert_equal(d2s, np.arange(10) + 10)
      np.testing.assert_equal(d3s, np.arange(80) + 20)

    with self.assertRaises(tf.errors.OutOfRangeError):
      sess.run(d1_example)

    with self.assertRaises(tf.errors.OutOfRangeError):
      sess.run(d2_example)

    with self.assertRaises(tf.errors.OutOfRangeError):
      sess.run(d3_example)

  def test_food101_32x32(self):
    """Sanity check that food101 loads data."""
    dataset = datasets.tfds_load_dataset("food101_32x32", split="train")
    batch = dataset.make_one_shot_iterator().get_next()
    with self.cached_session() as sess:
      np_batch = sess.run(batch)
    self.assertEqual(tuple(np_batch["image"].shape), (32, 32, 3))


if __name__ == "__main__":
  tf.test.main()
