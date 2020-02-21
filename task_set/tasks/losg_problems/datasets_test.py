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

"""Tests for learning.brain.models.learned_optimizer.problems.datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from task_set.tasks.losg_problems import datasets
import tensorflow.compat.v1 as tf


class DatasetsTest(tf.test.TestCase):

  def testBatchIndices(self):
    ds = datasets.Dataset(data=xrange(50), labels=np.random.randint(2, size=50))
    self.assertEqual(50, ds.size)

    batch_indices = ds.batch_indices(100, 20)
    self.assertLen(batch_indices, 100)
    self.assertLen(batch_indices[0], 20)
    for first_batch, second_batch in zip(batch_indices[0], batch_indices[1]):
      self.assertNotEqual(first_batch, second_batch)

  def testBatchIndices_BadData(self):
    ds = datasets.Dataset(data=xrange(10), labels=np.random.randint(2, size=50))
    with self.assertRaises(ValueError):
      ds.batch_indices(5, 10)

  def testNoisyParityClass(self):
    n_samples = 4
    n_classes = 3
    n_context_ids = 5
    ds = datasets.noisy_parity_class(
        n_samples,
        n_classes=n_classes,
        n_context_ids=n_context_ids,
        noise_prob=0.25,
        random_seed=200)
    self.assertEqual(n_samples, ds.size)
    self.assertLen(ds.labels, n_samples)
    self.assertLen(ds.data[0], n_context_ids)
    output = '{}'.format(ds.labels)
    output += ' {}'.format([type(d) for d in ds.labels])
    self.assertTrue(all([d >= 0 and d < n_classes for d in ds.data[0]]))
    self.assertTrue(all([d >= 0 and d < n_classes and isinstance(d, np.int32)
                         for d in ds.labels]), msg=output)

  def testRandom(self):
    n_features = 3
    n_samples = 4
    n_classes = 3
    ds = datasets.random(
        n_features, n_samples, n_classes=n_classes, sep=1., random_seed=200)
    self.assertEqual(n_samples, ds.size)
    self.assertLen(ds.labels, n_samples)
    self.assertLen(ds.data[0], n_features)
    self.assertTrue(all([d >= 0 and d < n_classes and isinstance(d, np.int32)
                         for d in ds.labels]))

  def testRandomBinary(self):
    n_samples = 5
    n_features = 4
    ds = datasets.random_binary(n_features, n_samples, random_seed=200)

    self.assertEqual(n_samples, ds.size)
    self.assertEqual((n_samples, 1), ds.labels.shape)
    self.assertLen(ds.data[0], n_features)
    self.assertTrue(all([x[0] in [0, 1] for x in ds.labels]))  # binary

  def testRandomSymmetric(self):
    n_samples = 4
    n_features = 6
    ds = datasets.random_symmetric(n_features, n_samples, random_seed=200)

    self.assertEqual(n_samples, ds.size)
    self.assertEqual((n_samples, 1), ds.labels.shape)
    self.assertLen(ds.data[0], n_features)
    self.assertTrue(all([x[0] in [0, 1] for x in ds.labels]))  # binary

    # Check the symmetry.
    for i in range(n_samples // 2):
      self.assertListEqual(
          list(ds.data[i]), list(-1 * ds.data[n_samples // 2 + i]))

  def testRandomMlp(self):
    n_samples = 4
    n_features = 3
    n_layers = 2
    width = 10

    ds = datasets.random_mlp(
        n_features, n_samples, n_layers=n_layers, width=width, random_seed=200)
    self.assertEqual(n_samples, ds.size)
    self.assertLen(ds.labels, n_samples)
    self.assertLen(ds.data[0], n_features)
    self.assertTrue(all([x in [0, 1] for x in ds.labels]))  # binary


if __name__ == '__main__':
  tf.test.main()
