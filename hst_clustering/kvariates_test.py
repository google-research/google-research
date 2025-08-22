# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Tests for kvariates."""

import logging

from absl.testing import absltest
import numpy as np

from hst_clustering import kvariates


class KvariatesTest(absltest.TestCase):

  def test_hash(self):
    x = np.array([1, 2, 1])
    proj_matrix = np.array([[1, 1, 1], [1, -1, 0], [0, 1, -1]])
    expected_hash = int(2**0 + 2**2)
    self.assertEqual(expected_hash, kvariates._simhash_split(x, proj_matrix))

  def test_simhash_random(self):
    num_keys = 8
    d = 10
    data = np.random.normal(0, 1, (1000, d))
    logging.info(str(data))
    hashes = []
    for i in range(1000):
      hashes.append(kvariates.simhash_split(data[i,:], num_keys))

    for i, h in enumerate(hashes):
      self.assertLess(h[0], num_keys)
      self.assertEqual(np.sum(np.equal(h[1], data[i,:])), d)

  def test_calculate_collection_distance_to_centers(self):
    collection = np.array([[1., 2., 4.], [0.2, 0.3, 0.5]])
    c1 = np.array([0.1, 1., 2.])
    c2 = np.array([1., 2., 3.])
    d11 = np.sqrt((1 - 0.1)**2 + (2 - 1)**2 + (4 - 2)**2)
    d12 = np.sqrt((1 - 1.)**2 + (2 - 2)**2 + (4 - 3)**2)

    d21 = np.sqrt((0.2 - 0.1)**2 + (0.3 - 1)**2 + (0.5 - 2)**2)
    d22 = np.sqrt((0.2 - 1)**2 + (0.3 - 2)**2 + (0.5 - 3)**2)
    print([d11, d12, d21, d22])
    centers = np.concatenate([c1.reshape(1, -1), c2.reshape(1, -1)], axis=0)
    logging.info(str(centers))
    logging.info("Testing calculate distance to centers")
    logging.info(
        kvariates.calculate_collection_distance_to_centers(
            (0, collection), centers
        )
    )
    output = kvariates.calculate_collection_distance_to_centers(
        (0, collection), centers
    )
    self.assertEqual(0, output[0])
    self.assertAlmostEqual(
        np.minimum(d11, d12) + np.minimum(d21, d22), output[1])

  def test_calculate_normalizer(self):
    distances = [(0, 10.), (1, 20.), (3, 25.)]
    self.assertEqual(kvariates.calculate_normalizer((0, distances)), 55.)

  def test_get_distance_to_centers(self):
    centers = np.array([[1., 2., 4.], [0.2, 0.3, 0.5]])
    array = np.array([0.1, 1., 2.])
    d1 = np.sqrt((1 - 0.1)**2 + (2 - 1)**2 + (4 - 2)**2)
    d2 = np.sqrt((0.2 - 0.1)**2 + (0.3 - 1)**2 + (0.5 - 2)**2)
    self.assertEqual(
        kvariates.get_distance_to_centers(array, centers), np.minimum(d1, d2))


if __name__ == "__main__":
  absltest.main()
