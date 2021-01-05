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
"""Tests for soft sorting tensorflow layers."""

import tensorflow.compat.v2 as tf
from soft_sort.matrix_factorization import data


class DataTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    tf.random.set_seed(0)
    self._data = data.SyntheticData(
        num_features=50, num_individuals=200, low_rank=10)

  def test_make(self):
    matrix = self._data.make()
    self.assertEqual(matrix.shape, (50, 200))


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()

