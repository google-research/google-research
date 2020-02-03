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
"""Tests for compression_lib.kmeans_quantize."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

from graph_compression.compression_lib import kmeans_quantize


class KmeansQuantizeTest(absltest.TestCase):

  def test_basic_kmeans_quantization(self):
    matrix = np.random.randn(20, 20)
    [codebook, encoding] = kmeans_quantize.kmeans_quantize_block(
        matrix, levels=7, pruning_factor=0.0, block_size=4, is_padded=True)

    self.assertEqual(codebook.shape, (7, 4))
    self.assertEqual(encoding.shape, (100,))

  def test_kmeans_quantization_gaussian_mixture(self):
    gaussian_0 = np.random.randn(100, 4) + (-5) * np.eye(4)[0, :]
    gaussian_1 = np.random.randn(100, 4)
    gaussian_2 = np.random.randn(100, 4) + 5 * np.eye(4)[0, :]

    matrix = np.concatenate((gaussian_0, gaussian_1, gaussian_2), axis=0)
    matrix = np.reshape(matrix, (-1, 20))

    [codebook, encoding] = kmeans_quantize.kmeans_quantize_block(
        matrix, levels=3, pruning_factor=0.0, block_size=4, is_padded=True)

    self.assertEqual(codebook.shape, (3, 4))
    self.assertEqual(encoding.shape, (300,))

    recovered_matrix = codebook[encoding].reshape(matrix.shape)
    # Assert recovered matrix is close to original matrix.
    self.assertLess(np.linalg.norm((recovered_matrix - matrix).flatten())/300,
                    0.2)


if __name__ == '__main__':
  absltest.main()
