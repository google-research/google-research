# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
"""Tests for negative_cache.util."""

import tensorflow.compat.v2 as tf
from negative_cache import util


class LossesTest(tf.test.TestCase):

  def test_approximate_top_k_with_indices(self):
    negative_scores = tf.convert_to_tensor([[1.0, 2.0, 3.0, 4.0],
                                            [2.0, 1.0, 3.0, 4.0],
                                            [4.0, 1.0, 3.0, 2.0]])
    top_k_scores, top_k_indices = util.approximate_top_k_with_indices(
        negative_scores, k=2)
    top_k_scores_expected = tf.convert_to_tensor([[2.0, 4.0], [2.0, 4.0],
                                                  [4.0, 3.0]])
    top_k_indices_expected = tf.convert_to_tensor([[1, 3], [0, 3], [0, 2]])
    self.assertAllClose(top_k_scores_expected, top_k_scores)
    self.assertAllEqual(top_k_indices_expected, top_k_indices)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
