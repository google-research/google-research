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

"""Tests for kendall."""

from absl.testing import absltest
import numpy as np

from private_kendall import kendall


class KendallTest(absltest.TestCase):

  def test_kendall(self):
    features = [[1, -1], [2, -2], [3, -3]]
    labels = [4, 5, 6]
    correlations = kendall.kendall(features, labels)
    expected_correlations = np.asarray([3/2, -3/2])
    np.testing.assert_array_equal(correlations, expected_correlations)

  def test_dp_kendall_feature_selection_no_privacy(self):
    features = [[1, -1, 3, 1], [2, -2, 1, 1], [3, -3, 2, 1]]
    labels = [4, 5, 6]
    epsilon = 1000
    indices = kendall.dp_kendall(features, labels, 2, epsilon)
    expected_indices = np.asarray([0, 1])
    np.testing.assert_array_equal(indices, expected_indices)

  if __name__ == '__main__':
    absltest.main()
