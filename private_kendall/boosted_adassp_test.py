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

"""Tests for Boosted AdaSSP."""

from absl.testing import absltest
import numpy as np
from scipy import stats

from private_kendall import boosted_adassp


class BoostedAdaSSPTest(absltest.TestCase):

  def test_dp_to_gdp(self):
    epsilon_1 = 1.1
    delta_1 = 1e-5
    mu_1 = boosted_adassp.dp_to_gdp(epsilon_1, delta_1)
    mu_1_delta = stats.norm.cdf(-epsilon_1 / mu_1 + mu_1 / 2) - np.exp(
        epsilon_1
    ) * stats.norm.cdf(-epsilon_1 / mu_1 - mu_1 / 2)
    self.assertAlmostEqual(mu_1_delta, delta_1, 6)
    epsilon_2 = 0.1
    delta_2 = 1e-5
    mu_2 = boosted_adassp.dp_to_gdp(epsilon_2, delta_2)
    mu_2_delta = stats.norm.cdf(-epsilon_2 / mu_2 + mu_2 / 2) - np.exp(
        epsilon_2
    ) * stats.norm.cdf(-epsilon_2 / mu_2 - mu_2 / 2)
    self.assertAlmostEqual(mu_2_delta, delta_2, 6)
    epsilon_3 = 1.1
    delta_3 = 1e-10
    mu_3 = boosted_adassp.dp_to_gdp(epsilon_3, delta_3)
    mu_3_delta = stats.norm.cdf(-epsilon_3 / mu_3 + mu_3 / 2) - np.exp(
        epsilon_3
    ) * stats.norm.cdf(-epsilon_3 / mu_3 - mu_3 / 2)
    self.assertAlmostEqual(mu_3_delta, delta_3, 6)

  def test_clip(self):
    matrix = np.asarray([[1, 2, 3], [4, 5, 6]])
    clip_norm = 4
    clipped_matrix = boosted_adassp.clip(matrix, clip_norm)
    np.testing.assert_array_equal(clipped_matrix[0], matrix[0])
    scale = clip_norm / np.sqrt(16 + 25 + 36)
    expected_second_row = np.asarray([4 * scale, 5 * scale, 6 * scale])
    np.testing.assert_array_equal(clipped_matrix[1], expected_second_row)

  def test_boosted_adassp_no_privacy(self):
    eps = 25
    delta = 0.5
    num_points = 100
    features = np.ones((num_points, 2))
    features[:, 0] = np.arange(num_points)
    labels = np.sum(features, axis=1)
    num_rounds = 100
    feature_clip_norm = 100
    gradient_clip_norm = 100
    model = boosted_adassp.boosted_adassp(
        features, labels, num_rounds, feature_clip_norm, gradient_clip_norm,
        eps, delta
    )
    self.assertAlmostEqual(model[0, 0], 1, 2)
    self.assertAlmostEqual(model[1, 0], 1, 2)


if __name__ == '__main__':
  absltest.main()
