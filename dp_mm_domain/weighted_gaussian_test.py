# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

from absl.testing import absltest
import numpy as np

from dp_mm_domain import gaussian
from dp_mm_domain import weighted_gaussian


class WGMTest(absltest.TestCase):

  def test_get_weighted_hist(self):
    input_data = [
        [1, 2, 3],
        [1, 2, 4],
        [5],
        [1, 5, 6],
        [7, 8],
    ]
    expected_output = {
        1: 3 * (1.0 / np.sqrt(3)),
        2: 2 * (1.0 / np.sqrt(3)),
        3: 1.0 / np.sqrt(3),
        4: 1.0 / np.sqrt(3),
        5: 1.0 + 1.0 / np.sqrt(3),
        6: 1.0 / np.sqrt(3),
        7: 1.0 / np.sqrt(2),
        8: 1.0 / np.sqrt(2),
    }
    output = weighted_gaussian.get_weighted_hist(input_data)
    self.assertDictAlmostEqual(output, expected_output, delta=1e-6)

  def test_get_weighted_hist_duplicate_elements(self):
    input_data = [[1, 1, 3], [2, 2, 2], [1, 2, 3]]
    expected_output = {
        1: 1 / np.sqrt(2) + 1 / np.sqrt(3),
        2: 1.0 + 1 / np.sqrt(3),
        3: 1 / np.sqrt(2) + 1.0 / np.sqrt(3),
    }
    output = weighted_gaussian.get_weighted_hist(input_data)
    self.assertDictAlmostEqual(output, expected_output, delta=1e-6)

  def test_get_weighted_gaussian_threshold__high_eps(self):
    # The expected threshold is calculated by running an external implementation
    # of get_weighted_gaussian_sigma_and_threshold.
    eps = 1.0
    delta = 1e-5
    l0_bound = 3
    expected_threshold = 17.38400956867444
    _, threshold = weighted_gaussian.get_weighted_gaussian_sigma_and_threshold(
        eps, delta, l0_bound
    )
    self.assertAlmostEqual(threshold, expected_threshold)

  def test_get_weighted_gaussian_threshold_low_eps(self):
    # The expected threshold is calculated by running an external implementation
    # of get_weighted_gaussian_sigma_and_threshold.
    eps = 0.10
    delta = 1e-5
    l0_bound = 10
    expected_threshold = 146.48399729851198
    _, threshold = weighted_gaussian.get_weighted_gaussian_sigma_and_threshold(
        eps, delta, l0_bound
    )
    self.assertAlmostEqual(threshold, expected_threshold)

  def test_get_weighted_gaussian_sigma(self):
    eps = 1.1
    delta = 0.1
    l0_bound = 5
    expected_sigma = gaussian.get_gaussian_sigma(eps, delta, 1.0)
    sigma, _ = weighted_gaussian.get_weighted_gaussian_sigma_and_threshold(
        eps, delta, l0_bound
    )
    self.assertAlmostEqual(sigma, expected_sigma, places=4)

  def test_get_noisy_weighted_hist_above_threshold_no_noise_no_threshold(self):
    input_data = [[1, 4]] * 10 + [[2, 3], [1, 2, 3]]
    sigma = 0
    threshold = 0
    expected_output = {
        1: 10 / np.sqrt(2) + 1 / np.sqrt(3),
        2: 1 / np.sqrt(3) + 1 / np.sqrt(2),
        3: 1 / np.sqrt(3) + 1 / np.sqrt(2),
        4: 10 / np.sqrt(2),
    }
    output = weighted_gaussian.get_noisy_weighted_hist_above_threshold(
        input_data, sigma, threshold
    )
    self.assertDictAlmostEqual(output, expected_output, delta=1e-6)

  def test_get_noisy_weighted_hist_above_threshold_no_noise_threshold(self):
    input_data = [[1, 4]] * 10 + [[2, 3], [1, 2, 3]]
    sigma = 0
    threshold = 4
    expected_output = {
        1: 10 / np.sqrt(2) + 1 / np.sqrt(3),
        4: 10 / np.sqrt(2),
    }
    output = weighted_gaussian.get_noisy_weighted_hist_above_threshold(
        input_data, sigma, threshold
    )
    self.assertDictAlmostEqual(output, expected_output, delta=1e-6)


if __name__ == '__main__':
  absltest.main()
