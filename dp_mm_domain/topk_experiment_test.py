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
from dp_mm_domain import topk_experiment

# Input data for testing. The true top 3 elements are 1, 2, 5 with counts 3, 2,
# and 2, respectively. The remaining elements 3, 4, 6, 8 have count 1.
INPUT_DATA = [
    [1, 2, 3],
    [1, 2, 4],
    [5],
    [1, 5, 6],
    [7, 8],
]


class TopkExperimentTest(absltest.TestCase):

  def test_compute_topk_missing_mass_zero(self):
    output = [1, 2, 5]
    k = 3
    expected_topk_missing_mass = 0.0
    topk_missing_mass = topk_experiment.compute_topk_missing_mass(
        INPUT_DATA, output, k
    )
    self.assertAlmostEqual(topk_missing_mass, expected_topk_missing_mass)

  def test_compute_topk_missing_mass_non_zero(self):
    output = [1, 2, 6]
    k = 3
    expected_topk_missing_mass = (7 - 6) / 12
    topk_missing_mass = topk_experiment.compute_topk_missing_mass(
        INPUT_DATA, output, k
    )
    self.assertAlmostEqual(topk_missing_mass, expected_topk_missing_mass)

  def test_compute_topk_missing_mass_less_than_k_items(self):
    output = [1]
    k = 3
    expected_topk_missing_mass = (7 - 3) / 12
    topk_missing_mass = topk_experiment.compute_topk_missing_mass(
        INPUT_DATA, output, k
    )
    self.assertAlmostEqual(topk_missing_mass, expected_topk_missing_mass)

  def test_compute_topk_l1_loss_zero(self):
    output = [1, 2, 5]
    k = 3
    expected_topk_l1_loss = 0.0
    topk_l1_loss = topk_experiment.compute_topk_l1_loss(INPUT_DATA, output, k)
    self.assertAlmostEqual(topk_l1_loss, expected_topk_l1_loss)

  def test_compute_topk_l1_loss_non_zero(self):
    output = [2, 1, 5]
    k = 3
    expected_topk_l1_loss = 1.0 + 1.0
    topk_l1_loss = topk_experiment.compute_topk_l1_loss(INPUT_DATA, output, k)
    self.assertAlmostEqual(topk_l1_loss, expected_topk_l1_loss)

  def test_compute_topk_l1_loss_less_than_k_items(self):
    output = [2]
    k = 3
    expected_topk_l1_loss = 1.0 + 2.0 + 2.0
    topk_l1_loss = topk_experiment.compute_topk_l1_loss(INPUT_DATA, output, k)
    self.assertAlmostEqual(topk_l1_loss, expected_topk_l1_loss)

  def test_compare_methods_raises_error_for_unsupported_method(self):
    methods = ["unsupported_method"]
    k_range = [1, 2, 3]
    epsilon = 1.0
    delta = 1e-5
    l0_bound = 10
    num_trials = 1
    with self.assertRaises(ValueError):
      topk_experiment.compare_methods(
          INPUT_DATA, methods, k_range, epsilon, delta, l0_bound, num_trials
      )

  def test_compare_methods_raises_error_for_missing_params_for_limited_domain(
      self,
  ):
    methods = [topk_experiment.TopKMethod.LIMITED_DOMAIN]
    k_range = [1, 2, 3]
    epsilon = 1.0
    delta = 1e-5
    l0_bound = 10
    num_trials = 1
    with self.assertRaises(ValueError):
      topk_experiment.compare_methods(
          INPUT_DATA, methods, k_range, epsilon, delta, l0_bound, num_trials
      )

if __name__ == "__main__":
  absltest.main()
