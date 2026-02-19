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
from dp_mm_domain import hitting_set_experiment


class HittingSetExperimentTest(absltest.TestCase):

  def test_compute_missed_users(self):
    input_data = [[1, 2, 3], [1, 2, 4], [1, 5], [6, 7], [8, 9], [5, 3, 2]]
    output_set = {1, 6}
    missed_users = hitting_set_experiment.compute_missed_users(
        input_data, output_set
    )
    self.assertEqual(missed_users, 2)

  def test_compute_missed_users_no_missed_users(self):
    input_data = [[1, 2, 3], [1, 2, 4], [1, 5], [6, 7]]
    output_set = {1, 6}
    missed_users = hitting_set_experiment.compute_missed_users(
        input_data, output_set
    )
    self.assertEqual(missed_users, 0)

  def test_compare_methods_raises_error_for_unsupported_method(self):
    input_data = [[1, 2, 3], [1, 2, 4], [1, 5], [6, 7], [6, 9]]
    methods = ["unsupported_method"]
    k_range = [1, 2, 3]
    epsilon = 1.0
    delta = 1e-5
    l0_bound = 10
    num_trials = 1
    with self.assertRaises(ValueError):
      hitting_set_experiment.compare_methods(
          input_data, methods, k_range, epsilon, delta, l0_bound, num_trials
      )


if __name__ == "__main__":
  absltest.main()
