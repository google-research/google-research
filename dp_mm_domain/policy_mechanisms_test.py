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

from dp_mm_domain import policy_mechanisms


class PolicyMechanismsTest(absltest.TestCase):

  def test_run_greedy_descent_step_items_above_threshold_unchanged(self):
    policy_hist = {1: 1.75, 2: 1.85, 3: 0.5, 4: 0.5}
    items = [1, 2, 3, 4]
    threshold = 1.0
    policy_hist = policy_mechanisms.run_greedy_descent_step(
        policy_hist, items, threshold
    )
    self.assertEqual(policy_hist[1], 1.75)
    self.assertEqual(policy_hist[2], 1.85)

  def test_run_greedy_descent_step_no_items_updated(self):
    policy_hist = {1: 1.5, 2: 1.5, 3: 1.5, 4: 1.5}
    items = [1, 2, 3, 4]
    threshold = 1.0
    policy_hist = policy_mechanisms.run_greedy_descent_step(
        policy_hist, items, threshold
    )
    self.assertEqual(policy_hist[1], 1.50)
    self.assertEqual(policy_hist[2], 1.50)
    self.assertEqual(policy_hist[3], 1.50)
    self.assertEqual(policy_hist[4], 1.50)

  def test_run_greedy_descent_step_all_items_updated(self):
    policy_hist = {1: 0.75, 2: 0.75, 3: 0.75, 4: 0.75}
    items = [1, 1, 1, 2, 2, 3, 4]
    threshold = 1.0
    policy_hist = policy_mechanisms.run_greedy_descent_step(
        policy_hist, items, threshold
    )
    self.assertEqual(policy_hist[1], 1.0)
    self.assertEqual(policy_hist[2], 1.0)
    self.assertEqual(policy_hist[3], 1.0)
    self.assertEqual(policy_hist[4], 1.0)

  def test_run_greedy_descent_step_ordered_updates(self):
    policy_hist = {1: 0.70, 2: 0.10, 3: 0.05, 4: 0.02}
    items = [1, 1, 2, 2, 2, 3, 4]
    threshold = 1.0
    policy_hist = policy_mechanisms.run_greedy_descent_step(
        policy_hist, items, threshold
    )
    self.assertAlmostEqual(policy_hist[1], 0.80)
    self.assertAlmostEqual(policy_hist[2], 1.0)
    self.assertAlmostEqual(policy_hist[3], 0.05)
    self.assertAlmostEqual(policy_hist[4], 0.02)

  def test_run_greedy_descent_step_new_item_fully_updated(self):
    policy_hist = {1: 0.70, 2: 0.10, 3: 0.05, 4: 0}
    items = [1, 2, 3, 4, 4]
    threshold = 1.0
    policy_hist = policy_mechanisms.run_greedy_descent_step(
        policy_hist, items, threshold
    )
    self.assertEqual(policy_hist[1], 0.70)
    self.assertEqual(policy_hist[2], 0.10)
    self.assertEqual(policy_hist[3], 0.05)
    self.assertEqual(policy_hist[4], 1.0)

  def test_get_policy_hist_greedy(self):
    input_data = [[1, 2], [1, 3], [1, 2]]
    threshold = 2.0
    policy_hist = policy_mechanisms.get_policy_hist(
        input_data, threshold, policy_mechanisms.Policy.GREEDY
    )
    self.assertEqual(policy_hist[1], 2.0)
    self.assertEqual(policy_hist[2], 1.0)
    self.assertEqual(policy_hist[3], 0.0)
    self.assertEqual(policy_hist[4], 0.0)

  def test_run_l1_norm_descent_step_above_threshold_unchanged(self):
    policy_hist = {1: 0.75, 2: 0.85, 3: 0.5, 4: 0.5}
    items = [1, 2, 3, 4]
    threshold = 0.75
    policy_hist = policy_mechanisms.run_lnorm_descent_step(
        policy_hist, items, threshold, policy_mechanisms.Descent.L1
    )
    self.assertEqual(policy_hist[1], 0.75)
    self.assertEqual(policy_hist[2], 0.85)

  def test_run_l1_norm_descent_step_all_items_updated(self):
    policy_hist = {1: 0.50, 2: 0.50, 3: 0.5, 4: 0.5}
    items = [1, 2, 3, 4]
    threshold = 0.75
    policy_hist = policy_mechanisms.run_lnorm_descent_step(
        policy_hist, items, threshold, policy_mechanisms.Descent.L1
    )
    self.assertEqual(policy_hist[1], 0.75)
    self.assertEqual(policy_hist[2], 0.75)
    self.assertEqual(policy_hist[3], 0.75)
    self.assertEqual(policy_hist[4], 0.75)

  def test_run_l1_norm_descent_step_low_norm_update(self):
    policy_hist = {1: 0.75, 2: 0.50, 3: 0.50, 4: 0.50}
    items = [1, 2, 3, 4]
    threshold = 1.0
    policy_hist = policy_mechanisms.run_lnorm_descent_step(
        policy_hist, items, threshold, policy_mechanisms.Descent.L1
    )

    self.assertEqual(policy_hist[1], 1.0)
    self.assertEqual(policy_hist[2], 1.0)
    self.assertEqual(policy_hist[3], 1.0)
    self.assertEqual(policy_hist[4], 1.0)

  def test_run_l1_norm_descent_step_high_norm_update_uniform_spread(self):
    policy_hist = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    items = [1, 2, 3, 4]
    threshold = 2.0
    policy_hist = policy_mechanisms.run_lnorm_descent_step(
        policy_hist, items, threshold, policy_mechanisms.Descent.L1
    )
    self.assertEqual(policy_hist[1], 1.5)
    self.assertEqual(policy_hist[2], 1.5)
    self.assertEqual(policy_hist[3], 1.5)
    self.assertEqual(policy_hist[4], 1.5)

  def test_run_l1_descent_step_high_norm_update_uneven_spread(self):
    policy_hist = {1: 2.75, 2: 1.0, 3: 1.0, 4: 1.0}
    items = [1, 2, 3, 4]
    threshold = 3
    policy_hist = policy_mechanisms.run_lnorm_descent_step(
        policy_hist, items, threshold, policy_mechanisms.Descent.L1
    )

    self.assertEqual(policy_hist[1], 3.0)
    self.assertEqual(policy_hist[2], 1.0 + (1.0-0.25**2) / np.sqrt(3))
    self.assertEqual(policy_hist[3], 1.0 + (1.0-0.25**2) / np.sqrt(3))
    self.assertEqual(policy_hist[4], 1.0 + (1.0-0.25**2) / np.sqrt(3))

  def test_get_policy_hist_gaussian_l1(self):
    input_data = [[1, 2], [1, 3]]
    threshold = 2.0
    policy_hist = policy_mechanisms.get_policy_hist(
        input_data, threshold, policy_mechanisms.Policy.GAUSSIAN_L1
    )
    print(policy_hist)
    self.assertEqual(policy_hist[1], 2.0/np.sqrt(2))
    self.assertEqual(policy_hist[2], 1/np.sqrt(2))
    self.assertEqual(policy_hist[3], 1/np.sqrt(2))

  def test_run_l2_norm_descent_above_threshold_unchanged(self):
    policy_hist = {1: 0.75, 2: 0.85, 3: 0.5, 4: 0.5}
    items = [1, 2, 3, 4]
    threshold = 0.75
    policy_hist = policy_mechanisms.run_lnorm_descent_step(
        policy_hist, items, threshold, descent=policy_mechanisms.Descent.L2
    )
    self.assertEqual(policy_hist[1], 0.75)
    self.assertEqual(policy_hist[2], 0.85)

  def test_run_l2_norm_descent_low_norm_update(self):
    policy_hist = {1: 0.50, 2: 0.50, 3: 0.50, 4: 0.50}
    items = [1, 2, 3, 4]
    threshold = 0.75
    policy_hist = policy_mechanisms.run_lnorm_descent_step(
        policy_hist, items, threshold, descent=policy_mechanisms.Descent.L2
    )
    self.assertEqual(policy_hist[1], 0.75)
    self.assertEqual(policy_hist[2], 0.75)
    self.assertEqual(policy_hist[3], 0.75)
    self.assertEqual(policy_hist[4], 0.75)

  def test_run_l2_norm_descent_high_norm_update(self):
    policy_hist = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}
    items = [1, 2, 3, 4]
    threshold = 2.5
    policy_hist = policy_mechanisms.run_lnorm_descent_step(
        policy_hist, items, threshold, descent=policy_mechanisms.Descent.L2
    )
    self.assertEqual(policy_hist[1], 1.0)
    self.assertEqual(policy_hist[2], 1.0)
    self.assertEqual(policy_hist[3], 1.0)
    self.assertEqual(policy_hist[4], 1.0)

  def test_run_l2_norm_descent_high_norm_update_uneven_spread(self):
    policy_hist = {1: 2.75, 2: 1.0, 3: 1.0, 4: 1.0}
    items = [1, 2, 3, 4]
    threshold = 3
    policy_hist = policy_mechanisms.run_lnorm_descent_step(
        policy_hist, items, threshold, descent=policy_mechanisms.Descent.L2
    )
    z = np.sqrt(0.25**2 + 3*2**2)
    self.assertEqual(policy_hist[1], 2.75 + 0.25/z)
    self.assertEqual(policy_hist[2], 1.0 + 2.0/z)
    self.assertEqual(policy_hist[3], 1.0 + 2.0/z)
    self.assertEqual(policy_hist[4], 1.0 + 2.0/z)

  def test_get_policy_hist_gaussian_l2(self):
    input_data = [[1, 2], [1, 3]]
    threshold = 2.0
    policy_hist = policy_mechanisms.get_policy_hist(
        input_data, threshold, policy_mechanisms.Policy.GAUSSIAN_L2
    )
    z = (2 - (1 / np.sqrt(2))) ** 2 + 4.0
    self.assertEqual(
        policy_hist[1], 1.0 / np.sqrt(2) + (2.0 - 1.0 / np.sqrt(2)) / np.sqrt(z)
    )
    self.assertEqual(policy_hist[2], 1 / np.sqrt(2))
    self.assertEqual(policy_hist[3], 2 / np.sqrt(z))

  def test_policy_gaussian_mechanism_unknown_descent_raises_error(self):
    input_data = [[1, 2], [1]]
    l0_bound = 2
    eps = 1.0
    delta = 1e-5
    descent = "unknown"
    alpha = 3
    with self.assertRaises(ValueError):
      policy_mechanisms.policy_gaussian_mechanism(
          input_data, l0_bound, eps, delta, descent, alpha
      )

  def test_get_policy_hist_unknown_policy_raises_error(self):
    input_data = [[1, 2], [1]]
    threshold = 2.0
    policy = "unknown"

    with self.assertRaises(ValueError):
      policy_mechanisms.get_policy_hist(input_data, threshold, policy)

if __name__ == "__main__":
  absltest.main()
