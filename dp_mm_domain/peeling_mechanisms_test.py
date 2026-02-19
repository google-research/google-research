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

from dp_mm_domain import peeling_mechanisms
from dp_mm_domain import utils


class PeelingMechanismsTest(absltest.TestCase):

  def test_cdp_peeling_no_noise(self):
    input_data = [[1, 2], [1]]
    freq_hist = utils.get_hist(input_data)
    domain = set([1, 2])
    k = 1
    expected_items = [1]
    items = peeling_mechanisms.cdp_peeling_mechanism(
        freq_hist, domain, k, eps=1e6, delta=0.1
    )
    self.assertEqual(items, expected_items)

  def test_cdp_peeling_with_noise(self):
    k = 1
    epsilon = 1.1
    trials = 100000

    input_data = [[1]]
    freq_hist = utils.get_hist(input_data)
    domain = set([1, 2])
    counts = np.array([1, 0])

    probs = np.exp(epsilon * counts)
    probs = probs / sum(probs)

    expected_correctness = probs[0]

    correct = 0
    for _ in range(trials):
      items = peeling_mechanisms.cdp_peeling_mechanism(
          freq_hist, domain, k=k, eps=epsilon, delta=0.1
      )
      if items[0] == 1:
        correct += 1

    real_correctness = correct / trials
    self.assertAlmostEqual(expected_correctness, real_correctness, places=2)

  def test_user_peeling_mechanism_no_noise_full_domain(self):
    input_data = [[1, 2, 3], [1, 2, 4], [1, 5], [6, 7], [6, 9]]
    eps = 1e6
    delta = 1.0
    domain = {1, 2, 3, 4, 5, 6, 7, 8, 9}
    output_set = peeling_mechanisms.user_peeling_mechanism(
        input_data, domain, 2, eps, delta
    )
    self.assertSetEqual(output_set, {1, 6})

  def test_user_peeling_mechanism_no_noise_partial_domain(self):
    input_data = [[1, 2, 3], [1, 2, 4, 5], [1, 5], [6, 7], [6, 9]]
    eps = 1e6
    delta = 0.50
    domain = {5, 6, 7}
    output_set = peeling_mechanisms.user_peeling_mechanism(
        input_data, domain, 2, eps, delta
    )
    self.assertSetEqual(output_set, {5, 6})

  def test_user_peeling_mechanism_no_noise_domain_smaller_than_k(self):
    input_data = [[1, 2, 3], [1, 2, 4, 5], [1, 5], [6, 7], [6, 9]]
    eps = 1e6
    delta = 0.50
    domain = {5, 6, 7}
    output_set = peeling_mechanisms.user_peeling_mechanism(
        input_data, domain, 5, eps, delta
    )
    self.assertSetEqual(output_set, {5, 6, 7})

  def test_user_peeling_mechanism_no_noise_all_users_removed(self):
    input_data = [[1, 2, 3], [1, 2, 3], [1], [1, 2]]
    eps = 1e10
    delta = 1.0
    domain = {1, 2, 3}
    output_set = peeling_mechanisms.user_peeling_mechanism(
        input_data, domain, 1, eps, delta,
    )
    self.assertSetEqual(output_set, {1})

  def test_wgm_then_peel_mechanism_no_noise_item_peeling(self):
    input_data = [[1]] * 10 + [[2, 3], [1, 4, 5]]
    k = 1
    expected_items = [1]
    items = peeling_mechanisms.wgm_then_peel_mechanism(
        input_data, k, [1e6, 1e6], [1.0, 1.0], 5, peel_users=False
    )
    self.assertEqual(items, expected_items)

  def test_wgm_then_peel_mechanism_no_noise_user_peeling(self):
    input_data = (
        [[1, 2, 3], [1, 2, 4], [1, 5], [2, 7], [2, 9]] + [[1]] * 10 + [[2]] * 10
    )
    eps_schedule = [1e6, 1e6]
    delta_schedule = [1.0, 1.0]
    l0_bound = 10
    output_set = peeling_mechanisms.wgm_then_peel_mechanism(
        input_data, 2, eps_schedule, delta_schedule, l0_bound, peel_users=True
    )
    self.assertSetEqual(output_set, {1, 2})

if __name__ == "__main__":
  absltest.main()
