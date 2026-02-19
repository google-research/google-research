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

from dp_mm_domain import top_k


class TopKTest(absltest.TestCase):

  def test_get_hbot_small_kbar(self):
    sorted_hist = [(1, 1), (2, 2), (3, 3)]
    k_bar = 2
    delta = 0.1
    eps = 1.1
    l0_bound = 5
    expected_h_bot = (
        sorted_hist[k_bar][1] + 1 + np.log(min(l0_bound, k_bar) / delta) / eps
    )
    h_bot = top_k.get_hbot(sorted_hist, k_bar, delta, eps, l0_bound)
    self.assertAlmostEqual(h_bot, expected_h_bot, places=4)

  def test_get_hbot_large_kbar(self):
    sorted_hist = [(1, 1), (2, 2), (3, 3)]
    k_bar = 4
    delta = 0.1
    eps = 1.1
    l0_bound = 5
    expected_h_bot = 1 + np.log(min(l0_bound, k_bar) / delta) / eps
    h_bot = top_k.get_hbot(sorted_hist, k_bar, delta, eps, l0_bound)
    self.assertAlmostEqual(h_bot, expected_h_bot, places=4)

  def test_compute_limited_domain_eps_k_one(self):
    local_eps = 1.1
    delta = 0.1
    k = 1
    eps = top_k.compute_limited_domain_eps(local_eps, delta, k)

    self.assertAlmostEqual(eps, local_eps, places=6)

  def test_compute_limited_domain_eps_k_ten(self):
    # The expected value is calculated by running an external implementation
    # of compute_limited_domain_eps.
    local_eps = 1.1
    delta = 0.1
    k = 10
    eps = top_k.compute_limited_domain_eps(local_eps, delta, k)
    expected_eps = 9.782377233428313

    self.assertAlmostEqual(expected_eps, eps, places=6)

  def test_get_local_eps_delta_eps_k_one(self):
    eps = 1.1
    delta = 0.1
    delta_prime = 0.05
    k = 1
    local_eps, _ = top_k.get_local_eps_delta(eps, delta, delta_prime, k)
    expected_local_eps = eps
    self.assertAlmostEqual(expected_local_eps, local_eps, places=4)

  def test_get_local_eps_delta_eps_k_ten_less_than_eps(self):
    eps = 1.1
    delta = 0.1
    delta_prime = 0.05
    k = 10
    local_eps, _ = top_k.get_local_eps_delta(eps, delta, delta_prime, k)
    overall_eps = top_k.compute_limited_domain_eps(local_eps, delta_prime, k)
    self.assertLessEqual(overall_eps, eps)

  def test_get_local_eps_delta_eps_k_ten_value(self):
    # The expected local eps is calculated by running an external implementation
    # of get_local_eps.
    eps = 1.1
    delta = 0.1
    delta_prime = 0.05
    k = 10
    local_eps, _ = top_k.get_local_eps_delta(eps, delta, delta_prime, k)
    expected_local_eps = 0.22107791900634766
    self.assertAlmostEqual(local_eps, expected_local_eps, places=4)

  def test_get_local_eps_delta_delta_prime(self):
    eps = 1.1
    delta = 0.1
    delta_prime = 0.05
    k = 10
    _, local_delta = top_k.get_local_eps_delta(eps, delta, delta_prime, k)
    self.assertAlmostEqual(local_delta, delta - delta_prime, places=4)

  def test_get_local_eps_delta_delta_prime_raises_error(self):
    eps = 1.1
    delta = 0.1
    delta_prime = 1.0
    k = 10
    with self.assertRaises(ValueError):
      top_k.get_local_eps_delta(eps, delta, delta_prime, k)

  def test_limited_domain_mechanism_no_noise(self):
    input_data = [[1]] * 10 + [[2, 3], [1, 4, 5]]
    k = 1
    k_bar = 2
    expected_items = [1]
    items = top_k.limited_domain_mechanism(input_data, k, k_bar, 1e6, 0.1, 5)
    self.assertEqual(items, expected_items)


if __name__ == "__main__":
  absltest.main()
