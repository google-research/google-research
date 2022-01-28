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

"""Tests for baseline_mechanisms."""

from absl.testing import absltest
import numpy as np

from dp_topk import baseline_mechanisms
from dp_topk.differential_privacy import NeighborType


class BaselineMechanismsTest(absltest.TestCase):

  def test_sorted_top_k_returns_correct_when_input_is_ascending(self):
    counts = np.arange(10)
    k = 3
    expected_items = np.array([9, 8, 7])
    items = baseline_mechanisms.sorted_top_k(counts, k)
    self.assertSequenceAlmostEqual(items, expected_items)

  def test_sorted_top_k_returns_correct_when_input_is_descending(self):
    counts = np.arange(10)[::-1]
    k = 3
    expected_items = np.arange(k)
    items = baseline_mechanisms.sorted_top_k(counts, k)
    self.assertSequenceAlmostEqual(items, expected_items)

  def test_laplace_no_noise_add_remove(self):
    counts = np.arange(2)
    k = 1
    c = 10
    expected_items = [1]
    items = baseline_mechanisms.laplace_mechanism(
        counts, k, c, epsilon=1e6, neighbor_type=NeighborType.ADD_REMOVE)
    self.assertSequenceAlmostEqual(items, expected_items)

  def test_laplace_no_noise_swap(self):
    counts = np.arange(2)
    k = 1
    c = 10
    expected_items = [1]
    items = baseline_mechanisms.laplace_mechanism(
        counts, k, c, epsilon=1e6, neighbor_type=NeighborType.SWAP)
    self.assertSequenceAlmostEqual(items, expected_items)

  def test_laplace_with_noise_add_remove(self):
    k = 1
    epsilon = 1.1
    trials = 100000
    z1 = 1 + np.random.laplace(0, k / epsilon, size=trials)
    z2 = np.random.laplace(0, k / epsilon, size=trials)

    expected_correctness = sum(z1 > z2) / trials

    counts = np.array([1, 0])

    correct = 0
    for _ in range(trials):
      items = baseline_mechanisms.laplace_mechanism(
          item_counts=counts,
          k=k,
          c=2,
          epsilon=epsilon,
          neighbor_type=NeighborType.ADD_REMOVE)
      if items[0] == 0:
        correct += 1

    real_correctness = correct / trials
    self.assertAlmostEqual(expected_correctness, real_correctness, places=2)

  def test_laplace_with_noise_swap(self):
    k = 1
    epsilon = 1.1
    trials = 100000
    z1 = 1 + np.random.laplace(0, 2 * k / epsilon, size=trials)
    z2 = np.random.laplace(0, 2 * k / epsilon, size=trials)

    expected_correctness = sum(z1 > z2) / trials

    counts = np.array([1, 0])

    correct = 0
    for _ in range(trials):
      items = baseline_mechanisms.laplace_mechanism(
          item_counts=counts,
          k=k,
          c=2,
          epsilon=epsilon,
          neighbor_type=NeighborType.SWAP)
      if items[0] == 0:
        correct += 1

    real_correctness = correct / trials
    self.assertAlmostEqual(expected_correctness, real_correctness, places=2)

  def test_em_epsilon_cdp_delta_zero(self):
    k = 10
    epsilon = 1.1
    delta = 0
    local_epsilon = baseline_mechanisms.em_epsilon_cdp(epsilon, delta, k)

    self.assertAlmostEqual(epsilon / k, local_epsilon, places=4)

  def test_em_epsilon_cdp_k_one(self):
    k = 1
    epsilon = 1.1
    delta = 0.1
    local_epsilon = baseline_mechanisms.em_epsilon_cdp(epsilon, delta, k)

    self.assertAlmostEqual(epsilon, local_epsilon, places=4)

  def test_em_epsilon_cdp_k_ten(self):
    k = 10
    epsilon = 1.1
    delta = 0.1
    local_epsilon = baseline_mechanisms.em_epsilon_cdp(epsilon, delta, k)

    self.assertAlmostEqual(0.29264, local_epsilon, places=4)

  def test_cdp_peeling_no_noise(self):
    counts = np.arange(2)
    k = 1
    expected_items = [1]
    items = baseline_mechanisms.cdp_peeling_mechanism(
        counts, k, epsilon=1e6, delta=0.1)
    self.assertEqual(items, expected_items)

  def test_cdp_peeling_with_noise(self):
    k = 1
    epsilon = 1.1
    trials = 100000

    counts = np.array([1, 0])

    probs = np.exp(epsilon * counts)
    probs = probs / sum(probs)

    expected_correctness = probs[0]

    correct = 0
    for _ in range(trials):
      items = baseline_mechanisms.cdp_peeling_mechanism(
          item_counts=counts, k=k, epsilon=epsilon, delta=0.1)
      if items[0] == 0:
        correct += 1

    real_correctness = correct / trials
    self.assertAlmostEqual(expected_correctness, real_correctness, places=2)

  def test_pnf_peeling_no_noise(self):
    counts = np.arange(2)
    k = 1
    expected_items = [1]
    items = baseline_mechanisms.pnf_peeling_mechanism(counts, k, epsilon=1e6)
    self.assertEqual(items, expected_items)

  def test_pnf_peeling_with_noise(self):
    k = 1
    epsilon = 1.1
    trials = 100000

    counts = np.array([1, 0])

    # PNF randomly permutes the items and then iterates through the permuted
    # items in order, flipping a coin for each item r with
    # P[heads] = exp(epsilon * [score of r - optimal score] / sensitivity)
    # until a heads appears, then outputting that item. Thus the probability of
    # outputting item 1 is P[item 1 is first in the permutation] +
    # P[item 2 is first but returns tails]. See Algorithm 1 in
    # https://arxiv.org/abs/2010.12603 for details. Note that we omit the factor
    # of 2 in the denominator inside the exponent of the heads probability
    # because our utility function is monotonic.
    expected_correctness = 0.5 + 0.5 * (1 - np.exp(-epsilon))

    correct = 0
    for _ in range(trials):
      items = baseline_mechanisms.pnf_peeling_mechanism(
          item_counts=counts, k=k, epsilon=epsilon)
      if items[0] == 0:
        correct += 1

    real_correctness = correct / trials
    self.assertAlmostEqual(expected_correctness, real_correctness, places=2)

  def test_gamma_no_noise(self):
    counts = np.arange(2)
    k = 1
    expected_items = [1]
    items = baseline_mechanisms.gamma_mechanism(counts, k, epsilon=1e6)
    self.assertEqual(items, expected_items)

  def test_gamma_with_noise(self):
    k = 1
    epsilon = 1.1
    trials = 100000

    counts = np.array([1, 0])
    radius = np.random.gamma(
        shape=(2 + 1), scale=1 / epsilon, size=trials)
    noise = np.random.uniform(-1, 1, (trials, 2)) * radius[:, np.newaxis]
    noisy_counts = counts + noise

    expected_correctness = sum(noisy_counts[:, 0] > noisy_counts[:, 1]) / trials

    correct = 0
    for _ in range(trials):
      items = baseline_mechanisms.gamma_mechanism(
          item_counts=counts, k=k, epsilon=epsilon)
      correct += (items[0] == 0)

    real_correctness = correct / trials
    self.assertAlmostEqual(expected_correctness, real_correctness, places=2)


if __name__ == '__main__':
  absltest.main()
