# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Tests for the Policy."""

from absl.testing import absltest
import jax
import jax.numpy as np
import jax.test_util

from grouptesting import policy
from grouptesting import state
from grouptesting.group_selectors import random
from grouptesting.group_selectors import split


class PolicyTest(jax.test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.policy = policy.Policy(
        [random.RandomSelector(), split.SplitSelector()])

  def test_next_selector(self):
    self.assertEqual(self.policy.index, 0)
    self.assertIsInstance(self.policy.get_selector(), random.RandomSelector)
    self.assertIsInstance(self.policy.next_selector, split.SplitSelector)

  def test_act(self):
    num_patients = 40
    num_tests_per_cycle = 4
    s = state.State(num_patients, num_tests_per_cycle,
                    max_group_size=5, prior_infection_rate=0.05,
                    prior_specificity=0.95, prior_sensitivity=0.80)
    self.assertEqual(np.size(s.groups_to_test), 0)
    self.assertEqual(self.policy.index, 0)
    self.policy.act(self.rng, s)
    self.assertGreater(np.size(s.groups_to_test), 0)
    self.assertEqual(s.groups_to_test.shape[1], num_patients)
    self.assertGreater(s.groups_to_test.shape[0], 0)
    self.assertEqual(self.policy.index, 1)


if __name__ == '__main__':
  absltest.main()
