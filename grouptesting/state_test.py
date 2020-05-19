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
"""Tests for the Simulator."""

from absl.testing import absltest
import jax
import jax.numpy as np
import jax.test_util

from grouptesting import state


class StateTestCase(jax.test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.state = state.State(num_patients=72,
                             num_tests_per_cycle=3,
                             max_group_size=5,
                             prior_infection_rate=0.05,
                             prior_specificity=0.95,
                             prior_sensitivity=0.75)
    self.num_groups = 4
    self.rng, *rngs = jax.random.split(self.rng, 3)
    self.groups = jax.random.uniform(
        rngs[0], (self.num_groups, self.state.num_patients)) > 0.3
    self.results = jax.random.uniform(rngs[1], (self.num_groups,)) > 0.2

  def test_add_groups_to_test(self):
    self.assertEqual(np.size(self.state.groups_to_test), 0)
    self.state.add_groups_to_test(self.groups)
    self.assertTrue(np.all(self.state.groups_to_test == self.groups))
    self.state.add_groups_to_test(self.groups)
    self.assertEqual(self.state.groups_to_test.shape,
                     (2 * self.num_groups, self.state.num_patients))
    self.assertGreater(self.state.num_groups_left_to_test, 0)
    self.state.reset()
    self.assertEqual(np.size(self.state.groups_to_test), 0)
    self.assertEqual(self.state.num_groups_left_to_test, 0)

  def test_add_test_results(self):
    self.assertEqual(np.size(self.state.groups_to_test), 0)
    self.state.add_groups_to_test(self.groups)
    self.state.add_past_groups(self.groups)
    self.state.add_test_results(self.results)
    self.assertTrue(np.all(self.state.past_groups == self.groups))
    self.assertTrue(np.all(self.state.past_test_results == self.results))

  def test_next_groups_to_test(self):
    self.state.add_groups_to_test(self.groups)
    groups = self.state.next_groups_to_test()
    self.assertEqual(groups.shape,
                     (self.state.num_tests_per_cycle, self.state.num_patients))


if __name__ == '__main__':
  absltest.main()
