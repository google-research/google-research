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
"""Tests for the different group selectors."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as np
import jax.test_util

from grouptesting import state
from grouptesting.group_selectors import bayes_oed
from grouptesting.group_selectors import informative_dorfman
from grouptesting.group_selectors import mutual_information
from grouptesting.group_selectors import random
from grouptesting.group_selectors import split
from grouptesting.samplers import sequential_monte_carlo


class GroupSelectorsTest(jax.test_util.JaxTestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.num_patients = 20
    self.num_tests_per_cycle = 3
    self.state = state.State(
        self.num_patients, self.num_tests_per_cycle, max_group_size=4,
        prior_infection_rate=0.30, prior_specificity=0.9, prior_sensitivity=0.7)

    self.rng = jax.random.PRNGKey(0)

  @parameterized.parameters([
      random.RandomSelector(), split.SplitSelector(split_factor=2),
  ])
  def test_selector_without_particles(self, selector):
    groups = selector.get_groups(self.rng, self.state)
    self.assertGreater(groups.shape[0], 0)
    self.assertEqual(groups.shape[1], self.num_patients)
    self.assertEqual(np.size(self.state.groups_to_test), 0)
    selector(self.rng, self.state)
    self.assertGreater(np.size(self.state.groups_to_test), 0)

  @parameterized.parameters([
      mutual_information.MaxMutualInformation(),
      informative_dorfman.InformativeDorfman(),
      bayes_oed.BayesOED(),
      bayes_oed.BayesOED(utility_fn=bayes_oed.auc()),
      bayes_oed.BayesOED(utility_fn=bayes_oed.entropy()),
      bayes_oed.BayesOED(utility_fn=bayes_oed.mean_sensitivity_specificity()),
  ])
  def test_selector_with_particles(self, selector):
    sampler = sequential_monte_carlo.SmcSampler(num_particles=100)
    rngs = jax.random.split(self.rng, 2)
    sampler.produce_sample(rngs[0], self.state)
    self.state.update_particles(sampler)
    self.assertEqual(np.size(self.state.groups_to_test), 0)
    selector(rngs[1], self.state)
    self.assertGreater(np.size(self.state.groups_to_test), 0)


if __name__ == '__main__':
  absltest.main()
