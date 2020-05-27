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

from grouptesting import policy
from grouptesting import simulator
from grouptesting import wet_lab
from grouptesting.group_selectors import random


class SimulatorTestCase(jax.test_util.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.num_patients = 10
    self.wetlab = wet_lab.WetLab(self.num_patients, freeze_diseased=True)
    self.num_simulations = 1
    self.max_test_cycles = 4
    self.policy = policy.Policy([random.RandomSelector()])
    self.simulator = simulator.Simulator(
        workdir=None, wetlab=self.wetlab, policy=self.policy,
        num_simulations=self.num_simulations,
        max_test_cycles=self.max_test_cycles,
        num_tests_per_cycle=4, max_group_size=5)
    self.rng = jax.random.PRNGKey(0)

  def test_reset(self):
    self.simulator.reset(self.rng)
    self.assertIsNotNone(self.wetlab.diseased)
    self.assertEqual(self.simulator.state.curr_cycle, 0)

  def test_sampler(self):
    self.assertIsNotNone(self.simulator.sampler)
    self.assertIs(self.simulator.sampler, self.simulator._samplers[1])

  def test_run(self):
    """A setup were we find a solution before the last cycle."""
    sim = simulator.Simulator(
        None, wet_lab.WetLab(num_patients=100),
        num_simulations=self.num_simulations,
        max_test_cycles=self.max_test_cycles,
        num_tests_per_cycle=4, max_group_size=5)
    sim.run(0)
    last_groups = sim.metrics.groups[0, -1]
    self.assertFalse(np.all(np.isnan(last_groups)))


if __name__ == '__main__':
  absltest.main()
