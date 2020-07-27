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
"""An interactive loop to go back and forth from colab to spreadsheet."""

import time

import jax
import numpy as np

from grouptesting import policy as group_policy
from grouptesting import simulator
from grouptesting import wet_lab
from grouptesting.samplers import sequential_monte_carlo as smc


class InteractiveSimulator(simulator.Simulator):
  """Interactive group testing loop, reading an write into a spreadsheet."""

  def __init__(self, sheet):
    wetlab = wet_lab.WetLab(num_patients=sheet.num_patients)
    sampler = smc.SmcSampler(num_particles=sheet.num_patients * 150,
                             resample_at_each_iteration=True)

    policy_name = sheet.params['policy']
    if policy_name == 'G-MIMAX':
      policy = group_policy.MimaxPolicy(forward_iterations=2,
                                        backward_iterations=1)
    elif policy_name == 'InformativeDorfman':
      policy = group_policy.InformativeDorfmanPolicy(cut_off_high=0.95,
                                                     cut_off_low=0.001)
    elif policy_name == 'Random':
      policy = group_policy.MezardPolicy()
    else:
      raise ValueError(f'Unsupported policy {policy_name}')

    super().__init__(
        wetlab=wetlab,
        sampler=sampler,
        policy=policy,
        num_simulations=1,
        num_tests_per_cycle=sheet.params['tests per cycle'],
        max_test_cycles=sheet.params['cycles'],
        max_group_size=sheet.params['max group size'],
        prior_specificity=sheet.params['specificity'],
        prior_sensitivity=sheet.params['sensitivity'],
        prior_infection_rate=sheet.priors)
    self.sheet = sheet

  def reset(self, rng):
    """Maybe reset simulator's state based on the current spreadsheet."""
    rngs = jax.random.split(rng, 2)
    super().reset(rngs[0])
    groups = self.sheet.read_groups()
    if np.size(groups):
      self.state.add_past_groups(groups)
      tests_results = self.sheet.read_results()
      self.process_tests_results(rngs[1], tests_results)
      self.sheet.write_marginals(self.marginal)

  def run(self, rngkey=None):
    """Starts an interaction loop."""
    rngkey = int(time.time()) if rngkey is None else rngkey
    rng = jax.random.PRNGKey(rngkey)

    rng, rng_reset = jax.random.split(rng, 2)
    self.reset(rng_reset)

    while not self.is_over:
      rng, *rngs = jax.random.split(rng, 3)
      print(f'---> Cycle {self.state.curr_cycle}')

      groups = self.get_new_groups(rngs[0])
      self.sheet.write_groups(groups)

      tests_results = None
      while tests_results is None:
        input(f'Please enters group results in\n{self.sheet.groups_url}\n')
        try:
          tests_results = self.sheet.read_results(groups.shape[0])
        except ValueError:
          print('Error: some group are lacking test results.')
          tests_results = None
          continue

      self.process_tests_results(rngs[1], tests_results)
      self.sheet.write_marginals(self.marginal)

      self._on_iteration_end(0, self.state.curr_cycle, groups)
      self.state.curr_cycle += 1
