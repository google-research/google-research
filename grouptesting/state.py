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
"""A state of the group testing algorithm."""

from typing import Dict
import jax
import jax.numpy as np
from jax.scipy import special


class State:
  """The state of the group testing system.

  Attributes:
   num_patients: the number of patients to be tested.
   num_tests_per_cycle: how many tests can we run in one cycle (
    depends on the PCR machines and the machine availability).
   max_group_size: (int) the maximum size of a group in this experiment.
   curr_cycle: the index of the current test cycle.
   extra_tests_needed: are there remaining tests neeed to be completed before
     moving on to the next selector.
   prior_infection_rate: what we believe to be the infection rate.
   prior_specificity: what we believe to be the specificity of the PCR machine.
   prior_sensitivity: what we believe to be the specificity of the PCR machine.
   log_prior_specificity: log of the prior specificity.
   log_prior_1msensitivity: log(1-prior_sensitivity).
   logit_prior_specificity: same but in logit.
   logit_prior_sensitivity: same but in logit.
   past_groups: np.ndarray<bool>[num_groups_tested, num_patients] the groups
    that were built in so far.
   past_test_results: np.ndarray<bool>[num_groups_tested]. The outcomes of the
    tests for the past_groups.
   groups_to_test: np.array<bool>[num_groups, num_patients] who to test next.
   particle_weights: np.ndarray<float>[num_particles]. In particle filters, the
    weight of each particle.
   particles: np.ndarray<float>[num_particles, num_patients]. In particle
    filters, the value of the particle as a possible scenario of who is
    diseased and who is not.
   to_clear_positives: some methods (Dorfman) need this information.
   all_cleared: bool, some methods consider perfect tests and stop when every
    one has been cleared.
   marginals: store different ways to compute the marginal.
   num_groups_left_to_test: (int) the number of groups that were decided and
    still waiting to be tested.
   log_posterior_params: a dictionary that contains the parameters used for
    computing the log posterior probabilities of the particles.
  """

  def __init__(self,
               num_patients,
               num_tests_per_cycle,
               max_group_size,
               prior_infection_rate,
               prior_specificity,
               prior_sensitivity):
    self.num_patients = num_patients
    self.num_tests_per_cycle = num_tests_per_cycle
    self.max_group_size = max_group_size

    self.prior_infection_rate = np.atleast_1d(prior_infection_rate)
    self.prior_specificity = np.atleast_1d(prior_specificity)
    self.prior_sensitivity = np.atleast_1d(prior_sensitivity)
    self.log_prior_specificity = np.log(self.prior_specificity)
    self.log_prior_1msensitivity = np.log(1 - self.prior_sensitivity)
    self.logit_prior_sensitivity = special.logit(self.prior_sensitivity)
    self.logit_prior_specificity = special.logit(self.prior_specificity)

    self.curr_cycle = 0
    self.past_groups = None
    self.past_test_results = None
    self.groups_to_test = None
    self.particle_weights = None
    self.particles = None
    self.to_clear_positives = None
    self.all_cleared = False
    self.marginals = {}
    self.reset()  # Initializes the attributes above.

  def reset(self):
    """Reset the state."""
    self.curr_cycle = 0
    self.past_groups = np.empty((0, self.num_patients), dtype=bool)
    self.past_test_results = np.empty((0,), dtype=bool)
    self.groups_to_test = np.empty((0, self.num_patients), dtype=bool)

    # Those are specific to some methods. They are not always used or filled.
    self.particle_weights = None
    self.particles = None
    self.to_clear_positives = np.empty((0,), dtype=bool)
    self.all_cleared = False

    # In case we store marginals computed in different ways.
    self.marginals = {}

  def add_test_results(self, test_results):
    self.past_test_results = np.concatenate(
        (self.past_test_results, test_results), axis=0)
    self.to_clear_positives = np.concatenate(
        (self.to_clear_positives, test_results), axis=0)

  def add_groups_to_test(self, groups):
    self.groups_to_test = np.concatenate((self.groups_to_test, groups), axis=0)

  def add_past_groups(self, groups):
    self.past_groups = np.concatenate((self.past_groups, groups), axis=0)

  def update_to_clear_positives(self):
    self.to_clear_positives = jax.ops.index_update(
        self.to_clear_positives, jax.ops.index[self.to_clear_positives],
        False)

  @property
  def num_groups_left_to_test(self):
    return self.groups_to_test.shape[0]

  @property
  def extra_tests_needed(self):
    """Number of tests left unused in next testing cycle."""
    return self.num_tests_per_cycle - self.num_groups_left_to_test

  def next_groups_to_test(self):
    """Moves the next batch from groups_to_test to past_group."""
    num_groups = np.minimum(
        self.groups_to_test.shape[0], self.num_tests_per_cycle)
    result = self.groups_to_test[:num_groups, :]
    self.groups_to_test = self.groups_to_test[num_groups:, :]
    self.add_past_groups(result)
    return result

  def update_particles(self, sampler):
    """Keep as current particules the ones of a particle sampler."""
    self.particle_weights = sampler.particle_weights
    self.particles = sampler.particles

  @property
  def log_posterior_params(self):
    return dict(
        past_test_results=self.past_test_results,
        past_groups=self.past_groups,
        log_prior_specificity=self.log_prior_specificity,
        log_prior_1msensitivity=self.log_prior_1msensitivity,
        prior_infection_rate=self.prior_infection_rate)
