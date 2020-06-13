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
import jax.numpy as np
from jax.scipy import special


class State:
  """The state of the group testing system.

  Attributes:
   num_patients: the number of patients to be tested.
   num_tests_per_cycle: how many tests can we run in one cycle ( depends on the
     PCR machines and the machine availability).
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
   past_groups: np.ndarray<bool>[num_groups_tested, num_patients] groups that
     have been tested so far.
   past_test_results: np.ndarray<bool>[num_groups_tested]. Outcomes of tests for
     all past_groups, from start of testing campaign.
   sampled_up_to: int, keeps track of what tests have been included in sampling
     so far
   groups_to_test: np.array<bool>[num_groups, num_patients] groups scheduled for
     testing in next cycle(s).
   particle_weights: np.ndarray<float>[num_particles]. In particle filters, the
     weight of each particle.
   particles: np.ndarray<float>[num_particles, num_patients]. In particle
     filters, the description of num_particles, where each particle is a boolean
     vector describing who is diseased and who is not.
   to_clear_positives: some methods (e.g. Dorfman) need this information to keep
     track of which groups need to be tested next.
   all_cleared: bool, some methods consider perfect tests and stop when every
     one has been cleared.
   marginals: store one (or more) marginal approximations from samplers.
   num_groups_left_to_test: (int) the number of groups that were decided and
     still waiting to be tested.
   log_posterior_params: a dictionary that contains the parameters used to
     evaluate the log posterior probabilities of each particle.
  """

  def __init__(self, num_patients, num_tests_per_cycle,
               max_group_size, prior_infection_rate,
               prior_specificity, prior_sensitivity):
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
    """Update state with results from recently tested groups."""
    self.past_test_results = np.concatenate(
        (self.past_test_results, test_results), axis=0)

    missing_entries_in_to_clear = (
        np.size(self.past_test_results) - np.size(self.to_clear_positives))
    if missing_entries_in_to_clear > 0:
      # we should update the list of groups that have been tested positives.
      # this information is used by some strategies, notably Dorfman type ones.
      # if some entries are missing, they come by default from the latest wave
      # of tests carried out.
      self.to_clear_positives = np.concatenate(
          (self.to_clear_positives,
           test_results[-missing_entries_in_to_clear:]),
          axis=0)

  def add_groups_to_test(self,
                         groups,
                         results_need_clearing = False):
    """Add groups to test to state.

    Args:
      groups: np.ndarray <bool> : matrix of new tests to add
      results_need_clearing: bool, if false, one does not expect results
        returned for these groups to form the basis of further retesting. To
        impact this, a buffer of False values (as many as new tests added) is
        appended to the vector to_clear_positives. If logging is required, these
        results will be added by default when calling add_test_results.
    """
    self.groups_to_test = np.concatenate((self.groups_to_test, groups), axis=0)
    if not results_need_clearing:
      # if we do not need to record whether groups will be reused if positive,
      # we add to vector to_clear_positives as many False values.
      self.to_clear_positives = np.concatenate(
          (self.to_clear_positives, np.zeros((groups.shape[0],), dtype=bool)),
          axis=0)

  def add_past_groups(self, groups):
    self.past_groups = np.concatenate((self.past_groups, groups), axis=0)

  def update_to_clear_positives(self):
    self.to_clear_positives = np.zeros_like(self.to_clear_positives, dtype=bool)

  @property
  def num_groups_left_to_test(self):
    return self.groups_to_test.shape[0]

  @property
  def extra_tests_needed(self):
    """Number of tests left unused in next testing cycle."""
    return self.num_tests_per_cycle - self.num_groups_left_to_test

  def next_groups_to_test(self):
    """Moves the next batch from groups_to_test to past_group."""
    num_groups = np.minimum(self.groups_to_test.shape[0],
                            self.num_tests_per_cycle)
    result = self.groups_to_test[:num_groups, :]
    self.groups_to_test = self.groups_to_test[num_groups:, :]
    self.add_past_groups(result)
    return result

  def update_particles(self, sampler):
    """Keep as current particules the ones of a particle sampler."""
    self.particle_weights = sampler.particle_weights
    self.particles = sampler.particles

  def log_posterior_params(self,
                           sampling_from_scratch=True,
                           start_from_prior=False,
                           sampled_up_to=0):
    """Outputs parameters used to compute log posterior.

    Two scenarios are possible, depending on whether one wants to update an
    existing posterior approximation, or whether one wants to resample it from
    scratch
    Args:
      sampling_from_scratch: bool, flag to select all tests / prior seen so far
        or only the last wave of tests results.
      start_from_prior: bool, flag to indicate whether the first particles have
        been sampled from prior (True) or from a uniform measure.
      sampled_up_to: indicates what tests were used previously to generate
        samples. used when sampling_from_scratch is False


    Returns:
      a dict structure with fields relevant to evaluate Bayes.log_posterior
    """
    log_posterior_params = dict(
        log_prior_specificity=self.log_prior_specificity,
        log_prior_1msensitivity=self.log_prior_1msensitivity)
    # if past groups are same as latest, use by all past, including prior.
    if sampling_from_scratch or sampled_up_to == 0:
      log_posterior_params.update(
          test_results=self.past_test_results,
          groups=self.past_groups)
      if start_from_prior:
        log_posterior_params.update(
            prior_infection_rate=None)
      else:
        log_posterior_params.update(
            prior_infection_rate=self.prior_infection_rate)
    # if only using latest wave of tests, use no prior in posterior and only
    # use tests added since sampler was last asked to produce_sample.
    else:
      log_posterior_params.update(
          test_results=self.past_test_results[sampled_up_to:],
          groups=self.past_groups[sampled_up_to:],
          prior_infection_rate=None)
    return log_posterior_params

  def log_base_measure_params(self,
                              sampling_from_scratch=True,
                              start_from_prior=False,
                              sampled_up_to=0
                             ):
    """Outputs parameters used to compute log probability of base measure.

    Two scenarios are possible, depending on whether one wants to update an
    existing posterior approximation, or whether one wants to resample it from
    scratch
    Args:
      sampling_from_scratch: bool, flag to select all tests / prior seen so far
        or only the last wave of tests results.
      start_from_prior: bool, flag to indicate whether the first particles have
        been sampled from prior (True) or from a uniform measure.
      sampled_up_to: indicates what tests were used previously to generate
        samples. used when sampling_from_scratch is False

    Returns:
      a dict structure with fields relevant to evaluate Bayes.log_posterior
    """
    log_base_measure_params = dict(
        log_prior_specificity=self.log_prior_specificity,
        log_prior_1msensitivity=self.log_prior_1msensitivity)
    if sampling_from_scratch or sampled_up_to == 0:
      log_base_measure_params.update(
          test_results=None, groups=None)
      if start_from_prior:
        log_base_measure_params.update(
            prior_infection_rate=self.prior_infection_rate)
      else:
        log_base_measure_params.update(prior_infection_rate=None)

    else:
      past_minus_unused_tests = self.past_test_results[:sampled_up_to]
      past_minus_unused_groups = self.past_groups[:sampled_up_to]
      log_base_measure_params.update(
          test_results=past_minus_unused_tests,
          groups=past_minus_unused_groups,
          prior_infection_rate=self.prior_infection_rate)
    return log_base_measure_params
