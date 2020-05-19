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
"""Computes the prior, posterior, and likelihood."""

import jax
import jax.numpy as np
from jax.scipy import special
from grouptesting import utils


@jax.jit
def log_likelihood(particles,
                   test_results,
                   groups,
                   log_specificity,
                   log_1msensitivity):
  """Computes individual (parallel) log_likelihood of k_groups test results.

  Args:
   particles: np.ndarray<bool>[n_particles, n_patients]. Each one is a possible
    scenario of a disease status of n patients.
   test_results: np.ndarray<bool>[n_groups] the results given by the wet lab for
    each of the tested groups.
   groups: np.ndarray<bool>[num_groups, num_patients] the definition of the
    group that were tested.
   log_specificity: np.ndarray. Depending on the configuration, it can be an
    array of size one or more if we have different sensitivities per group size.
   log_1msensitivity: np.ndarray. Depending on the configuration, it can be an
    array of size one or more if we have different specificities per group size.

  Returns:
   The log likelihood of the particles given the test results.
  """
  positive_in_groups = np.dot(groups, np.transpose(particles)) > 0
  group_sizes = np.sum(groups, axis=1)
  log_specificity = utils.select_from_sizes(log_specificity, group_sizes)
  log_1msensitivity = utils.select_from_sizes(
      log_1msensitivity, group_sizes)
  logit_specificity = special.logit(np.exp(log_specificity))
  logit_sensitivity = - special.logit(np.exp(log_1msensitivity))
  gamma = log_1msensitivity - log_specificity
  add_logits = logit_specificity + logit_sensitivity
  ll = np.sum(
      positive_in_groups * (gamma + test_results * add_logits)[:, np.newaxis],
      axis=0)
  return ll + np.sum(log_specificity - test_results * logit_specificity)


@jax.jit
def log_prior(particles,
              base_infection_rate):
  """Computes log of prior probability of state using infection rate."""
  # here base_infection can be either a single number per patient or an array
  if np.size(base_infection_rate) == 1:  # only one rate
    return (np.sum(particles, axis=-1) * special.logit(base_infection_rate) +
            particles.shape[0] * np.log(1 - base_infection_rate))
  elif base_infection_rate.shape[0] == particles.shape[-1]:  # prior per patient
    return np.sum(
        particles * special.logit(base_infection_rate)[
            np.newaxis, :] + np.log(1 - base_infection_rate)[np.newaxis, :],
        axis=-1)
  else:
    raise ValueError("Vector of prior probabilities is not of correct size")


@jax.jit
def log_posterior(particles,
                  past_test_results,
                  past_groups,
                  log_prior_specificity,
                  log_prior_1msensitivity,
                  prior_infection_rate):
  """Given past tests, outputs unnormalized log-posterior of each particle."""
  ll = log_likelihood(particles, past_test_results, past_groups,
                      log_prior_specificity, log_prior_1msensitivity)
  lp = log_prior(particles, prior_infection_rate)
  return ll + lp

