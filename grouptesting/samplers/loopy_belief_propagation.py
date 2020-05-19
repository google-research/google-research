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
"""Loopy belief propagation based sampler."""

import gin
import jax
import jax.nn
import jax.numpy as np
import jax.scipy.special

from grouptesting import utils
from grouptesting.samplers import sampler


@jax.jit
def loopy_belief_propagation(tests,
                             groups,
                             base_infection_rate,
                             sensitivity,
                             specificity,
                             n_iter = 200
                             ):
  """LBP approach to compute approximate marginal of posterior distribution.

  Outputs marginal approximation of posterior distribution using all tests'
  history and test setup parameters.

  Args:
    tests : np.ndarray<bool>[n_groups] results stored as a vector of booleans
    groups : np.ndarray<bool>[n_groups, n_patients] matrix of groups
    base_infection_rate : np.ndarray<float> [1,] or [n_patients,] infection rate
    sensitivity : np.ndarray<float> [?,] of sensitivity per group size
    specificity : np.ndarray<float> [?,] of specificity per group size
    n_iter : int, number of loops in belief propagation.
  Returns:
    a vector of marginal probabilities for all n_patients.
  """
  if np.size(groups) == 0:
    if np.size(base_infection_rate) == 1:  # only one rate
      return base_infection_rate * np.ones(groups.shape[1])
    elif np.size(base_infection_rate) == groups.shape[1]:
      return base_infection_rate
    else:
      raise ValueError("Improper size for vector of base infection rates")

  n_groups, n_patients = groups.shape
  mu = -jax.scipy.special.logit(base_infection_rate)

  groups_size = np.sum(groups, axis=1)
  sensitivity = utils.select_from_sizes(sensitivity, groups_size)
  specificity = utils.select_from_sizes(specificity, groups_size)
  gamma0 = np.log(sensitivity + specificity - 1) - np.log(1 - sensitivity)
  gamma1 = np.log(sensitivity + specificity - 1) - np.log(sensitivity)
  gamma = tests * gamma1 + (1 - tests) * gamma0
  test_sign = 1 - 2 * tests[:, np.newaxis]

  # Initialization
  alphabeta = np.zeros((2, n_groups, n_patients))

  # lbp loop
  def lbp_loop(_, alphabeta):
    alpha = alphabeta[0, :, :]
    beta = alphabeta[1, :, :]
    # update alpha
    beta_bar = np.sum(beta, axis=0)
    alpha = jax.nn.log_sigmoid(beta_bar - beta + mu)
    alpha *= groups

    # update beta
    alpha_bar = np.sum(alpha, axis=1, keepdims=True)
    beta = np.log1p(
        test_sign * np.exp(-alpha + alpha_bar + gamma[:, np.newaxis]))
    beta *= groups
    return np.stack((alpha, beta), axis=0)

  # Run LBP loop
  alphabeta = jax.lax.fori_loop(0, n_iter, lbp_loop, alphabeta)

  # return marginals
  beta_bar = np.sum(alphabeta[1, :, :], axis=0)
  return jax.scipy.special.expit(-beta_bar - mu)


@gin.configurable
class LbpSampler(sampler.Sampler):
  """Loopy Belief Propagation approximation to Marginal."""

  NAME = "LBP"

  def __init__(self, num_iterations = 50):
    super().__init__()
    self.num_iterations = num_iterations

  def produce_sample(self, rng, state):
    """Produces only "one" fractional particle state: a marginal."""
    marginal = loopy_belief_propagation(
        state.past_test_results,
        state.past_groups,
        state.prior_infection_rate,
        state.prior_sensitivity,
        state.prior_specificity,
        n_iter=self.num_iterations)
    self.particles = np.expand_dims(marginal, axis=0)
    self.particle_weights = np.array([1])
