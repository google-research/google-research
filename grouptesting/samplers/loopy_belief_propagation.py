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

from absl import logging

import gin
import jax
import jax.nn
import jax.numpy as np
import jax.scipy.special

from grouptesting import utils
from grouptesting.samplers import sampler


def loopy_belief_propagation(tests, groups,
                             base_infection_rate,
                             sensitivity, specificity,
                             min_iterations, max_iterations,
                             atol):
  """LBP approach to compute approximate marginal of posterior distribution.

  Outputs marginal approximation of posterior distribution using all tests'
  history and test setup parameters.

  Args:
    tests : np.ndarray<bool>[n_groups] results stored as a vector of booleans
    groups : np.ndarray<bool>[n_groups, n_patients] matrix of groups
    base_infection_rate : np.ndarray<float> [1,] or [n_patients,] infection rate
    sensitivity : np.ndarray<float> [?,] of sensitivity per group size
    specificity : np.ndarray<float> [?,] of specificity per group size
    min_iterations: int, min number of belief propagation iterations
    max_iterations: int, max number of belief propagation iterations
    atol: float, elementwise tolerance for the difference between two
      consecutive iterations.

  Returns:
    two vectors of marginal probabilities for all n_patients, obtained
    as consecutive evaluations of the LBP algorithm after n_iter and n_iter+1
    iterations.
  """
  n_groups, n_patients = groups.shape
  if np.size(groups) == 0:
    if np.size(base_infection_rate) == 1:  # only one rate
      marginal = base_infection_rate * np.ones(n_patients)
      return marginal, 0
    elif np.size(base_infection_rate) == n_patients:
      return base_infection_rate, 0
    else:
      raise ValueError("Improper size for vector of base infection rates")

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
  alpha_beta_iteration = [alphabeta, 0]

  # return marginal from alphabeta
  def marginal_from_alphabeta(alphabeta):
    beta_bar = np.sum(alphabeta[1, :, :], axis=0)
    return jax.scipy.special.expit(-beta_bar - mu)

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
    beta = np.log1p(test_sign *
                    np.exp(-alpha + alpha_bar + gamma[:, np.newaxis]))
    beta *= groups
    return np.stack((alpha, beta), axis=0)

  def cond_fun(alpha_beta_iteration):
    alphabeta, iteration = alpha_beta_iteration
    marginal = marginal_from_alphabeta(alphabeta)
    marginal_plus_one_iteration = marginal_from_alphabeta(
        lbp_loop(0, alphabeta))
    converged = np.allclose(marginal, marginal_plus_one_iteration, atol=atol)
    return (not converged) and (iteration < max_iterations)

  def body_fun(alpha_beta_iteration):
    alphabeta, iteration = alpha_beta_iteration
    alphabeta = jax.lax.fori_loop(0, min_iterations, lbp_loop, alphabeta)
    iteration += min_iterations
    return [alphabeta, iteration]

  # Run LBP while loop
  while cond_fun(alpha_beta_iteration):
    alpha_beta_iteration = body_fun(alpha_beta_iteration)

  alphabeta, _ = alpha_beta_iteration

  # Compute two consecutive marginals
  marginal = marginal_from_alphabeta(alphabeta)
  marginal_plus_one_iteration = marginal_from_alphabeta(lbp_loop(0, alphabeta))

  return marginal, np.amax(np.abs(marginal - marginal_plus_one_iteration))


@gin.configurable
class LbpSampler(sampler.Sampler):
  """Loopy Belief Propagation approximation to Marginal."""

  NAME = "LBP"

  def __init__(self,
               min_iterations = 10,
               max_iterations = 1000,
               atol = 1e-4,
               gaptol = 1e-2):
    """Initialize LbpSampler with parameters passed on to LBP algorithm.

    Args:
      min_iterations : int, minimal number of executions per loop of LBP
      max_iterations : int, maximal number of iterations LBP updates marginal
      atol : float, tolerance parameter used to measure discrepancy between two
        consecutive iterations with np.allclose to decide termination of LBP
        loops.
      gaptol : float, tolerance used to decide whether the gap returned by an
        LBP sampler is acceptable or requires resampling.
    """
    super().__init__()
    self.min_iterations = min_iterations
    self.max_iterations = max_iterations
    self.atol = atol
    self.gaptol = gaptol

  def produce_sample(self, rng, state):
    """Produces only "one" fractional particle state: a marginal.

    Args:
      rng : random PRNG key
      state : state object containing all relevant information to produce sample

    Returns:
      a measure of the quality of convergence, here gap_between_consecutives
      also updates particle_weights and particles members.
    """

    self.particle_weights = np.array([1])
    marginal, gap_between_consecutives = loopy_belief_propagation(
        state.past_test_results, state.past_groups, state.prior_infection_rate,
        state.prior_sensitivity, state.prior_specificity, self.min_iterations,
        self.max_iterations, self.atol)

    # record convergence of LBP in sampler
    self.convergence_metric = gap_between_consecutives
    if gap_between_consecutives < self.gaptol:
      # if LBP has converged and is not oscillating, return marginal
      self.particles = np.expand_dims(marginal, axis=0)
    else:
      # if LBP has not converged, return a vector of NaNs
      vector_of_nans = np.zeros_like(marginal)
      vector_of_nans = jax.ops.index_update(vector_of_nans, jax.ops.index[:],
                                            np.nan)
      self.particles = np.expand_dims(vector_of_nans, axis=0)
      logging.info("LBP Sampler has not converged and/or oscillates.")
      logging.info("gap: %8.4f", gap_between_consecutives)
