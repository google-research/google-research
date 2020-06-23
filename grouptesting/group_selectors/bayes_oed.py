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
"""Generic group optimization with Bayes optimal experimental design (OED) .

This file implements a generic Bayes OED function to optimize groups. Given a
utility function such as area under the ROC curve (AUC), the Bayes OED
approach attempts to find groups that maximize the expected objective function
under the current posterior.
"""

import itertools
import gin
import jax
import jax.numpy as np

from grouptesting import metrics
from grouptesting import utils
from grouptesting.group_selectors import group_selector
from grouptesting.group_selectors import mutual_information


@gin.configurable
def entropy():
  """Entropy as an objective function."""

  @jax.jit
  def entropy_utility(particle_weights, particles):
    """Entropy of the distribution as utility.

    This function returns the entropy utility of the distribution of weights,
    defined as the Kullback-Leibler divergence between the uniform distribution
    and the distribution encoded by the particles.
    Note that this requires all particles to be distinct; one should therefore
    be careful to call this function only after after collapsing together
    particles that may be repeated in the 'particles' array.

    Args:
     particle_weights: weights of particles
     particles: particles summarizing belief about infection status

    Returns:
     The entropy utility of the distribution of weights.
    """
    return particles.shape[1]*np.log(2) - metrics.entropy(particle_weights)

  return entropy_utility


@gin.configurable
def auc():
  """Area under the curve as an objective function."""

  @jax.jit
  def auc_utility(particle_weights, particles):
    """Expected AUC of the marginal predictor as utility.

    This function returns the AUC utility of a distribution encoded as a
    weighted sum of Dirac measures at particles. The AUC utility is the expected
    AUC of the marginal distribution as predictor.

    Args:
     particle_weights: weights of particles
     particles: particles summarizing belief about infection status

    Returns:
     The AUC utility of the distribution.
    """
    marginal = np.sum(particle_weights[:, np.newaxis] * particles, axis=0)
    sorted_particles = particles[:, np.argsort(marginal)]
    false_count = np.cumsum(1 - sorted_particles, axis=1)
    area = np.sum(sorted_particles * false_count, axis=1)
    aucs = area / (
        false_count[:, -1] * (sorted_particles.shape[1] - false_count[:, -1]))
    return np.nansum(aucs * particle_weights)
  return auc_utility


@gin.configurable
def mean_sensitivity_specificity(threshold=0.1):
  """Mean of sensitivity and specificity as an objective function.

  Args:
   threshold: threshold on the marginal to make a positive or negative
    prediction.

  Returns:
   A function that takes two parameters the weights and the particles and
   computes the objective function.
  """

  @jax.jit
  def _mean_sen_spe_utility(particle_weights, particles):
    """Expected mean sensitivity/specificity of the marginal predictor.

    This function returns the mean sensitivity/specificity utility of a
    distribution encoded as a weighted sum of Dirac measures at particles. The
    mean sensitivity/specificity utility is the expected mean
    sensitivity/specificity of the marginal distribution thresholded at
    'threshold' as predictor.

    Args:
     particle_weights: weights of particles
     particles: particles summarizing belief about infection status

    Returns:
     The mean sensitivity/specificity utility of the distribution.
    """
    num_patients = particles.shape[1]
    marginal = np.sum(particle_weights[:, np.newaxis] * particles, axis=0)
    y_pred = marginal > threshold
    total_pos = np.sum(particles, axis=1)
    total_neg = num_patients - total_pos
    sensitivities = np.sum(y_pred * particles, axis=1) / total_pos
    specifities = np.sum((1-y_pred) * (1-particles), axis=1) / total_neg
    sum_sen_spe = sensitivities + specifities
    return np.nansum(sum_sen_spe * particle_weights) / 2
  return _mean_sen_spe_utility


def group_utility(particle_weights,
                  particles,
                  groups,
                  group_sensitivities,
                  group_specificities,
                  utility_fun):
  """Compute the utility of a set of groups.

  This function computes the utility of a set of groups, given a distribution
  over the population status encoded as a weighted sum of Dirac measures on
  particles, the specificities and sensitivities of tests, and a utility
  function.

  Args:
   particle_weights: weights of particles
   particles: particles summarizing belief about infection status
   groups: set of groups to be tested
   group_sensitivities: sensitivies of test for each group
   group_specificities: specificities of test for each group
   utility_fun: a utility function that takes as input (particle_weights,
      particles) and output the utility of the distribution

  Returns:
   The expected utility (over the test results) of the posterior
  """
  num_groups = groups.shape[0]
  proba_y_is_one_given_x = (np.matmul(particles, np.transpose(groups))
                            * (group_sensitivities + group_specificities - 1)
                            + 1.0 - group_specificities)
  proba_y_is_one_given_x = np.expand_dims(proba_y_is_one_given_x, axis=2)
  test_res = np.array(list(itertools.product([0, 1], repeat=num_groups)))
  test_res = np.expand_dims(np.transpose(test_res), axis=0)
  proba_y_given_x = np.product(test_res * proba_y_is_one_given_x + (1-test_res)
                               * (1-proba_y_is_one_given_x), axis=1)
  proba_y_and_x = proba_y_given_x * np.expand_dims(particle_weights, 1)
  proba_y = np.sum(proba_y_and_x, axis=0)
  proba_x_given_y = proba_y_and_x / np.expand_dims(proba_y, 0)
  vutility_fun = jax.vmap(utility_fun, [1, None])
  utility_x_given_y = vutility_fun(proba_x_given_y, particles)
  return np.dot(proba_y, utility_x_given_y)


def next_best_group(particle_weights,
                    particles,
                    previous_groups,
                    cur_group,
                    sensitivity,
                    specificity,
                    utility_fun,
                    backtracking):
  """Performs greedy utility optimization to compute the next best group.

  Given a set of groups previous_groups, and a current candidate group
  cur_group, this function computes the utility of the combination of
  previous_groups and cur_group modified by adding (if backtracking = True) or
  adding (if backtracking = False) on element to cur_group, and returns the
  combination with largest utility.

  Args:
   particle_weights: weights of particles
   particles: particles summarizing belief about infection status
   previous_groups: groups already chosen
   cur_group: group that we wish to optimize
   sensitivity: value (vector) of sensitivity(-ies depending on group size).
   specificity: value (vector) of specificity(-ies depending on group size).
   utility_fun: function to compute the utility of a set of groups
   backtracking: (bool), True if removing rather than adding individuals.

  Returns:
   best_group : cur_group updated with best choice
   utility: utility of best_group
  """
  if backtracking:
    # Backward mode: test groups obtained by removing an item to cur_group
    candidate_groups = np.logical_not(
        mutual_information.add_ones_to_line(np.logical_not(cur_group)))
  else:
    # Forward mode: test groups obtained by adding an item to cur_group
    candidate_groups = mutual_information.add_ones_to_line(cur_group)
  n_candidates = candidate_groups.shape[0]

  # Combine past groups with candidate groups
  candidate_sets = np.concatenate(
      (np.repeat(previous_groups[:, :, np.newaxis], n_candidates, axis=2),
       np.expand_dims(np.transpose(candidate_groups), axis=0)),
      axis=0)

  # Compute utility of each candidate group
  group_sizes = np.sum(candidate_sets[:, :, 0], axis=1)
  group_sensitivities = utils.select_from_sizes(sensitivity, group_sizes)
  group_specificities = utils.select_from_sizes(specificity, group_sizes)
  group_util_fun = lambda x: group_utility(particle_weights, particles, x,
                                           group_sensitivities,
                                           group_specificities, utility_fun)
  mgroup_util_fun = jax.vmap(group_util_fun, in_axes=2)
  objectives = mgroup_util_fun(candidate_sets)

  # Greedy selection of largest value
  index = np.argmax(objectives)
  return (candidate_groups[index, :], objectives[index])


@gin.configurable
class BayesOED(group_selector.GroupSelector):
  """Uses generic Bayed OED to choose groups."""

  NEEDS_POSTERIOR = True

  def __init__(self,
               forward_iterations=1,
               backward_iterations=0,
               utility_fn=auc()):
    if forward_iterations <= backward_iterations:
      raise ValueError('Forward should be greater than backward.')
    super().__init__()
    self.forward_iterations = forward_iterations
    self.backward_iterations = backward_iterations
    self.utility_fn = utility_fn

  def get_groups(self, rng, state):
    """A greedy forward-backward algorithm to pick groups with large utility."""
    particle_weights, particles = mutual_information.collapse_particles(
        rng, state.particle_weights, state.particles)
    n_patients = particles.shape[1]
    iterations = [self.forward_iterations, self.backward_iterations]

    chosen_groups = np.empty((0, n_patients), dtype=bool)
    added_groups_counter = 0
    while added_groups_counter < state.extra_tests_needed:
      # start forming a new group, and improve it greedily
      proposed_group = np.zeros((n_patients,), dtype=bool)
      obj_old = -1
      while np.sum(proposed_group) < state.max_group_size:
        for steps, backtrack in zip(iterations, [False, True]):
          for _ in range(steps):
            # Extract candidate with largest utility
            proposed_group, obj_new = next_best_group(particle_weights,
                                                      particles,
                                                      chosen_groups,
                                                      proposed_group,
                                                      state.prior_sensitivity,
                                                      state.prior_specificity,
                                                      self.utility_fn,
                                                      backtracking=backtrack)
            if obj_new > obj_old + 1e-6:
              cur_group = proposed_group
              obj_old = obj_new
            else:
              break
      # stop adding, form next group
      chosen_groups = np.concatenate((chosen_groups, cur_group[np.newaxis, :]),
                                     axis=0)
      added_groups_counter += 1
    return chosen_groups
