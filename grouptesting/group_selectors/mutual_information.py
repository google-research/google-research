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
"""Defines a greedy selector based on maximizing the mutual information."""

from absl import logging
import gin
import jax
import jax.numpy as np
from grouptesting import metrics
from grouptesting import utils
from grouptesting.group_selectors import group_selector


def collapse_particles(rng, particle_weights, particles):
  """Collapses identical particles and recompute their weights."""
  n_particles, num_patients = particles.shape
  if n_particles < 2:
    return particle_weights, particles

  alpha = jax.random.normal(rng, shape=((1, num_patients)))
  random_projection = np.sum(particles * alpha, axis=-1)
  indices = np.argsort(random_projection)
  particles = particles[indices, :]
  particle_weights = particle_weights[indices]
  cumsum_particle_weights = np.cumsum(particle_weights)
  random_projection = random_projection[indices]
  indices_singles, = np.where(np.flip(np.diff(np.flip(random_projection)) != 0))
  indices_singles = np.append(indices_singles, n_particles - 1)
  new_weights = np.diff(cumsum_particle_weights[indices_singles])
  new_weights = np.append(
      np.array(cumsum_particle_weights[indices_singles[0]]), new_weights)
  new_particles = particles[indices_singles, :]
  return new_weights, new_particles


def joint_mi_criterion_mg(particle_weights,
                          particles,
                          cur_group,
                          cur_positives,
                          previous_groups_prob_particles_states,
                          previous_groups_cumcond_entropy,
                          sensitivity,
                          specificity,
                          backtracking):
  """Compares the benefit of adding one group to previously selected ones.

  Groups are formed iteratively by considering all possible individuals
  that can be considered to add (or remove if backtracking).

  If the sensitivity and/or specificity parameters are group size dependent,
  we take that into account in our optimization.
  Here all groups have the same size, hence they all share the same
  specificity / sensitivity setting. We just replace the vector by its value at
  the appropriate coordinate.

  The size of the group considered here will be the size of cur_group + 1 if
  going forward / -1 if backtracking.

  Args:
   particle_weights: weights of particles
   particles: particles summarizing belief about infection status
   cur_group: group currently considered to add to former groups.
   cur_positives: stores which particles would test positive w.r.t cur_group
   previous_groups_prob_particles_states: particles x test outcome probabilities
   previous_groups_cumcond_entropy: previous conditional entropies
   sensitivity: value (vector) of sensitivity(-ies depending on group size).
   specificity: value (vector) of specificity(-ies depending on group size).
   backtracking: (bool), True if removing rather than adding individuals.

  Returns:
    cur_group : group updated with best choice
    cur_positives : bool vector keeping trace of whether particles
                             would test or not positive
    new_objective : MI reached with this new group
    prob_particles_states : if cur_group were to be selected, this matrix
      would keep track of probability of seeing one of 2^j possible test
      outcomes across all particles.
    new_cond_entropy : if cur_group were to be selected, this constant would be
      added to store the conditional entropies of all tests carried out thusfar
  """
  group_size = np.atleast_1d(np.sum(cur_group) + 1 - 2 * backtracking)
  sensitivity = utils.select_from_sizes(sensitivity, group_size)
  specificity = utils.select_from_sizes(specificity, group_size)
  if backtracking:
    # if backtracking, we recompute the truth table for all proposed groups,
    # namely run the np.dot below
    # TODO(cuturi)? If we switch to integer arithmetic we may be able to
    # save on this iteration by keeping track of how many positives there
    # are, and not just on whether there is or not one positive.
    candidate_groups = np.logical_not(
        add_ones_to_line(np.logical_not(cur_group)))
    positive_in_groups = np.dot(candidate_groups, np.transpose(particles)) > 0
  else:
    # in forward mode, candidate groups are recovered by adding
    # a 1 instead of zeros. Therefore, we can use previous vector of positive
    # in groups to simply compute all positive in groups for candidates
    indices_of_false_in_cur_group, = np.where(np.logical_not(cur_group))
    positive_in_groups = np.logical_or(
        cur_positives[:, np.newaxis],
        particles[:, indices_of_false_in_cur_group])
    # recover a candidates x n_particles matrix
    positive_in_groups = np.transpose(positive_in_groups)

  entropy_spec = metrics.binary_entropy(specificity)
  gamma = metrics.binary_entropy(sensitivity) - entropy_spec
  cond_entropy = previous_groups_cumcond_entropy + entropy_spec + gamma * np.sum(
      particle_weights[np.newaxis, :] * positive_in_groups, axis=1)
  rho = specificity + sensitivity - 1

  # positive_in_groups defines probability of two possible outcomes for the test
  # of each new candidate group.
  probabilities_new_test = np.stack(
      (specificity - rho * positive_in_groups,
       1 - specificity + rho * positive_in_groups),
      axis=-1)
  # we now incorporate previous probability of all previous groups added so far
  # and expand x 2 the state space of possible test results.
  new_plus_previous_groups_prob_particles_states = np.concatenate(
      (probabilities_new_test[:, :, 0][:, :, np.newaxis] *
       previous_groups_prob_particles_states[np.newaxis, :, :],
       probabilities_new_test[:, :, 1][:, :, np.newaxis] *
       previous_groups_prob_particles_states[np.newaxis, :, :]),
      axis=2)

  # average over particles to recover probability of all 2^j possible
  # test results
  new_plus_previous_groups_prob_states = np.sum(
      particle_weights[np.newaxis, :, np.newaxis] *
      new_plus_previous_groups_prob_particles_states,
      axis=1)

  whole_entropy = metrics.entropy(
      new_plus_previous_groups_prob_states, axis=1)

  # exhaustive way to compute cond entropy, useful to check
  # computations.
  # cond_entropy_old = np.sum(
  #     particle_weights[np.newaxis, :] *
  #     entropy(new_plus_previous_groups_prob_particles_states, axis=2),
  #     axis=1)
  objectives = whole_entropy - cond_entropy

  # greedy selection of largest/smallest value
  index = np.argmax(objectives)

  if backtracking:
    # return most promising group by recovering it from the matrix directly
    logging.info('backtracking, candidate_groups size: %i',
                 candidate_groups.shape)
    cur_group = candidate_groups[index, :]

  else:
    # return most promising group by adding a 1
    cur_group = jax.ops.index_update(
        cur_group, indices_of_false_in_cur_group[index], True)

  # refresh the status of vector positives
  cur_positives = positive_in_groups[index, :]
  new_objective = objectives[index]
  prob_particles_states = new_plus_previous_groups_prob_particles_states[
      index, :, :]
  new_cond_entropy = cond_entropy[index]
  return (cur_group, cur_positives, new_objective,
          prob_particles_states, new_cond_entropy)


def add_ones_to_line(single_group):
  indices, = np.where(np.logical_not(single_group))
  k_zeros = len(indices)
  groups = (
      np.zeros((k_zeros, single_group.shape[0]), dtype=bool)
      + single_group[np.newaxis, :])
  groups = jax.ops.index_update(
      groups, jax.ops.index[np.arange(0, k_zeros), indices], True)
  return groups


@gin.configurable
class MaxMutualInformation(group_selector.GroupSelector):
  """Uses MI greedy optimization to choose groups."""

  NEEDS_POSTERIOR = True

  def __init__(self, forward_iterations=1, backward_iterations=0):
    if forward_iterations <= backward_iterations:
      raise ValueError('Forward should be greater than backward.')

    super().__init__()
    self.forward_iterations = forward_iterations
    self.backward_iterations = backward_iterations

  def get_groups(self, rng, state):
    """A greedy forward-backward algorithm to pick groups with large MI."""
    particle_weights, particles = collapse_particles(
        rng, state.particle_weights, state.particles)
    n_particles, n_patients = particles.shape

    previous_groups_prob_partstates = np.ones((n_particles, 1))
    previous_groups_cumcond_entropy = 0
    chosen_groups = np.empty((0, n_patients), dtype=bool)
    added_groups_counter = 0
    while added_groups_counter < state.extra_tests_needed:
      # start forming a new group, and evaluate all possible groups
      proposed_group = np.zeros((n_patients,), dtype=bool)
      positives_in_proposed_group = np.zeros((n_particles,), dtype=bool)
      proposed_group_size = 0
      obj_old = -1

      while proposed_group_size < state.max_group_size:
        iterations = [self.forward_iterations, self.backward_iterations]
        for steps, backtrack in zip(iterations, [False, True]):
          for _ in range(steps):
            # extract candidate with largest potential
            (proposed_group, positives_in_proposed_group, obj_new,
             proposed_group_prob_partstates,
             proposed_group_cumcond_entropy) = joint_mi_criterion_mg(
                 particle_weights,
                 particles,
                 proposed_group,
                 positives_in_proposed_group,
                 previous_groups_prob_partstates,
                 previous_groups_cumcond_entropy,
                 state.prior_sensitivity,
                 state.prior_specificity,
                 backtrack)

        if obj_new > obj_old + 1e-6:  # 1st update or minimal increment
          # accept update to cur_group and keep track of quantities that
          # might be used if this cur_group is added to the list of groups.
          cur_group = proposed_group
          cur_groups_cumcond_entropy = proposed_group_cumcond_entropy
          cur_groups_recorded_prob_partstates = proposed_group_prob_partstates
          # cur_positives = positives_in_proposed_group
          obj_old = obj_new
          proposed_group_size += 1
        else:
          break
      # stop adding, form next group
      previous_groups_prob_partstates = cur_groups_recorded_prob_partstates
      previous_groups_cumcond_entropy = cur_groups_cumcond_entropy
      chosen_groups = np.concatenate(
          (chosen_groups, cur_group[np.newaxis, :]), axis=0)
      added_groups_counter += 1
    return chosen_groups
