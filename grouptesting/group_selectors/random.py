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
"""Defines several random group selection strategies."""

import itertools
from typing import Tuple

import gin
import jax
import jax.numpy as np
import numpy as onp
import scipy.special

from grouptesting.group_selectors import group_selector


@gin.configurable
class RandomSelector(group_selector.GroupSelector):
  """Selects groups randomly."""

  def get_groups(self, rng, state):
    shape = (state.extra_tests_needed, state.num_patients)
    threshold = state.max_group_size / state.num_patients
    return jax.random.uniform(rng, shape=shape) < threshold


@gin.configurable
class Mezard(group_selector.GroupSelector):
  """Selects groups randomly with predefined/adaptive-to-infection-rate size."""

  def __init__(self, group_size = None):
    super().__init__()
    self.group_size = group_size

  def get_groups(self, rng, state):
    """Produces random design matrix fixed number of 1s per line.

    Args:
     rng: np.ndarray<int>[2]: the random key.
     state: the current state.State of the system.

    Returns:
     A np.array<bool>[num_groups, patients].
    """
    if self.group_size is None:
      # if no size has been defined, we compute it adaptively
      # in the simple case where prior is uniform.
      if np.size(state.prior_infection_rate) == 1:
        group_size = np.ceil(
            (np.log(state.prior_sensitivity - .5) -
             np.log(state.prior_sensitivity + state.prior_specificity - 1)) /
            np.log(1 - state.prior_infection_rate))
        group_size = np.minimum(group_size, state.max_group_size)
      # if prior is not uniform, pick max size.
      else:
        group_size = self.max_group_size
    else:
      group_size = self.group_size
    group_size = int(np.squeeze(group_size))
    new_groups = np.empty((0, state.num_patients), dtype=bool)
    for _ in range(state.extra_tests_needed):
      rng, rng_shuffle = jax.random.split(rng, 2)
      vec = np.zeros((1, state.num_patients), dtype=bool)
      idx = jax.random.permutation(rng_shuffle, np.arange(state.num_patients))
      vec = jax.ops.index_update(vec, [0, idx[0:group_size]], True)
      new_groups = np.concatenate((new_groups, vec), axis=0)
    return new_groups


def macula_matrix(d, k, n):
  """Produces d-separable design matrix."""
  # https://core.ac.uk/download/pdf/82758506.pdf
  n_groups = int(scipy.special.comb(n, d))
  n_cols = int(scipy.special.comb(n, k))
  new_groups = np.zeros((n_groups, n_cols), dtype=bool)
  comb_groups = itertools.combinations(range(n), d)
  comb_cols = itertools.combinations(range(n), k)
  d_vec = np.zeros((n_groups, n), dtype=bool)
  k_vec = np.zeros((n_cols, n), dtype=bool)
  for i, comb_g in enumerate(comb_groups):
    d_vec[i, comb_g] = True
  for j, comb_c in enumerate(comb_cols):
    k_vec[j, comb_c] = True

  for i in range(n_groups):
    for j in range(n_cols):
      new_groups[i, j] = np.all(
          np.logical_or(np.logical_not(d_vec[i, :]), k_vec[j, :]))
  return new_groups


def sample_groups_of_size(shape, group_size):
  """Creates a matrix with k rows, of size n with g ones randomly placed."""
  random_groups = onp.zeros(shape, dtype=bool)
  num_tests, num_patients = shape
  for i in range(num_tests):
    random_subset = onp.random.choice(num_patients, group_size, replace=False)
    random_groups[i, random_subset] = True
  return random_groups


@gin.configurable
def count_mean(values_arr):
  return onp.mean(values_arr)


@gin.configurable
def count_min(values_arr):
  return onp.min(values_arr)


def eval_disjoint(mat, d, count_fn):
  """Evaluates how disjoint a matrix is based on count_fn."""
  num_rows, num_cols = mat.shape
  count = 0
  print('num_cols ', num_cols, ' d ', d)
  for s in itertools.combinations(range(num_cols), d):
    boolean_sum_subset = onp.sum(mat[:, s], axis=1) > 0
    boolean_sum_mat = onp.broadcast_to(boolean_sum_subset[:, None],
                                       (num_rows, num_cols - d))
    complement_subset_mat = onp.delete(mat, s, axis=1)
    ng_diag = onp.amax(complement_subset_mat > boolean_sum_mat, axis=0)
    count += count_fn(1 * ng_diag)
  return count


def is_disjoint(mat, d):
  """Checks that a matrix is d-disjoint."""
  num_rows, num_cols = mat.shape
  for s in itertools.combinations(range(num_cols), d):
    boolean_sum_subset = onp.sum(mat[:, s], axis=1) > 0
    boolean_sum_mat = onp.broadcast_to(boolean_sum_subset[:, None],
                                       (num_rows, num_cols - d))
    complement_subset_mat = onp.delete(mat, s, axis=1)
    ng_diag = onp.amax(complement_subset_mat > boolean_sum_mat, axis=0)
    if not min(ng_diag):
      return False
  return True


def sample_disjoint_matrix(num_cols,
                           num_rows,
                           n_max_test,
                           d,
                           max_tries=1e2):
  """Samples matrix without replacement until disjoint check passes."""
  attempt = 0
  while attempt < max_tries:
    attempt += 1
    groups = sample_groups_of_size((num_rows, num_cols), n_max_test)
    if is_disjoint(groups, d):
      return groups
  return None


def sample_maxeval_disjoint_matrix(num_cols,
                                   num_rows,
                                   n_max_test,
                                   d,
                                   max_tries=100,
                                   count_fn=count_mean):
  """Samples matrices, returns the most count-disjoint matrix."""
  random_groups = sample_groups_of_size((num_rows, num_cols), n_max_test)
  count = eval_disjoint(random_groups, d, count_fn=count_fn)
  max_count = int(scipy.special.comb(num_cols, d))
  attempt = 0
  while attempt < max_tries and count < max_count:
    attempt += 1
    random_groups_iter = sample_groups_of_size((num_rows, num_cols), n_max_test)
    count_iter = eval_disjoint(random_groups, d, count_fn=count_fn)
    print('attempt', attempt, 'count_iter', count_iter, 'count ', count)
    if count_iter > count:
      count = count_iter
      random_groups = random_groups_iter
      if count == max_count:
        return random_groups, count
  return random_groups, count


@gin.configurable
class RandomDisjoint(group_selector.GroupSelector):
  """Selects groups randomly with predefined size."""

  def __init__(self, max_iter=1, method='count', count_fn=count_mean):
    super().__init__()
    self.max_iter = max_iter
    self.method = method
    self.count_fn = count_fn

  def get_groups(self, state):
    """Produces random design matrix with nmax 1s per line.

    Args:
     state: the current state.State of the system.

    Returns:
     A np.array<bool>[num_groups, patients].
    """
    if self.method == 'single':
      new_groups = sample_groups_of_size(
          (state.num_patients, state.extra_tests_needed), state.max_group_size)
    # if prior_infection_rate is a scalar take average otherwise sum
    if np.size(self.prior_infection_rate) == 1:
      max_infected = int(
          np.ceil(self.prior_infection_rate * state.num_patients))
    elif np.size(self.prior_infection_rate) == state.num_patients:
      max_infected = int(np.sum(self.prior_infection_rate))

    if self.method == 'disjoint':
      new_groups = sample_disjoint_matrix(state.num_patients,
                                          state.extra_tests_needed,
                                          state.max_group_size, max_infected,
                                          self.max_iter)
      if new_groups is None:
        raise ValueError('No satisfying matrix found after max iterations')
    if self.method == 'count':
      new_groups, _ = sample_maxeval_disjoint_matrix(
          state.num_patients, state.extra_tests_needed, state.max_group_size,
          max_infected, self.max_iter, self.count_fn)

    return np.array(new_groups)
