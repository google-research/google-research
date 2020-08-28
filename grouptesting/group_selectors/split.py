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
"""Defines several split group selectors."""

from absl import logging
import gin
import jax.numpy as np
import numpy as onp

from grouptesting.group_selectors import group_selector


@gin.configurable
class SplitSelector(group_selector.GroupSelector):
  """Split the patients into sub groups."""

  def __init__(self, split_factor=None):
    super().__init__()
    self.split_factor = split_factor

  def get_groups(self, rng, state):
    if self.split_factor is None:
      # if no factor is given by default, we use prior infection rate.
      if np.size(state.prior_infection_rate) > 1:
        raise ValueError(
            'Dorfman Splitting cannot be used with individual infection rates.'+
            ' Consider using Informative Dorfman instead.')

      # set group size to value defined by Dorfman testing
      group_size = 1 + np.ceil(1 / np.sqrt(np.squeeze(
          state.prior_infection_rate)))
      # adjust to take into account testing limits
      group_size = min(group_size, state.max_group_size)
      split_factor = -(-state.num_patients // group_size)
    else:
      # ensure the split factor does not produce groups that are too large
      min_splits = -(-state.num_patients // state.max_group_size)
      split_factor = np.maximum(self.split_factor, min_splits)

    indices = onp.array_split(np.arange(state.num_patients), split_factor)
    new_groups = onp.zeros((len(indices), state.num_patients))
    for i in range(len(indices)):
      new_groups[i, indices[i]] = True
    return np.array(new_groups, dtype=bool)

  def __call__(self, rng, state):
    new_groups = self.get_groups(rng, state)
    state.add_groups_to_test(
        new_groups, results_need_clearing=True)
    return state


@gin.configurable
class SplitPositive(group_selector.GroupSelector):
  """First looks for previous groups that were tested positive and not cleared.

  Select and split them using split_factor.
  """

  def __init__(self, split_factor=2):
    super().__init__()
    self.split_factor = split_factor

  def _split_groups(self, groups):
    """Splits the groups."""
    # if split_factor is None, we do exhaustive split,
    # i.e. we test everyone as in Dorfman groups
    use_split_factor = self.split_factor
    # make sure this is a matrix
    groups = onp.atleast_2d(groups)
    n_groups, n_patients = groups.shape

    # we form new groups one by one now. initialize matrix first
    new_groups = onp.empty((0, n_patients), dtype=bool)
    for i in range(n_groups):
      group_i = groups[i, :]
      # test if there is one individual to test
      if np.sum(group_i) > 1:
        indices, = np.where(group_i)
        if self.split_factor is None:
          use_split_factor = np.size(indices)
        indices = onp.array_split(indices, use_split_factor)
        newg = onp.zeros((len(indices), n_patients))
        for j in range(len(indices)):
          newg[j, indices[j]] = 1
      new_groups = onp.concatenate((new_groups, newg), axis=0)
    return np.array(new_groups, dtype=bool)

  def get_groups(self, rng, state):
    if np.size(state.past_groups) > 0 and np.size(state.to_clear_positives) > 0:
      to_split = state.past_groups[state.to_clear_positives, :]
      # we can only split groups that have more than 1 individual
      to_split = to_split[np.sum(to_split, axis=-1) > 1, :]
      if np.size(to_split) > 0:
        if np.ndim(to_split) == 1:
          to_split = onp.expand_dims(to_split, axis=0)
        # each group indexed by indices will be split in split_factor terms
        return self._split_groups(to_split)
      else:
        logging.info('only singletons')

  def __call__(self, rng, state):
    new_groups = self.get_groups(rng, state)

    if new_groups is not None:
      state.add_groups_to_test(new_groups,
                               results_need_clearing=True)
      state.update_to_clear_positives()

    else:
      state.all_cleared = True
    return state
