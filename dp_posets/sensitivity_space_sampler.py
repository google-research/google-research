# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Functions to sample from the sensitivity space of a poset (P,<).

This module introduces a sampler to draw a uniform point from the sensitivity
space described by a poset (P U {r}, <) where p < r for all p in P. Details for
the relevant algorithms can be found in
https://braintex.goog/project/65e5a9ff24c22900a89644ee.
"""

import numpy as np
import scipy


def get_maximal_element_index(order):
  """Returns index for maximal element in the poset specified by `order`.

  Args:
    order: Matrix where order[i,j] = 1 if poset elements p_i <= p_j and 0
      otherwise.

  Returns:
    int index for the maximal element with the minimum index.
  """
  # Check how many larger elements each element has
  counts = np.count_nonzero(order == 1, axis=1)
  maximals = np.where(counts == 1)[0]
  index_maximal = maximals[0]
  return index_maximal


def get_max_inf_bound(
    extended_subposet,
    target_element,
    order,
    ground_set,
):
  """Returns maximal infimum bound of `target_element` in `extended_subposet`.

  Given an extended subposet of a poset and an element, returns the largest
  element in `extended_subposet` that is smaller than `target_element`.
  `extended_subposet` is totally ordered, so there is at most one such element.

  Args:
    extended_subposet: List with elements from the ground set of the poset, in
      ascending order. This is a linear extension of a subset of the ground set.
    target_element: Element in the ground set of the poset.
    order: Matrix where order[i,j] = 1 if poset elements p_i <= p_j and 0
      otherwise.
    ground_set: List of elements in the ground set of the poset.

  Returns:
    The index of the largest element less than `target_element` in the linear
    extension `extended_subposet`. If no such element exists, returns -1, for
    insertion as the new largest element.
  """
  target_element_index = ground_set.index(target_element)
  for i in range(1, len(extended_subposet) + 1):
    element = extended_subposet[-i]
    element_index = ground_set.index(element)
    if order[element_index, target_element_index] > 0:
      return len(extended_subposet) - i
  return -1


def get_bipartition(ground_set, order):
  """Uniformly samples a bipartition of the poset of `ground_set` and `order`.

  Let (ground_set, order) be a poset. This method samples (extended) bipartition
  (partition_a, partition_b) of ground_set such that each subset is an ordered
  list of some elements in ground_set, forming a linear extension of order to
  the subset.

  Args:
    ground_set: List representing ground set of the poset.
    order: Matrix where order[i,j] = 1 if poset elements p_i <= p_j and 0
      otherwise.

  Returns:
    Tuple (partition_a, partition_b) representing a bipartition of the poset.
    Each subset is an ordered list extending `order`.
  """
  # Singleton base case
  if len(ground_set) == 1:
    coin = np.random.randint(2)
    if coin:
      return ground_set, []
    return [], ground_set

  # Sample smaller bipartition
  index_maximal = get_maximal_element_index(order)
  maximal_element = ground_set[index_maximal]
  new_set = ground_set[:index_maximal] + ground_set[index_maximal + 1 :]
  new_order = np.delete(
      np.delete(order, index_maximal, axis=0), index_maximal, axis=1
  )
  partition_a, partition_b = get_bipartition(new_set, new_order)

  # Build larger partition from smaller bipartition by inserting maximal element
  # in random valid position
  a_min_position = (
      get_max_inf_bound(partition_a, maximal_element, order, ground_set) + 1
  )
  b_min_position = (
      get_max_inf_bound(partition_b, maximal_element, order, ground_set) + 1
  )

  num_insertions_a = len(partition_a) + 1
  num_insertions_b = len(partition_b) + 1

  num_valid_insertions_a = num_insertions_a - a_min_position
  num_valid_insertions_b = num_insertions_b - b_min_position
  total_num_valid_insertions = num_valid_insertions_a + num_valid_insertions_b

  weight_a = num_valid_insertions_a / total_num_valid_insertions
  weight_b = num_valid_insertions_b / total_num_valid_insertions

  subset = np.random.choice(['a', 'b'], p=[weight_a, weight_b])
  if subset == 'a':
    maximal_position = np.random.choice(
        np.arange(a_min_position, num_insertions_a)
    )
    partition_a.insert(maximal_position, maximal_element)
  else:
    maximal_position = np.random.choice(
        np.arange(b_min_position, num_insertions_b)
    )
    partition_b.insert(maximal_position, maximal_element)
  return partition_a, partition_b


def get_filters_chain_from_extended_subposet(
    extended_subposet, ground_set, order
):
  """Returns a chain of filters as a list according to `extended_subposet`.

  Lemma 3.13 in the paper constructs a bijection between bipartitions (N+, N-)
  of a poset (ground_set, order) and non interfering pairs of chains of filters
  (C+, C-). This method recovers the chain C+ corresponding to a given subset
  C+, or C- corresponding to C- respectively, in time O(|ground_set|^2) using
  Lemma 3.13.

  Args:
    extended_subposet: List with elements from the ground set of the poset, in
      ascending order. This is a linear extension of a subset of the ground set.
    ground_set: List of all elements in the ground set of the poset.
    order: Matrix where order[i,j] = 1 if poset elements p_i <= p_j and 0
      otherwise.
  """
  ground_set = np.array(ground_set)
  filters = [[]]
  for i in range(1, len(extended_subposet) + 1):
    new_filter = filters[-1].copy()
    new_element = extended_subposet[-i]
    new_element_idx = np.where(ground_set == new_element)[0][0]
    ancestor_indices = np.where(order[new_element_idx, :] == 1)
    new_filter.extend(list(ground_set[ancestor_indices]))
    filters.append(list(set(new_filter)))
  return filters


def get_vertices_from_filters_chain(
    chain, ground_set
):
  """Gets the vertices of the simplex defined by the filters in `chain`.

  Args:
    chain: Chain of filters as a list.
    ground_set: List of all elements in the ground set of the poset.

  Returns:
    Matrix where each row represents a vertex of the simplex defined by the
    filters in `chain`.
  """
  vertices = np.zeros((len(chain), len(ground_set)))
  ground_set = np.array(ground_set)
  for i, f in enumerate(chain):
    indices = [np.where(ground_set == x)[0][0] for x in f]
    vertices[i, indices] = 1
  return vertices


def sample_point_from_simplex(d, vertices):
  """Returns point sampled uniformly from simplex given by vertices.

  Args:
    d: int for the dimension of the simplex.
    vertices: Matrix where each row represents a vertex of the simplex.
  """
  c = scipy.stats.dirichlet.rvs([1] * (d + 1))
  return np.sum(np.transpose(c) * vertices, axis=0)


def _get_vertices_from_extended_subposet(
    extended_subposet, ground_set, order
):
  """Returns the vertices of the simplex defined by the filters in `chain`.

  Args:
    extended_subposet: List with elements from the ground set of the poset, in
      ascending order. This is a linear extension of a subset of the ground set.
    ground_set: List of elements in the ground set of the poset, without the
      root.
    order: Matrix where order[i,j] = 1 if poset elements p_i <= p_j and 0
      otherwise, without the root.

  Returns:
    Matrix where each row represents a vertex of the simplex, with the root
    added back.
  """
  filters = get_filters_chain_from_extended_subposet(
      extended_subposet, ground_set, order
  )
  vertices = get_vertices_from_filters_chain(filters, ground_set)
  # Append Cayley sum (root) dimension.
  vertices = np.hstack([np.ones((vertices.shape[0], 1)), vertices])
  return vertices


def sample_poset_ball(ground_set, order):
  """Returns a uniform sample from the poset ball of (ground_set, order).

  The function assumes that ground_set and order omit the root (see Assumption
  2.8 in the paper). The random bipartition is sampled without the root, which
  is then added back inside _get_vertices_from_subset.

  Args:
    ground_set: List of elements in the ground set of the poset, without the
      root.
    order: Matrix where order[i,j] = 1 if poset elements p_i <= p_j and 0
      otherwise, without the root.
  """
  partition_a, partition_b = get_bipartition(ground_set, order)
  vertices_a = _get_vertices_from_extended_subposet(
      partition_a, ground_set, order
  )
  vertices_b = _get_vertices_from_extended_subposet(
      partition_b, ground_set, order
  )
  vertices_b = -vertices_b

  simplex_vertices = np.concatenate([vertices_a, vertices_b], axis=0)
  sample = sample_point_from_simplex(len(ground_set) + 1, simplex_vertices)
  return sample
