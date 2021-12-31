# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Tree and AggTree methods for computing multiple DP quantiles.

Code is modeled after the quantile trees implementation in this Java library:
https://github.com/google/differential-privacy/blob/main/java/main/com/google/privacy/differentialprivacy/BoundedQuantiles.java
The method is essentially using range trees to answer rank queries, as in the
mechanism presented in Section 7.2 of "Private and Continual Release of
Statistics" by Chan et al.: https://eprint.iacr.org/2010/076.pdf.
The AggTree method further aggregates its noisy values to improve the accuracy
of its estimates as described in
"Efficient Use of Differentially Private Binary Trees" by Honaker:
http://hona.kr/papers/files/privatetrees.pdf.
"""

import collections
import enum
import numpy as np

# Smallest value difference that is considered significant.
_NUMERICAL_TOLERANCE = 1e-6

# Index of the root of the tree.
_ROOT_INDEX = 0


class PrivateQuantileTree:
  """Tree structure for computing DP quantiles."""

  class NoiseType(enum.Enum):
    LAPLACE = 1
    GAUSSIAN = 2

  def __init__(self,
               noise_type,
               epsilon,
               delta,
               data_low,
               data_high,
               swap,
               tree_height=4,
               branching_factor=16):
    """Initializes an empty tree and creates a noise generator.

    Leaf nodes of the tree can be thought of as bins that uniformly partition
    the [data_low, data_high] range.

    Args:
      noise_type: Sepecifies a value from the NoiseType enum.
      epsilon: Differential privacy parameter epsilon.
      delta: Differential privacy parameter delta.
      data_low: Smallest possible value for data points; any data points with
        smaller values will be clamped at data_low.
      data_high: Largest possible value for data points; any data points with
        larger values will be clamped at data_high.
      swap: If true, uses swap dp sensitivity, otherwise uses add-remove.
      tree_height: Depth of the tree structure.  Must be greater than or equal
        to one; height zero corresponds to a tree that is just a single node.
      branching_factor: Number of children of each internal tree node.  Must be
        at least two.
    Throws: ValueError if any input arg does not conform to the above
      specifications.
    """
    if data_low >= data_high:
      raise ValueError("Invalid data bounds [{}, {}]; data_low must be smaller "
                       "than data_high.".format(data_low, data_high))
    self._data_low = data_low
    self._data_high = data_high

    if tree_height < 1:
      raise ValueError(
          "Invalid value of {} for tree_height input; height must be at least"
          " 1.".format(tree_height))
    self._tree_height = tree_height

    if branching_factor < 2:
      raise ValueError("Invalid value of {} for branching_factor input; factor "
                       "must be at least 2.".format(branching_factor))
    self._branching_factor = branching_factor

    self._tree = collections.Counter()
    self._noised_tree = {}

    self._num_leaves = branching_factor**tree_height
    num_nodes = ((branching_factor**(tree_height + 1)) - 1) / (
        branching_factor - 1)
    self._leftmost_leaf_index = (int)(num_nodes - self._num_leaves)

    self._range = self._data_high - self._data_low

    self._finalized = False

    # Create noise generator function.
    # For sensitivity computations: We assume each user contributes one data
    # point, which means that each user contributes a count of one to one node
    # in each level of the tree.  L1 and L2 sensitivity are thus identical.
    scaling = 2 if swap else 1
    sensitivity = scaling * self._tree_height
    if noise_type == PrivateQuantileTree.NoiseType.LAPLACE:
      scale = sensitivity / epsilon
      self._gen_noise = lambda n: np.random.laplace(loc=0, scale=scale, size=n)
    elif noise_type == PrivateQuantileTree.NoiseType.GAUSSIAN:
      scale = np.sqrt(2 * np.log(1.32 / delta)) * sensitivity / epsilon
      self._gen_noise = lambda n: np.random.normal(loc=0, scale=scale, size=n)
    else:
      raise ValueError(
          "Invalid value of {} for noise_type input.".format(noise_type))

  def get_leaf_indices(self, values):
    """Returns the indices of the leaf node bins into which the values fall.

    Leaf nodes uniformly partition the [data_low, data_high] range.

    Args:
      values: Array of values, assumed to lie in [data_low, data_high].
    """
    range_fracs = (values - self._data_low) / self._range
    leaf_indices = np.trunc(range_fracs * self._num_leaves)
    high_values = values == self._data_high
    leaf_indices[high_values] -= 1
    return self._leftmost_leaf_index + leaf_indices

  def get_parents(self, child_indices):
    """Returns the indices of the parents of the child_indices nodes.

    Args:
      child_indices: Array of child indices.
    """
    return np.trunc((child_indices - 1) / self._branching_factor).astype(int)

  def add_data(self, data):
    """"Inserts data into the tree.

    Args:
      data: Array of data points.

    Raises:
      RuntimeError: If this method is called after tree is finalized.
    """
    if self._finalized:
      raise RuntimeError("Cannot add data once tree is finalized.")
    if data.size == 0:
      return

    clipped_data = np.clip(data, self._data_low, self._data_high)

    # Increment counts at leaf nodes and then iterate upwards, incrementing
    # counts at all ancestors on the path to the root (but not the root itself).
    indices = self.get_leaf_indices(clipped_data)
    indices, counts = np.unique(indices, return_counts=True)
    index_count_map = dict(zip(indices, counts))
    while indices[0] != _ROOT_INDEX:
      self._tree.update(index_count_map)

      new_indices = self.get_parents(indices)
      new_index_count_map = collections.Counter()
      for i in range(len(indices)):
        new_index_count_map[new_indices[i]] += index_count_map[indices[i]]
      indices = np.unique(new_indices)
      index_count_map = new_index_count_map
    return

  def finalize(self):
    """Disables calling add_data, and enables calling compute_quantile."""
    self._finalized = True
    return

  def get_leftmost_child(self, parent_index):
    """Returns the leftmost (lowest-numbered) child of the parent_index node.

    Args:
      parent_index: Index of the parent node.
    """
    return parent_index * self._branching_factor + 1

  def get_rightmost_child(self, parent_index):
    """Returns the rightmost (highest-numbered) child of the parent_index node.

    Args:
      parent_index: Index of the parent node.
    """
    return (parent_index + 1) * self._branching_factor

  def get_left_value(self, index):
    """Returns the minimum value that is mapped to index's subtree.

    Args:
      index: Index of a node in the tree.
    """
    # Find the smallest-index leaf node in this subtree.
    while index < self._leftmost_leaf_index:
      index = self.get_leftmost_child(index)
    return self._data_low + self._range * (
        index - self._leftmost_leaf_index) / self._num_leaves

  def get_right_value(self, index):
    """Returns the maximum value that is mapped to index's subtree.

    Args:
      index: Index of a node in the tree.
    """
    # Find the largest-index leaf node in this subtree.
    while index < self._leftmost_leaf_index:
      index = self.get_rightmost_child(index)
    return self._data_low + self._range * (index - self._leftmost_leaf_index +
                                           1) / self._num_leaves

  def get_noised_count(self, index):
    """Returns a noised version of the count for the given index.

    Note that if the count has previously been noised, the same value as before
    is returned.

    Args:
      index: Index of a node in the tree.
    """
    if index in self._noised_tree:
      return self._noised_tree[index]
    noised_count = self._tree[index] + self._gen_noise(1)
    self._noised_tree[index] = noised_count
    return noised_count

  def get_noised_counts(self, leftmost_child_index):
    """Returns noised version of counts at leftmost_child_index and siblings.

    Like get_noised_count, if the counts have previously been noised, the same
    value as before is returned.

    Args:
      leftmost_child_index: Index of a node in the tree, assumed to be the
        leftmost child of its parent.
    """
    return np.asarray([
        self.get_noised_count(i)
        for i in range(leftmost_child_index, leftmost_child_index +
                       self._branching_factor)
    ])

  def compute_quantile(self, quantile):
    """Returns a differentially private estimate of the quantile.

    Args:
      quantile: A value in [0, 1].
    """
    # Ensure no data can be added once a quantile has been computed.
    if not self._finalized:
      self.finalize()

    if quantile < 0.0 or quantile > 1.0:
      raise ValueError(
          "Quantile must be in [0, 1]; requested quantile {}.".format(quantile))

    # Find the (approximate) index of the leaf node containing the quantile.
    index = _ROOT_INDEX
    while index < self._leftmost_leaf_index:
      leftmost_child_index = self.get_leftmost_child(index)

      # Sum all child nodes' noisy counts.
      noised_counts = self.get_noised_counts(leftmost_child_index)
      total = np.sum(noised_counts)

      # If all child nodes are "empty", return rank value of current subtree.
      if total <= 0.0:
        break

      # Find the child whose subtree contains the quantile.
      partial_count = 0.0
      for i in range(self._branching_factor):
        ith_count = noised_counts[i]
        partial_count += ith_count

        # Break if the current child's subtree contains the quantile.
        if partial_count / total >= quantile - _NUMERICAL_TOLERANCE:
          quantile = (total * quantile -
                      (partial_count - ith_count)) / ith_count
          # Truncate at 1; calculated quantile may be larger than 1, due to
          # the subtraction of the numerical tolerance value above.
          quantile = min(quantile, 1.0)
          index = i + leftmost_child_index
          break

    # Linearly interpolate between the min and max values associated with the
    # node of the current index.
    return (1 - quantile) * self.get_left_value(
        index) + quantile * self.get_right_value(index)


class PrivateQuantileAggTree(PrivateQuantileTree):
  """Aggregated tree structure for computing DP quantiles."""

  def __init__(self,
               noise_type,
               epsilon,
               delta,
               data_low,
               data_high,
               swap,
               tree_height=4,
               branching_factor=16):
    """Initializes an empty aggregated tree and creates a noise generator.

    Leaf nodes of the tree can be thought of as bins that uniformly partition
    the [data_low, data_high] range.

    Args:
      noise_type: Sepecifies a value from the NoiseType enum.
      epsilon: Differential privacy parameter epsilon.
      delta: Differential privacy parameter delta.
      data_low: Smallest possible value for data points; any data points with
        smaller values will be clamped at data_low.
      data_high: Largest possible value for data points; any data points with
        larger values will be clamped at data_high.
      swap: If true, uses swap dp sensitivity, otherwise uses add-remove.
      tree_height: Depth of the tree structure.  Must be greater than or equal
        to one; height zero corresponds to a tree that is just a single node.
      branching_factor: Number of children of each internal tree node.  Must be
        at least two.
    Throws: ValueError if any input arg does not conform to the above
      specifications.
    """
    super(PrivateQuantileAggTree,
          self).__init__(noise_type, epsilon, delta, data_low, data_high, swap,
                         tree_height, branching_factor)
    self._data_size = 0
    self._num_nodes = (int)(
        ((branching_factor**(tree_height + 1)) - 1) / (branching_factor - 1))
    self._noised_tree = np.empty(self._num_nodes)
    # t_minus re-estimates each node's value using using lower nodes' values.
    self._t_minus = np.empty(self._num_nodes)
    # t_plus re-estimates each node's value using higher nodes' values.
    self._t_plus = np.empty(self._num_nodes)
    # t_star aggregates the node value estimates from t_minus and t_plus.
    self._t_star = np.empty(self._num_nodes)
    # sigma_minus is used to compute the correct weight per level when
    # aggregating from lower nodes to compute t_minus.
    self._sigma_minus = np.empty(tree_height + 1)
    # sigma_plus is used to compute the correct weight per level when
    # aggregating from upper nodes to compute t_plus.
    self._sigma_plus = np.empty(tree_height + 1)
    # sigma_star is used to compute the correct weight per level when
    # aggregating from t_minus and t_plus to compute t_star.
    self._sigma_star = np.empty(tree_height + 1)
    self._swap = swap
    scaling = 2 if swap else 1
    sensitivity = scaling * self._tree_height
    if noise_type == PrivateQuantileTree.NoiseType.LAPLACE:
      self._noise_stddev = np.sqrt(2) * sensitivity / epsilon
    elif noise_type == PrivateQuantileTree.NoiseType.GAUSSIAN:
      self._noise_stddev = np.sqrt(
          2 * np.log(1.32 / delta)) * sensitivity / epsilon

  def noise_counts(self):
    """Add noise to each node in tree."""
    self._noised_tree = self._gen_noise(self._num_nodes)
    self._noised_tree[[int(x) for x in list(self._tree.keys())
                      ]] += list(self._tree.values())
    return

  def sum_children_t_minus(self, index):
    """Sums the t_minus values of the children of the node at index.

    Args:
      index: Node whose child t_minus values will be summed.

    Returns:
      Sum of the t_minus values of the children of the node at index.
    """
    leftmost_child = self.get_leftmost_child(index)
    return np.sum(self._t_minus[leftmost_child:leftmost_child +
                                self._branching_factor])

  def get_parent(self, child_index):
    """Returns the index of the parent of the child_index node.

    Args:
      child_index: Index of child node.
    """
    return self.get_parents(np.asarray([child_index]))[0]

  def add_data(self, data):
    """"Inserts data into the tree.

    Args:
      data: Array of data points.

    Raises:
      RuntimeError: If this method is called after tree is finalized.
    """
    super(PrivateQuantileAggTree, self).add_data(data)
    self._data_size += data.size
    return

  def add_minuses(self):
    """Executes ``estimation from below'' to aggregate node values."""
    self._sigma_minus[self._tree_height] = self._noise_stddev
    leaf_indices = np.arange(self._leftmost_leaf_index, self._num_nodes)
    self._t_minus[leaf_indices] = self._noised_tree[leaf_indices]
    sigma_inv_sq = 1.0 / self._noise_stddev**2
    previous_leftmost = self._leftmost_leaf_index
    previous_rightmost = self._num_nodes - 1
    current_leftmost = self.get_parent(self._leftmost_leaf_index)
    current_rightmost = self.get_parent(self._num_nodes - 1)
    current_level = self._tree_height - 1
    while current_level >= 0:
      w = sigma_inv_sq / (
          sigma_inv_sq + (1.0 / (self._branching_factor *
                                 (self._sigma_minus[current_level + 1]**2))))
      self._sigma_minus[current_level] = self._noise_stddev * np.sqrt(w)
      indices_at_current_level = np.arange(current_leftmost,
                                           current_rightmost + 1)
      summed_child_values = np.sum(
          np.split(self._t_minus[previous_leftmost:previous_rightmost + 1],
                   len(indices_at_current_level)),
          axis=1)
      self._t_minus[indices_at_current_level] = w * self._noised_tree[
          indices_at_current_level] + (1 - w) * summed_child_values
      previous_leftmost = current_leftmost
      previous_rightmost = current_rightmost
      current_leftmost = self.get_parent(current_leftmost)
      current_rightmost = self.get_parent(current_rightmost)
      current_level -= 1
    return

  def add_pluses(self):
    """Executes ``estimation from above'' to aggregate node values."""
    if self._swap:
      self._t_plus[0] = self._data_size
      self._sigma_plus[0] = 0
    else:
      self._t_plus[0] = self._noised_tree[0]
      self._sigma_plus[0] = self._noise_stddev
    sigma_inv_sq = 1.0 / self._noise_stddev**2
    parent_t_pluses = np.repeat(self._t_plus[0], self._branching_factor)
    current_leftmost = 1
    current_rightmost = self._branching_factor
    current_level = 1
    while current_level <= self._tree_height:
      w = sigma_inv_sq / (
          sigma_inv_sq + (1.0 / (self._sigma_plus[current_level - 1]**2 +
                                 (self._branching_factor - 1) *
                                 (self._sigma_minus[current_level]**2))))
      self._sigma_plus[current_level] = self._noise_stddev * np.sqrt(w)
      indices_at_current_level = np.arange(current_leftmost,
                                           current_rightmost + 1)
      sibling_t_minus_sums = np.repeat(
          np.sum(
              np.split(self._t_minus[indices_at_current_level],
                       len(indices_at_current_level) / self._branching_factor),
              axis=1), self._branching_factor)
      self._t_plus[indices_at_current_level] = (
          w * self._noised_tree[indices_at_current_level]) + (1 - w) * (
              parent_t_pluses - sibling_t_minus_sums +
              self._t_minus[indices_at_current_level])
      parent_t_pluses = np.repeat(indices_at_current_level,
                                  self._branching_factor)
      current_leftmost = self.get_leftmost_child(current_leftmost)
      current_rightmost = self.get_rightmost_child(current_rightmost)
      current_level += 1
    return

  def add_stars(self):
    """Executes final aggregation of node values from both above and below."""
    parent_t_pluses = np.repeat(self._t_plus[0], self._branching_factor)
    current_leftmost = 1
    current_rightmost = self._branching_factor
    current_level = 1
    while current_level <= self._tree_height:
      sigma_minus_inv_sq = 1.0 / self._sigma_minus[current_level]**2
      w = sigma_minus_inv_sq / (
          sigma_minus_inv_sq + (1.0 / (self._sigma_plus[current_level - 1]**2 +
                                       (self._branching_factor - 1) *
                                       (self._sigma_minus[current_level]**2))))
      self._sigma_star[
          current_level] = self._sigma_minus[current_level] * np.sqrt(w)
      indices_at_current_level = np.arange(current_leftmost,
                                           current_rightmost + 1)
      sibling_t_minus_sums = np.repeat(
          np.sum(
              np.split(self._t_minus[indices_at_current_level],
                       len(indices_at_current_level) / self._branching_factor),
              axis=1), self._branching_factor)
      parent_t_pluses = self._t_plus[self.get_parents(indices_at_current_level)]
      self._t_star[indices_at_current_level] = (
          w * self._t_minus[indices_at_current_level]) + (1 - w) * (
              parent_t_pluses - sibling_t_minus_sums +
              self._t_minus[indices_at_current_level])
      parent_t_pluses = np.repeat(indices_at_current_level,
                                  self._branching_factor)
      current_leftmost = self.get_leftmost_child(current_leftmost)
      current_rightmost = self.get_rightmost_child(current_rightmost)
      current_level += 1
    return

  def get_noised_counts(self, leftmost_child_index):
    """Returns t_star values at leftmost_child_index and siblings.

    Args:
      leftmost_child_index: Index of a node in the tree, assumed to be the
        leftmost child of its parent.
    """
    return self._t_star[leftmost_child_index:leftmost_child_index +
                        self._branching_factor]

  def finalize(self):
    """Disables calling add_data, and enables calling compute_quantile."""
    self.noise_counts()
    self.add_minuses()
    self.add_pluses()
    self.add_stars()
    self._finalized = True
    return


def tree(sampled_data,
         data_low,
         data_high,
         qs,
         eps,
         delta,
         swap,
         tree_height=4,
         branching_factor=16):
  """Computes (eps, delta)-differentially private quantile estimates for qs.

    Creates a PrivateQuantileTree with Laplace noise when delta is zero, and
    Gaussian noise otherwise.

  Args:
    sampled_data: Array of data points.
    data_low: Lower bound for data.
    data_high: Upper bound for data.
    qs: Increasing array of quantiles in (0,1).
    eps: Privacy parameter epsilon.
    delta: Privacy parameter delta.
    swap: If true, uses swap dp sensitivity, otherwise uses add-remove.
    tree_height: Height for PrivateQuantileTree.
    branching_factor: Branching factor for PrivateQuantileTree.

  Returns:
    Array o where o[i] is the quantile estimate corresponding to quantile q[i].
  """
  noise_type = (
      PrivateQuantileTree.NoiseType.LAPLACE
      if delta == 0 else PrivateQuantileTree.NoiseType.GAUSSIAN)

  t = PrivateQuantileTree(
      noise_type=noise_type,
      epsilon=eps,
      delta=delta,
      data_low=data_low,
      data_high=data_high,
      swap=swap,
      tree_height=tree_height,
      branching_factor=branching_factor)
  t.add_data(sampled_data)

  results = np.empty(len(qs))
  for i in range(len(qs)):
    results[i] = t.compute_quantile(qs[i])
  return results


def agg_tree(sampled_data, data_low, data_high, qs, eps, delta, swap,
             tree_height, branching_factor):
  """Computes (eps, delta)-DP quantile estimates for qs using aggregated counts.

    Creates a PrivateQuantileAggTree with Laplace noise when delta is zero, and
    Gaussian noise otherwise.

  Args:
    sampled_data: Array of data points.
    data_low: Lower bound for data.
    data_high: Upper bound for data.
    qs: Increasing array of quantiles in (0,1).
    eps: Privacy parameter epsilon.
    delta: Privacy parameter delta.
    swap: If true, uses swap dp sensitivity, otherwise uses add-remove.
    tree_height: Height for PrivateQuantileTree.
    branching_factor: Branching factor for PrivatQuantileTree,

  Returns:
    Array o where o[i] is the quantile estimate corresponding to quantile q[i].
  """
  noise_type = (
      PrivateQuantileTree.NoiseType.LAPLACE
      if delta == 0 else PrivateQuantileTree.NoiseType.GAUSSIAN)

  t = PrivateQuantileAggTree(
      noise_type=noise_type,
      epsilon=eps,
      delta=delta,
      data_low=data_low,
      data_high=data_high,
      tree_height=tree_height,
      branching_factor=branching_factor,
      swap=swap)
  t.add_data(sampled_data)
  t.finalize()

  results = np.empty(len(qs))
  for i in range(len(qs)):
    results[i] = t.compute_quantile(qs[i])
  return results
