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

"""Tree method for computing multiple DP quantiles.

Code is modeled after the quantile trees implementation in this Java library:
https://github.com/google/differential-privacy/blob/main/java/main/com/google/privacy/differentialprivacy/BoundedQuantiles.java
The method is essentially using range trees to answer rank queries, as in the
mechanism presented in Section 7.2 of "Private and Continual Release of
Statistics" by Chan et al.: https://eprint.iacr.org/2010/076.pdf
"""

import collections
import enum
import numpy as np

# Smallest value difference that is considered significant.
_NUMERICAL_TOLERANCE = 1e-6

# Index of the root of the tree.
_ROOT_INDEX = 0

# Heuristic for filtering out empty nodes.  Suppose that the total sum of a
# node's noisy value and all of its siblings' noisy values is t.  Then, if the
# node's value is less than _ALPHA * t, it will be discarded, and a new sum t'
# will be computed excluding it.  Setting _ALPHA to zero implies no filtering.
_ALPHA = 0.005


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
      self._gen_noise = lambda: np.random.laplace(loc=0.0, scale=scale)
    elif noise_type == PrivateQuantileTree.NoiseType.GAUSSIAN:
      stdev = np.sqrt(2 * sensitivity * np.log(1.32 / delta)) / epsilon
      self._gen_noise = lambda: np.random.normal(loc=0.0, scale=stdev)
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
    return np.trunc((child_indices - 1) / self._branching_factor)

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
    noised_count = self._tree[index] + self._gen_noise()
    self._noised_tree[index] = noised_count
    return noised_count

  def compute_quantile(self, quantile):
    """Returns a differentially private estimate of the quantile.

    Args:
      quantile: A value in [0, 1].
    """
    # Ensure no data can be added once a quantile has been computed.
    self.finalize()

    if quantile < 0.0 or quantile > 1.0:
      raise ValueError(
          "Quantile must be in [0, 1]; requested quantile {}.".format(quantile))

    # Find the (approximate) index of the leaf node containing the quantile.
    index = _ROOT_INDEX
    while index < self._leftmost_leaf_index:
      leftmost_child_index = self.get_leftmost_child(index)
      rightmost_child_index = self.get_rightmost_child(index)

      # Sum all child nodes' noisy counts.
      noised_counts = np.asarray([
          self.get_noised_count(i)
          for i in range(leftmost_child_index, rightmost_child_index + 1)
      ])
      total = np.sum(noised_counts)

      # If all child nodes are "empty", return rank value of current subtree.
      if total <= 0.0:
        break

      # Sum again, but only noisy counts exceeding min_value_cutoff.
      min_value_cutoff = total * _ALPHA
      passes_cutoff = noised_counts >= min_value_cutoff
      filtered_counts = noised_counts[passes_cutoff]
      adjusted_total = np.sum(filtered_counts)
      if adjusted_total == 0.0:
        break

      # Find the child whose subtree contains the quantile.
      partial_count = 0.0
      for i in range(self._branching_factor):
        # Skip nodes whose contributions are too small.
        if passes_cutoff[i]:
          ith_count = noised_counts[i]
          partial_count += ith_count

          # Break if the current child's subtree contains the quantile.
          if partial_count / adjusted_total >= quantile - _NUMERICAL_TOLERANCE:
            quantile = (adjusted_total * quantile -
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
