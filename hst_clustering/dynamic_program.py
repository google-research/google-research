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

"""Library to generate an HST to use for finding centers in a kmed problem."""

from __future__ import annotations

from typing import Callable, Optional, List

import numpy as np
import pandas as pd

from hst_clustering import dp_one_median as dp_med
from hst_clustering import kmedian_plus_plus as kmed


class Node:
  """Encodes a node in the tree."""

  def __init__(self, weight, diameter, node_id):
    self.weight = weight  # Number of points in the cell
    self.diameter = diameter  # Diameter of the cell
    self.node_id = node_id
    self.right_child = None
    self.left_child = None
    self.cost_matrix = []
    self.right_optimal = []
    self.left_optimal = []
    self.is_root = False

  def mark_as_root(self):
    self.is_root = True

  def is_root(self):
    return self.is_root

  def is_leaf(self):
    return self.left_child is None and self.right_child is None

  def is_valid(self):
    return True

  def initialize_cost_matrix_and_facilities(self, k):
    self.cost_matrix = np.zeros(k + 1)
    self.right_optimal = np.zeros(k + 1, dtype=int)
    self.left_optimal = np.zeros(k + 1, dtype=int)

  def set_right_child(self, node):
    self.right_child = node

  def set_left_child(self, node):
    self.left_child = node

  def __str__(self):
    right_child_id = -1
    left_child_id = -1
    if self.right_child is not None:
      right_child_id = self.right_child.node_id
    if self.left_child is not None:
      left_child_id = self.left_child.node_id
    cost_matrix = ",".join([str(c) for c in self.cost_matrix])
    right_optimal = ",".join([str(o) for o in self.right_optimal])
    left_optimal = ",".join([str(o) for o in self.left_optimal])
    return f"""Node id {self.node_id}
    right_child {right_child_id}
    left_child {left_child_id}
    weight {self.weight}
    diameter {self.diameter}
    cost_matrix {cost_matrix}
    right_optimal {right_optimal}
    left_optimal {left_optimal}
    """


class HST:
  """An HST is a collection of nodes.

  The HST class solves the k median optimization problem and outputs The
  optimal leafs and maps them to the original space.
  """

  def __init__(self):
    self.nodes = {}
    self.k = -1  # This is the k for k median
    self.root_id = None

  def add_node(self, node_id, node):
    if node_id in self.nodes.keys():
      raise IndexError("Node id %d already in tree" % node_id)
    self.nodes[node_id] = node

  def add_root(self, root_id, node):
    if root_id in self.nodes.keys():
      raise IndexError("Tree already has root")
    self.nodes[root_id] = node
    self.root_id = root_id
    node.is_root = True

  def get_node(self, node_id):
    if node_id not in self.nodes.keys():
      raise IndexError("Node id %s not in tree" % node_id)
    return self.nodes[node_id]

  def _validate_tree(self, base_node):
    """Checks whether the tree nodes inserted form a binary tree.

    Args:
      base_node: Root of subtree to validate.

    Returns:
      False if the tree is not valid (for instance if a node has no parent).
      True and all nodes already visited by this function.
    """
    if base_node.node_id not in self.nodes:
      print("Base node id %s not in nodes" % base_node.node_id)
      return False, []
    if not base_node.is_valid():
      print("Base node %s not valid" % base_node.node_id)
      return False, []
    if base_node.is_leaf():
      return True, [base_node.node_id]
    if (base_node.right_child is not None and
        base_node.right_child.node_id not in self.nodes.keys()):
      print(
          "Right child id %s not in nodes" % base_node.right_child.node_id
          + str(base_node)
      )
      return False, []
    if (base_node.left_child is not None and
        base_node.left_child.node_id not in self.nodes.keys()):
      print(
          "Left child id %s not in nodes" % base_node.left_child.node_id
          + str(base_node)
      )
      return False, []

    right_child = base_node.right_child
    left_child = base_node.left_child
    right_valid = True
    left_valid = True
    right_visited_nodes = []
    left_visited_nodes = []
    if right_child is not None:
      right_valid, right_visited_nodes = self._validate_tree(right_child)
    if not right_valid:
      print("At node %s right tree is invalid" % base_node.node_id)
      return False, []

    if left_child is not None:
      left_valid, left_visited_nodes = self._validate_tree(left_child)
    if not left_valid:
      print("At node %s left tree is invalid" % base_node.node_id)
      return False, []
    visited_nodes = ([base_node.node_id] + right_visited_nodes +
                     left_visited_nodes)
    return True, visited_nodes

  def validate_tree(self, remove_unreachable_nodes):
    """Validates the tree from the root.

    Args:
      remove_unreachable_nodes: Boolean if True, removes all nodes that have no
        parents from the tree.

    Returns:
      True or False depending on whether the tree is valid. It also returns
      the number of reachable nodes.
    """
    if self.root_id not in self.nodes:
      print("Tree does not have a root")
      return False
    valid, visited_nodes = self._validate_tree(self.nodes[self.root_id])
    print("Finished tree traversal")
    visited_nodes_set = set(visited_nodes)
    if len(visited_nodes) != len(self.nodes):
      if not remove_unreachable_nodes:
        print("There exist unreachable nodes. Not removing them. Use " +
              "remove_unreachable_nodes to remove.")
        return False
      else:
        removed_nodes = [
            node_id
            for node_id in self.nodes
            if node_id not in visited_nodes_set
        ]
        self.nodes = {
            node_id: node
            for node_id, node in self.nodes.items()
            if node_id in visited_nodes_set
        }
        print("Removed %d unreachable nodes" % len(removed_nodes))
    return valid, len(visited_nodes)

  def initialize_dp(self, k):
    """Initializes the dynamic program.

    Args:
      k: Number of centers to return.
    """
    for node in self.nodes.values():
      node.initialize_cost_matrix_and_facilities(k)
    self.k = k

  def _solve_dp(self, base_node, exp):
    """Solves dynamic program up to base node.

    Args:
      base_node: Root node up to which we solve the DP problem.
      exp: The exponent to which every distance is raised in the cost formula.
    """
    if not base_node.is_leaf():
      # Solve the problem for all children
      if base_node.right_child is not None:
        self._solve_dp(base_node.right_child, exp)
      if base_node.left_child is not None:
        self._solve_dp(base_node.left_child, exp)

    for kprime in range(self.k + 1):
      if kprime == 0:
        base_node.cost_matrix[
            kprime] = 2 * (base_node.diameter ** exp) * base_node.weight
        continue
      if base_node.is_leaf():
        # This is the terminal condition. If we hit a leaf we simply
        # assign the cost to be the diameter times the number of points in the
        # leaf.
        base_node.cost_matrix[
            kprime] = (base_node.diameter ** exp) * base_node.weight
        base_node.left_optimal[kprime] = -1
        base_node.right_optimal[kprime] = -1
        continue
      right_child = base_node.right_child
      left_child = base_node.left_child
      opt = 2**32
      opt_k1 = -1
      opt_k2 = -1
      # Find the optimal decomposition
      for k1 in range(kprime + 1):
        k2 = kprime - k1
        right_cost = 2**32 if k1 > 0 else 0
        left_cost = 2**32 if k2 > 0 else 0
        if right_child is not None:
          right_cost = right_child.cost_matrix[k1]
        if left_child is not None:
          left_cost = left_child.cost_matrix[k2]
        cost = right_cost + left_cost
        if cost < opt:
          opt_k1 = k1
          opt_k2 = k2
          opt = cost
      if opt_k1 + opt_k2 != kprime:
        raise AttributeError(
            "Error in DP at node %d optimal decomposition doesn't add up" %
            base_node.node_id)
      base_node.cost_matrix[kprime] = opt
      base_node.right_optimal[kprime] = int(opt_k1)
      base_node.left_optimal[kprime] = int(opt_k2)

  def solve_dp(self, k, exp=1.0):
    """Solve dyinamic program on the whole tree.

    Args:
      k: Number of centers.
      exp: The exponent to which every distance is raised in the cost formula.
    Returns:
      Tree cost of the optimal solution.
    """
    self.initialize_dp(k)
    if not self.validate_tree(False):
      raise AttributeError("Invalid tree cannot run dynamic program")
    self._solve_dp(self.nodes[self.root_id], exp=exp)

    return self.nodes[self.root_id].cost_matrix[self.k]

  def _recover_solution(self, base_node, k):
    """Finds the optimal centers in the tree.

    Must have called solve_dp before calling this function.

    Args:
      base_node: The root for the subtree over which we want to find the
        solution.
      k: Number of centers in the optimal solution.

    Returns:
      Optimal centers.
    """
    if base_node is None:
      print("None node passed to recover solution")
      return []
    if k == 0:
      return []
    if base_node.is_leaf():
      return [base_node.node_id]
    optimal_left = base_node.left_optimal[k]
    optimal_right = base_node.right_optimal[k]
    left_solution = self._recover_solution(base_node.left_child, optimal_left)
    right_solution = self._recover_solution(base_node.right_child,
                                            optimal_right)

    return left_solution + right_solution

  def recover_dp_solution(self):
    """Recovers the optimal centers for the full tree."""
    return self._recover_solution(self.nodes[self.root_id], self.k)


def euclidean(x, y):
  """Euclidean distance between two points.

  Args:
    x: One point in R^d.
    y: One point in R^d.

  Returns:
    Euclidean distance between x and y.
  """
  return np.sqrt(np.sum((x - y) ** 2))


def l2(x, y):
  """Quadratic distance between two points.

  Args:
    x: One point.
    y: One point.

  Returns:
    Quadratic distance between x and y.
  """
  return np.sum(((x - y) ** 2))


def find_center(
    x,
    centers,
    distance,
):
  """Finds closest center to x.

  Args:
    x: Point.
    centers: 2d array with one row per center.
    distance: Distance function.

  Returns:
    Row index of the closest center to x.
  """
  return np.argmin(
      [distance(x, centers[i, :]) for i in range(centers.shape[0])])


def eval_clustering(data, clusters, exp=1):
  """Evaluate a clustering.

  Args:
    data: Data to run the evaluation on.
    clusters: Centers as 2d array one center per row.
    exp: The exponent to which every distance is raised in the cost formula.

  Returns:
    The sum of distances from each point to its closest center.
  """
  n = data.shape[0]
  k = clusters.shape[0]
  print("Building distance array of size %d %d " % (n, k))
  distances = kmed.find_distances(data, clusters)
  return np.sum(np.power(distances, exp))


def _get_center(data, node_id, feature_columns):
  """Finds a center in the original space.

  Args:
    data: Corresponds to the centers of cells built by the HST pipeline. It is a
      pandas data frame with a column called id. This column must match the
      node_ids used to build the tree. The algorithm expects the id column to be
      binary strings.
    node_id: The node id of the center we want to recover.
    feature_columns: Columns in data that correspond to the encoding of the
      features.

  Returns:
    A Numpy array representing the center in the original space.
  """
  if node_id in data.id.values:
    return data[data.id == node_id][feature_columns].values
  else:
    return _get_center(data, node_id[:-1], feature_columns)


def _get_centers(
    data, node_ids, feature_columns
):
  centers = []
  for node in node_ids:
    centers.append(_get_center(data, node, feature_columns))
  return np.vstack(centers)


def get_centers_from_hst(
    data, hst, feature_columns
):
  solution = hst.recover_dp_solution()
  return _get_centers(data, solution, feature_columns)


def eval_hst_solution(
    raw_data,
    data,
    hst,
    feature_columns,
    lloyd_iters = 0,
    privacy_params = None,
    exp = 1.0,
):
  """Evaluates the HST algorithm.

  Important: The privacy params object passed is used to apply Lloyd's algorithm
  privately. Each iteration uses the privacy params object. The final epsilon
  guarantee needs to be properly accounted by the user a
  lloyd_iters * privacy_params.epsilon.

  Args:
    raw_data: Original dataset we want to cluster.
    data: Pandas data frame that represents the cell centers of the HST. Input
      is expected to have a column called id. The format of the id should be a
      binary string. All other columns should represent the center in R^d: one
      column per dimension.
    hst: An HST object.
    feature_columns: Array of names for the columns that represent the features
      of the cell centers in data.
    lloyd_iters: If positive, it will run this number of lloyd iterations after
      running the HST algorithm.
    privacy_params: A PrivacyParams object (see dp_one_median.py)
    exp: The exponent to which every distance is raised in the cost formula.

  Returns:
    The kmedian objective of the solution returned by the algorithm.
  """

  centers = get_centers_from_hst(data, hst, feature_columns)
  if lloyd_iters > 0:
    print("Before lloyd obj %f\n " % eval_clustering(raw_data, centers))
    centers, _ = kmed.lloyd_iter(
        raw_data,
        centers,
        iters=lloyd_iters,
        weights=None,
        max_num_points=20000,
        privacy_params=privacy_params)
  return eval_clustering(raw_data, centers, exp=exp), centers


def private_centers(
    data, norm, epsilon, delta
):
  """Finds the private average of data.

  Uses the Gaussian mechanism to find the average.

  Args:
    data: Data to calculate average. One row per data point.
    norm: Max norm of a point in the dataset.
    epsilon: Epsilon of differential privacy.
    delta: Delta of differential privacy.

  Returns:
    Private average of data.
  """
  sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
  n, d = data.shape
  return np.mean(data, 0) + norm / n * np.random.normal(0, sigma, d)


def private_center_recovery(
    raw_data,
    cluster_assignment,
    norm,
    epsilon,
    delta,
):
  """Recovers the centers (averages) of a cluster assignment.

  Args:
    raw_data: Data we want to cluster.
    cluster_assignment: array of cluster ids. cluster_assignment[i] Corresponds
      to the cluster id of raw_data[i,:].
    norm: Max norm of points in raw_data.
    epsilon: Epsilon of DP.
    delta: Delta of DP.

  Returns:
    2d np.array of centers. One center per row.
  """

  cluster_ids = np.unique(cluster_assignment)
  _, d = raw_data.shape
  centers = []
  for _ in range(max(cluster_ids) + 1):
    centers.append(np.zeros(d))

  for c_id in cluster_ids:
    points = raw_data[c_id == cluster_assignment]
    if np.size(points) == 0:
      centers[c_id] = np.random.uniform(-norm, norm)
    centers[c_id] = private_centers(points, norm, epsilon, delta)
  return np.vstack(centers)


def eval_tree_with_projection(
    raw_data,
    projected_data,
    data_frame,
    hst,
    feature_columns,
    norm,
    epsilon,
    delta,
    exp = 1.0,
):
  """Evaluates the quality of a HST when data has been projected down.

  Evaluation is done on the original dimension of the data.

  Args:
    raw_data: Data in the original dimension
    projected_data: Data projected down to a lower dimension
    data_frame: Data frame encoding the centers of the HST
    hst: HST object with a dp_solution found.
    feature_columns: Name of the feature columns in data frame
    norm: Max norm of the data
    epsilon: Epsilon of differential privacy
    delta: Delta of differential privacy
    exp: The exponent to which every distance is raised in the cost formula.

  Returns:
    The k-median cost (in the original high dimensional space) of the solution
    of the HST.
  """
  centers = get_centers_from_hst(data_frame, hst, feature_columns)
  assignment = np.array(
      [
          find_center(projected_data[i, :], centers, euclidean)
          for i in range(projected_data.shape[0])
      ],
      dtype=np.int32,
  )
  private_centers_array = private_center_recovery(raw_data, assignment, norm,
                                                  epsilon, delta)
  return (eval_clustering(raw_data, private_centers_array,
                          exp=exp), private_centers_array)
