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

"""User-friendly class to generate stochastic block model graphs with features.

A stochastic block model is a random graph model that defines edge probabilities
based on "blocks" to which nodes are assigned. The set of blocks is a disjoint
and exhaustive partition of the nodes. The likelihood of this model is often
used in algorithms to estimate community structure in graphs. The model is also
used as a generative model for producing graphs with community structure. This
class is for the latter use-case, and uses graph_tool.generation.generate_sbm
under-the-hood, but has an interface with a more standard set of parameters. The
implementation of the package simulator is optimized, but still can be slow for
graphs with a large number of edges.

This class also includes functionality to generate node features with varying
correlation to the blocks, which is useful for benchmarking graph convolutional
neural networks (see https://arxiv.org/abs/2006.16904).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import enum
import math
import random
import typing
from graph_tool import generation
from graph_tool import Graph
import numpy as np

# Types
Any = typing.Any
Dict = typing.Dict
Enum = enum.Enum
List = typing.List
Set = typing.Set
Text = typing.Text


class MatchType(Enum):
  """Indicates type of feature/graph membership matching to do.

    RANDOM: feature memberships are generated randomly.
    NESTED: for # feature groups >= # graph groups. Each feature cluster is a
      sub-cluster of a graph cluster. Multiplicity of sub-clusters per
      graph cluster is kept as uniform as possible.
    GROUPED: for # feature groups <= # graph groups. Each graph cluster is a
      sub-cluster of a feature cluster. Multiplicity of sub-clusters per
      feature cluster is kept as uniform as possible.
  """
  RANDOM = 1
  NESTED = 2
  GROUPED = 3


# TODO(palowitch): ensure feature generators are not called before making graph
class SbmSimulator(object):
  """Generates stochastic block model graphs."""

  def _GetNestingMap(self, large_k, small_k):
    """Given two group sizes, computes a "nesting map" between groups.

    This function will produce a bipartite map between two sets of "group nodes"
    that will be used downstream to partition nodes in a bigger graph. The map
    encodes which groups from the larger set are nested in certain groups from
    the smaller set.

    As currently implemented, nesting is assigned as evenly as possible. If
    large_k is an integer multiple of small_k, each smaller-set group will be
    mapped to exactly (large_k/small_k) larger-set groups. If there is a
    remainder r, the first r smaller-set groups will each have one extra nested
    larger-set group.


    Arguments:
      large_k: (int) size of the larger group set
      small_k: (int) size of the smaller group set

    Returns:
      nesting_map: (dict) map from larger group set indices to lists of
        smaller group set indices

    """
    min_multiplicity = int(math.floor(large_k / small_k))
    max_bloated_group_index = large_k - small_k * min_multiplicity - 1
    nesting_map = collections.defaultdict(list)
    pos = 0
    for i in range(small_k):
      for _ in range(min_multiplicity + int(i <= max_bloated_group_index)):
        nesting_map[i].append(pos)
        pos += 1
    return nesting_map

  def _GenerateFeatureMemberships(
      self,
      graph_memberships = None,
      num_groups = None,
      match_type = MatchType.RANDOM):
    """Generates a feature membership assignment.

    Args:
      graph_memberships: (list) the integer memberships for the graph SBM
      num_groups: (int) number of groups. If None, defaults to number of unique
        values in graph_memberships.
      match_type: (MatchType) see the enum class description.

    Returns:
      memberships: a int list - index i contains feature group of node i.
    """
    # Parameter checks
    if num_groups is not None and num_groups == 0:
      raise ValueError("argument num_groups must be None or positive")
    graph_num_groups = len(set(graph_memberships))
    if num_groups is None:
      num_groups = graph_num_groups

    # Simulate memberships
    memberships = []
    if match_type == MatchType.GROUPED:
      if num_groups > graph_num_groups:
        raise ValueError(
            "for match type GROUPED, must have num_groups <= graph_num_groups")
      nesting_map = self._GetNestingMap(graph_num_groups, num_groups)
      reverse_nesting_map = {}
      for feature_cluster, graph_cluster_list in nesting_map.items():
        for cluster in graph_cluster_list:
          reverse_nesting_map[cluster] = feature_cluster
      for cluster in graph_memberships:
        memberships.append(reverse_nesting_map[cluster])
    elif match_type == MatchType.NESTED:
      if num_groups < graph_num_groups:
        raise ValueError(
            "for match type NESTED, must have num_groups >= graph_num_groups")
      nesting_map = self._GetNestingMap(num_groups, graph_num_groups)
      for cluster in graph_memberships:
        memberships.append(random.choice(nesting_map[cluster]))
    else:  # MatchType.RANDOM
      memberships = random.choices(range(num_groups), k=len(graph_memberships))
    return memberships

  def _ComputeExpectedEdgeCounts(self, num_edges, num_vertices,
                                 pi,
                                 prop_mat):
    """Computes expected edge counts within and between communities.

    Args:
      num_edges: expected number of edges in the graph.
      num_vertices: number of nodes in the graph
      pi: interable of non-zero community size proportions. Must sum to 1.0, but
        this check is left to the caller of this internal function.
      prop_mat: square, symmetric matrix of community edge count rates. Entries
        must be non-negative, but this check is left to the caller.

    Returns:
      symmetric matrix with shape prop_mat.shape giving expected edge counts.
    """
    scale = np.matmul(pi, np.matmul(prop_mat, pi)) * num_vertices**2
    prob_mat = prop_mat * num_edges / scale
    return np.outer(pi, pi) * prob_mat * num_vertices**2

  def _GenerateNodeMemberships(self, num_vertices,
                               pi):
    """Gets node memberships for sbm.

    Args:
      num_vertices: number of nodes in graph.
      pi: interable of non-zero community size proportions. Must sum to 1.0, but
        this check is left to the caller of this internal function.

    Returns:
      np vector of ints representing community indices.
    """
    community_sizes = np.random.multinomial(num_vertices, pi / np.sum(pi))
    memberships = np.zeros(num_vertices, dtype=int)
    node = 0
    for i in range(community_sizes.shape[0]):
      memberships[range(node, node + community_sizes[i])] = i
      node += community_sizes[i]
    return memberships

  def SimulateSbm(self,
                  num_vertices,
                  num_edges,
                  pi,
                  prop_mat,
                  out_degs = None):
    """Generates a stochastic block model.

    This function uses graph_tool.generate_sbm. Refer to that
    documentation for more information on the model and parameters.

    Args:
      num_vertices: (int) number of nodes in the graph.
      num_edges: (int) expected number of edges in the graph.
      pi: interable of non-zero community size proportions. Must sum to 1.0.
      prop_mat: square, symmetric matrix of community edge count rates.
      out_degs: Out-degree propensity for each node. If not provided, a constant
        value will be used. Note that the values will be normalized inside each
        group, if they are not already so.
    Returns: (none)
    """
    if np.sum(pi) != 1.0:
      raise ValueError("entries of pi must sum to 1.0")
    if prop_mat.shape[0] != len(pi) or prop_mat.shape[1] != len(pi):
      raise ValueError("prop_mat must be k x k where k = len(pi)")
    self.memberships = self._GenerateNodeMemberships(num_vertices, pi)
    edge_counts = self._ComputeExpectedEdgeCounts(num_edges,
                                                  num_vertices, pi, prop_mat)
    self.graph = generation.generate_sbm(self.memberships, edge_counts,
                                         out_degs)

  def SimulateFeatures(self,
                       center_var,
                       feature_dim,
                       num_groups,
                       match_type = MatchType.RANDOM,
                       cluster_var = 1.0):
    """Generates a multi-Normal mixture model with randomly-generated centers.

    Feature data is stored as a member variable named 'features'.

    Args:
      center_var: (float) variance of feature cluster centers. When this is 0.0,
        the signal-to-noise ratio is 0.0. When equal to cluster_var, SNR is 1.0.
      feature_dim: (int) dimension of the multivariate normal.
     num_groups: (int) number of centers. Generated by a multivariate d-normal
         with mean zero and covariance matrix cluster_var * I_{feature_dim}.
      match_type: (MatchType) see the enum class description.
      cluster_var: (float) variance of feature clusters around their centers.
    Returns: (None)
    """
    # Get memberships
    self.feature_memberships = self._GenerateFeatureMemberships(
        graph_memberships=self.memberships,
        num_groups=num_groups,
        match_type=match_type)

    # Get centers
    centers = []
    center_cov = np.identity(feature_dim) * center_var
    cluster_cov = np.identity(feature_dim) * cluster_var
    for _ in range(num_groups):
      center = np.random.multivariate_normal(
          np.zeros(feature_dim), center_cov, 1)[0]
      centers.append(center)
    self.features = []
    for cluster_index in self.feature_memberships:
      feature = np.random.multivariate_normal(
          centers[cluster_index], cluster_cov, 1)[0]
      self.features.append(feature)

  def SimulateEdgeFeatures(self,
                           feature_dim,
                           center_distance = 0.0,
                           cluster_variance = 1.0):
    """Generates edge feature distribution via inter-class vs intra-class.

    Edge feature data is stored as a member variable named 'edge_features', a
    dict from 2-tuples of node IDs to numpy vectors.

    Edge features have two centers: one at (0, 0, ....) and one at
    (center_distance, center_distance, ....) for inter-class and intra-class
    edges (respectively). They are generated from a multivariate normal with
    covariance matrix = cluster_variance * I_d.

    Args:
      feature_dim: (int) dimension of the multivariate normal.
      center_distance: (float) per-dimension distance between the intra-class
        and inter-class means. Increasing this makes the edge feature signal
        stronger.
      cluster_variance: (float) variance of clusters around their centers.
    """
    if not hasattr(self, "graph"):
      self.edge_features = None
      return None
    self.edge_features = {}
    center0 = np.zeros(shape=(feature_dim,))
    center1 = np.ones(shape=(feature_dim,)) * center_distance
    covariance = np.identity(feature_dim) * cluster_variance
    for edge in self.graph.edges():
      vertex1 = int(edge.source())
      vertex2 = int(edge.target())
      edge_tuple = tuple(sorted((vertex1, vertex2)))
      if self.memberships[vertex1] == self.memberships[vertex2]:
        center = center1
      else:
        center = center0
      self.edge_features[edge_tuple] = np.random.multivariate_normal(
          center, covariance, 1)[0]
