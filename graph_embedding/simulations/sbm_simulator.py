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
import graph_tool.generation as generation
import graph_tool.Graph
import numpy as np

# Types
Any = typing.Any
Dict = typing.Dict
Enum = enum.Enum
List = typing.List
Set = typing.Set
Text = typing.Text

# Constants
MEMBERSHIPS_FILENAME = "memberships.txt"
MEMBERSHIPS_SSTABLE = "memberships"
GRAPH_FILENAME = "graph.txt"
SHARDED_GRAPH_FILENAME = "graph@20"


class MatchType(Enum):
  """Indicates type of graph membership matching to do.

    RANDOM: memberships are generated randomly.
    NESTED: for k >= number of memberships in g_memberships (fails if this is
      not true). Each cluster is a sub-cluster of a cluster in g_memberships.
      Multiplicity of sub-clusters per g_membership cluster is kept as uniform
      as possible.
    GROUPED: for k <= number of memberships in g_memberships (fails if this is
      not true). Each cluster is a super-cluster of a cluster in g_memberships.
      Multiplicity of graph sub-clusters per feature cluster is kept as uniform
      as possible.
  """
  RANDOM = 1
  NESTED = 2
  GROUPED = 3


# TODO(palowitch): add tests for this class
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
      n,
      g_memberships = None,
      k = None,
      match_type = MatchType.RANDOM):
    """Generates a feature membership assignment.

    Args:
      n: (int) number of samples
      g_memberships: (list) the integer memberships for the graph SBM
      k: (int) number of groups. If None, defaults to number of groups in
        g_memberships.
      match_type: (MatchType) see the enum class description.
    Returns:
      memberships: a int list - index i contains feature group of node i.
    """
    # Parameter checks
    if k is not None and k == 0:
      raise ValueError("argument k must be None or positive")
    graph_k = len(set(g_memberships))
    if k is None:
      k = graph_k

    # Simulate memberships
    memberships = []
    if match_type == MatchType.GROUPED:
      if k > graph_k:
        raise ValueError("for match type GROUPED, must have k <= graph_k")
      nesting_map = self._GetNestingMap(graph_k, k)
      reverse_nesting_map = {}
      for feature_c, graph_c_list in nesting_map.items():
        for c in graph_c_list:
          reverse_nesting_map[c] = feature_c
      for c in g_memberships:
        memberships.append(reverse_nesting_map[c])
    elif match_type == MatchType.NESTED:
      if k < graph_k:
        raise ValueError("for match type NESTED, must have k >= graph_k")
      nesting_map = self._GetNestingMap(k, graph_k)
      for c in g_memberships:
        memberships.append(random.choice(nesting_map[c]))
    else:  # MatchType.RANDOM
      memberships = random.choices(range(k), k=n)
    return memberships

  def _ComputeExpectedEdgeCounts(self, m, n, pi,
                                 prop_mat):
    """Computes expected edge counts within and between communities.

    Args:
      m: expected number of edges in the graph.
      n: number of nodes in the graph
      pi: interable of non-zero community size proportions. Must sum to 1.0, but
        this check is left to the caller of this internal function.
      prop_mat: square, symmetric matrix of community edge count rates. Entries
        must be non-negative, but this check is left to the caller.
    Returns:
      symmetric matrix with shape prop_mat.shape giving expected edge counts.
    """
    alpha = m / (np.matmul(pi, np.matmul(prop_mat, pi)) * n**2)
    prob_mat = prop_mat * alpha
    return np.outer(pi, pi) * prob_mat * n**2

  def _GenerateNodeMemberships(self, n, pi):
    """Gets node memberships for sbm.

    Args:
      n: number of nodes in graph.
      pi: interable of non-zero community size proportions. Must sum to 1.0, but
        this check is left to the caller of this internal function.
    Returns:
      np vector of ints representing community indices.
    """
    community_sizes = np.random.multinomial(n, pi / np.sum(pi))
    memberships = np.zeros(n, dtype=int)
    node = 0
    for i in range(community_sizes.shape[0]):
      memberships[range(node, node + community_sizes[i])] = i
      node += community_sizes[i]
    return memberships

  def SimulateSbm(self,
                  n,
                  m,
                  pi,
                  prop_mat,
                  out_degs = None):
    """Generates a stochastic block model.

    This function uses graph_tool.generation.generate_sbm. Refer to that
    documentation for more information on the model and parameters.

    Args:
      n: (int) number of nodes in the graph.
      m: (int) expected number of edges in the graph.
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
    self.memberships = self._GenerateNodeMemberships(n, pi)
    edge_counts = self._ComputeExpectedEdgeCounts(m, n, pi, prop_mat)
    self.graph = generation.generate_sbm(self.memberships, edge_counts,
                                         out_degs)

  def SimulateFeatures(self,
                       center_var,
                       d,
                       k,
                       match_type = MatchType.RANDOM,
                       cluster_var = 1.0):
    """Generates a multivariate d-normal with k randomly-generated centers.

    Feature data is stored as a member variable named 'features'.

    Args:
      center_var: (float) variance of feature cluster centers. When this is 0.0,
        the classical "signal-to-noise ratio" is 0.0. When equal to cluster_var,
        the SNR is 1.0.
      d: (int) dimension of the multivariate normal.
      k: (int) number of centers. Generated by a multivariate d-normal with mean
         zero and covariance matrix center_rate * sigma^2 * I_d.
      match_type: (MatchType) see the enum class description.
      cluster_var: (float) variance of feature clusters around their centers.
    Returns: (None)
    """
    # Get memberships
    self.feature_memberships = self._GenerateFeatureMemberships(
        n=self.graph.num_vertices(),
        g_memberships=self.memberships,
        k=k,
        match_type=match_type)

    # Get centers
    centers = []
    center_cov = np.identity(d) * center_var
    cluster_cov = np.identity(d) * cluster_var
    for _ in range(k):
      centers.append(
          np.random.multivariate_normal(np.zeros(d), center_cov, 1)[0])
    self.features = []
    for c in self.feature_memberships:
      self.features.append(
          np.random.multivariate_normal(centers[c], cluster_cov, 1)[0])
