# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Library for stochastic block models (SBMs) with node features.

SimulateSbm, SimulateFeatures, and SimulateEdgeFeatures are top-level library
functions used by GenerateStochasticBlockModel in simulations.py. You can call
these separately to generate various parts of an SBM with features.
"""
import collections
import dataclasses
import enum
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import graph_tool
from graph_tool import generation
import networkx as nx
import numpy as np

from graph_embedding.simulations import heterogeneous_sbm_utils as hsu

# pylint: disable=g-explicit-length-test


class MatchType(enum.Enum):
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


@dataclasses.dataclass
class EdgeProbabilityProfile:
  """Stores p-to-q ratios for Stochastic Block Model.

  Attributes:
    p_to_q_ratio1: Probability of in-cluster edges divided by probability of
      out-cluster edges, for type 1 nodes. If the SBM is homogeneous, this
      is the global p_to_q_ratio.
    p_to_q_ratio2: Probability of in-cluster edges divided by probability of
      out-cluster edges, for type 2 nodes.
    p_to_q_ratio_cross: Probability of in-cluster edges divided by probability
      of out-cluster edges, for node clusters that are linked across-type.
  """
  p_to_q_ratio1: float = Ellipsis
  p_to_q_ratio2: Optional[float] = 0.0
  p_to_q_ratio_cross: Optional[float] = 0.0


@dataclasses.dataclass
class StochasticBlockModel:
  """Stores data for stochastic block model (SBM) graphs with features.

  This class supports heterogeneous SBMs, in which each node is assumed to be
  exactly one of two types. In this model, the following extra fields are used:
    * type1_clusters: list of cluster indices for type 1 nodes. (For single-type
        graphs, this contains the list of all cluster indices.)
    * type2_clusters: list of cluster indices for type 2 nodes.
    * cross_links: tuples of cluster indices that are linked cross-type.
    * node_features2: features for type 2 nodes. (node_features1 is used as the
        sole feature field for single-type SBM.)

  Attributes:
    graph: graph-tool Graph object.
    graph_memberships: list of integer node classes.
    node_features1: numpy array of node features for nodes of type 1. Features
      for node with index i is in row i.
    node_features2: numpy array of node features for nodes of type 2. Features
      for node with index i is in row i - (# of nodes of type 1).
    feature_memberships: list of integer node feature classes.
    edge_features: map from edge tuple to numpy array. Only stores undirected
      edges, i.e. (0, 1) will be in the map, but (1, 0) will not be.
    cross_links: list of 2-tuples, each tuple a pair of cluster indices which
      are cross-correlated between the types. (i, j) included in this list means
      the i-th cluster from type 1 is correlated with the j-th cluster from type
      2.
    type1_clusters: list of the indices of type 1 clusters.
    type2_clusters: list of the indices of type 2 clusters.
    cross_links: list of cluster index pairs, each pair coding that the clusters
      are linked across types.
  """
  graph: graph_tool.Graph = Ellipsis
  graph_memberships: np.ndarray = Ellipsis
  node_features1: np.ndarray = Ellipsis
  node_features2: Optional[np.ndarray] = Ellipsis
  feature_memberships: np.ndarray = Ellipsis
  edge_features: Dict[Tuple[int, int], np.ndarray] = Ellipsis
  type1_clusters: Optional[List[int]] = Ellipsis
  type2_clusters: Optional[List[int]] = Ellipsis
  cross_links: Optional[List[Tuple[int, int]]] = Ellipsis


def _GetNestingMap(large_k, small_k):
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


  Args:
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
    graph_memberships,
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

  # Compute memberships
  memberships = []
  if match_type == MatchType.GROUPED:
    if num_groups > graph_num_groups:
      raise ValueError(
          "for match type GROUPED, must have num_groups <= graph_num_groups")
    nesting_map = _GetNestingMap(graph_num_groups, num_groups)
    # Creates deterministic map from (smaller) graph clusters to (larger)
    # feature clusters.
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
    nesting_map = _GetNestingMap(num_groups, graph_num_groups)
    # Creates deterministic map from (smaller) feature clusters to (larger)
    # graph clusters.
    for graph_cluster_id, feature_cluster_ids in nesting_map.items():
      sorted_feature_cluster_ids = sorted(feature_cluster_ids)
      num_feature_groups = len(sorted_feature_cluster_ids)
      feature_pi = np.ones(num_feature_groups) / num_feature_groups
      num_graph_cluster_nodes = np.sum(
          [i == graph_cluster_id for i in graph_memberships])
      sub_memberships = _GenerateNodeMemberships(num_graph_cluster_nodes,
                                                 feature_pi)
      sub_memberships = [sorted_feature_cluster_ids[i] for i in sub_memberships]
      memberships.extend(sub_memberships)
  else:  # MatchType.RANDOM
    memberships = random.choices(range(num_groups), k=len(graph_memberships))
  return np.array(sorted(memberships))


def _ComputeExpectedEdgeCounts(num_edges, num_vertices,
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


def _ComputeCommunitySizes(num_vertices, pi):
  """Helper function of GenerateNodeMemberships to compute group sizes.

  Args:
    num_vertices: number of nodes in graph.
    pi: interable of non-zero community size proportions.

  Returns:
    community_sizes: np vector of group sizes. If num_vertices * pi[i] is a
      whole number (up to machine precision), community_sizes[i] will be that
      number. Otherwise, this function accounts for rounding errors by making
      group sizes as balanced as possible (i.e. increasing smallest groups by
      1 or decreasing largest groups by 1 if needed).
  """
  community_sizes = [int(x * num_vertices) for x in pi]
  if sum(community_sizes) != num_vertices:
    size_order = np.argsort(community_sizes)
    delta = sum(community_sizes) - num_vertices
    adjustment = np.sign(delta)
    if adjustment == 1:
      size_order = np.flip(size_order)
    for i in range(int(abs(delta))):
      community_sizes[size_order[i]] -= adjustment
  return community_sizes


def _GenerateNodeMemberships(num_vertices,
                             pi):
  """Gets node memberships for sbm.

  Args:
    num_vertices: number of nodes in graph.
    pi: interable of non-zero community size proportions. Must sum to 1.0, but
      this check is left to the caller of this internal function.

  Returns:
    np vector of ints representing community indices.
  """
  community_sizes = _ComputeCommunitySizes(num_vertices, pi)
  memberships = np.zeros(num_vertices, dtype=int)
  node = 0
  for i in range(len(pi)):
    memberships[range(node, node + community_sizes[i])] = i
    node += community_sizes[i]
  return memberships


def SimulateSbm(sbm_data,
                num_vertices,
                num_edges,
                pi,
                prop_mat,
                out_degs = None,
                pi2 = None):
  """Generates a stochastic block model, storing data in sbm_data.graph.

  This function uses graph_tool.generate_sbm. Refer to that
  documentation for more information on the model and parameters.

  This function can generate a heterogeneous SBM graph, meaning each node is
  exactly one of two types (and both types are present). To generate a
  heteroteneous SBM graph, `pi2` must be supplied, and additional fields of
  `sbm_data` will be filled. See the StochasticBlockModel dataclass for details.

  Args:
    sbm_data: StochasticBlockModel dataclass to store result data.
    num_vertices: (int) number of nodes in the graph.
    num_edges: (float) expected number of edges in the graph.
    pi: iterable of non-zero community size relative proportions. Community i
      will be pi[i] / pi[j] times larger than community j.
    prop_mat: square, symmetric matrix of community edge count rates.
    out_degs: Out-degree propensity for each node. If not provided, a constant
      value will be used. Note that the values will be normalized inside each
      group, if they are not already so.
    pi2: This is the pi vector for the vertices of type 2. Type 2 community k
      will be pi2[k] / pi[j] times larger than type 1 community j. Supplying
      this argument produces a heterogeneous model.
  Returns: (none)
  """
  if pi2 is None: pi2 = []
  k1, k2 = len(pi), len(pi2)
  pi = np.array(list(pi) + list(pi2)).astype(np.float64)
  pi /= np.sum(pi)
  if prop_mat.shape[0] != len(pi) or prop_mat.shape[1] != len(pi):
    raise ValueError("prop_mat must be k x k; k = len(pi1) + len(pi2)")
  sbm_data.graph_memberships = _GenerateNodeMemberships(num_vertices, pi)
  sbm_data.type1_clusters = sorted(list(set(sbm_data.graph_memberships)))
  if len(pi2) > 0:
    sbm_data.cross_links = hsu.GetCrossLinks([k1, k2], 0, 1)
    type1_clusters, type2_clusters = zip(*sbm_data.cross_links)
    sbm_data.type1_clusters = sorted(list(set(type1_clusters)))
    sbm_data.type2_clusters = sorted(list(set(type2_clusters)))
  edge_counts = _ComputeExpectedEdgeCounts(
      num_edges, num_vertices, pi, prop_mat)
  sbm_data.graph = generation.generate_sbm(sbm_data.graph_memberships,
                                           edge_counts, out_degs)
  graph_tool.stats.remove_self_loops(sbm_data.graph)
  graph_tool.stats.remove_parallel_edges(sbm_data.graph)
  sbm_data.graph.reindex_edges()


def _GetFeatureCenters(num_groups, center_var, feature_dim):
  """Helper function to generate multivariate Normal feature centers.

  Args:
    num_groups: number of centers to generate.
    center_var: diagonal element of the covariance matrix (off-diagonals = 0).
    feature_dim: the dimension of each center.
  Returns:
    centers: numpy array with feature group centers as rows.
  """
  centers = np.random.multivariate_normal(
      np.zeros(feature_dim), np.identity(feature_dim) * center_var,
      num_groups)
  return centers


def SimulateFeatures(sbm_data,
                     center_var,
                     feature_dim,
                     num_groups = None,
                     match_type = MatchType.RANDOM,
                     cluster_var = 1.0,
                     center_var2 = 0.0,
                     feature_dim2 = 0,
                     type_correlation = 0.0,
                     type_center_var = 0.0):
  """Generates node features using multivate normal mixture model.

  This function does nothing and throws a warning if
  sbm_data.graph_memberships is empty. Run SimulateSbm to fill that field.

  Feature data is stored as an attribute of sbm_data named 'node_features1'.

  If the `type2_clusters` field in the input `sbm_data` is filled, this function
  produces node features for a heterogeneous SBM. Specifically:
   * Handling differing # graph clusters and # feature clusters is not
     implemented for heterogeneous SBMs. `num_groups` and must equal the
     length of sbm_data.type1_clusters (raises RuntimeWarning if not).
   * The node_features{1,2} fields of the input sbm_data will store the features
     generated for type {1,2} nodes.

  Args:
    sbm_data: StochasticBlockModel dataclass to store result data.
    center_var: (float) variance of feature cluster centers. When this is 0.0,
      the signal-to-noise ratio is 0.0. When equal to cluster_var, SNR is 1.0.
    feature_dim: (int) dimension of the multivariate normal.
    num_groups: (int) number of centers. Generated by a multivariate normal with
      mean zero and covariance matrix cluster_var * I_{feature_dim}. This is
      ignored if the input sbm_data is heterogeneous. Feature cluster counts
      will be set equal to the graph cluster counts. If left as default (None),
      and input sbm_data is homogeneous, set to len(sbm_data.type1_clusters).
    match_type: (MatchType) see sbm_simulator.MatchType for details.
    cluster_var: (float) variance of feature clusters around their centers.
    center_var2: (float) center_var for nodes of type 2. Not needed if sbm_data
      is not heterogeneous (see above).
    feature_dim2: (int) feature_dim for nodes of type 2. Not needed if sbm_data
      is not heterogeneous (see above).
    type_correlation: (float) proportion of each cluster's center vector that
      is shared with other clusters linked across types. Not needed if sbm_data
      is not heterogeneous (see above).
    type_center_var: (float) center_var for center vectors that are shared with
      clusters linked across types. Not used if input sbm_data is not
      heterogeneous.

  Raises:
    RuntimeWarning:
      * if sbm_data no graph, no graph_memberships, or type1_clusters fields.
      * if len(sbm_data.type2_clusters) > 0 and sbm_data.cross_links is not a
        list.
  """
  if sbm_data.graph is None or sbm_data.graph is Ellipsis:
    raise RuntimeWarning("No graph found: no features generated. "
                         "Run SimulateSbm to generate a graph.")
  if sbm_data.graph_memberships is None or sbm_data.graph_memberships is Ellipsis:
    raise RuntimeWarning("No graph_memberships found: no features generated. "
                         "Run SimulateSbm to generate graph_memberships.")
  if sbm_data.type1_clusters is None or sbm_data.type1_clusters is Ellipsis:
    raise RuntimeWarning("No type1_clusters found: no features generated. "
                         "Run SimulateSbm to generate type1_clusters.")
  if num_groups is None:
    num_groups = len(sbm_data.type1_clusters)
  centers = list(_GetFeatureCenters(num_groups, center_var, feature_dim))
  num_groups2 = (0 if sbm_data.type2_clusters is Ellipsis
                 else len(sbm_data.type2_clusters))
  if num_groups2 > 0:
    # The SBM is heterogeneous. Check input and adjust variables.
    if not isinstance(sbm_data.cross_links, list):
      raise RuntimeWarning(
          ("len(sbm_data.type2_clusters) > 0, implying heterogeneous SBM, but "
           "heterogeneous data `cross_links` is unfilled."))

    # Generate heterogeneous feature centers.
    centers += list(_GetFeatureCenters(num_groups2, center_var2, feature_dim2))
    correspondence_graph = nx.Graph()
    correspondence_graph.add_edges_from(sbm_data.cross_links)
    connected_components = list(
        nx.algorithms.connected_components(correspondence_graph))
    cross_type_feature_dim = min(feature_dim, feature_dim2)
    component_center_cov = np.identity(cross_type_feature_dim) * type_center_var
    for component in connected_components:
      component_center = np.random.multivariate_normal(
          np.zeros(cross_type_feature_dim), component_center_cov, 1)[0]
      for cluster_index in component:
        centers[cluster_index][:cross_type_feature_dim] = (
            component_center * type_correlation
            + centers[cluster_index][:cross_type_feature_dim] *
            (1 - type_correlation))

  # Get memberships
  sbm_data.feature_memberships = _GenerateFeatureMemberships(
      graph_memberships=sbm_data.graph_memberships,
      num_groups=num_groups,
      match_type=match_type)
  cluster_indices = sbm_data.feature_memberships
  if num_groups2 > 0:
    cluster_indices = sbm_data.graph_memberships

  features1 = []
  features2 = []
  cluster_cov1 = np.identity(feature_dim) * cluster_var
  cluster_cov2 = np.identity(feature_dim2) * cluster_var
  for cluster_index in cluster_indices:
    cluster_cov = cluster_cov1
    if num_groups2 > 0 and cluster_index in sbm_data.type2_clusters:
      cluster_cov = cluster_cov2
    feature = np.random.multivariate_normal(centers[cluster_index], cluster_cov,
                                            1)[0]
    if cluster_index in sbm_data.type1_clusters:
      features1.append(feature)
    else:
      features2.append(feature)
  sbm_data.node_features1 = np.array(features1)
  if num_groups2 > 0:
    sbm_data.node_features2 = np.array(features2)


def SimulateEdgeFeatures(sbm_data,
                         feature_dim,
                         center_distance = 0.0,
                         cluster_variance = 1.0):
  """Generates edge feature distribution via inter-class vs intra-class.

  Edge feature data is stored as an sbm_data attribute named `edge_feature`, a
  dict from 2-tuples of node IDs to numpy vectors.

  Edge features have two centers: one at (0, 0, ....) and one at
  (center_distance, center_distance, ....) for inter-class and intra-class
  edges (respectively). They are generated from a multivariate normal with
  covariance matrix = cluster_variance * I_d.

  Requires non-None `graph` and `graph_memberships` attributes in sbm_data.
  Use SimulateSbm to generate them. Throws warning if either are None.

  Args:
    sbm_data: StochasticBlockModel dataclass to store result data.
    feature_dim: (int) dimension of the multivariate normal.
    center_distance: (float) per-dimension distance between the intra-class and
      inter-class means. Increasing this makes the edge feature signal stronger.
    cluster_variance: (float) variance of clusters around their centers.

  Raises:
    RuntimeWarning: if simulator has no graph or a graph with no nodes.
  """
  if sbm_data.graph is None:
    raise RuntimeWarning("SbmSimulator has no graph: no features generated.")
  if sbm_data.graph.num_vertices() == 0:
    raise RuntimeWarning("graph has no nodes: no features generated.")
  if sbm_data.graph_memberships is None:
    raise RuntimeWarning("graph has no memberships: no features generated.")

  center0 = np.zeros(shape=(feature_dim,))
  center1 = np.ones(shape=(feature_dim,)) * center_distance
  covariance = np.identity(feature_dim) * cluster_variance
  sbm_data.edge_features = {}
  for edge in sbm_data.graph.edges():
    vertex1 = int(edge.source())
    vertex2 = int(edge.target())
    edge_tuple = tuple(sorted((vertex1, vertex2)))
    if (sbm_data.graph_memberships[vertex1] ==
        sbm_data.graph_memberships[vertex2]):
      center = center1
    else:
      center = center0
    sbm_data.edge_features[edge_tuple] = np.random.multivariate_normal(
        center, covariance, 1)[0]
