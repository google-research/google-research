# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

This function can generate node features with varying correlation to the blocks,
which is useful for benchmarking graph convolutional neural networks (see
https://arxiv.org/abs/2006.16904).

This function can also generate a heterogeneous SBM with two types. Specify the
number of nodes of type 2 via num_vertices2, as well as other parameters that
are analogous to the parameter for type 1 nodes. Cross-type graph clusters can
be induced in the model via the prop_mat parameter, or more easily with the
edge_probability_profile input. Cross-type feature clusters can be induced by
setting feature_type_correlation above 0.0 (up to 1.0). This does not control
the actual Pearson correlation of features across types, but does interpolate
between entirely intra-type feature correlation (0.0) and entirely inter-type
feature correlation (1.0). See the methods in sbm_simulator.py for details.
"""

import typing

import numpy as np

from graph_embedding.simulations import heterogeneous_sbm_utils as hsu
from graph_embedding.simulations import sbm_simulator

# Types
Sequence = typing.Sequence
Optional = typing.Optional
MatchType = sbm_simulator.MatchType


def GenerateStochasticBlockModelWithFeatures(
    num_vertices,
    num_edges,
    pi,
    prop_mat = None,
    out_degs = None,
    feature_center_distance = 0.0,
    feature_dim = 0,
    num_feature_groups = None,
    feature_group_match_type = MatchType.RANDOM,
    feature_cluster_variance = 1.0,
    edge_feature_dim = 0,
    edge_center_distance = 0.0,
    edge_cluster_variance = 1.0,
    pi2 = None,
    feature_center_distance2 = 0.0,
    feature_dim2 = 0,
    feature_type_correlation = 0.0,
    feature_type_center_distance = 0.0,
    edge_probability_profile = None
):
  """Generates stochastic block model (SBM) with node features.

  Args:
    num_vertices: number of nodes in the graph.
    num_edges: expected number of edges in the graph.
    pi: iterable of non-zero community size relative proportions. Community i
      will be pi[i] / pi[j] times larger than community j.
    prop_mat: square, symmetric matrix of community edge count rates. Example:
      if diagonals are 2.0 and off-diagonals are 1.0, within-community edges are
      twices as likely as between-community edges.
    out_degs: Out-degree propensity for each node. If not provided, a constant
      value will be used. Note that the values will be normalized inside each
      group, if they are not already so.
    feature_center_distance: distance between feature cluster centers. When this
      is 0.0, the signal-to-noise ratio is 0.0. When equal to
      feature_cluster_variance, SNR is 1.0.
    feature_dim: dimension of node features.
    num_feature_groups: number of feature clusters. This is ignored if
      num_vertices2 is provided, as the internal feature generators will assume
      a heterogeneous SBM model, which does not support differing # feature
      clusters from # graph clusters. In this case, # feature clusters
      will be set equal to # graph clusters. If left as default (None),
      and input sbm_data is homogeneous, set to len(pi1).
    feature_group_match_type: see sbm_simulator.MatchType.
    feature_cluster_variance: variance of feature clusters around their centers.
      centers. Increasing this weakens node feature signal.
    edge_feature_dim: dimension of edge features.
    edge_center_distance: per-dimension distance between the intra-class and
      inter-class means. Increasing this strengthens the edge feature signal.
    edge_cluster_variance: variance of edge clusters around their centers.
      Increasing this weakens the edge feature signal.
    pi2: This is the pi vector for the vertices of type 2. Type 2 community k
      will be pi2[k] / pi[j] times larger than type 1 community j. Supplying
      this argument produces a heterogeneous model.
    feature_center_distance2: feature_center_distance for type 2 nodes. Not used
      if len(pi2) = 0.
    feature_dim2: feature_dim for nodes of type 2. Not used if len(pi2) = 0.
    feature_type_correlation: proportion of each cluster's center vector that
      is shared with other clusters linked across types. Not used if len(pi2) =
      0.
    feature_type_center_distance: the variance of the generated centers for
      feature vectors that are shared across types. Not used if len(pi2) = 0.
    edge_probability_profile: This can be provided instead of prop_mat. If
      provided, prop_mat will be built according to the input p-to-q ratios. If
      prop_mat is provided, it will be preferred over this input.
  Returns:
    result: a StochasticBlockModel data class.
  Raises:
    ValueError: if neither of prop_mat or edge_probability_profile are provided.
  """
  result = sbm_simulator.StochasticBlockModel()
  if prop_mat is None and edge_probability_profile is None:
    raise ValueError(
        "One of prop_mat or edge_probability_profile must be provided.")
  if prop_mat is None and edge_probability_profile is not None:
    prop_mat = hsu.GetPropMat(
        num_clusters1=len(pi),
        p_to_q_ratio1=edge_probability_profile.p_to_q_ratio1,
        num_clusters2=0 if pi2 is None else len(pi2),
        p_to_q_ratio2=edge_probability_profile.p_to_q_ratio2,
        p_to_q_ratio_cross=edge_probability_profile.p_to_q_ratio_cross)

  sbm_simulator.SimulateSbm(result, num_vertices, num_edges, pi,
                            prop_mat, out_degs, pi2)
  sbm_simulator.SimulateFeatures(result, feature_center_distance,
                                 feature_dim, num_feature_groups,
                                 feature_group_match_type,
                                 feature_cluster_variance,
                                 feature_center_distance2,
                                 feature_dim2,
                                 feature_type_correlation,
                                 feature_type_center_distance)
  if edge_feature_dim > 0:
    sbm_simulator.SimulateEdgeFeatures(result, edge_feature_dim,
                                       edge_center_distance,
                                       edge_cluster_variance)
  return result
