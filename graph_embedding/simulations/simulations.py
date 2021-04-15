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

import typing

import numpy as np

from graph_embedding.simulations import sbm_simulator

# Types
List = typing.List
MatchType = sbm_simulator.MatchType


def GenerateStochasticBlockModelWithFeatures(
    num_vertices,
    num_edges,
    pi,
    prop_mat,
    out_degs = None,
    feature_center_distance = 0.0,
    feature_dim = 0,
    num_feature_groups = 1,
    feature_group_match_type = MatchType.RANDOM,
    feature_cluster_variance = 1.0,
    edge_feature_dim = 0,
    edge_center_distance = 0.0,
    edge_cluster_variance = 1.0):
  """Generates stochastic block model (SBM) with node features.

  Args:
    num_vertices: number of nodes in the graph.
    num_edges: expected number of edges in the graph.
    pi: interable of non-zero community size proportions. Must sum to 1.0.
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
    num_feature_groups: number of feature clusters.
    feature_group_match_type: see sbm_simulator.MatchType.
    feature_cluster_variance: variance of feature clusters around their centers.
      centers. Increasing this weakens node feature signal.
    edge_feature_dim: dimension of edge features.
    edge_center_distance: per-dimension distance between the intra-class and
      inter-class means. Increasing this strengthens the edge feature signal.
    edge_cluster_variance: variance of edge clusters around their centers.
      Increasing this weakens the edge feature signal.
  Returns:
    result: a StochasticBlockModel data class.
  """
  result = sbm_simulator.StochasticBlockModel()
  sbm_simulator.SimulateSbm(result, num_vertices, num_edges, pi,
                            prop_mat, out_degs)
  sbm_simulator.SimulateFeatures(result, feature_center_distance,
                                 feature_dim, num_feature_groups,
                                 feature_group_match_type,
                                 feature_cluster_variance)
  sbm_simulator.SimulateEdgeFeatures(result, edge_feature_dim,
                                     edge_center_distance,
                                     edge_cluster_variance)
  return result
