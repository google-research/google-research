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

"""Library for heterogeneous SBMs with node features."""
import math
from typing import List, Tuple

import numpy as np


def GetCrossLinks(num_clusters1, num_clusters2):
  """Given two clustering sizes, returns an ordered linking between clusterings.

  The linking is given as a list of tuples. Each tuple contains two cluster
  indices. Indices are continuously ordered, so that every index from 0 to
  num_clusters1 + num_clusters2 - 1 is present in the output. If the inputs
  are the same, this function returns a 1-1 map:
    (0, num_clusters1)
    (1, num_clusters1 + 1)
    ...
    (num_clusters1 - 1, num_clusters1 + num_clusters2 - 1)
  If the inputs are different, the smaller number restarts when looping through
  the output tuples.

  Arguments:
    num_clusters1: number of clusters in the first clustering.
    num_clusters2: number of clusters in the second clustering.

  Returns:
    cross_links: list of cluster index tuples
  """
  if num_clusters1 <= num_clusters2:
    cluster_indices1 = list(range(num_clusters1))
    cluster_indices2 = list(range(num_clusters2))
    cluster1_index_list = []
    for _ in range(math.floor(num_clusters2 / num_clusters1)):
      cluster1_index_list.extend(cluster_indices1)
    cluster1_index_list += cluster_indices1[:(num_clusters2 % num_clusters1)]
    cluster2_index_list = [j + num_clusters1 for j in cluster_indices2]
    return list(zip(cluster1_index_list, cluster2_index_list))
  else:
    reversed_result = GetCrossLinks(num_clusters2, num_clusters1)  # pylint: disable=arguments-out-of-order
    return [(j - num_clusters2, i + num_clusters1) for (i, j) in reversed_result
           ]


def _GetHomogeneousPropMat(num_clusters, p_to_q_ratio):
  """Generates a proportion matrix within a type."""
  base_prop_mat = np.ones(shape=(num_clusters, num_clusters))
  np.fill_diagonal(base_prop_mat, p_to_q_ratio)
  return base_prop_mat


def _GetCrossPropMat(num_clusters1, num_clusters2, cross_links, p_to_q_ratio):
  """Helper function to generate a proporation matrix across types."""
  base_prop_mat = np.ones(shape=(num_clusters1, num_clusters2))
  for link in cross_links:
    base_prop_mat[link[0], link[1] - num_clusters1] = p_to_q_ratio
  return base_prop_mat


def GetPropMat(num_clusters1, p_to_q_ratio1,
               num_clusters2 = 0, p_to_q_ratio2 = 0,
               p_to_q_ratio_cross = 0.0):
  """Generates a proportion matrix for the heterogeneous SBM.

  Arguments:
    num_clusters1: Number of clusters of nodes of type 1.
    p_to_q_ratio1: Probability of in-cluster edges divided by probability of
      out-cluster edges, for type 1 nodes.
    num_clusters2: Number of clusters of nodes of type 2.
    p_to_q_ratio2: Probability of in-cluster edges divided by probability of
      out-cluster edges, for type 2 nodes.
    p_to_q_ratio_cross: Probability of in-cluster edges divided by probability
      of out-cluster edges, for node clusters that are linked across-type.

  Returns:
    prop_mat: proportion matrix for input to
      simulations.GenerateStochasticBlockModelWithFeatures.
  """
  base_prop_mat = np.zeros(
      shape=(num_clusters1 + num_clusters2, num_clusters1 + num_clusters2))
  base_prop_mat[0:num_clusters1,
                0:num_clusters1] = _GetHomogeneousPropMat(num_clusters1,
                                                          p_to_q_ratio1)
  if num_clusters2 == 0:
    return base_prop_mat
  cross_links = GetCrossLinks(num_clusters1, num_clusters2)
  base_prop_mat[
      (num_clusters1):(num_clusters1 + num_clusters2),
      (num_clusters1):(num_clusters1 + num_clusters2)] = _GetHomogeneousPropMat(
          num_clusters2, p_to_q_ratio2)
  cross_prop_mat = _GetCrossPropMat(num_clusters1, num_clusters2, cross_links,
                                    p_to_q_ratio_cross)
  base_prop_mat[0:num_clusters1,
                (num_clusters1):(num_clusters1 +
                                 num_clusters2)] = cross_prop_mat
  base_prop_mat[(num_clusters1):(num_clusters1 + num_clusters2),
                0:num_clusters1] = cross_prop_mat.T
  return base_prop_mat
