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

"""Library for heterogeneous SBMs with node features."""
from typing import List, Tuple

import numpy as np


def GetClusterTypeComponents(
    num_clusters_list):
  """Given a list of # clusters per-type, compute cross-type cluster components.

  This function expands num_clusters_lists into a list of cluster index lists --
  one list of size num_clusters_list[i] for each i-th entry. It then assigns
  each member of each list to a unique component out of min(num_clusters_list)
  components.

  For example, an input [3, 4, 2] implies three clusters for type-1 nodes, four
  clusters for type-2 nodes, and two clusters for type-1 nodes. This function
  will return two type components, and evenly (or as-evenly-as-possible) divide
  the cluster indices of each type among the two components. See the test file
  for the expected outputs from this example.

  Arguments:
    num_clusters_list: list of the number of clusters for each node type.
  Returns:
    output: a 2-tuple with the following elements:
      cluster_index_lists: a list of cluster index lists.
      type_components: a list of cluster index sets, giving the type components.
  """
  # Compute the cluster_index_lists.
  offset = 0
  cluster_index_lists = []
  for num_clusters in num_clusters_list:
    cluster_index_lists.append(list(range(offset, offset + num_clusters)))
    offset += num_clusters

  # Compute type_components.
  num_components = np.min(num_clusters_list)
  type_components = [list() for _ in range(num_components)]
  for cluster_index_list in cluster_index_lists:
    for i in cluster_index_list:
      type_components[i % num_components].append(i)

  return (cluster_index_lists, type_components)


def GetCrossLinks(num_clusters_list,
                  type_index1,
                  type_index2):
  """Returns the cross-type component linking between two specified clusterings.

  The linking is given as a list of tuples. Each tuple contains two cluster
  indices. Indices are decided based on the first output when num_clusters_list
  is passed to GetClusterTypeComponents. Each tuple is contained in the second
  output of the same function.

  Arguments:
    num_clusters_list: list of the number of clusters for each node type.
    type_index1: the first node type to return in the linking.
    type_index2: the second node type to return in the linking.

  Returns:
    cross_links: list of cluster index tuples.
  """
  cluster_index_lists, type_components = GetClusterTypeComponents(
      num_clusters_list)
  cross_links = []
  for component in type_components:
    for cluster_index1 in component:
      for cluster_index2 in component:
        if (cluster_index1 in cluster_index_lists[type_index1] and
            cluster_index2 in cluster_index_lists[type_index2]):
          cross_links.append((cluster_index1, cluster_index2))
  return cross_links


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
  cross_links = GetCrossLinks([num_clusters1, num_clusters2], 0, 1)
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
