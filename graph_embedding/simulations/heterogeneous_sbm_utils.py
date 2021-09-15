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

"""Library for heterogeneous SBMs with node features.
"""
import math

from typing import List, Tuple


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
    return [(j - num_clusters2, i + num_clusters1) for
            (i, j) in reversed_result]
