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

"""Utils for clustering with weak and strong signals."""

import collections


class Clusters:
  """Stores clusters. Format: {ex_id: c_id}."""

  def __init__(self, assignments=None):
    self.assignments = assignments if assignments else {}

  def same_cluster(self, ex_id1, ex_id2):
    if not self.is_assigned(ex_id1) or not self.is_assigned(ex_id2):
      print('Clusters.same_cluster called with unassigned ex_ids.')
      return False
    return self.assignments[ex_id1] == self.assignments[ex_id2]

  def add(self, ex_ids, c_id):
    if ex_ids.__class__ == int:
      ex_ids = [ex_ids]
    for ex_id in ex_ids:
      self.assignments[ex_id] = c_id

  def is_assigned(self, ex_id):
    return ex_id in self.assignments

  def c_id_to_ex_id(self):
    c2e = collections.defaultdict(lambda: [])
    for ex_id, c_id in self.assignments.items():
      c2e[c_id].append(ex_id)
    return c2e

  def largest_cluster_size(self):
    return max(len(ex_ids) for _, ex_ids in self.c_id_to_ex_id().items())

  def __repr__(self):
    clusters_repr = []
    for c_id, ex_ids in self.c_id_to_ex_id().items():
      ex_ids_str = ' '.join([str(ex_id) for ex_id in ex_ids])
      clusters_repr.append(f'Cluster {c_id}: {ex_ids_str}')
    return '\n'.join(clusters_repr)


class ClusterPairs:
  """Stores pairs belonging to the same cluster. Format: {ex_id: [...]}."""

  def __init__(self):
    self.pairs = collections.defaultdict(lambda: [])

  def add(self, ex_id1, ex_id2):
    self.pairs[ex_id1] += [ex_id2]
    self.pairs[ex_id2] += [ex_id1]

  def all_examples_in_same_cluster(self, ex_id):
    """Given ex_id returns all examples in the same cluster as ex_id."""
    visited = set()

    def dfs(ex_id):
      if ex_id in visited:
        return []
      result = [ex_id]
      visited.add(ex_id)
      for same_cluster_ex_id in self.pairs[ex_id]:
        result.extend(dfs(same_cluster_ex_id))
      return result

    return dfs(ex_id)

  def convert_cluster_pairs_to_clusters(self, num_examples):
    """Converts a ClusterPairs object to a Clusters object."""
    clusters = Clusters()
    c_id = 0
    for ex_id in range(num_examples):
      if not clusters.is_assigned(ex_id):
        clusters.add(self.all_examples_in_same_cluster(ex_id), c_id)
        c_id += 1
    return clusters
