// Copyright 2025 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "utils.h"

#include <stdlib.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <ostream>
#include <vector>

uint64_t ComputeCost::ComputeClusteringCost(
    const std::vector<std::vector<int>> neighbors,
    const std::vector<int>& clustering) {
  uint64_t sum_degrees = 0;
  uint64_t intercluster_edges = 0;
  std::map<int, int> cluster_size;
  // Compute the cost of the positive edges between clusters.
  for (int i = 0; i < neighbors.size(); i++) {
    uint64_t clean_degree = 0;
    for (int j = 0; j < neighbors[i].size(); j++) {
      if (clustering[neighbors[i][j]] != clustering[i]) {
        intercluster_edges++;
      }
      if (i != neighbors[i][j]) {
        ++clean_degree;
      }
    }
    sum_degrees += clean_degree;
    cluster_size[clustering[i]]++;
  }
  intercluster_edges /= 2;
  uint64_t total_possible_in_cluster_edges = 0;
  for (const auto [cluster_id, size] : cluster_size) {
    total_possible_in_cluster_edges += (size * (size - 1)) / 2;
  }
  uint64_t cost = intercluster_edges;
  cost += total_possible_in_cluster_edges -
          ((sum_degrees / 2) - intercluster_edges);

  return cost;
}

int RecourseCalculator::RecourseCostUsingMaxOverlap(
    std::vector<int> clustering_old, std::vector<int> clustering_new) {
  std::map<std::pair<int, int>, int> old_to_new_intersection_edges;
  std::vector<bool> old_cluster_is_matched(clustering_old.size(), false);
  std::vector<bool> new_cluster_is_matched(clustering_new.size(), false);
  std::vector<std::vector<int>> new_cluster_id_to_node_ids(
      clustering_new.size(), std::vector<int>());
  recourse_per_node_seen_.resize(clustering_new.size(), 0);

  for (int i = 0; i < clustering_old.size(); i++) {
    int old_cluster = clustering_old[i];
    int new_cluster = clustering_new[i];
    const auto& clustering_pair = std::make_pair(old_cluster, new_cluster);

    auto insersection_edge_iter = old_to_new_intersection_edges.find(
        std::make_pair(old_cluster, new_cluster));
    if (insersection_edge_iter == old_to_new_intersection_edges.end()) {
      old_to_new_intersection_edges.emplace(clustering_pair, 1);
    } else {
      ++insersection_edge_iter->second;
    }
    new_cluster_id_to_node_ids[new_cluster].push_back(i);
  }

  std::vector<std::pair<std::pair<int, int>, int>> intersection_edges_sorted;
  intersection_edges_sorted.reserve(old_to_new_intersection_edges.size());
  for (const auto& [key_pair, value] : old_to_new_intersection_edges) {
    intersection_edges_sorted.push_back(
        std::make_pair(std::make_pair(key_pair.first, key_pair.second), value));
  }
  std::sort(intersection_edges_sorted.begin(), intersection_edges_sorted.end(),
            [](const std::pair<std::pair<int, int>, int>& l,
               const std::pair<std::pair<int, int>, int>& r) {
              return l.second > r.second;
            });

  int total_missmatch = 0;
  for (const auto& [key_pair, value] : intersection_edges_sorted) {
    int old_cluster_id = key_pair.first;
    int new_cluster_id = key_pair.second;

    // If the clusters on both sides are not matched yet, then we have a new
    // match!
    if (old_cluster_is_matched[old_cluster_id] ||
        new_cluster_is_matched[new_cluster_id])
      continue;

    old_cluster_is_matched[old_cluster_id] = true;
    new_cluster_is_matched[new_cluster_id] = true;
    // The nodes of the Y-side cluster that are not intersecting the X-side
    // cluster need to change their cluster id.
    total_missmatch +=
        new_cluster_id_to_node_ids[new_cluster_id].size() - value;
    for (int node_id : new_cluster_id_to_node_ids[new_cluster_id]) {
      if (clustering_old[node_id] != old_cluster_id) {
        ++recourse_per_node_seen_[node_id];
      }
    }
  }
  // All unmatched clusters on the Y side contribute to difference of the two
  // clusterings as they need to be assigned a new cluster id.
  for (int cluster_id = 0; cluster_id < new_cluster_is_matched.size();
       cluster_id++) {
    if (new_cluster_is_matched[cluster_id]) {
      continue;
    } else {
      total_missmatch += new_cluster_id_to_node_ids[cluster_id].size();
      for (int node_id : new_cluster_id_to_node_ids[cluster_id]) {
        ++recourse_per_node_seen_[node_id];
      }
    }
  }
  return total_missmatch;
}

std::vector<int>
AgreementReconcileClustering::AgreeementClusteringTransformCost(
    std::vector<int> clustering_old, std::vector<int> clustering_new) {
  std::vector<std::vector<int>> cluster_id_to_node_id_old;
  std::vector<std::vector<int>> cluster_id_to_node_id_new;

  if (clustering_old.size() < clustering_new.size()) {
    clustering_old.push_back(clustering_old.size());
  }
  cluster_id_to_node_id_old.resize(clustering_old.size(), std::vector<int>());
  for (int node_id = 0; node_id < clustering_old.size(); node_id++) {
    cluster_id_to_node_id_old[clustering_old[node_id]].push_back(node_id);
  }

  cluster_id_to_node_id_new.resize(clustering_new.size(), std::vector<int>());
  for (int node_id = 0; node_id < clustering_new.size(); node_id++) {
    cluster_id_to_node_id_new[clustering_new[node_id]].push_back(node_id);
  }

  origin_cluster_size_.resize(clustering_new.size(), 0);
  while (maintained_clustering_.size() < clustering_new.size()) {
    int node_id = maintained_clustering_.size();
    int new_cluster_id = clustering_new[node_id];
    maintained_clustering_.push_back(new_cluster_id);
  }

  // Rule 1: The nodes in singleton clusters maintain their cluster ids.
  // The nodes in non-singleton clusters in both clusterings maintain their
  // cluster ids.
  std::vector<int> non_singleton_cluster_id_mapping(
      cluster_id_to_node_id_new.size(), -1);
  for (int cluster_id = 0; cluster_id < cluster_id_to_node_id_new.size();
       cluster_id++) {
    const std::vector<int>& new_cluster = cluster_id_to_node_id_new[cluster_id];

    if (new_cluster.size() == 1) {
      continue;
    } else if (new_cluster.size() > 1) {
      for (int node_id : new_cluster) {
        int old_node_cluster_id = clustering_old[node_id];
        // if the same node was part of a non-trivial cluster.
        if (cluster_id_to_node_id_old[old_node_cluster_id].size() > 1) {
          // maintained_clustering_[node_id] = old_node_cluster_id;
          non_singleton_cluster_id_mapping[cluster_id] = old_node_cluster_id;
          continue;
        }
      }
    }
  }

  // Find the new clusters whose nodes were singletons in the old clustering.
  std::vector<int> num_old_singletons_in_new_cluster(clustering_new.size(), 0);
  for (int node_id = 0; node_id < clustering_old.size() - 1; node_id++) {
    if (cluster_id_to_node_id_old[clustering_old[node_id]].size() == 1) {
      ++num_old_singletons_in_new_cluster[clustering_new[node_id]];
    }
  }

  // Rule 2: all newly formed clusters (are formed only by singletons from the
  // old cluster) get a new id.
  for (int cluster_id = 0; cluster_id < cluster_id_to_node_id_new.size();
       cluster_id++) {
    if (cluster_id_to_node_id_new[cluster_id].size() > 1 &&
        num_old_singletons_in_new_cluster[cluster_id] ==
            cluster_id_to_node_id_new[cluster_id].size()) {
      std::map<int, int> count_cluster_id_occurences;
      int max_value = 0;
      int cluster_max_value = 0;
      for (int node_id : cluster_id_to_node_id_new[cluster_id]) {
        int maintained_cluster_id = maintained_clustering_[node_id];
        auto iter = count_cluster_id_occurences.find(maintained_cluster_id);
        if (iter == count_cluster_id_occurences.end()) {
          count_cluster_id_occurences.emplace(maintained_cluster_id, 1);
        } else {
          ++iter->second;
          if (iter->second > max_value) {
            max_value = iter->second;
            cluster_max_value = maintained_cluster_id;
          }
        }
      }

      for (int node_id : cluster_id_to_node_id_new[cluster_id]) {
        maintained_clustering_[node_id] = cluster_max_value;
      }
    }
  }

  // Rule 3: all singletons getting into a new non-singleton cluster, get the
  // previous id of the cluster.
  for (int cluster_id = 0; cluster_id < cluster_id_to_node_id_new.size();
       cluster_id++) {
    if (num_old_singletons_in_new_cluster[cluster_id] <
            cluster_id_to_node_id_new[cluster_id].size() &&
        cluster_id_to_node_id_new[cluster_id].size() > 1) {
      for (int node_id : cluster_id_to_node_id_new[cluster_id]) {
        if (cluster_id_to_node_id_old[clustering_old[node_id]].size() == 1) {
          if (non_singleton_cluster_id_mapping[cluster_id] == -1) {
            maintained_clustering_[node_id] = cluster_id;
          } else {
            maintained_clustering_[node_id] =
                non_singleton_cluster_id_mapping[cluster_id];
          }
        }
      }
    }
  }

  // All clusters whose origin cluster has grown a lot, need to change cluster
  // ids.
  // Store the maintained clusters in vectors to be able to iterate over them.
  std::vector<std::vector<int>> cluster_id_to_node_id_maintained(
      maintained_clustering_.size(), std::vector<int>());
  for (int node_id = 0; node_id < maintained_clustering_.size(); node_id++) {
    cluster_id_to_node_id_maintained[maintained_clustering_[node_id]].push_back(
        node_id);
  }

  // Check which cluster ids, in the range 0...#nodes-1, we can re-use.
  std::vector<int> unused_cluster_ids;
  std::vector<bool> cluster_id_is_used(clustering_new.size(), false);
  for (int node_id = 0; node_id < maintained_clustering_.size(); node_id++) {
    cluster_id_is_used[maintained_clustering_[node_id]] = true;
  }
  for (int cluster_id = 0; cluster_id < cluster_id_is_used.size();
       cluster_id++) {
    if (!cluster_id_is_used[cluster_id]) {
      unused_cluster_ids.push_back(cluster_id);
      origin_cluster_size_[cluster_id] = 0;
    }
  }

  for (int cluster_id = 0; cluster_id < cluster_id_to_node_id_new.size();
       cluster_id++) {
    if (cluster_id_to_node_id_new[cluster_id].size() <= 1) continue;
    int maintained_cluster_id =
        maintained_clustering_[cluster_id_to_node_id_new[cluster_id][0]];
    if (cluster_id_to_node_id_new[cluster_id].size() >
        3.0 * origin_cluster_size_[maintained_cluster_id] / 2.0) {
      int new_cluster_id = unused_cluster_ids.back();
      unused_cluster_ids.pop_back();
      origin_cluster_size_[new_cluster_id] =
          cluster_id_to_node_id_new[cluster_id].size();
      for (int node_id : cluster_id_to_node_id_new[cluster_id]) {
        maintained_clustering_[node_id] = new_cluster_id;
      }
    }
  }

  // Make sure that no node has origin cluster of size 0; that could only be the
  // case for the newly added node.
  for (int node_id = 0; node_id < maintained_clustering_.size(); node_id++) {
    if (origin_cluster_size_[maintained_clustering_[node_id]] == 0)
      origin_cluster_size_[maintained_clustering_[node_id]] = 1;
  }

  return maintained_clustering_;
}
