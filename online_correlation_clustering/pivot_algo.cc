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

#include "pivot_algo.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <ostream>
#include <vector>

#include "random_handler.h"

std::vector<int> PivotAlgorithm::Cluster() {
  node_id_to_cluster_id_.resize(neighbors_.size(), -1);
  is_pivot_.resize(neighbors_.size(), false);
  // A random order of the node ids. The i-th value contains the id of the i-th
  // node.
  for (int i = 0; i < neighbors_.size(); i++) {
    double x = (RandomHandler::eng_() % std::numeric_limits<uint64_t>::max()) /
               std::numeric_limits<uint64_t>::max();
    nodes_ordered_by_rank_.emplace(x, i);
  }
  for (const auto& [rank, node_id] : nodes_ordered_by_rank_) {
    if (node_id_to_cluster_id_[node_id] == -1) {
      node_id_to_cluster_id_[node_id] = node_id;
      is_pivot_[node_id] = true;
      for (int j = 0; j < neighbors_[node_id].size(); j++) {
        if (node_id_to_cluster_id_[neighbors_[node_id][j]] == -1) {
          node_id_to_cluster_id_[neighbors_[node_id][j]] = node_id;
        }
      }
    }
  }
  return node_id_to_cluster_id_;
}

// Clusters the given graph based on the Pivot algorithm.
std::vector<int> PivotAlgorithm::InsertNodeToClustering(
    const std::vector<int>& new_nodes_neighbors) {
  int new_node_id = neighbors_.size();
  neighbors_.push_back(new_nodes_neighbors);
  for (int neighbor_id : new_nodes_neighbors) {
    if (neighbor_id != new_node_id) {
      neighbors_[neighbor_id].push_back(new_node_id);
    }
  }
  node_id_to_cluster_id_.push_back(-1);
  is_pivot_.push_back(false);

  double new_node_rank =
      (RandomHandler::eng_() % std::numeric_limits<uint64_t>::max()) /
      static_cast<double>(std::numeric_limits<uint64_t>::max());
  nodes_ordered_by_rank_.emplace(new_node_rank, new_node_id);
  rank_.push_back(new_node_rank);

  int num_nodes_recomputed = 0;
  for (const auto& [rank, node_id] : nodes_ordered_by_rank_) {
    if (rank < new_node_rank) continue;
    node_id_to_cluster_id_[node_id] = -1;
    is_pivot_[node_id] = false;
    ++num_nodes_recomputed;
  }

  for (const auto& [rank, node_id] : nodes_ordered_by_rank_) {
    if (rank < new_node_rank) continue;
    double min_rank = std::numeric_limits<double>::max();
    int min_rank_node_id;
    for (int j = 0; j < neighbors_[node_id].size(); j++) {
      if (is_pivot_[neighbors_[node_id][j]] &&
          rank_[neighbors_[node_id][j]] < min_rank) {
        min_rank = rank_[neighbors_[node_id][j]];
        min_rank_node_id = neighbors_[node_id][j];
      }
    }
    if (min_rank < rank_[node_id]) {
      node_id_to_cluster_id_[node_id] =
          node_id_to_cluster_id_[min_rank_node_id];
    }
    if (node_id_to_cluster_id_[node_id] == -1) {
      is_pivot_[node_id] = true;
      node_id_to_cluster_id_[node_id] = node_id;
    }
  }

  return node_id_to_cluster_id_;
}
