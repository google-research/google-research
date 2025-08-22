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

#include "graph_handler.h"

#include <stdlib.h>

#include <algorithm>
#include <cstdio>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "random_handler.h"

void GraphHandler::ReadGraph() {
  int num_vertex = 0;
  // Endpoints of the edges.
  char s[50], t[50];
  while (scanf("%s\t%s", s, t) == 2) {
    std::string source = s;
    std::string target = t;
    if (id_map_.find(source) == id_map_.end()) {
      id_map_[source] = num_vertex++;
      neighbors_.push_back(std::vector<int>());
    }
    if (id_map_.find(target) == id_map_.end()) {
      id_map_[target] = num_vertex++;
      neighbors_.push_back(std::vector<int>());
    }
    int source_id = id_map_[source], target_id = id_map_[target];
    if (edges_.find(std::pair<int, int>(source_id, target_id)) ==
        edges_.end()) {
      neighbors_[source_id].push_back(target_id);
      if (source_id != target_id) {
        neighbors_[target_id].push_back(source_id);
      }

      edges_.insert(std::pair<int, int>(source_id, target_id));
      if (source_id != target_id) {
        edges_.insert(std::pair<int, int>(target_id, source_id));
      }
    }
  }
  num_nodes_ = num_vertex;
}

void GraphHandler::ReadGraphAndOrderByNodeId() {
  // Endpoints of the edges.
  char s[50], t[50];

  std::set<std::string> node_set;
  std::vector<std::pair<std::string, std::string>> edge_set;
  std::map<int, std::string> timestamp_to_node_map;
  while (scanf("%s\t%s", s, t) == 2) {
    std::string source = s;
    std::string target = t;

    int source_num = std::stoi(source);
    int target_num = std::stoi(target);
    source_num = source_num <= 9999999 ? source_num : source_num - 110000000;
    source_num = source_num < 9000000 ? source_num + 10000000
                                                 : source_num;
    source = std::to_string(source_num);
    target_num = target_num <= 9999999 ? target_num : target_num - 110000000;
    target_num = target_num < 9000000 ? target_num + 10000000
                                                 : target_num;

    if (node_set.find(source) == node_set.end()) {
      node_set.emplace(source);
      timestamp_to_node_map.emplace(source_num, source);
    }
    if (node_set.find(target) == node_set.end()) {
      node_set.emplace(target);
      timestamp_to_node_map.emplace(target_num, target);
    }
    edge_set.push_back(std::pair<std::string, std::string>(source, target));
  }

  int num_vertex = 0;
  for (const auto& [timestamp_num, node_name] : timestamp_to_node_map) {
    if (node_set.find(node_name) != node_set.end() &&
        id_map_.find(node_name) == id_map_.end()) {
      id_map_[node_name] = num_vertex++;
      neighbors_.push_back(std::vector<int>());
    }
  }

  for (const auto& [source, target] : edge_set) {
    int source_id = id_map_[source], target_id = id_map_[target];
    if (edges_.find(std::pair<int, int>(source_id, target_id)) ==
        edges_.end()) {
      neighbors_[source_id].push_back(target_id);
      if (source_id != target_id) {
        neighbors_[target_id].push_back(source_id);
      }

      edges_.insert(std::pair<int, int>(source_id, target_id));
      if (source_id != target_id) {
        edges_.insert(std::pair<int, int>(target_id, source_id));
      }
    }
  }

  num_nodes_ = num_vertex;
}

void GraphHandler::AddMissingSelfLoops() {
  for (int i = 0; i < num_nodes_; i++) {
    if (std::find(neighbors_[i].begin(), neighbors_[i].end(), i) ==
        neighbors_[i].end()) {
      neighbors_[i].push_back(i);
    }
  }
}

void GraphHandler::StartMaintainingOnlineGraphInstance(bool shuffle_order) {
  online_neighbors_.clear();
  node_to_order_.clear();
  online_num_edges_ = 0;
  for (int i = 0; i < num_nodes_; i++) {
    node_to_order_.push_back(i);
  }

  if (shuffle_order) {
    for (int i = 0; i < node_to_order_.size(); i++) {
      int x = RandomHandler::eng_() % node_to_order_.size();
      std::swap(node_to_order_[i], node_to_order_[x]);
    }
  }

  order_to_node_.resize(num_nodes_);
  for (int i = 0; i < node_to_order_.size(); i++) {
    order_to_node_[node_to_order_[i]] = i;
  }
}

void GraphHandler::RemoveAllOnlineNodes() {
  online_neighbors_.clear();
  online_num_edges_ = 0;
}

std::vector<int> GraphHandler::AddNextOnlineNode() {
  if (online_neighbors_.size() >= num_nodes_) exit(1);
  int next_order = online_neighbors_.size();
  int next_node = order_to_node_[next_order];
  std::vector<int> neighbors_of_next_node;

  for (const int neighbor_id : neighbors_[next_node]) {
    int neighbor_order = node_to_order_[neighbor_id];
    if (neighbor_order <= next_order) {
      neighbors_of_next_node.push_back(neighbor_order);
      if (neighbor_order != next_order) {
        online_neighbors_[neighbor_order].push_back(next_order);
      }
    }
  }
  online_neighbors_.push_back(neighbors_of_next_node);

  online_num_edges_ += neighbors_of_next_node.size();
  return neighbors_of_next_node;
}

bool GraphHandler::NextOnlineNodeExists() {
  if (online_neighbors_.size() >= num_nodes_) return false;
  return true;
}
