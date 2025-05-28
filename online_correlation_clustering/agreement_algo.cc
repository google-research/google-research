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

#include "agreement_algo.h"

#include <stdlib.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <set>
#include <utility>
#include <vector>

int AgreementCorrelationClustering::SortedVectorIntersection(
    const std::vector<int>& x, const std::vector<int>& y) {
  int pointer_1 = 0, pointer_2 = 0;
  int intersection_size = 0;
  while (pointer_1 < x.size() && pointer_2 < y.size()) {
    if (x[pointer_1] == y[pointer_2]) {
      intersection_size++;
      pointer_1++;
      pointer_2++;
    } else if (x[pointer_1] < y[pointer_2]) {
      pointer_1++;
    } else {
      pointer_2++;
    }
  }
  return intersection_size;
}

void AgreementCorrelationClustering::AddNodeToMaintainedGraph(
    const int new_node_id, const std::vector<int>& neighbors_of_new_node) {
  if (new_node_id != neighbors_.size()) {
    exit(1);
  }
  for (const int neighbor_id : neighbors_of_new_node) {
    if (neighbor_id == new_node_id) continue;
    neighbors_[neighbor_id].push_back(new_node_id);
  }
  neighbors_.push_back(neighbors_of_new_node);
  std::sort(neighbors_[new_node_id].begin(), neighbors_[new_node_id].end());
  maintained_agreement_neighbors_.push_back(std::set<int>());
  maintained_filtered_graph_.push_back(std::set<int>());
}

void BFS(const std::vector<std::set<int>>& neighbors, int start,
         std::vector<int>& connected_component) {
  std::vector<int> heap;
  heap.push_back(start);
  connected_component[start] = start;
  int head = 0;
  while (head < heap.size()) {
    int node = heap[head];
    for (int j : neighbors[node]) {
      if (connected_component[j] == -1) {
        heap.push_back(j);
        connected_component[j] = start;
      }
    }
    head++;
  }
}
void AgreementCorrelationClustering::UpdateCommonNeighborsOfEdgeEndpoints(
    const int new_node_id) {
  const std::set<int> neighbors_of_new_node(neighbors_[new_node_id].begin(),
                                            neighbors_[new_node_id].end());
  for (int neighbor : neighbors_of_new_node) {
    if (neighbor == new_node_id) continue;
    const auto& neighboring_pair = std::make_pair(
        std::min(new_node_id, neighbor), std::max(new_node_id, neighbor));
    common_neighbors_of_pairs_[neighboring_pair] =
        SortedVectorIntersection(neighbors_[new_node_id], neighbors_[neighbor]);
    for (const int neighbor_of_neighbor : neighbors_[neighbor]) {
      if (neighbor_of_neighbor == new_node_id) continue;
      if (neighbors_of_new_node.find(neighbor_of_neighbor) !=
              neighbors_of_new_node.end() &&
          neighbor_of_neighbor > neighbor) {
        common_neighbors_of_pairs_[std::make_pair(
            std::min(neighbor_of_neighbor, neighbor),
            std::max(neighbor_of_neighbor, neighbor))]++;
      }
    }
  }

  for (int neighbor : neighbors_of_new_node) {
    if (neighbor == new_node_id) continue;
    const auto& neighboring_pair = std::make_pair(
        std::min(new_node_id, neighbor), std::max(new_node_id, neighbor));
    double max_degree =
        std::max(neighbors_[new_node_id].size(), neighbors_[neighbor].size());
    if (neighbors_[new_node_id].size() + neighbors_[neighbor].size() -
            2 * common_neighbors_of_pairs_[neighboring_pair] <=
        beta_ * max_degree) {
      maintained_agreement_neighbors_[new_node_id].insert(neighbor);
      maintained_agreement_neighbors_[neighbor].insert(new_node_id);
    } else {
      maintained_agreement_neighbors_[new_node_id].erase(neighbor);
      maintained_agreement_neighbors_[neighbor].erase(new_node_id);
    }
    for (const int neighbor_of_neighbor : neighbors_[neighbor]) {
      if (neighbor_of_neighbor == new_node_id ||
          neighbor_of_neighbor == neighbor)
        continue;
      const auto& neighboring_pair =
          std::make_pair(std::min(neighbor_of_neighbor, neighbor),
                         std::max(neighbor_of_neighbor, neighbor));
      double max_degree = std::max(neighbors_[neighbor_of_neighbor].size(),
                                   neighbors_[neighbor].size());
      if (neighbors_[neighbor].size() +
              neighbors_[neighbor_of_neighbor].size() -
              2 * common_neighbors_of_pairs_[neighboring_pair] <=
          beta_ * max_degree) {
        maintained_agreement_neighbors_[neighbor_of_neighbor].insert(neighbor);
        maintained_agreement_neighbors_[neighbor].insert(neighbor_of_neighbor);
      } else {
        maintained_agreement_neighbors_[neighbor_of_neighbor].erase(neighbor);
        maintained_agreement_neighbors_[neighbor].erase(neighbor_of_neighbor);
      }
    }
  }
}

void AgreementCorrelationClustering::ComputeLightNodes() {
  for (int i = 0; i < num_nodes_; i++) {
    light_nodes_.push_back(neighbors_[i].size() - (
                               maintained_agreement_neighbors_[i].size() + 1) >
                           lambda_ * neighbors_[i].size());
  }
}

void AgreementCorrelationClustering::RemoveLightEdges() {
  for (int i = 0; i < maintained_agreement_neighbors_.size(); i++) {
    maintained_filtered_graph_[i].clear();
    bool i_is_light = light_nodes_[i];
    std::vector<int> neighbors_to_remove;
    for (int j : maintained_agreement_neighbors_[i]) {
      if (!i_is_light || !light_nodes_[j]) {
        maintained_filtered_graph_[i].insert(j);
        maintained_filtered_graph_[j].insert(i);
      }
    }
  }
}

void AgreementCorrelationClustering::ComputeConnectedComponents() {
  connected_components_ = std::vector<int>(num_nodes_, -1);
  for (int i = 0; i < num_nodes_; i++) {
    if (connected_components_[i] == -1 && !light_nodes_[i])
      BFS(maintained_filtered_graph_, i, connected_components_);
  }
  for (int i = 0; i < num_nodes_; i++) {
    if (connected_components_[i] == -1 && light_nodes_[i])
      connected_components_[i] = i;
  }
}

std::vector<int> AgreementCorrelationClustering::Cluster(
    const int new_node_id, const std::vector<int> neighbors_of_new_node) {
  AddNodeToMaintainedGraph(new_node_id, neighbors_of_new_node);
  num_nodes_++;
  num_edges_ += neighbors_of_new_node.size();

  light_nodes_.clear();

  UpdateCommonNeighborsOfEdgeEndpoints(new_node_id);
  ComputeLightNodes();
  RemoveLightEdges();
  ComputeConnectedComponents();

  return connected_components_;
}

void AgreementCorrelationClustering::RestartUpdateSequence() {
  neighbors_.clear();
  light_nodes_.clear();
  common_neighbors_of_pairs_.clear();
  maintained_agreement_neighbors_.clear();
  num_edges_ = 0;
  num_nodes_ = 0;
}
