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

#include "gklx_algorithm.h"

#include <math.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <new>
#include <ostream>
#include <queue>
#include <utility>
#include <vector>

#include "random_handler.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"

double GKLXAlgorithm::CurrentClusteringCost() {
  AssignClientsToFacilities();
  double cost_sum = 0.0;
  for (const auto& [cluster_id, cost] : cluster_cost_) {
    cost_sum += cost;
  }
  return cost_sum;
}

void GKLXAlgorithm::CheckCorrectStatusOfDatastructures() {
  absl::flat_hash_map<int, int> num_of_unprocessed_descendants;
  absl::flat_hash_map<int, bool> has_marked_strict_descendants;
  for (const auto& [node, parent] : parent_in_hst_) {
    if (node == parent) continue;
    has_marked_strict_descendants[node] = false;
    if (!num_of_unprocessed_descendants.contains(parent)) {
      num_of_unprocessed_descendants[parent] = 1;
    } else {
      num_of_unprocessed_descendants[parent]++;
    }
  }

  std::vector<int> nodes_to_process;
  for (const auto& [node, parent] : parent_in_hst_) {
    if (!num_of_unprocessed_descendants.contains(node)) {
      nodes_to_process.push_back(node);
    }
    if (node_is_marked_[node]) {
      has_marked_strict_descendants[parent] = true;
    }
  }

  for (const auto& [facility_id, cost] : facility_cost_) {
    CHECK_EQ(clients_attached_to_facility_[facility_id].size(),
             num_clients_at_facility_[facility_id])
        << ", facility : " << facility_id << std::endl;
  }

  while (!nodes_to_process.empty()) {
    int node_to_process = nodes_to_process.back();
    nodes_to_process.pop_back();
    if (!has_marked_strict_descendants[node_to_process] &&
        node_is_marked_[node_to_process]) {
      CHECK(IsMarked(node_to_process))
          << "Node : " << node_to_process << std::endl;
      CHECK(node_is_open_[node_to_process])
          << "Node : " << node_to_process << std::endl;
    }

    int parent_of_node = parent_in_hst_[node_to_process];
    if (node_is_marked_[node_to_process]) {
      CHECK(node_is_marked_[parent_of_node])
          << "Node: " << node_to_process << ", parent: " << parent_of_node
          << std::endl;
    }
    if (has_marked_strict_descendants[node_to_process] ||
        node_is_marked_[node_to_process]) {
      has_marked_strict_descendants[parent_of_node] = true;
    }

    if (node_to_process == parent_of_node) {
      CHECK(num_of_unprocessed_descendants.empty());
      break;
    }
    num_of_unprocessed_descendants[parent_of_node]--;
    if (num_of_unprocessed_descendants[parent_of_node] == 0) {
      nodes_to_process.push_back(parent_of_node);
      num_of_unprocessed_descendants.erase(parent_of_node);
    }
  }
  CHECK(num_of_unprocessed_descendants.empty());
}

void GKLXAlgorithm::AssignClientsToFacilities() {
  absl::flat_hash_map<int, absl::flat_hash_set<int>> children_of_node;
  absl::flat_hash_map<int, int> node_to_connect_descendant;
  absl::flat_hash_map<int, int> num_of_unprocessed_descendants;
  int root;
  for (const auto& [node, parent] : parent_in_hst_) {
    if (node == parent) {
      root = node;
      continue;
    }
    if (!children_of_node.contains(parent)) {
      children_of_node[parent] = absl::flat_hash_set<int>();
    }
    children_of_node[parent].insert(node);

    if (!num_of_unprocessed_descendants.contains(parent)) {
      num_of_unprocessed_descendants[parent] = 1;
    } else {
      num_of_unprocessed_descendants[parent]++;
    }
  }

  // First inform all ancestors of each node that they have a descendant open at
  // the highest level.
  std::vector<int> nodes_to_process;
  for (const auto& [node, parent] : parent_in_hst_) {
    if (!num_of_unprocessed_descendants.contains(node)) {
      nodes_to_process.push_back(node);
    }
  }

  while (!nodes_to_process.empty()) {
    int node_to_process = nodes_to_process.back();
    nodes_to_process.pop_back();
    int parent_of_node = parent_in_hst_[node_to_process];

    if (node_is_open_[node_to_process]) {
      node_to_connect_descendant[node_to_process] = node_to_process;
    }

    if (node_to_process == parent_of_node) {
      CHECK_EQ(num_of_unprocessed_descendants.size(), 0);
      break;
    }

    if (node_to_connect_descendant.contains(node_to_process)) {
      if (!node_to_connect_descendant.contains(parent_of_node)) {
        node_to_connect_descendant[parent_of_node] =
            node_to_connect_descendant[node_to_process];
      } else {
        if (level_in_hst_[node_to_connect_descendant[node_to_process]] >
            level_in_hst_[node_to_connect_descendant[parent_of_node]]) {
          node_to_connect_descendant[parent_of_node] =
              node_to_connect_descendant[node_to_process];
        }
      }
    }

    num_of_unprocessed_descendants[parent_of_node]--;
    if (num_of_unprocessed_descendants[parent_of_node] == 0) {
      nodes_to_process.push_back(parent_of_node);
      num_of_unprocessed_descendants.erase(parent_of_node);
    }
  }
  CHECK(num_of_unprocessed_descendants.empty());
  // Next, propagate to the subtree of each node x the node in
  // node_to_connect_descendant[x], unless there is someother node to propagate
  // along the path to each leaf.
  std::queue<int> BFS_queue;
  cluster_cost_.clear();
  center_of_client_.clear();
  BFS_queue.push(root);
  if (!node_to_connect_descendant.contains(root)) return;
  num_of_open_facilities_ = 0;
  total_openning_cost_ = 0.0;
  total_connection_cost_ = 0.0;
  while (!BFS_queue.empty()) {
    int next_node = BFS_queue.front();
    BFS_queue.pop();
    CHECK(node_to_connect_descendant.contains(next_node));
    int node_to_propagate = node_to_connect_descendant[next_node];
    if (!children_of_node[next_node].empty()) {
      for (int child : children_of_node[next_node]) {
        BFS_queue.push(child);
        if (!node_to_connect_descendant.contains(child)) {
          node_to_connect_descendant[child] = node_to_propagate;
        }
      }
    } else {
      for (const int client : clients_attached_to_facility_[next_node]) {
        int facility_of_propagated_node =
            facility_of_tree_node_[node_to_propagate];
        center_of_client_[client] = facility_of_propagated_node;
        if (!cluster_cost_.contains(facility_of_propagated_node)) {
          cluster_cost_[facility_of_propagated_node] =
              facility_cost_[facility_of_propagated_node];

          num_of_open_facilities_++;
          total_openning_cost_ += cluster_cost_[facility_of_propagated_node];
        }
        cluster_cost_[facility_of_propagated_node] +=
            online_client_to_facilities_neighbors_[client]
                                                  [facility_of_propagated_node];
        total_connection_cost_ +=
            online_client_to_facilities_neighbors_[client]
                                                  [facility_of_propagated_node];
      }
    }
  }

  std::cout << "Total openning cost: " << total_openning_cost_ << " ("
            << num_of_open_facilities_
            << " facilities), total connection cost: " << total_connection_cost_
            << std::endl;
}

void GKLXAlgorithm::SetFacilities(
    const absl::flat_hash_map<int, double> facility_costs) {
  for (const auto& [facility, cost] : facility_costs) {
    facility_cost_[facility] = cost;
    num_clients_at_facility_[facility] = 0;
    num_conditional_clients_at_node_[facility] = 0;
    alphas_[facility] = 1;
    betas_[facility] = 1;
  }
}

int GKLXAlgorithm::NearestFacilityOfClient(
    const int client_id, absl::flat_hash_map<int, double> neighbors) {
  double min_distance = std::numeric_limits<double>::max();
  int arg_min;
  for (const auto& [neighbor, distance] : neighbors) {
    if (distance < min_distance) {
      min_distance = distance;
      arg_min = neighbor;
    }
  }
  return arg_min;
}

bool GKLXAlgorithm::ShouldBeOpen(const int node_id) {
  bool is_leaf_node =
      online_facility_to_facilities_neighbors_.contains(node_id);
  if (is_leaf_node) {
    return node_is_marked_[node_id];
  }
  if ((double)num_conditional_clients_at_node_[node_id] *
          pow(2.0, (double)level_in_hst_[node_id]) >
      (double)cost_of_tree_node_[node_id] /
          (double)(alphas_[node_id] * betas_[node_id])) {
    return true;
  }
  return false;
}

void GKLXAlgorithm::CloseNode(const int node_id) {
  node_is_open_[node_id] = false;
  betas_[node_id] = 1;
}

void GKLXAlgorithm::OpenNode(const int node_id) {
  node_is_open_[node_id] = true;
  betas_[node_id] = 2;
}

void GKLXAlgorithm::MarkNode(const int node_id) {
  node_is_marked_[node_id] = true;
  alphas_[node_id] = 2;
  betas_[node_id] = 1;
  int parent = parent_in_hst_[node_id];
  if (parent != node_id) {
    num_conditional_clients_at_node_[parent] -=
        num_clients_at_facility_[node_id];
  }
}

void GKLXAlgorithm::UnmarkNode(const int node_id) {
  node_is_marked_[node_id] = false;
  alphas_[node_id] = 1;
  betas_[node_id] = 1;
  int parent = parent_in_hst_[node_id];
  if (parent != node_id) {
    num_conditional_clients_at_node_[parent] +=
        num_clients_at_facility_[node_id];
  }
}

void GKLXAlgorithm::InsertPoint(int client_id,
                                absl::flat_hash_map<int, double> neighbors) {
  int nearest_facility = NearestFacilityOfClient(client_id, neighbors);
  online_client_to_facilities_neighbors_[client_id] =
      absl::flat_hash_map<int, double>(neighbors.begin(), neighbors.end());
  nearest_facility_of_client_[client_id] = nearest_facility;
  if (!clients_attached_to_facility_.contains(nearest_facility)) {
    clients_attached_to_facility_[nearest_facility] =
        absl::flat_hash_set<int>();
  }
  clients_attached_to_facility_[nearest_facility].insert(client_id);

  std::vector<int> facility_to_root_path;
  std::vector<int> root_to_facility_path;
  int current_node = nearest_facility;

  num_clients_at_facility_[current_node]++;
  facility_to_root_path.push_back(current_node);
  if (!node_is_marked_[current_node] &&
      current_node != parent_in_hst_[current_node]) {
    num_conditional_clients_at_node_[parent_in_hst_[current_node]]++;
  }

  while (parent_in_hst_[current_node] != current_node) {
    current_node = parent_in_hst_[current_node];
    num_clients_at_facility_[current_node]++;
    if (!node_is_marked_[current_node] &&
        current_node != parent_in_hst_[current_node]) {
      num_conditional_clients_at_node_[parent_in_hst_[current_node]]++;
    }
    facility_to_root_path.push_back(parent_in_hst_[current_node]);
  }

  for (int i = facility_to_root_path.size() - 1; i >= 0; i--) {
    root_to_facility_path.push_back(facility_to_root_path[i]);
  }

  for (int i = 0; i < root_to_facility_path.size(); i++) {
    int node = root_to_facility_path[i];
    if (!node_is_marked_[node]) {
      if (IsMarked(node)) {
        MarkNode(node);
      }
      int parent = parent_in_hst_[node];
      if (parent != node && node_is_open_[parent] && !ShouldBeOpen(parent)) {
        CloseNode(parent);
      }
      if (!node_is_open_[node] && ShouldBeOpen(node)) {
        OpenNode(node);
      }
    }
  }
}

void GKLXAlgorithm::DeletePoint(int client_id) {
  int nearest_facility = nearest_facility_of_client_[client_id];
  clients_attached_to_facility_[nearest_facility].erase(client_id);
  nearest_facility_of_client_.erase(client_id);

  std::vector<int> facility_to_root_path;
  int current_node = nearest_facility;

  num_clients_at_facility_[current_node]--;
  facility_to_root_path.push_back(current_node);
  if (!node_is_marked_[current_node] &&
      current_node != parent_in_hst_[current_node]) {
    num_conditional_clients_at_node_[parent_in_hst_[current_node]]--;
  }

  while (parent_in_hst_[current_node] != current_node) {
    current_node = parent_in_hst_[current_node];
    num_clients_at_facility_[current_node]--;
    facility_to_root_path.push_back(current_node);
    if (!node_is_marked_[current_node] &&
        current_node != parent_in_hst_[current_node]) {
      num_conditional_clients_at_node_[parent_in_hst_[current_node]]--;
    }
  }

  for (int i = 0; i < facility_to_root_path.size(); i++) {
    int node = facility_to_root_path[i];
    if (node_is_marked_[node]) {
      if (!IsMarked(node)) {
        UnmarkNode(node);
      }

      int parent = parent_in_hst_[node];
      if (parent != node && !node_is_open_[parent] && ShouldBeOpen(parent)) {
        OpenNode(parent);
      }
      if (node_is_open_[node] && !ShouldBeOpen(node)) {
        CloseNode(node);
      }
    }
  }
}

bool GKLXAlgorithm::IsMarked(int node_id) {
  bool ret = (double)num_clients_at_facility_[node_id] *
                 pow(2.0, (double)level_in_hst_[node_id]) >
             (double)cost_of_tree_node_[node_id] / alphas_[node_id];
  return ret;
}

void GKLXAlgorithm::SetFacilityToFacilityDistances(
    absl::flat_hash_map<int, absl::flat_hash_map<int, double>>
        facility_to_facility_distances) {
  online_facility_to_facilities_neighbors_.reserve(
      facility_to_facility_distances.size());
  for (const auto& [facility, neighbors] : facility_to_facility_distances) {
    online_facility_to_facilities_neighbors_[facility] =
        absl::flat_hash_map<int, double>(neighbors.size());
    for (const auto& [neighbor, distance] : neighbors) {
      online_facility_to_facilities_neighbors_[facility][neighbor] = distance;
    }
  }
}

void GKLXAlgorithm::SetMinAndMaxDistance() {
  double min_distance = std::numeric_limits<double>::max();
  double max_distance = std::numeric_limits<double>::min();
  for (const auto& [facility_id, neighbors_of_facility] :
       online_facility_to_facilities_neighbors_) {
    for (const auto& [neighbor, distance] : neighbors_of_facility) {
      if (distance < min_distance) {
        min_distance = distance;
      }
      if (distance > max_distance) {
        max_distance = distance;
      }
    }
  }
  for (const auto& [facility_id, cost] : facility_cost_) {
    if (cost > max_distance) {
      max_distance = cost;
    }
  }
  min_distance_ = min_distance;
  max_distance_ = max_distance;
}

double GKLXAlgorithm::DrawBetaFromDistribution() {
  double random_number = log(2.0) * (log(4.0) - log(2.0)) *
                         (double)RandomHandler::eng_() /
                         (double)std::numeric_limits<uint_fast64_t>::max();
  return 1.0 / (random_number * log(2.0));
}

absl::flat_hash_map<int, std::vector<int>>
GKLXAlgorithm::ComputedDominanceSequencePerNode(
    absl::flat_hash_map<int, int> node_ranks) {
  SetMinAndMaxDistance();
  absl::flat_hash_map<int, std::vector<int>> dominance_sequence_per_facility;
  double beta = DrawBetaFromDistribution();
  std::vector<double> dominance_distance_buckets;
  dominance_distance_buckets.reserve(ceil(log2(max_distance_)));
  for (int i = 0; i < ceil(log2(max_distance_)); i++) {
    dominance_distance_buckets.push_back(beta * pow(2.0, i));
  }

  for (const auto& [facility_id, cost] : facility_cost_) {
    std::vector<int> dominators;
    dominators.reserve(dominance_distance_buckets.size());
    std::vector<std::pair<int, double>> neighbors(
        online_facility_to_facilities_neighbors_[facility_id].begin(),
        online_facility_to_facilities_neighbors_[facility_id].end());

    std::sort(neighbors.begin(), neighbors.end(), [](auto& left, auto& right) {
      return left.second < right.second;
    });

    int current_index_of_distance_bucket = 0;
    int current_smallest_rank = node_ranks[facility_id];
    int current_node_with_smallest_rank = facility_id;
    for (const auto& [node_id, distance] : neighbors) {
      while (distance >
             dominance_distance_buckets[current_index_of_distance_bucket]) {
        dominators.push_back(current_node_with_smallest_rank);
        current_index_of_distance_bucket++;
      }
      if (node_ranks[node_id] < current_smallest_rank) {
        current_smallest_rank = node_ranks[node_id];
        current_node_with_smallest_rank = node_id;
      }
      CHECK_LE(current_index_of_distance_bucket,
               dominance_distance_buckets.size() - 1);
    }
    while (current_index_of_distance_bucket <
           dominance_distance_buckets.size()) {
      dominators.push_back(current_node_with_smallest_rank);
      current_index_of_distance_bucket++;
    }

    dominators.push_back(current_node_with_smallest_rank);
    dominance_sequence_per_facility[facility_id] = dominators;
  }
  return dominance_sequence_per_facility;
}

void GKLXAlgorithm::ComputedFacilityHST() {
  CHECK_EQ(facility_cost_.size(),
           online_facility_to_facilities_neighbors_.size());
  parent_in_hst_.reserve(facility_cost_.size());

  std::vector<int> index_to_facility;
  for (const auto& [facility_id, cost] : facility_cost_) {
    index_to_facility.push_back(facility_id);
  }

  for (int i = 0; i < index_to_facility.size(); i++) {
    int x = RandomHandler::eng_() % index_to_facility.size();
    std::swap(index_to_facility[i], index_to_facility[x]);
  }

  absl::flat_hash_map<int, int> node_ranks;
  node_ranks.reserve(index_to_facility.size());
  int max_facility_id = 0;
  for (int i = 0; i < index_to_facility.size(); i++) {
    node_ranks[index_to_facility[i]] = i;
    if (index_to_facility[i] > max_facility_id) {
      max_facility_id = index_to_facility[i];
    }
  }
  std::cout << " Max facility id: " << max_facility_id << std::endl;

  int next_node_id = max_facility_id + 1;
  absl::flat_hash_map<int, std::vector<int>> dominance_sequence_per_node =
      ComputedDominanceSequencePerNode(node_ranks);

  CHECK_EQ(dominance_sequence_per_node.size(), facility_cost_.size());
  absl::flat_hash_map<std::pair<int, int>, int> id_of_node_level_pair;

  for (const auto& [node, dominance_sequence] : dominance_sequence_per_node) {
    // As we construct the ancestors of `node`, this int keeps track of the id
    // of the ancestors of `node` at the highest level that we have seen so far.
    int highest_seen_ancestor = node;
    level_in_hst_[node] = -1;
    for (int i = 0; i < dominance_sequence.size(); i++) {
      if (id_of_node_level_pair.contains(
              std::make_pair(dominance_sequence[i], i))) {
        parent_in_hst_[highest_seen_ancestor] =
            id_of_node_level_pair.at(std::make_pair(dominance_sequence[i], i));
        break;
      }
      int newly_created_node_id = next_node_id++;
      level_in_hst_[newly_created_node_id] = i;
      id_of_node_level_pair[std::make_pair(dominance_sequence[i], i)] =
          newly_created_node_id;
      parent_in_hst_[highest_seen_ancestor] = newly_created_node_id;
      highest_seen_ancestor = newly_created_node_id;
    }
    if (!parent_in_hst_.contains(highest_seen_ancestor)) {
      parent_in_hst_[highest_seen_ancestor] = highest_seen_ancestor;
    }
  }

  absl::flat_hash_map<int, int> num_of_unprocessed_descendants;
  for (const auto& [node, parent] : parent_in_hst_) {
    if (node == parent) continue;
    if (!num_of_unprocessed_descendants.contains(parent)) {
      num_of_unprocessed_descendants[parent] = 1;
    } else {
      num_of_unprocessed_descendants[parent]++;
    }
    cost_of_tree_node_[node] = std::numeric_limits<double>::max();
  }

  for (const auto& [facility, cost] : facility_cost_) {
    cost_of_tree_node_[facility] = cost;
    facility_of_tree_node_[facility] = facility;
  }

  std::vector<int> nodes_to_process;
  for (const auto& [node, parent] : parent_in_hst_) {
    if (!num_of_unprocessed_descendants.contains(node)) {
      nodes_to_process.push_back(node);
    }
  }

  while (!nodes_to_process.empty()) {
    int node_to_process = nodes_to_process.back();
    nodes_to_process.pop_back();
    CHECK(cost_of_tree_node_.contains(node_to_process));
    int parent_of_node = parent_in_hst_[node_to_process];
    double cost_of_node = cost_of_tree_node_[node_to_process];
    if (!cost_of_tree_node_.contains(parent_of_node)) {
      cost_of_tree_node_[parent_of_node] = cost_of_node;
      facility_of_tree_node_[parent_of_node] =
          facility_of_tree_node_[node_to_process];
    } else {
      if (cost_of_node < cost_of_tree_node_[parent_of_node]) {
        cost_of_tree_node_[parent_of_node] = cost_of_node;
        facility_of_tree_node_[parent_of_node] =
            facility_of_tree_node_[node_to_process];
      }
    }

    if (node_to_process == parent_of_node) {
      CHECK_EQ(num_of_unprocessed_descendants.size(), 0)
          << "Node = " << node_to_process;
      break;
    }
    num_of_unprocessed_descendants[parent_of_node]--;
    if (num_of_unprocessed_descendants[parent_of_node] == 0) {
      nodes_to_process.push_back(parent_of_node);
      num_of_unprocessed_descendants.erase(parent_of_node);
    }
  }
  CHECK(num_of_unprocessed_descendants.empty());

  for (const auto& [node, parent] : parent_in_hst_) {
    alphas_[node] = 1;
    betas_[node] = 1;
    num_conditional_clients_at_node_[node] = 0;
    node_is_marked_[node] = false;
    node_is_open_[node] = false;
    num_clients_at_facility_[node] = 0;
    num_conditional_clients_at_node_[node] = 0;
  }
}

int GKLXAlgorithm::ComputeRecourseAndUpdateReference() {
  int recourse_sum = 0;
  for (const auto& [client, cluster_id] : center_of_client_) {
    if (prev_cluster_center_of_client_.contains(client) &&
        center_of_client_[cluster_id] !=
            prev_cluster_center_of_client_[client]) {
      ++recourse_sum;
    }
    prev_cluster_center_of_client_[client] = center_of_client_[cluster_id];
  }

  return recourse_sum;
}
