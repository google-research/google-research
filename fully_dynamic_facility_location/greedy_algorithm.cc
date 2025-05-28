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

#include "greedy_algorithm.h"

#include <math.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <ostream>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"

double GreedyAlgorithm::CurrentClusteringCost() {
  ComputeClustering();
  double cost_sum = 0.0;
  for (const auto& [cluster_id, cost] : cluster_cost_) {
    cost_sum += cost;
  }
  return cost_sum;
}

double GreedyAlgorithm::CurrentConnectionCost() {
  double total_connection_cost = 0.0;

  for (const auto& [client_id, facility_id] : cluster_of_client_) {
    total_connection_cost +=
        online_neighbors_of_facilities_[facility_id][client_id];
  }
  return total_connection_cost;
}

double GreedyAlgorithm::CurrentOpenningCost() {
  double total_openning_cost = 0.0;

  for (const auto& [facility_id, cluster] : clustering_) {
    total_openning_cost += facility_cost_[facility_id];
  }

  return total_openning_cost;
}

int GreedyAlgorithm::NearestOpenFacilityOfClient(const int client_id) {
  double min_distance = std::numeric_limits<double>::max();
  int arg_min;
  for (const auto& [neighbor, distance] :
       online_neighbors_of_clients_[client_id]) {
    if (!clustering_.contains(neighbor)) continue;
    if (distance < min_distance) {
      min_distance = distance;
      arg_min = neighbor;
    }
  }
  return arg_min;
}

int GreedyAlgorithm::LevelOfDistance(const double distance) {
  return ceil(log2(distance));
}

void GreedyAlgorithm::SetFacilities(
    const absl::flat_hash_map<int, double> facility_costs) {
  for (const auto& [facility, cost] : facility_costs) {
    facility_cost_[facility] = cost;
  }
}

void GreedyAlgorithm::InsertPoint(const int client_id,
                                  absl::flat_hash_map<int, double> neighbors) {
  // std::cout << "Insert(" << client_id << ")" << std::endl;
  online_neighbors_of_clients_[client_id] = neighbors;
  for (const auto& [facility, distance] : neighbors) {
    online_neighbors_of_facilities_[facility].insert(
        std::pair<int, double>(client_id, distance));
  }
}

void GreedyAlgorithm::DeletePoint(const int client_id) {
  for (const auto& [facility_id, distance] :
       online_neighbors_of_clients_[client_id]) {
    online_neighbors_of_facilities_[facility_id].erase(client_id);
  }
  online_neighbors_of_clients_.erase(client_id);
}

void GreedyAlgorithm::ComputeClustering() {
  cluster_of_client_.clear();
  cluster_cost_.clear();
  clustering_.clear();

  while (cluster_of_client_.size() < online_neighbors_of_clients_.size()) {
    std::tuple<int, double, std::vector<int>> best_score_and_cluster;
    const auto& optional_blocking_cluster = FindBlockingCluster();

    if (optional_blocking_cluster.has_value()) {
      best_score_and_cluster = optional_blocking_cluster.value();
    }

    for (const auto& [client_id, neighborhood] : online_neighbors_of_clients_) {
      if (clustering_.empty()) break;
      if (cluster_of_client_.contains(client_id)) continue;
      int nearest_facility = NearestOpenFacilityOfClient(client_id);
      double best_ordinary = std::numeric_limits<double>::max();
      if (neighborhood.at(nearest_facility) < best_ordinary) {
        best_ordinary = neighborhood.at(nearest_facility);
      }
      if (neighborhood.at(nearest_facility) <
          std::get<1>(best_score_and_cluster)) {
        best_score_and_cluster =
            std::make_tuple(nearest_facility, neighborhood.at(nearest_facility),
                            std::vector<int>({client_id}));
      }
    }
    int center_facility = std::get<0>(best_score_and_cluster);
    std::vector<int> cluster = std::get<2>(best_score_and_cluster);
    if (clustering_.contains(center_facility)) {
      clustering_[center_facility].insert(cluster[0]);
      cluster_cost_[center_facility] +=
          online_neighbors_of_clients_[cluster[0]][center_facility];
      cluster_of_client_[cluster[0]] = center_facility;
    } else {
      clustering_[center_facility] = absl::flat_hash_set<int>();
      cluster_cost_[center_facility] = facility_cost_[center_facility];

      for (int client : cluster) {
        clustering_[center_facility].insert(client);
        cluster_cost_[center_facility] +=
            online_neighbors_of_clients_[client][center_facility];
        cluster_of_client_[client] = center_facility;
      }
    }
  }
}

std::optional<std::pair<double, std::vector<int>>>
GreedyAlgorithm::IsBlockingAtLevel(const int facility_id) {
  double best_that_this_facility_can_do = std::numeric_limits<double>::max();
  std::vector<int> best_cluster;

  double current_total_cost = facility_cost_[facility_id];
  int current_cluster_size = 0;
  std::vector<int> new_cluster;
  std::vector<std::pair<int, double>> facility_neighbors(
      online_neighbors_of_facilities_[facility_id].begin(),
      online_neighbors_of_facilities_[facility_id].end());
  std::sort(facility_neighbors.begin(), facility_neighbors.end(),
            [](auto& left, auto& right) { return left.second < right.second; });

  for (const auto& [client, distance] : facility_neighbors) {
    if (cluster_of_client_.contains(client)) continue;
    new_cluster.push_back(client);
    current_total_cost += distance;
    current_cluster_size++;

    double current_average_cost = current_total_cost / current_cluster_size;
    if (current_average_cost < best_that_this_facility_can_do) {
      best_that_this_facility_can_do = current_average_cost;
      best_cluster.clear();
      best_cluster = new_cluster;
    }
  }

  if (best_that_this_facility_can_do < std::numeric_limits<double>::max()) {
    return std::make_pair(best_that_this_facility_can_do, best_cluster);
  }
  return std::nullopt;
}

absl::optional<std::tuple<int, double, std::vector<int>>>
GreedyAlgorithm::FindBlockingCluster() {
  std::vector<int> best_blocking_cluster;
  double best_blocking_cost = std::numeric_limits<double>::max();
  int best_facility;
  for (const auto& [facility, cost] : facility_cost_) {
    if (clustering_.contains(facility)) continue;
    const auto& is_blocking = IsBlockingAtLevel(facility);
    if (is_blocking.has_value()) {
      if (is_blocking.value().first < best_blocking_cost) {
        best_blocking_cost = is_blocking.value().first;
        best_facility = facility;
        best_blocking_cluster = is_blocking.value().second;
      }
    }
  }
  if (best_blocking_cost < std::numeric_limits<double>::max()) {
    return std::make_tuple(best_facility, best_blocking_cost,
                           best_blocking_cluster);
  }
  return std::nullopt;
}

int GreedyAlgorithm::ComputeRecourseAndUpdateReference() {
  int recourse_sum = 0;
  for (const auto& [client, cluster_id] : cluster_of_client_) {
    if (prev_cluster_center_of_client_.contains(client) &&
        cluster_id != prev_cluster_center_of_client_[client]) {
      ++recourse_sum;
    }
    prev_cluster_center_of_client_[client] = cluster_id;
  }

  return recourse_sum;
}
