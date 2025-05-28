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

#include "nice_clustering_algorithm.h"

#include <math.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <ostream>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "util/task/status.h"
#include "util/task/status_macros.h"

double NiceClusteringAlgorithm::CurrentClusteringCost() {
  double cost_sum = 0.0;

  int num_of_open_facilities = 0;
  double total_openning_cost = 0.0;
  double total_connection_cost = 0.0;

  for (const auto& [cluster_id, cost] : cluster_cost_) {
    cost_sum += cost;
    if (cluster_is_critical_[cluster_id]) {
      num_of_open_facilities++;
      total_openning_cost += facility_cost_[cluster_center_[cluster_id]];
    }
  }
  for (const auto& [client_id, cluster_id] : cluster_of_client_) {
    total_connection_cost +=
        online_neighbors_of_facilities_[cluster_center_[cluster_id]][client_id];
  }

  std::cout << ">Total openning cost: " << total_openning_cost << " ("
            << num_of_open_facilities
            << " facilities), total connection cost: " << total_connection_cost
            << std::endl;
  return cost_sum;
}

double NiceClusteringAlgorithm::CurrentConnectionCost() {
  double total_connection_cost = 0.0;

  for (const auto& [client_id, cluster_id] : cluster_of_client_) {
    total_connection_cost +=
        online_neighbors_of_facilities_[cluster_center_[cluster_id]][client_id];
  }
  return total_connection_cost;
}

double NiceClusteringAlgorithm::CurrentOpenningCost() {
  double total_openning_cost = 0.0;

  for (const auto& [cluster_id, cost] : cluster_cost_) {
    if (cluster_is_critical_[cluster_id]) {
      total_openning_cost += facility_cost_[cluster_center_[cluster_id]];
    }
  }

  return total_openning_cost;
}

int NiceClusteringAlgorithm::CurrentNumFacilities() {
  int num_of_open_facilities = 0;

  for (const auto& [cluster_id, cost] : cluster_cost_) {
    if (cluster_is_critical_[cluster_id]) {
      num_of_open_facilities++;
    }
  }
  return num_of_open_facilities;
}

int NiceClusteringAlgorithm::LevelOfDistance(const double distance) {
  return ceil(log2(distance) / log2(1.0 + epsilon_));
}

int NiceClusteringAlgorithm::NearestFacilityOfClient(const int client_id) {
  double min_distance = std::numeric_limits<double>::max();
  int arg_min;
  for (const auto& [neighbor, distance] :
       online_neighbors_of_clients_[client_id]) {
    if (distance < min_distance) {
      min_distance = distance;
      arg_min = neighbor;
    }
  }
  return arg_min;
}

int NiceClusteringAlgorithm::NearestOpenFacilityOfClient(const int client_id) {
  double min = std::numeric_limits<double>::max();
  int arg_min;
  for (const auto& [neighbor, distance] :
       online_neighbors_of_clients_[client_id]) {
    if (!critical_cluster_of_facility_.contains(neighbor)) continue;
    if (distance < min) {
      min = distance;
      arg_min = neighbor;
    }
  }
  return arg_min;
}

void NiceClusteringAlgorithm::SetFacilities(
    const absl::flat_hash_map<int, double> facility_costs) {
  for (const auto& [facility, cost] : facility_costs) {
    facility_cost_[facility] = cost;
    TryUpdatingLevelBounds(LevelOfDistance(cost));
    facilities_sorted_by_cost_.push_back(std::make_pair(facility, cost));
  }
  std::sort(facilities_sorted_by_cost_.begin(),
            facilities_sorted_by_cost_.end(),
            [](auto& left, auto& right) { return left.second < right.second; });
}

void NiceClusteringAlgorithm::TestAllDataStructures() {
  for (const auto& [cluster_id, cluster] : clustering_) {
    for (const auto& client : cluster) {
      CHECK(cluster_of_client_[client] == cluster_id);
    }
  }
  for (const auto& [client_id, cluster_id] : cluster_of_client_) {
    CHECK(clustering_[cluster_id].contains(client_id))
        << " cluster id = " << cluster_id << ", client id = " << client_id;
  }

  if (clusters_in_level_.contains(0)) {
    CHECK(!clusters_in_level_[0].contains(0));
  }
  CHECK(!cluster_level_.contains(0));
  for (const auto& [level, clusters] : clusters_in_level_) {
    for (const auto& cluster_id : clusters) {
      CHECK(cluster_level_[cluster_id] == level);
    }
  }
  for (const auto& [cluster_id, level] : cluster_level_) {
    CHECK(clusters_in_level_[level].contains(cluster_id))
        << "clusters_in_level_[" << level << "] vs cluster_id " << cluster_id
        << std::endl;
  }

  for (const auto& [cluster_id, cluster] : clustering_) {
    int cluster_center = cluster_center_[cluster_id];
    double cost = facility_cost_[cluster_center];
    for (const auto& client : cluster) {
      cost += online_neighbors_of_clients_[client][cluster_center];
    }
    CHECK(cluster_is_critical_.contains(cluster_id))
        << " cluster id " << cluster_id << ", when checked and had side "
        << cluster.size();
    if (cluster_is_critical_[cluster_id]) {
      CHECK(critical_cluster_of_facility_[cluster_center] == cluster_id);
    } else {
      CHECK(clustering_[cluster_id].size() == 1);
    }
  }

  for (const auto& [facility, cluster_id] : critical_cluster_of_facility_) {
    CHECK(cluster_center_[cluster_id] == facility)
        << " facility: " << facility << ", cluster id: " << cluster_id
        << std::endl;
    CHECK(cluster_is_critical_[cluster_id]);
  }
  for (const auto& [cluster_id, is_critical] : cluster_is_critical_) {
    CHECK(cluster_center_.contains(cluster_id));
    if (is_critical) {
      CHECK(
          critical_cluster_of_facility_.contains(cluster_center_[cluster_id]));
      CHECK(critical_cluster_of_facility_[cluster_center_[cluster_id]] ==
            cluster_id);
    }
  }
  for (const auto& [cluster_id, cluster_center] : cluster_center_) {
    if (cluster_is_critical_[cluster_id]) {
      CHECK(critical_cluster_of_facility_[cluster_center] == cluster_id);
    }
  }

  for (const auto& [facility, neighbors] : online_neighbors_of_facilities_) {
    absl::flat_hash_map<int, int> num_neighbors_at_level;
    num_neighbors_at_level.clear();
    for (const auto& [client_id, distance] : neighbors) {
      const int cluster_id = cluster_of_client_[client_id];
      const int cluster_level = cluster_level_[cluster_id];

      if (num_neighbors_at_level.contains(cluster_level)) {
        num_neighbors_at_level[cluster_level]++;
      } else {
        num_neighbors_at_level[cluster_level] = 1;
      }
    }

    for (const auto& [level, neighbor_count] : num_neighbors_at_level) {
      int sum_per_bucket = 0;
      for (const auto& [bucket_level, num_neighbors_at_bucket] :
           facility_neighbors_breakdown_[facility]) {
        if (level != bucket_level) continue;
        for (const auto& [bucketed_distance, count] : num_neighbors_at_bucket) {
          sum_per_bucket += count;
        }
      }
      CHECK(neighbor_count == sum_per_bucket);

      absl::flat_hash_map<int, int> num_neighbors_at_bucket;
      for (const auto& [client_id, distance] : neighbors) {
        if (cluster_level_[cluster_of_client_[client_id]] != level) {
          continue;
        } else {
          if (num_neighbors_at_bucket.contains(LevelOfDistance(distance))) {
            num_neighbors_at_bucket[LevelOfDistance(distance)]++;
          } else {
            num_neighbors_at_bucket[LevelOfDistance(distance)] = 1;
          }
        }
      }
      for (const auto& [bucket, count] : num_neighbors_at_bucket) {
        CHECK(count == facility_neighbors_breakdown_[facility][level][bucket]);
      }
    }
  }

  CHECK(cluster_level_.size() == clustering_.size());
  CHECK(cluster_level_.size() == cluster_cost_.size());
  CHECK(cluster_level_.size() == cluster_center_.size());
  CHECK(online_neighbors_of_clients_.size() == cluster_of_client_.size());
  CHECK(online_neighbors_of_facilities_.size() == facility_cost_.size());
  CHECK(facility_cost_.size() == facility_neighbors_breakdown_.size());
}

void NiceClusteringAlgorithm::TryUpdatingLevelBounds(const int level) {
  if (level < min_level_) min_level_ = level;
  if (level > max_level_) max_level_ = level;
}

void NiceClusteringAlgorithm::InsertPoint(
    const int client_id, absl::flat_hash_map<int, double> neighbors) {
  online_neighbors_of_clients_[client_id] = neighbors;
  for (const auto& [facility, distance] : neighbors) {
    online_neighbors_of_facilities_[facility].insert(
        std::pair<int, double>(client_id, distance));
  }
  nearest_facility_[client_id] = NearestFacilityOfClient(client_id);
  if (clustering_.empty()) {
    double min = std::numeric_limits<double>::max();
    int arg_min = -1;
    for (const auto& [facility, cost] : facility_cost_) {
      double cost_of_creating_cluster =
          cost + online_neighbors_of_clients_[client_id][facility];
      if (cost_of_creating_cluster < min) {
        min = cost_of_creating_cluster;
        arg_min = facility;
      }
    }
    CreateCluster({client_id}, arg_min, LevelOfDistance(min),
                  /*critical=*/true);
    TryUpdatingLevelBounds(LevelOfDistance(min));
  } else {
    int cluster_center = NearestOpenFacilityOfClient(client_id);
    int appropriate_level = LevelOfDistance(
        online_neighbors_of_clients_[client_id][cluster_center]);

    int unique_critical_cluster_id =
        critical_cluster_of_facility_[cluster_center];

    if (cluster_level_[unique_critical_cluster_id] < appropriate_level) {
      CreateCluster({client_id}, cluster_center, appropriate_level,
                    /*critical=*/false);

    } else {
      clustering_[unique_critical_cluster_id].insert(client_id);
      cluster_cost_[unique_critical_cluster_id] +=
          online_neighbors_of_clients_[client_id][cluster_center];

      cluster_of_client_[client_id] = unique_critical_cluster_id;
      UpdateNeighborsOfClient(client_id, absl::nullopt,
                              cluster_level_[unique_critical_cluster_id]);
    }
  }
  FixClustering();
  // TestAllDataStructures(); // Expensive operation; only for testing.
}

void NiceClusteringAlgorithm::AssignNewCriticalCluster(const int facility_id) {
  int min_level = std::numeric_limits<int>::max();
  int min_arg = 0;
  for (const auto& [cluster, center] : cluster_center_) {
    if (center != facility_id) continue;
    if (cluster_level_[cluster] < min_level) {
      min_level = cluster_level_[cluster];
      min_arg = cluster;
    }
  }

  if (min_level != std::numeric_limits<int>::max()) {
    critical_cluster_of_facility_[facility_id] = min_arg;
    cluster_is_critical_[min_arg] = true;
    cluster_cost_[min_arg] += facility_cost_[facility_id];
  }
}

void NiceClusteringAlgorithm::DeletePoint(const int client_id) {
  int cluster_id = cluster_of_client_[client_id];
  UpdateNeighborsOfClient(client_id, cluster_level_[cluster_id], std::nullopt);
  if (clustering_[cluster_id].size() == 1) {
    bool was_critical = cluster_is_critical_[cluster_id];
    int cluster_center = cluster_center_[cluster_id];
    EraseCluster(cluster_id);
    if (was_critical) {
      AssignNewCriticalCluster(cluster_center);
    }
  } else {
    clustering_[cluster_id].erase(client_id);
    cluster_cost_[cluster_id] -=
        online_neighbors_of_facilities_[cluster_center_[cluster_id]][client_id];
  }
  cluster_of_client_.erase(client_id);

  for (const auto& [facility_id, distance] :
       online_neighbors_of_clients_[client_id]) {
    online_neighbors_of_facilities_[facility_id].erase(client_id);
  }
  online_neighbors_of_clients_.erase(client_id);
  FixClustering();
  // TestAllDataStructures();
}

void NiceClusteringAlgorithm::FixClustering() {
  bool invariant1_violation = true;
  bool invariant2_violation = true;
  bool move_client = false;

  while (invariant1_violation || invariant2_violation || move_client) {
    invariant1_violation = false;
    invariant2_violation = false;

    const auto& optional_blocking_cluster = FindBlockingClusterCenter();
    if (optional_blocking_cluster.has_value()) {
      invariant2_violation = true;
      FixBlockingClusterCenter(optional_blocking_cluster.value().first,
                               optional_blocking_cluster.value().second);
      continue;
    }
    levels_to_consider_for_facility_.clear();

    for (const auto& [cluster_id, cluster_cost] : cluster_cost_) {
      int cluster_level = cluster_level_[cluster_id];
      int cluster_size = clustering_[cluster_id].size();
      while (
          (LevelOfDistance(cluster_cost / cluster_size) <
               (cluster_level - num_levels_slack_) ||
           LevelOfDistance(cluster_cost / cluster_size) > cluster_level) &&
          (cluster_is_critical_[cluster_id] ||
           cluster_level_[critical_cluster_of_facility_
                              [cluster_center_[cluster_id]]] < cluster_level)) {
        invariant1_violation = true;
        FixLevel(cluster_id);
        cluster_level = cluster_level_[cluster_id];
        cluster_size = clustering_[cluster_id].size();
      }
    }
  }
}

std::optional<double> NiceClusteringAlgorithm::IsBlockingAtLevel(
    const int facility_id, const int target_level) {
  if (critical_cluster_of_facility_.contains(facility_id)) {
    if (cluster_level_[critical_cluster_of_facility_[facility_id]] <
        target_level) {
      return std::nullopt;
    }
  }
  double best_that_this_facility_can_do = std::numeric_limits<double>::max();
  bool is_blocking = false;
  double target_cost_upperbound =
      pow((1.0 + epsilon_), target_level - (num_levels_slack_ - 1));
  double current_total_cost = 0.0;
  int64_t current_cluster_size = 0;

  // Decide on whether the cost should account for the opening of the facility
  // or not.
  if (critical_cluster_of_facility_.contains(facility_id)) {
    if (cluster_level_[critical_cluster_of_facility_[facility_id]] >
        target_level) {
      current_total_cost = facility_cost_[facility_id];
    } else {
      // That means that this is an ordinary blocking cluster.
      // Ordinary blocking clusters are found elsewhere...
      return std::nullopt;
    }
  } else {
    current_total_cost = facility_cost_[facility_id];
  }

  for (const auto& [level, num_neighbors_at_bucketed_distance] :
       facility_neighbors_breakdown_[facility_id]) {
    // Look only at nodes who are at level above the one of the target_level.
    // The rest do not satisfy the conditions of a blocking cluster.
    if (level <= target_level) continue;

    for (const auto& [bucketed_distance, num_neighbors] :
         num_neighbors_at_bucketed_distance) {
      // Look only at nodes whose distance from the current facility is less tat
      // the (1.0+epsilon)^(target_level - num_levels_slack_). The rest do not
      // satisfy the conditions of a blocking cluster.
      if (bucketed_distance + (num_levels_slack_ - 1) > target_level) continue;
      current_total_cost +=
          pow((1.0 + epsilon_), bucketed_distance) * num_neighbors;
      current_cluster_size += num_neighbors;

      double current_average_cost = current_total_cost / current_cluster_size;
      if (/*current_average_cost > target_cost_lowerbound &&*/
          current_average_cost <= target_cost_upperbound &&
          current_cluster_size > 0) {
        if (current_average_cost < best_that_this_facility_can_do) {
          best_that_this_facility_can_do = current_average_cost;
          is_blocking = true;
        }
      }
    }
  }
  if (is_blocking) return best_that_this_facility_can_do;

  if (current_cluster_size == 0 || current_total_cost == 0.0)
    return std::nullopt;

  // Didn't find a ball forming a cluster with cost below the upper bound.
  if (best_that_this_facility_can_do > target_cost_upperbound) {
    return std::nullopt;
  }
  // Didn't find a ball forming a cluster with cost above the lower bound.
  // This should never happen
  CHECK(false) << "Final average cost "
               << best_that_this_facility_can_do
               // << ", target lowerbound " << target_cost_lowerbound
               << "  target average upperbound " << target_cost_upperbound
               << std::endl;
  return std::nullopt;
}

std::optional<std::pair<int, int>>
NiceClusteringAlgorithm::FindBlockingClusterCenter() {
  std::pair<int, int> best_blocking_at_level;
  double best_blocking_cost = std::numeric_limits<double>::max();
  for (const auto& [facility, cost] : facilities_sorted_by_cost_) {
    if (!critical_cluster_of_facility_.contains(facility)) continue;
    bool found_something_for_facility = false;
    for (int i : levels_to_consider_for_facility_[facility]) {
      const auto& is_blocking = IsBlockingAtLevel(facility, i);
      if (is_blocking.has_value()) {
        CHECK(levels_to_consider_for_facility_.contains(facility));
        if (!levels_to_consider_for_facility_[facility].contains(i)) {
          (void)IsBlockingAtLevel(facility, i);
        }
        CHECK(levels_to_consider_for_facility_[facility].contains(i));
        found_something_for_facility = true;
        if (is_blocking.value() < best_blocking_cost) {
          best_blocking_cost = is_blocking.value();
          best_blocking_at_level = std::make_pair(facility, i);
        }
      }
    }
    if (!found_something_for_facility) {
      levels_to_consider_for_facility_.erase(facility);
    }
  }
  if (best_blocking_cost < std::numeric_limits<double>::max()) {
    return best_blocking_at_level;
  }

  for (const auto& [facility, cost] : facilities_sorted_by_cost_) {
    if (critical_cluster_of_facility_.contains(facility)) continue;
    bool found_something_for_facility = false;
    for (int i : levels_to_consider_for_facility_[facility]) {
      const auto& is_blocking = IsBlockingAtLevel(facility, i);
      if (is_blocking.has_value()) {
        CHECK(levels_to_consider_for_facility_.contains(facility));
        if (!levels_to_consider_for_facility_[facility].contains(i)) {
          (void)IsBlockingAtLevel(facility, i);
        }
        CHECK(levels_to_consider_for_facility_[facility].contains(i))
            << " level " << i << ", elvel of facility : "
            << cluster_level_[critical_cluster_of_facility_[facility]]
            << std::endl;
        found_something_for_facility = true;
        if (is_blocking.value() < best_blocking_cost) {
          best_blocking_cost = is_blocking.value();
          best_blocking_at_level = std::make_pair(facility, i);
        }
      }
    }
    if (!found_something_for_facility) {
      levels_to_consider_for_facility_.erase(facility);
    }
  }
  if (best_blocking_cost < std::numeric_limits<double>::max()) {
    return best_blocking_at_level;
  }
  return std::nullopt;
}

void NiceClusteringAlgorithm::FixBlockingClusterCenter(
    const int32_t facility_id, const int32_t target_level) {
  std::vector<std::pair<int, double>> facility_neighbors(
      online_neighbors_of_facilities_[facility_id].begin(),
      online_neighbors_of_facilities_[facility_id].end());
  std::sort(facility_neighbors.begin(), facility_neighbors.end(),
            [](auto& left, auto& right) { return left.second < right.second; });

  // Create a new blocking cluster.
  std::vector<int> new_cluster;

  double target_cost_upperbound =
      pow((1.0 + epsilon_), target_level - (num_levels_slack_ - 1));
  double current_total_cost = facility_cost_[facility_id];
  std::vector<int> best_cluster;
  int new_cluster_iterator = 0;
  int64_t current_cluster_size = 0;
  bool found_cluster = false;

  for (const auto& [client, distance] : facility_neighbors) {
    // If the client is already at a lower level, then ignore it.
    if (cluster_level_[cluster_of_client_[client]] <= target_level) continue;
    int bucketed_distance = LevelOfDistance(distance);

    // If the distance is not much lower than target_level, ignore it.
    if (bucketed_distance + num_levels_slack_ - 1 > target_level) continue;
    new_cluster.push_back(client);
    current_total_cost += distance;
    current_cluster_size++;

    double current_average_cost = current_total_cost / current_cluster_size;
    if (current_average_cost <= target_cost_upperbound) {
      for (int i = new_cluster_iterator; i < new_cluster.size(); i++) {
        best_cluster.push_back(new_cluster.at(i));
      }
      new_cluster_iterator = new_cluster.size();
      found_cluster = true;
      break;
    }
  }
  CHECK(found_cluster);

  (void)CreateCluster(best_cluster, facility_id, target_level,
                      /*critical=*/ true);
}

void NiceClusteringAlgorithm::Merge(const int cluster_id) {
  int cluster_level = cluster_level_[cluster_id];
  TryUpdatingLevelBounds(cluster_level);
  int center_facility = cluster_center_[cluster_id];
  for (const int cluster_at_level : clusters_in_level_[cluster_level]) {
    // Only consider clusters that have the same center.
    if (cluster_center_[cluster_at_level] != center_facility ||
        cluster_at_level == cluster_id) {
      continue;
    }

    for (int client : clustering_[cluster_at_level]) {
      clustering_[cluster_id].insert(client);
      cluster_cost_[cluster_id] +=
          online_neighbors_of_facilities_[center_facility][client];
      cluster_of_client_[client] = cluster_id;
    }
    EraseCluster(cluster_at_level);
  }
}

void NiceClusteringAlgorithm::FixLevel(const int cluster_id) {
  int level = cluster_level_[cluster_id];
  double score = cluster_cost_[cluster_id];
  double average_score = score / clustering_[cluster_id].size();
  if (average_score < pow((1.0 + epsilon_), level - num_levels_slack_)) {
    if (!cluster_is_critical_[cluster_id]) {
      CHECK(clustering_[cluster_id].size() == 1);
      int new_level = std::max(
          LevelOfDistance(average_score) - num_levels_slack_,
          cluster_level_
              [critical_cluster_of_facility_[cluster_center_[cluster_id]]]);
      cluster_level_[cluster_id] = new_level;
      clusters_in_level_[level].erase(cluster_id);
      clusters_in_level_[new_level].insert(cluster_id);
      for (const int client : clustering_[cluster_id]) {
        UpdateNeighborsOfClient(client, level, new_level);
      }
      return;
    }

  } else if (average_score > pow((1.0 + epsilon_), level)) {
    cluster_level_[cluster_id]++;
    clusters_in_level_[level].erase(cluster_id);
    if (clusters_in_level_[level].empty()) clusters_in_level_.erase(level);
    if (!clusters_in_level_.contains(level + 1))
      clusters_in_level_[level + 1] = absl::flat_hash_set<int>();
    clusters_in_level_[level + 1].insert(cluster_id);
    for (const int client : clustering_[cluster_id]) {
      UpdateNeighborsOfClient(client, level, level + 1);
    }
    if (cluster_is_critical_[cluster_id]) Merge(cluster_id);
  }
}

void NiceClusteringAlgorithm::UpdateBreakDownNeighborhoodEntry(
    const int facility, const int level, const int bucketed_distance,
    const int delta) {
  if (!facility_neighbors_breakdown_.contains(facility)) {
    facility_neighbors_breakdown_[facility] =
        absl::flat_hash_map<int, absl::flat_hash_map<int, int>>();
  }
  if (!facility_neighbors_breakdown_[facility].contains(level)) {
    facility_neighbors_breakdown_[facility][level] =
        absl::flat_hash_map<int, int>();
  }
  if (!facility_neighbors_breakdown_[facility][level].contains(
          bucketed_distance)) {
    facility_neighbors_breakdown_[facility][level][bucketed_distance] = delta;
  } else {
    facility_neighbors_breakdown_[facility][level][bucketed_distance] += delta;
  }
  if (facility_neighbors_breakdown_[facility][level][bucketed_distance] == 0) {
    facility_neighbors_breakdown_[facility][level].erase(bucketed_distance);
    if (facility_neighbors_breakdown_[facility][level].empty()) {
      facility_neighbors_breakdown_[facility].erase(level);
    }
  }
}

void NiceClusteringAlgorithm::UpdateNeighborsOfClient(
    const int client_id, std::optional<int> old_level,
    std::optional<int> new_level) {
  if (old_level != new_level) {
    for (const auto& [neighboring_facility, distance] :
         online_neighbors_of_clients_[client_id]) {
      const int rounded_distance = LevelOfDistance(distance);
      if (old_level.has_value()) {
        UpdateBreakDownNeighborhoodEntry(
            neighboring_facility, old_level.value(), rounded_distance, -1);
      }
      if (new_level.has_value()) {
        UpdateBreakDownNeighborhoodEntry(
            neighboring_facility, new_level.value(), rounded_distance, +1);
      }
    }
  }

  if (new_level.has_value()) {
    for (const auto& [neighboring_facility, distance] :
         online_neighbors_of_clients_[client_id]) {
      double distance_to_nearest_facility =
          online_neighbors_of_clients_[client_id][nearest_facility_[client_id]];
      int lowest_blocking_level = LevelOfDistance(distance_to_nearest_facility);
      for (int i = lowest_blocking_level - num_levels_slack_;
           i < new_level.value(); i++) {
        if (!levels_to_consider_for_facility_.contains(neighboring_facility)) {
          levels_to_consider_for_facility_[neighboring_facility] =
              absl::flat_hash_set<int>();
        }
        levels_to_consider_for_facility_[neighboring_facility].insert(i);
      }
    }
  }
}

void NiceClusteringAlgorithm::EraseCluster(const int cluster_id) {
  int level = cluster_level_[cluster_id];
  int cluster_center = cluster_center_[cluster_id];
  cluster_center_.erase(cluster_id);
  clustering_[cluster_id].clear();
  clustering_.erase(cluster_id);
  cluster_level_.erase(cluster_id);
  clusters_in_level_[level].erase(cluster_id);
  cluster_cost_.erase(cluster_id);
  if (cluster_is_critical_[cluster_id]) {
    critical_cluster_of_facility_.erase(cluster_center);
  }
  cluster_is_critical_.erase(cluster_id);
}

int NiceClusteringAlgorithm::CreateCluster(std::vector<int> new_cluster,
                                           const int center_facility,
                                           const int level,
                                           bool critical = true) {
  int new_cluster_id = next_cluster_id_++;

  // Remove points from their old clusters
  absl::flat_hash_set<int> clusters_that_lost_nodes;
  for (const int client : new_cluster) {
    absl::optional<int> old_cluster_level = absl::nullopt;
    if (cluster_of_client_.contains(client)) {
      int old_cluster_id = cluster_of_client_[client];
      int old_cluster_center = cluster_center_[old_cluster_id];
      old_cluster_level = cluster_level_[old_cluster_id];

      clustering_[old_cluster_id].erase(client);
      cluster_cost_[old_cluster_id] -=
          online_neighbors_of_clients_[client][old_cluster_center];
      clusters_that_lost_nodes.insert(old_cluster_id);
    }
    UpdateNeighborsOfClient(client, old_cluster_level, level);
  }
  for (const int cluster_id : clusters_that_lost_nodes) {
    if (clustering_[cluster_id].empty()) {
      bool was_critical = cluster_is_critical_[cluster_id];
      int cluster_center = cluster_center_[cluster_id];
      EraseCluster(cluster_id);
      if (was_critical && center_facility != cluster_center)
        AssignNewCriticalCluster(cluster_center);
    } else {
      if (cluster_center_[cluster_id] == center_facility) {
        ConvertIntoOrdinary(cluster_id);
      }
    }
  }
  if (critical && critical_cluster_of_facility_.contains(center_facility)) {
    ConvertIntoOrdinary(critical_cluster_of_facility_[center_facility]);
  }

  double current_total_cost = critical ? facility_cost_[center_facility] : 0;
  for (const int client : new_cluster) {
    cluster_of_client_[client] = new_cluster_id;
    // Add the distance.
    current_total_cost +=
        online_neighbors_of_facilities_[center_facility][client];
  }

  cluster_is_critical_[new_cluster_id] = critical;
  clustering_.emplace(
      new_cluster_id,
      absl::flat_hash_set<int>(new_cluster.begin(), new_cluster.end()));
  cluster_level_[new_cluster_id] = level;
  cluster_cost_[new_cluster_id] = current_total_cost;
  clusters_in_level_[level].insert(new_cluster_id);
  cluster_center_[new_cluster_id] = center_facility;
  if (critical) critical_cluster_of_facility_[center_facility] = new_cluster_id;

  TryUpdatingLevelBounds(level);
  return new_cluster_id;
}

void NiceClusteringAlgorithm::ConvertIntoOrdinary(const int cluster_id) {
  CHECK(cluster_is_critical_[cluster_id] == true);
  int cluster_center = cluster_center_[cluster_id];
  int cluster_level = cluster_level_[cluster_id];
  const std::vector<int> former_cluster(clustering_[cluster_id].begin(),
                                        clustering_[cluster_id].end());

  EraseCluster(cluster_id);
  for (const int client : former_cluster) {
    int new_cluster_id = next_cluster_id_++;
    cluster_of_client_[client] = new_cluster_id;
    // Add the distance.
    double current_total_cost =
        online_neighbors_of_facilities_[cluster_center][client];

    cluster_is_critical_[new_cluster_id] = false;
    clustering_.emplace(new_cluster_id, absl::flat_hash_set<int>({client}));
    cluster_level_[new_cluster_id] = cluster_level;
    cluster_cost_[new_cluster_id] = current_total_cost;
    clusters_in_level_[cluster_level].insert(new_cluster_id);
    cluster_center_[new_cluster_id] = cluster_center;
  }
}

int NiceClusteringAlgorithm::ComputeRecourseAndUpdateReference() {
  int recourse_sum = 0;
  for (const auto& [client, cluster_id] : cluster_of_client_) {
    if (prev_cluster_center_of_client_.contains(client) &&
        cluster_center_[cluster_id] != prev_cluster_center_of_client_[client]) {
      ++recourse_sum;
    }
    prev_cluster_center_of_client_[client] = cluster_center_[cluster_id];
  }

  return recourse_sum;
}
