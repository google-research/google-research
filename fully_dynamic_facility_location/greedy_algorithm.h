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

#ifndef FULLY_DYNAMIC_FACILITY_LOCATION_GREEDY_ALGORITHM_H_
#define FULLY_DYNAMIC_FACILITY_LOCATION_GREEDY_ALGORITHM_H_

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

typedef absl::flat_hash_map<int, int> NumNeighborsAtBucketedDistance;
typedef absl::flat_hash_map<int, NumNeighborsAtBucketedDistance>
    NumNeighborsAtLevel;

class GreedyAlgorithm {
 public:
  // Deletes a client from the dataset, and triggers the necessary functions to
  // update the solution.
  void DeletePoint(const int client_id);

  // Inserts a client to the dataset, and triggers the necessary functions to
  // update the solution.
  void InsertPoint(int client_id, absl::flat_hash_map<int, double> neighbors);

  // Returns the current total cost of the solution.
  double CurrentClusteringCost();

  // Returns the current connection cost of the solution.
  double CurrentConnectionCost();

  // Returns the current opening cost of the solution.
  double CurrentOpenningCost();

  // Returns the current number of facilities.
  int CurrentNumFacilities() { return clustering_.size(); }

  // Given the set of facilities, and their cost, initializes internal data
  // structures.
  void SetFacilities(const absl::flat_hash_map<int, double> facility_costs);

  // Computes the recourse with respect to the reference solution, and updates
  // the reference solution with the current solution.
  int ComputeRecourseAndUpdateReference();

 private:
  // Computes a solution.
  void ComputeClustering();

  // Searches for a blocking cluster.
  std::optional<std::tuple<int, double, std::vector<int>>>
  FindBlockingCluster();

  // Given a client ID, returns the nearest *open* facility to that client, by
  // scanning its neighborhood.
  int NearestOpenFacilityOfClient(const int client_id);

  // Returns the right level of the cluster hierarchy that the given distance
  // belongs to.
  int LevelOfDistance(const double distance);

  // Returns whether the given facility is blocking.
  std::optional<std::pair<double, std::vector<int>>> IsBlockingAtLevel(
      const int facility_id);

  absl::flat_hash_map<int, absl::flat_hash_set<int>> clustering_;
  absl::flat_hash_map<int, int> cluster_of_client_;
  absl::flat_hash_map<int, double> cluster_cost_;

  absl::flat_hash_map<int, int> prev_cluster_center_of_client_;

  // Represents the edges of the graph. The id of the nodes belongs to [0, n).
  // the i-th vector contains the neighbors of the node "i".
  absl::flat_hash_map<int, absl::flat_hash_map<int, double>>
      online_neighbors_of_facilities_;
  absl::flat_hash_map<int, double> facility_cost_;
  absl::flat_hash_map<int, absl::flat_hash_map<int, double>>
      online_neighbors_of_clients_;
};

#endif  // FULLY_DYNAMIC_FACILITY_LOCATION_GREEDY_ALGORITHM_H_
