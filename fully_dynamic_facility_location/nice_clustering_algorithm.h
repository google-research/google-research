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

#ifndef FULLY_DYNAMIC_FACILITY_LOCATION_NICE_CLUSTERING_ALGORITHM_H_
#define FULLY_DYNAMIC_FACILITY_LOCATION_NICE_CLUSTERING_ALGORITHM_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

typedef absl::flat_hash_map<int, int> NumNeighborsAtBucketedDistance;
typedef absl::flat_hash_map<int, NumNeighborsAtBucketedDistance>
    NumNeighborsAtLevel;

class NiceClusteringAlgorithm {
 public:
  explicit NiceClusteringAlgorithm(int num_levels_slack, double epsilon)
      : num_levels_slack_(num_levels_slack), epsilon_(epsilon) {}

  // Deletes a client from the dataset, and triggers the necessary functions to
  // update the solution.
  void DeletePoint(const int client_id);

  // Inserts a client into the dataset, and triggers the necessary functions to
  // update the solution.
  void InsertPoint(int client_id, absl::flat_hash_map<int, double> neighbors);

  // Returns the current cost of the solution.
  double CurrentClusteringCost();

  // Given a client ID, returns the nearest *open* facility to that client, by
  // scanning its neighborhood.
  int NearestOpenFacilityOfClient(const int client_id);

  // Given a client ID, returns the nearest facility to that client, by scanning
  // its neighborhood.
  int NearestFacilityOfClient(const int client_id);

  // Returns the current connection cost of the solution.
  double CurrentConnectionCost();

  // Returns the current openning cost of the solution.
  double CurrentOpenningCost();

  // Returns the number of open facilities of the solution.
  int CurrentNumFacilities();

  // Given the set of facilities, and their cost, initializes internal data
  // structures.
  void SetFacilities(const absl::flat_hash_map<int, double> facility_costs);

  // Computes the recourse with respect to the reference solution, and updates
  // the reference solution with the current solution.
  int ComputeRecourseAndUpdateReference();

 private:
  // Fixes the maintained solution so that all invariants are satisfied.
  void FixClustering();

  // Searches for a blocking cluster.
  std::optional<std::pair<int, int>> FindBlockingClusterCenter();

  // Given a facility that forms a blocking cluster, identifies the blocking
  // cluster and updates the solution.
  void FixBlockingClusterCenter(const int32_t facility_id,
                                const int32_t target_level);

  // Transform a cluster into a set of ordinary clusters.
  void ConvertIntoOrdinary(const int cluster_id);

  // Places the given cluster to the right level.
  void FixLevel(const int cluster_id);

  // Given a cluster, merges all other clusters in the same level that also have
  // the same center.
  void Merge(const int cluster_id);

  // Returns the right level of the cluster hierarchy that the given distance
  // belongs to.
  int LevelOfDistance(const double distance);

  // Returns whether the given facility is blocking at the specified level.
  std::optional<double> IsBlockingAtLevel(const int facility_id,
                                          const int level);

  // Moves client from the old level to the new level and updates the internal
  // data structures.
  void UpdateNeighborsOfClient(const int client_id,
                               std::optional<int> old_level,
                               std::optional<int> new_level);

  // Creates a new cluster and assigns it to the the specified level.
  int CreateCluster(std::vector<int> new_cluster, const int center_facility,
                    const int level, bool critical);

  // Deletes a cluster and update the necessary data structures.
  void EraseCluster(const int cluster_id);

  // Given a level, update the minimum and maximum known levels.
  void TryUpdatingLevelBounds(const int level);

  // Helper testing function that verifies that all data structures have the
  // right state, and no mistakes have been made. This is an expensive operation
  // and is used for testing and development purposes.
  void TestAllDataStructures();

  // Update internal data structures that are useful for efficiently testing for
  // violations of the invariants of the algorithm.

  void UpdateBreakDownNeighborhoodEntry(const int facility, const int level,
                                        const int bucketed_distance,
                                        const int delta);

  void AssignNewCriticalCluster(const int facility_id);

  absl::flat_hash_map<int, absl::flat_hash_set<int>> clustering_;
  absl::flat_hash_map<int, int> cluster_center_;
  absl::flat_hash_map<int, int> critical_cluster_of_facility_;
  absl::flat_hash_map<int, int> cluster_of_client_;
  absl::flat_hash_map<int, bool> cluster_is_critical_;
  absl::flat_hash_map<int, int> cluster_level_;
  absl::flat_hash_map<int, absl::flat_hash_set<int>> clusters_in_level_;
  // Total absolute cost of the cluster (i.e., not average cost).
  absl::flat_hash_map<int, double> cluster_cost_;
  absl::flat_hash_map<int, absl::flat_hash_set<int>>
      levels_to_consider_for_facility_;

  // Represents the edges of the graph. The id of the nodes belongs to [0, n).
  // The i-th vector contains the neighbors of the node "i".
  absl::flat_hash_map<int, absl::flat_hash_map<int, double>>
      online_neighbors_of_facilities_;
  absl::flat_hash_map<int, double> facility_cost_;
  std::vector<std::pair<int, double>> facilities_sorted_by_cost_;
  absl::flat_hash_map<int, NumNeighborsAtLevel> facility_neighbors_breakdown_;
  absl::flat_hash_map<int, absl::flat_hash_map<int, double>>
      online_neighbors_of_clients_;

  absl::flat_hash_map<int, int> nearest_facility_;
  absl::flat_hash_map<int, int> prev_cluster_center_of_client_;

  int min_level_ = 0;
  int max_level_ = 0;
  int32_t next_cluster_id_ = 1;
  int num_levels_slack_ = 10;
  double epsilon_ = 1.0;
};

#endif  // FULLY_DYNAMIC_FACILITY_LOCATION_NICE_CLUSTERING_ALGORITHM_H_
