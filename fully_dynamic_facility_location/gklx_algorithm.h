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

#ifndef FULLY_DYNAMIC_FACILITY_LOCATION_GKLX_ALGORITHM_H_
#define FULLY_DYNAMIC_FACILITY_LOCATION_GKLX_ALGORITHM_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

class GKLXAlgorithm {
 public:
  // Given the set of facilities, and their cost, initializes internal data
  // structures.
  void SetFacilities(const absl::flat_hash_map<int, double> facility_costs);

  // Deletes a client from the dataset, and triggers the necessary functions to
  // update the solution.
  void DeletePoint(const int client_id);

  // Inserts a client to the dataset, and triggers the necessary functions to
  // update the solution.
  void InsertPoint(int client_id, absl::flat_hash_map<int, double> neighbors);

  // Computes an HST on the set of facilities.
  void ComputedFacilityHST();

  // Computes the recourse with respect to the reference solution, and updates
  // the reference solution with the current solution.
  int ComputeRecourseAndUpdateReference();

  // Initializes the distances between facilities.
  void SetFacilityToFacilityDistances(
      absl::flat_hash_map<int, absl::flat_hash_map<int, double>>
          facility_to_facility_distances);

  // Returns the current total cost of the solution.
  double CurrentClusteringCost();

  // Returns the current connection cost of the solution.
  double CurrentConnectionCost() { return total_connection_cost_; }

  // Returns the opening total cost of the solution.
  double CurrentOpenningCost() { return total_openning_cost_; }

  // Returns the current number of facilities in the solution.
  int CurrentNumFacilities() { return num_of_open_facilities_; }

  // Verifies that all the internal data structures are correctly updated.
  void CheckCorrectStatusOfDatastructures();

 private:
  // Assumes a full metric.
  absl::flat_hash_map<int, std::vector<int>> ComputedDominanceSequencePerNode(
      absl::flat_hash_map<int, int> node_ranks);

  bool ShouldBeOpen(const int node_id);
  void SetMinAndMaxDistance();
  bool IsMarked(int node_id);
  void CloseNode(const int node_id);
  void OpenNode(const int node_id);
  void MarkNode(const int node_id);
  void UnmarkNode(const int node_id);

  // Given a client ID, returns the nearest facility to that client, by scanning
  // its neighborhood.
  int NearestFacilityOfClient(const int client_id,
                              absl::flat_hash_map<int, double> neighbors);

  void AssignClientsToFacilities();

  // Draws a random number x in the range [1,2] with probability 1/(x ln(2)).
  double DrawBetaFromDistribution();

  absl::flat_hash_map<int, int> nearest_facility_of_client_;
  absl::flat_hash_map<int, absl::flat_hash_set<int>>
      clients_attached_to_facility_;

  absl::flat_hash_map<int, double> cluster_cost_;
  absl::flat_hash_map<int, int> center_of_client_;

  absl::flat_hash_map<int, int> prev_cluster_center_of_client_;
  absl::flat_hash_map<int, absl::flat_hash_map<int, double>>
      online_facility_to_facilities_neighbors_;

  absl::flat_hash_map<int, absl::flat_hash_map<int, double>>
      online_client_to_facilities_neighbors_;

  absl::flat_hash_map<int, double> facility_cost_;
  absl::flat_hash_map<int, int> num_clients_at_facility_;
  absl::flat_hash_map<int, int> num_conditional_clients_at_node_;

  absl::flat_hash_map<int, int> parent_in_hst_;
  absl::flat_hash_map<int, int> level_in_hst_;
  absl::flat_hash_map<int, int> facility_of_tree_node_;
  absl::flat_hash_map<int, double> cost_of_tree_node_;
  absl::flat_hash_map<int, bool> node_is_marked_;
  absl::flat_hash_map<int, bool> node_is_open_;

  absl::flat_hash_map<int, int> alphas_;
  absl::flat_hash_map<int, int> betas_;

  double total_openning_cost_;
  double total_connection_cost_;
  int num_of_open_facilities_;

  double min_distance_;
  double max_distance_;
};

#endif  // FULLY_DYNAMIC_FACILITY_LOCATION_GKLX_ALGORITHM_H_
