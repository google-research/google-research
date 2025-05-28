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

#ifndef FULLY_DYNAMIC_FACILITY_LOCATION_GRAPH_HANDLER_H_
#define FULLY_DYNAMIC_FACILITY_LOCATION_GRAPH_HANDLER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

enum ArrivalStrategy { SLIDING_WINDOW, DELETION_PROBABILITY };


int NearestFacilityOfClient(const int client_id,
                            absl::flat_hash_map<int, double> neighbors);
class GraphHandler {
 public:
  // Reads the input. For each edge, the expected format is the endpoint of the
  // edge seperate with a white space. For instance, if we have three edges
  // (1, 5), (2, 3), (1, 3), an input can be:
  // 1 5
  // 2 3
  // 1 3
  // Ignores parallel edges and only keeps one copy. Notice that the name of the
  // endpoints can be strings or integers.
  void ReadClientToFacilityEdges();

  void ReadEmbeddings();

  void SetUpGraph(float percentage_of_facilities, int32_t target_num_nodes);

  void StartMaintainingDynamicUpdates(bool shuffle_order,
                                      double prob_of_deletion,
                                      int sliding_window_size,
                                      ArrivalStrategy arrival_strategy);

  void RestartStreamOfClients();
  int NextClientsDeletion();

  double GetFacilityWeight(int facility_id) {
    return facility_weight_[facility_id];
  }

  absl::flat_hash_map<int, double> GetClientsNeighbors(const int client_id);

  std::optional<std::pair<bool, int>> GetNextUpdate();

  // Represents the edges of the graph. The id of the nodes belongs to [0, n).
  // the i-th vector contains the neighbors of the node "i".
  absl::flat_hash_map<int, absl::flat_hash_map<int, double>>
      facility_neighbors_;

  absl::flat_hash_map<int, absl::flat_hash_map<int, double>>
      facility_to_facility_neighbors_;

  absl::flat_hash_map<int, absl::flat_hash_map<int, double>> client_neighbors_;

  absl::flat_hash_map<int, double> facility_weight_;

 private:
  int num_clients_ = 0;
  // The name of a vertex to its id.
  absl::flat_hash_map<std::string, int> node_id_map_;

  // Set of all the edges.
  absl::flat_hash_set<std::tuple<int, int, double>> client_to_facility_edges_;

  std::vector<std::pair<bool, int>> update_sequence_;

  std::vector<std::vector<double>> input_embeddings_;

  int next_update_to_execute_ = 0;
  double facility_cost_upperbound_ = 0;
};

#endif  // FULLY_DYNAMIC_FACILITY_LOCATION_GRAPH_HANDLER_H_
