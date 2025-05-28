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

#include <time.h>

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gklx_algorithm.h"
#include "graph_handler.h"
#include "greedy_algorithm.h"
#include "nice_clustering_algorithm.h"
#include "absl/container/flat_hash_map.h"

// First set of parameters for NiceClustering
constexpr int num_levels_slack_1 = 3;
constexpr double epsilon_1 = 1;

// Second set of parameters for NiceClustering
constexpr int num_levels_slack_2 = 1;
constexpr double epsilon_2 = 1;

// Third set of parameters for NiceClustering
constexpr int num_levels_slack_3 = 3;
constexpr double epsilon_3 = 0.05;

// Fourth set of parameters for NiceClustering
constexpr int num_levels_slack_4 = 1;
constexpr double epsilon_4 = 0.05;

// Parameters of the instance
constexpr bool random_order = false;
constexpr double prob_of_deletion = 0.3;
constexpr int sliding_window_size = 1000;
constexpr float percentage_of_facilities = 5;
constexpr int target_num_nodes = 5000;
constexpr ArrivalStrategy arrival_strategy = SLIDING_WINDOW;

int main(int argc, char* argv[]) {
  // Read and construct graph
  GraphHandler graph_handler;
  graph_handler.ReadEmbeddings();
  graph_handler.SetUpGraph(percentage_of_facilities, target_num_nodes);
  graph_handler.StartMaintainingDynamicUpdates(
      /*shuffle_order=*/random_order, prob_of_deletion, sliding_window_size,
      arrival_strategy);

  // Run the baseline where we assign each client to its nearest facility. If
  // the nearest facility of a point is not open, we open it.
  std::cout << "NearestNeighbor solution:" << std::endl;
  std::vector<double> nearest_neighbor_cost;
  std::vector<int> nearest_neighbor_recourse;
  std::vector<double> nearest_neighbor_time;
  std::vector<double> nearest_neighbor_connection_cost;
  std::vector<double> nearest_neighbor_opening_cost;
  std::vector<int> nearest_neighbor_num_open_facilities;

  graph_handler.RestartStreamOfClients();

  double cost_of_nearest_neighbor_solution = 0.0;
  int iteration = 0;
  std::optional<std::pair<double, int>> next_update =
      graph_handler.GetNextUpdate();
  absl::flat_hash_map<int, int> clients_at_facility;
  absl::flat_hash_map<int, int> facility_of_client;

  clock_t t;
  double total_time = 0;
  double connection_cost = 0;
  double openning_cost = 0;
  int num_open_facilities = 0;
  while (next_update.has_value()) {
    t = clock();
    int next_update_is_insertions = next_update.value().first;
    int next_client_id = next_update.value().second;
    absl::flat_hash_map<int, double> next_client_neighbors =
        graph_handler.GetClientsNeighbors(next_client_id);
    if (next_update_is_insertions) {  // point insertion.
      const int nearest_facility =
          NearestFacilityOfClient(next_client_id, next_client_neighbors);
      facility_of_client[next_client_id] = nearest_facility;
      if (clients_at_facility.contains(nearest_facility)) {
        clients_at_facility[nearest_facility]++;
      } else {  // Open the facility.
        clients_at_facility[nearest_facility] = 1;
        cost_of_nearest_neighbor_solution +=
            graph_handler.GetFacilityWeight(nearest_facility);
        num_open_facilities++;
        openning_cost += graph_handler.GetFacilityWeight(nearest_facility);
      }
      // Update cost.
      connection_cost +=
          graph_handler.facility_neighbors_[nearest_facility][next_client_id];
      cost_of_nearest_neighbor_solution +=
          graph_handler.facility_neighbors_[nearest_facility][next_client_id];
    } else {  // point deletion.
      const int nearest_facility = facility_of_client[next_client_id];
      clients_at_facility[nearest_facility]--;
      // If no other client is connected, then we close the facility.
      if (clients_at_facility[nearest_facility] == 0) {
        cost_of_nearest_neighbor_solution -=
            graph_handler.GetFacilityWeight(nearest_facility);
        clients_at_facility.erase(nearest_facility);
        num_open_facilities--;
        openning_cost -= graph_handler.GetFacilityWeight(nearest_facility);
      }
      // Update cost.
      cost_of_nearest_neighbor_solution -=
          graph_handler.facility_neighbors_[nearest_facility][next_client_id];
      connection_cost -=
          graph_handler.facility_neighbors_[nearest_facility][next_client_id];
    }

    // Store the stats for printing them later.
    nearest_neighbor_connection_cost.push_back(connection_cost);
    nearest_neighbor_opening_cost.push_back(openning_cost);
    nearest_neighbor_num_open_facilities.push_back(num_open_facilities);

    total_time += static_cast<double>(clock() - t) / CLOCKS_PER_SEC;
    // If you want to watch the stats as they are calculated uncomment the
    // following.
    // std::cout << iteration++ << "\t" << cost_of_nearest_neighbor_solution
    //           << "\t" << 0 << "\t" << total_time << std::endl;
    nearest_neighbor_cost.push_back(cost_of_nearest_neighbor_solution);
    nearest_neighbor_recourse.push_back(0);
    nearest_neighbor_time.push_back(total_time);
    next_update = graph_handler.GetNextUpdate();
  }

  // Run the first configuration of the Nice clustering algorithm.
  std::cout << "Nice clustering 1 solution:" << std::endl;
  std::vector<double> nice_clustering_1_cost;
  std::vector<int> nice_clustering_1_recourse;
  std::vector<double> nice_clustering_1_time;

  std::vector<double> nice_clustering_1_connection_cost;
  std::vector<double> nice_clustering_1_opening_cost;
  std::vector<int> nice_clustering_1_num_open_facilities;
  graph_handler.RestartStreamOfClients();
  NiceClusteringAlgorithm nice_clustering_algorithm_1(num_levels_slack_1,
                                                      epsilon_1);
  total_time = 0;
  t = clock();
  nice_clustering_algorithm_1.SetFacilities(graph_handler.facility_weight_);
  total_time += static_cast<double>(clock() - t) / CLOCKS_PER_SEC;
  next_update = graph_handler.GetNextUpdate();
  iteration = 0;
  int64_t total_cumulative_recourse = 0;
  while (next_update.has_value()) {
    bool next_update_is_insertions = next_update.value().first;
    int next_client_id = next_update.value().second;

    t = clock();
    if (next_update_is_insertions) {
      absl::flat_hash_map<int, double> next_client_neighbors =
          graph_handler.GetClientsNeighbors(next_client_id);
      nice_clustering_algorithm_1.InsertPoint(next_client_id,
                                              next_client_neighbors);
    } else {
      nice_clustering_algorithm_1.DeletePoint(next_client_id);
    }
    total_time += static_cast<double>(clock() - t) / CLOCKS_PER_SEC;
    total_cumulative_recourse +=
        nice_clustering_algorithm_1.ComputeRecourseAndUpdateReference();
    double cost = nice_clustering_algorithm_1.CurrentClusteringCost();
    // If you want to watch the stats as they are calculated uncomment the
    // following.
    // std::cout << iteration++ << "\t" << cost << "\t"
    //           << total_cumulative_recourse << "\t" << total_time <<
    //           std::endl;

    // Store the stats for printing them later.
    nice_clustering_1_cost.push_back(cost);
    nice_clustering_1_recourse.push_back(total_cumulative_recourse);
    nice_clustering_1_time.push_back(total_time);
    nice_clustering_1_num_open_facilities.push_back(
        nice_clustering_algorithm_1.CurrentNumFacilities());
    nice_clustering_1_opening_cost.push_back(
        nice_clustering_algorithm_1.CurrentOpenningCost());
    nice_clustering_1_connection_cost.push_back(
        nice_clustering_algorithm_1.CurrentConnectionCost());

    next_update = graph_handler.GetNextUpdate();
  }

  // Run the second configuration of the Nice clustering algorithm.
  std::cout << "Nice clustering 2 solution:" << std::endl;
  std::vector<double> nice_clustering_2_cost;
  std::vector<int> nice_clustering_2_recourse;
  std::vector<double> nice_clustering_2_time;

  std::vector<double> nice_clustering_2_connection_cost;
  std::vector<double> nice_clustering_2_opening_cost;
  std::vector<int> nice_clustering_2_num_open_facilities;
  graph_handler.RestartStreamOfClients();
  NiceClusteringAlgorithm nice_clustering_algorithm_2(num_levels_slack_2,
                                                      epsilon_2);
  total_time = 0;
  t = clock();
  nice_clustering_algorithm_2.SetFacilities(graph_handler.facility_weight_);
  total_time += static_cast<double>(clock() - t) / CLOCKS_PER_SEC;
  next_update = graph_handler.GetNextUpdate();
  iteration = 0;
  total_cumulative_recourse = 0;
  while (next_update.has_value()) {
    bool next_update_is_insertions = next_update.value().first;
    int next_client_id = next_update.value().second;

    t = clock();
    if (next_update_is_insertions) {
      absl::flat_hash_map<int, double> next_client_neighbors =
          graph_handler.GetClientsNeighbors(next_client_id);
      nice_clustering_algorithm_2.InsertPoint(next_client_id,
                                              next_client_neighbors);
    } else {
      nice_clustering_algorithm_2.DeletePoint(next_client_id);
    }
    total_time += static_cast<double>(clock() - t) / CLOCKS_PER_SEC;
    total_cumulative_recourse +=
        nice_clustering_algorithm_2.ComputeRecourseAndUpdateReference();
    double cost = nice_clustering_algorithm_2.CurrentClusteringCost();
    // std::cout << iteration++ << "\t" << cost << "\t"
    //           << total_cumulative_recourse << "\t" << total_time <<
    //           std::endl;

    // Store the stats for printing them later.
    nice_clustering_2_cost.push_back(cost);
    nice_clustering_2_recourse.push_back(total_cumulative_recourse);
    nice_clustering_2_time.push_back(total_time);
    nice_clustering_2_num_open_facilities.push_back(
        nice_clustering_algorithm_2.CurrentNumFacilities());
    nice_clustering_2_opening_cost.push_back(
        nice_clustering_algorithm_2.CurrentOpenningCost());
    nice_clustering_2_connection_cost.push_back(
        nice_clustering_algorithm_2.CurrentConnectionCost());

    next_update = graph_handler.GetNextUpdate();
  }

  // Run the third configuration of the Nice clustering algorithm.
  std::cout << "Nice clustering 3 solution:" << std::endl;
  std::vector<double> nice_clustering_3_cost;
  std::vector<int> nice_clustering_3_recourse;
  std::vector<double> nice_clustering_3_time;
  std::vector<double> nice_clustering_3_connection_cost;
  std::vector<double> nice_clustering_3_opening_cost;
  std::vector<int> nice_clustering_3_num_open_facilities;
  graph_handler.RestartStreamOfClients();
  total_time = 0;
  t = clock();
  NiceClusteringAlgorithm nice_clustering_algorithm_3(num_levels_slack_3,
                                                      epsilon_3);
  nice_clustering_algorithm_3.SetFacilities(graph_handler.facility_weight_);
  total_time += static_cast<double>(clock() - t) / CLOCKS_PER_SEC;
  next_update = graph_handler.GetNextUpdate();
  iteration = 0;
  total_cumulative_recourse = 0;
  while (next_update.has_value()) {
    bool next_update_is_insertions = next_update.value().first;
    int next_client_id = next_update.value().second;

    t = clock();
    if (next_update_is_insertions) {
      absl::flat_hash_map<int, double> next_client_neighbors =
          graph_handler.GetClientsNeighbors(next_client_id);
      nice_clustering_algorithm_3.InsertPoint(next_client_id,
                                              next_client_neighbors);
    } else {
      nice_clustering_algorithm_3.DeletePoint(next_client_id);
    }
    total_time += static_cast<double>(clock() - t) / CLOCKS_PER_SEC;
    total_cumulative_recourse +=
        nice_clustering_algorithm_3.ComputeRecourseAndUpdateReference();
    double cost = nice_clustering_algorithm_3.CurrentClusteringCost();
    // std::cout << iteration++ << "\t" << cost << "\t"
    //           << total_cumulative_recourse << "\t" << total_time <<
    //           std::endl;

    nice_clustering_3_cost.push_back(cost);
    nice_clustering_3_recourse.push_back(total_cumulative_recourse);
    nice_clustering_3_time.push_back(total_time);
    nice_clustering_3_num_open_facilities.push_back(
        nice_clustering_algorithm_3.CurrentNumFacilities());
    nice_clustering_3_opening_cost.push_back(
        nice_clustering_algorithm_3.CurrentOpenningCost());
    nice_clustering_3_connection_cost.push_back(
        nice_clustering_algorithm_3.CurrentConnectionCost());

    next_update = graph_handler.GetNextUpdate();
  }

  std::cout << "Nice clustering 4 solution:" << std::endl;
  std::vector<double> nice_clustering_4_cost;
  std::vector<int> nice_clustering_4_recourse;
  std::vector<double> nice_clustering_4_time;
  std::vector<double> nice_clustering_4_connection_cost;
  std::vector<double> nice_clustering_4_opening_cost;
  std::vector<int> nice_clustering_4_num_open_facilities;
  graph_handler.RestartStreamOfClients();
  NiceClusteringAlgorithm nice_clustering_algorithm_4(num_levels_slack_4,
                                                      epsilon_4);
  total_time = 0;
  t = clock();
  nice_clustering_algorithm_4.SetFacilities(graph_handler.facility_weight_);
  total_time += static_cast<double>(clock() - t) / CLOCKS_PER_SEC;
  next_update = graph_handler.GetNextUpdate();
  iteration = 0;
  total_cumulative_recourse = 0;
  while (next_update.has_value()) {
    bool next_update_is_insertions = next_update.value().first;
    int next_client_id = next_update.value().second;

    t = clock();
    if (next_update_is_insertions) {
      absl::flat_hash_map<int, double> next_client_neighbors =
          graph_handler.GetClientsNeighbors(next_client_id);
      nice_clustering_algorithm_4.InsertPoint(next_client_id,
                                              next_client_neighbors);
    } else {
      nice_clustering_algorithm_4.DeletePoint(next_client_id);
    }
    total_time += static_cast<double>(clock() - t) / CLOCKS_PER_SEC;
    total_cumulative_recourse +=
        nice_clustering_algorithm_4.ComputeRecourseAndUpdateReference();
    double cost = nice_clustering_algorithm_4.CurrentClusteringCost();
    // std::cout << iteration++ << "\t" << cost << "\t"
    //           << total_cumulative_recourse << "\t" << total_time <<
    //           std::endl;

    nice_clustering_4_cost.push_back(cost);
    nice_clustering_4_recourse.push_back(total_cumulative_recourse);
    nice_clustering_4_time.push_back(total_time);
    nice_clustering_4_num_open_facilities.push_back(
        nice_clustering_algorithm_4.CurrentNumFacilities());
    nice_clustering_4_opening_cost.push_back(
        nice_clustering_algorithm_4.CurrentOpenningCost());
    nice_clustering_4_connection_cost.push_back(
        nice_clustering_algorithm_4.CurrentConnectionCost());

    next_update = graph_handler.GetNextUpdate();
  }

  std::cout << "GKLX clustering solution:" << std::endl;
  std::vector<double> gklx_cost;
  std::vector<int> gklx_recourse;
  std::vector<double> gklx_time;
  std::vector<double> gklx_connection_cost;
  std::vector<double> gklx_opening_cost;
  std::vector<int> gklx_num_open_facilities;
  graph_handler.RestartStreamOfClients();
  GKLXAlgorithm gklx_algorithm;
  total_time = 0;
  t = clock();
  gklx_algorithm.SetFacilities(graph_handler.facility_weight_);
  gklx_algorithm.SetFacilityToFacilityDistances(
      graph_handler.facility_to_facility_neighbors_);
  gklx_algorithm.ComputedFacilityHST();
  total_time += static_cast<double>(clock() - t) / CLOCKS_PER_SEC;
  next_update = graph_handler.GetNextUpdate();
  iteration = 0;
  total_cumulative_recourse = 0;
  while (next_update.has_value()) {
    bool next_update_is_insertions = next_update.value().first;
    int next_client_id = next_update.value().second;

    t = clock();
    if (next_update_is_insertions) {
      absl::flat_hash_map<int, double> next_client_neighbors =
          graph_handler.GetClientsNeighbors(next_client_id);
      gklx_algorithm.InsertPoint(next_client_id, next_client_neighbors);
    } else {
      gklx_algorithm.DeletePoint(next_client_id);
    }
    total_time += static_cast<double>(clock() - t) / CLOCKS_PER_SEC;
    total_cumulative_recourse +=
        gklx_algorithm.ComputeRecourseAndUpdateReference();
    double cost = gklx_algorithm.CurrentClusteringCost();
    // std::cout << iteration++ << "\t" << cost << "\t"
    //           << total_cumulative_recourse << "\t" << total_time <<
    //           std::endl;

    gklx_cost.push_back(cost);
    gklx_recourse.push_back(total_cumulative_recourse);
    gklx_time.push_back(total_time);
    gklx_num_open_facilities.push_back(gklx_algorithm.CurrentNumFacilities());
    gklx_opening_cost.push_back(gklx_algorithm.CurrentOpenningCost());
    gklx_connection_cost.push_back(gklx_algorithm.CurrentConnectionCost());

    next_update = graph_handler.GetNextUpdate();
  }

  std::cout << "Greedy clustering solution:" << std::endl;
  std::vector<double> greedy_cost;
  std::vector<int> greedy_recourse;
  std::vector<double> greedy_time;
  std::vector<double> greedy_connection_cost;
  std::vector<double> greedy_opening_cost;
  std::vector<int> greedy_num_open_facilities;
  graph_handler.RestartStreamOfClients();
  GreedyAlgorithm greedy_algorithm;
  total_time = 0;
  t = clock();
  greedy_algorithm.SetFacilities(graph_handler.facility_weight_);
  total_time += static_cast<double>(clock() - t) / CLOCKS_PER_SEC;
  next_update = graph_handler.GetNextUpdate();
  iteration = 0;
  total_cumulative_recourse = 0;
  while (next_update.has_value()) {
    bool next_update_is_insertions = next_update.value().first;
    int next_client_id = next_update.value().second;

    t = clock();
    if (next_update_is_insertions) {
      absl::flat_hash_map<int, double> next_client_neighbors =
          graph_handler.GetClientsNeighbors(next_client_id);
      greedy_algorithm.InsertPoint(next_client_id, next_client_neighbors);
    } else {
      greedy_algorithm.DeletePoint(next_client_id);
    }
    // greedy_algorithm.CheckCorrectStatusOfDatastructures();
    double cost = greedy_algorithm.CurrentClusteringCost();
    total_cumulative_recourse +=
        greedy_algorithm.ComputeRecourseAndUpdateReference();
    total_time += static_cast<double>(clock() - t) / CLOCKS_PER_SEC;
    // std::cout << iteration++ << "\t" << cost << "\t"
    //           << total_cumulative_recourse << "\t" << total_time <<
    //           std::endl;

    greedy_cost.push_back(cost);
    greedy_recourse.push_back(total_cumulative_recourse);
    greedy_time.push_back(total_time);
    greedy_num_open_facilities.push_back(
        greedy_algorithm.CurrentNumFacilities());
    greedy_opening_cost.push_back(greedy_algorithm.CurrentOpenningCost());
    greedy_connection_cost.push_back(greedy_algorithm.CurrentConnectionCost());

    next_update = graph_handler.GetNextUpdate();
  }

  for (int i = 0; i < nearest_neighbor_cost.size(); i++) {
    std::cout << (i + 1) << "\t " << nearest_neighbor_cost.at(i) << "\t "
              << nearest_neighbor_recourse.at(i) << "\t "
              << nearest_neighbor_time.at(i) << "\t "
              << nice_clustering_1_cost.at(i) << "\t "
              << nice_clustering_1_recourse.at(i) << "\t "
              << nice_clustering_1_time.at(i) << "\t "
              << nice_clustering_2_cost.at(i) << "\t "
              << nice_clustering_2_recourse.at(i) << "\t "
              << nice_clustering_2_time.at(i) << "\t "
              << nice_clustering_3_cost.at(i) << "\t "
              << nice_clustering_3_recourse.at(i) << "\t "
              << nice_clustering_3_time.at(i) << "\t "
              << nice_clustering_4_cost.at(i) << "\t "
              << nice_clustering_4_recourse.at(i) << "\t "
              << nice_clustering_4_time.at(i) << "\t " << gklx_cost.at(i)
              << "\t " << gklx_recourse.at(i) << "\t " << gklx_time.at(i)
              << "\t " << greedy_cost.at(i) << "\t " << greedy_recourse.at(i)
              << "\t " << greedy_time.at(i) << std::endl;
  }

  for (int i = 0; i < nearest_neighbor_cost.size(); i++) {
    std::cout << (i + 1) << "\t " << nearest_neighbor_num_open_facilities.at(i)
              << "\t " << nearest_neighbor_opening_cost.at(i) << "\t "
              << nearest_neighbor_connection_cost.at(i) << "\t "
              << nice_clustering_1_num_open_facilities.at(i) << "\t "
              << nice_clustering_1_opening_cost.at(i) << "\t "
              << nice_clustering_1_connection_cost.at(i) << "\t "
              << nice_clustering_2_num_open_facilities.at(i) << "\t "
              << nice_clustering_2_opening_cost.at(i) << "\t "
              << nice_clustering_2_connection_cost.at(i) << "\t "
              << nice_clustering_3_num_open_facilities.at(i) << "\t "
              << nice_clustering_3_opening_cost.at(i) << "\t "
              << nice_clustering_3_connection_cost.at(i) << "\t "
              << nice_clustering_4_num_open_facilities.at(i) << "\t "
              << nice_clustering_4_opening_cost.at(i) << "\t "
              << nice_clustering_4_connection_cost.at(i) << "\t "
              << gklx_num_open_facilities.at(i) << "\t "
              << gklx_opening_cost.at(i) << "\t " << gklx_connection_cost.at(i)
              << "\t " << greedy_num_open_facilities.at(i) << "\t "
              << greedy_opening_cost.at(i) << "\t "
              << greedy_connection_cost.at(i) << std::endl;
  }

  return 0;
}
