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

#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "random_handler.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/numbers.h"
#include "util/math/mathutil.h"

constexpr int distance_exponent = 2;
constexpr int distance_multiplier = 1;

int NearestFacilityOfClient(const int client_id,
                            absl::flat_hash_map<int, double> neighbors) {
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

double LDistance(std::vector<double> embedding_1,
                 std::vector<double> embedding_2) {
  double sum = 0.0;
  for (int i = 0; i < embedding_1.size(); i++)
    sum += ::MathUtil::IPow<double>(abs(embedding_1.at(i) - embedding_2.at(i)),
                                    distance_exponent);
  return distance_multiplier * pow(sum, 1.0 / distance_exponent);
}

void GraphHandler::ReadEmbeddings() {
  // Endpoints of the edges.
  char c[50], f[50], w[50];

  CHECK_EQ(scanf("%s %s\n", f, w), 2);

  int n, d;
  CHECK(absl::SimpleAtoi(f, &n));
  CHECK(absl::SimpleAtoi(w, &d));

  for (int i = 0; i < n; i++) {
    input_embeddings_.emplace_back(std::vector<double>());
    for (int j = 0; j < d; j++) {
      CHECK_EQ(scanf("%s ", c), 1);
      double coordinate;
      CHECK(absl::SimpleAtod(c, &coordinate));
      input_embeddings_[i].emplace_back(coordinate);
    }
  }
}

void GraphHandler::SetUpGraph(const float percentage_of_facilities,
                              const int32_t target_num_nodes) {
  // Shuffle points.
  absl::flat_hash_set<int> added_points;
  std::vector<std::vector<double>> embeddings;
  for (int i = 0; i < target_num_nodes; i++) {
    int random_index = RandomHandler::eng_() % input_embeddings_.size();
    while (added_points.contains(random_index)) {
      random_index = RandomHandler::eng_() % input_embeddings_.size();
    }
    added_points.insert(random_index);
    embeddings.emplace_back(input_embeddings_[random_index]);
  }

  // Pick the facilities
  absl::flat_hash_set<int> facilities;
  for (int i = 0; i < target_num_nodes * (percentage_of_facilities / 100.0);
       i++) {
    int random_index = RandomHandler::eng_() % target_num_nodes;
    while (facilities.contains(random_index)) {
      random_index = RandomHandler::eng_() % target_num_nodes;
    }
    std::cout << "Indeed " << random_index << " picked as facility\n"
              << " " << i << " / "
              << (target_num_nodes * (percentage_of_facilities / 100.0))
              << std::endl;
    facilities.insert(random_index);
  }
  std::cout << "Done picking facilities\n";

  // Allocate memory for neighborhoods
  client_neighbors_.clear();
  facility_neighbors_.clear();
  facility_to_facility_neighbors_.clear();
  for (int i = 0; i < embeddings.size(); i++) {
    if (facilities.contains(i)) {
      facility_neighbors_[i] = absl::flat_hash_map<int, double>();
      facility_to_facility_neighbors_[i] = absl::flat_hash_map<int, double>();
    } else {
      num_clients_++;
      client_neighbors_[i] = absl::flat_hash_map<int, double>();
    }
  }

  std::cout << "Start the construction of the graph.\n";

  absl::flat_hash_map<int, int> edges_in_bucket;
  absl::flat_hash_map<int, int> nearest_facility_in_bucket;
  std::vector<double> nearest_facility_distance;
  double sum_distances = 0.0;
  int count_distances = 0;
  double sum_distances_to_nearest = 0.0;
  int count_distances_to_nearest = 0;

  for (int i = 0; i < embeddings.size(); i++) {
    if (facilities.contains(i)) {
      for (int facility_id : facilities) {
        if (facility_id == i) continue;
        double distance = LDistance(embeddings[i], embeddings[facility_id]);
        facility_to_facility_neighbors_[facility_id][i] =
            distance + 1.0 / target_num_nodes;
        facility_to_facility_neighbors_[i][facility_id] =
            distance + 1.0 / target_num_nodes;
      }
    } else {
      for (int facility_id : facilities) {
        double distance = LDistance(embeddings[i], embeddings[facility_id]);
        edges_in_bucket[ceil(log2(distance + 1.0 / target_num_nodes))]++;
        client_neighbors_[i][facility_id] = distance + 1.0 / target_num_nodes;
        facility_neighbors_[facility_id][i] = distance + 1.0 / target_num_nodes;
        sum_distances += distance + 1.0 / target_num_nodes;
        count_distances++;
      }
      int nearest_facility = NearestFacilityOfClient(i, client_neighbors_[i]);
      double distance = LDistance(embeddings[i], embeddings[nearest_facility]);
      nearest_facility_in_bucket[ceil(
          log2(distance + 1.0 / target_num_nodes))]++;
      sum_distances_to_nearest += distance + 1.0 / target_num_nodes;
      nearest_facility_distance.push_back(distance + 1.0 / target_num_nodes);
      count_distances_to_nearest++;
    }
  }
  std::cout << "avg distance: " << sum_distances / count_distances << std::endl;
  std::cout << "avg distances to nearest: "
            << sum_distances_to_nearest / count_distances_to_nearest
            << std::endl;

  std::vector<std::pair<int, int>> sorted_edges_in_bucket(
      edges_in_bucket.begin(), edges_in_bucket.end());

  int cumulative_count = 0;
  std::sort(sorted_edges_in_bucket.begin(), sorted_edges_in_bucket.end(),
            [](auto& left, auto& right) { return left.first < right.first; });
  for (const auto& [bucket, count] : sorted_edges_in_bucket) {
    cumulative_count += count;
    std::cout << "In bucket [" << pow(2, bucket - 1) << "," << pow(2, bucket)
              << "] got " << count << " edges. Cumulative count "
              << cumulative_count << std::endl;
  }

  std::cout << "Distribution of distances to nearest facility required for the "
               "GKLX algorithm."
            << std::endl;
  std::vector<std::pair<int, int>> sorted_nearest_facility_in_bucket(
      nearest_facility_in_bucket.begin(), nearest_facility_in_bucket.end());

  cumulative_count = 0;
  std::sort(sorted_nearest_facility_in_bucket.begin(),
            sorted_nearest_facility_in_bucket.end(),
            [](auto& left, auto& right) { return left.first < right.first; });

  for (const auto& [bucket, count] : sorted_nearest_facility_in_bucket) {
    cumulative_count += count;
    std::cout << "In bucket [" << pow(2, bucket - 1) << "," << pow(2, bucket)
              << "] got " << count
              << " edges. Cumulative count: " << cumulative_count << std::endl;
  }
  std::cout << "Done adding neighbors\n";
  std::sort(nearest_facility_distance.begin(), nearest_facility_distance.end());

  facility_cost_upperbound_ =
      (int)100.0 * sum_distances_to_nearest / count_distances_to_nearest;

  facility_weight_.clear();
  for (const int facility_id : facilities) {
    double random_weight = (RandomHandler::eng_() /
                            (double)std::numeric_limits<uint_fast32_t>::max()) *
                           (facility_cost_upperbound_);
    facility_weight_[facility_id] = random_weight;
    std::cout << "Cost [" << facility_id << "] = " << random_weight
              << std::endl;
  }
  next_update_to_execute_ = 0;
}

void GraphHandler::ReadClientToFacilityEdges() {
  int num_vertex = 0;
  // Endpoints of the edges.
  char c[50], f[50], w[50];
  int target_num_edges = 0;

  while (scanf("f\t%s\t%s", f, w) == 2) {
    std::string facility = f;
    std::string weight = w;
    CHECK(node_id_map_.find(facility) == node_id_map_.end());
    node_id_map_[facility] = num_vertex++;
    const int facility_id = node_id_map_[facility];
    facility_weight_[facility_id] = std::stod(weight);
  }

  while (scanf("c\t%s\t%s\t%s", c, f, w) == 3) {
    if (++target_num_edges >= 1000) break;
    std::string client = c;
    std::string facility = f;
    std::string weight_as_string = w;
    double weight = std::stod(weight_as_string);
    if (node_id_map_.find(client) == node_id_map_.end()) {
      node_id_map_[client] = num_vertex++;
      client_neighbors_[node_id_map_[client]] =
          absl::flat_hash_map<int, double>();
      num_clients_++;
    }
    CHECK(node_id_map_.find(facility) != node_id_map_.end());
    int client_id = node_id_map_[client], facility_id = node_id_map_[facility];
    if (client_to_facility_edges_.find(
            std::tuple<int, int, double>(client_id, facility_id, weight)) ==
        client_to_facility_edges_.end()) {
      facility_neighbors_[facility_id].emplace(client_id, weight);
      CHECK(client_id != facility_id);
      client_neighbors_[client_id].emplace(facility_id, weight);
      client_to_facility_edges_.insert(
          std::tuple<int, int, double>(client_id, facility_id, weight));
    }
  }
}

void GraphHandler::StartMaintainingDynamicUpdates(
    bool shuffle_order, double prob_of_deletion, int sliding_window_size,
    ArrivalStrategy arrival_strategy) {
  std::vector<int> order_to_client;
  absl::flat_hash_map<int, int> client_to_order;
  for (const auto& [client_id, neighbors] : client_neighbors_) {
    order_to_client.push_back(client_id);
  }

  if (shuffle_order) {
    for (int i = 0; i < order_to_client.size(); i++) {
      int x = RandomHandler::eng_() % order_to_client.size();
      std::swap(order_to_client[i], order_to_client[x]);
    }
  }

  for (int i = 0; i < order_to_client.size(); i++) {
    client_to_order[order_to_client[i]] = i;
  }
  update_sequence_.clear();
  if (arrival_strategy == SLIDING_WINDOW) {
    int num_inserted_nodes = 0;
    for (const int client : order_to_client) {
      if (num_inserted_nodes >= sliding_window_size) {
        update_sequence_.push_back(std::make_pair(
            false, order_to_client[num_inserted_nodes - sliding_window_size]));
      }
      update_sequence_.push_back(std::make_pair(true, client));
      num_inserted_nodes++;
    }
    for (int i = num_inserted_nodes - sliding_window_size;
         i < order_to_client.size(); i++) {
      update_sequence_.push_back(std::make_pair(false, order_to_client[i]));
    }
  } else if (arrival_strategy == DELETION_PROBABILITY) {
    int num_inserted_nodes = 0;
    absl::flat_hash_map<int, bool> is_deleted;
    for (const auto& [client, order] : client_to_order) {
      is_deleted[client] = false;
    }
    int num_of_nodes_present = 0;
    for (const int client : order_to_client) {
      update_sequence_.push_back(std::make_pair(true, client));
      num_inserted_nodes++;
      num_of_nodes_present++;

      double randomized_decission =
          (double)RandomHandler::eng_() /
          (double)std::numeric_limits<uint_fast64_t>::max();
      while (randomized_decission < prob_of_deletion &&
             num_of_nodes_present > 1) {
        num_of_nodes_present--;
        // std::cout << randomized_decission << std::endl;
        int random_index = RandomHandler::eng_() % num_inserted_nodes;
        int random_node = order_to_client[random_index];
        while (is_deleted[random_node]) {
          random_index = RandomHandler::eng_() % num_inserted_nodes;
          random_node = order_to_client[random_index];
        }
        is_deleted[random_node] = true;
        update_sequence_.push_back(std::make_pair(false, random_node));
        randomized_decission =
            (double)RandomHandler::eng_() /
            (double)std::numeric_limits<uint_fast64_t>::max();
      }
    }
  }
  next_update_to_execute_ = 0;
}

void GraphHandler::RestartStreamOfClients() { next_update_to_execute_ = 0; }

std::optional<std::pair<bool, int>> GraphHandler::GetNextUpdate() {
  if (next_update_to_execute_ >= update_sequence_.size() - 1)
    return std::nullopt;

  auto ret_value = std::pair(update_sequence_[next_update_to_execute_].first,
                             update_sequence_[next_update_to_execute_].second);
  next_update_to_execute_++;

  return ret_value;
}

absl::flat_hash_map<int, double> GraphHandler::GetClientsNeighbors(
    const int client_id) {
  absl::flat_hash_map<int, double> neighbors_of_next_client;

  for (const auto [facility_id, distance] : client_neighbors_[client_id]) {
    neighbors_of_next_client.emplace(facility_id, distance);
  }

  return neighbors_of_next_client;
}
