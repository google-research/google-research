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

#ifndef ONLINE_CORRELATION_CLUSTERING_PIVOT_ALGO_H_
#define ONLINE_CORRELATION_CLUSTERING_PIVOT_ALGO_H_

#include <map>
#include <vector>

class PivotAlgorithm {
 public:
  explicit PivotAlgorithm(const std::vector<std::vector<int>>& neighbors)
      : neighbors_(neighbors) {}
  // Clusters the given graph based on the Pivot algorithm.
  std::vector<int> Cluster();

  // Clusters the given graph based on the Pivot algorithm.
  std::vector<int> InsertNodeToClustering(
      const std::vector<int>& new_nodes_neighbors);

  // Stores the maintained graph.
  std::vector<std::vector<int>> neighbors_;

 private:
  std::vector<bool> is_pivot_;

  std::map<double, int> nodes_ordered_by_rank_;

  std::vector<double> rank_;

  std::vector<int> node_id_to_cluster_id_;
};

#endif  // ONLINE_CORRELATION_CLUSTERING_PIVOT_ALGO_H_
