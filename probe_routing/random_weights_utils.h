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

#ifndef RANDOM_WEIGHTS_UTILS_H_
#define RANDOM_WEIGHTS_UTILS_H_

#include <cstdlib>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "graph.h"
#include "graph_search.h"

namespace geo_algorithms {

absl::flat_hash_map<int, double> GetClusterCosts(
    const MultiCostGraph& graph, const std::vector<int>& cluster_assignment);

std::pair<MultiCostGraph*, double> GetMergedGraphAndRealClustersFraction(
    const MultiCostGraph& no_traffic_graph,
    const MultiCostGraph& resampled_traffic_graph,
    const MultiCostGraph& real_traffic_graph, double replacement_threshold,
    const std::vector<int>& cluster_assignment);

std::vector<double> EvaluatePathInNewGraph(
    const MultiCostGraph& old_graph, const MultiCostGraph& new_graph,
    const WeightedCostSearch::Result& path);

class PathCache {
 public:
  PathCache(MultiCostGraph* graph, const std::string& graph_name,
            const std::string& output_dir);

  absl::StatusOr<WeightedCostSearch::Result> GetPath(int source, int target);

  int GetClusterCountWithinDistanceAboveThreshold(
      int source, double distance_threshold, double weight_threshold,
      const std::vector<int>& cluster_assignment);

  void ReplaceMergedGraph(MultiCostGraph* merged_graph) {
    graph_ = merged_graph;
  }

 private:
  struct CountData {
    std::vector<double> distance;
    std::vector<double> weight;
    std::vector<int> count;
  };
  void SavePathToFiles(int source, int target,
                       absl::StatusOr<WeightedCostSearch::Result> path);
  absl::StatusOr<WeightedCostSearch::Result> ReadPathFromFiles(int source,
                                                               int target);

  CountData GetCounts(const std::vector<int>& cluster_assignment, int source);
  void SaveCountsToFile(int source, CountData counts);
  absl::StatusOr<CountData> ReadCountsFromFile(int source);

  std::string path_cache_filename(int source, int target);
  std::string cost_cache_filename(int source, int target);
  std::string cache_dir();
  std::string counts_filename(int source);
  const std::string output_dir_;
  MultiCostGraph* graph_;
  const std::string graph_name_;
};

}  // namespace geo_algorithms

#endif  // RANDOM_WEIGHTS_UTILS_H_
