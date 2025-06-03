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

#ifndef GRAPH_SEARCH_H_
#define GRAPH_SEARCH_H_

#include <memory>
#include <queue>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "graph.h"

namespace geo_algorithms {

class WeightedCostSearch {
 public:
  struct Result {
    std::vector<ArcIndex> path;
    std::vector<double> cost_vector;
  };

  struct Stats {
    int64_t num_edges_traversed = 0;
  };

  // Constructors run Dijkstra in time O(graph size)
  WeightedCostSearch(const MultiCostGraph& graph,
                     std::vector<double> cost_weights, int source)
      : graph_(graph), cost_weights_(cost_weights), sources_({source}) {
    GenerateSearchTree();
  }

  WeightedCostSearch(const MultiCostGraph& graph,
                     std::vector<double> cost_weights,
                     const absl::flat_hash_set<int> sources)
      : graph_(graph), cost_weights_(cost_weights), sources_(sources) {
    GenerateSearchTree();
  }

  // Returns cost vector for shortest path to target in O(1) time.
  absl::StatusOr<std::vector<double>> Cost(int target) const;
  // Returns the closest source to the target in O(1) time.
  absl::StatusOr<int> ClosestSource(int target) const;
  // Returns path to target in time O(length of path returned)
  absl::StatusOr<Result> Search(int target) const;
  // Returns all arcs in the shortest path tree generated during this
  // search.
  std::vector<ArcIndex> SearchTreeArcs() const;
  // Returns the minimum cost to this node found during the search (or nullopt
  // if the node wasn't reached). Must be called after Search().
  std::optional<std::vector<double>> GetCostFromSearch(int node) const;

  const Stats& stats() const { return stats_; }

 private:
  struct NodeVisit {
    int id = -1;
    int prev_id = -1;
    int source = -1;
    std::vector<double> cost_vector;
  };

  void GenerateSearchTree();
  void ProcessQueue();
  void Enqueue(int node, int prev_node, int source,
               const std::vector<double>& cost_vector);

  const MultiCostGraph& graph_;
  std::vector<double> cost_weights_;
  absl::flat_hash_set<int> sources_;

  std::priority_queue<std::pair<double, const NodeVisit*>> queue_;
  std::vector<std::unique_ptr<NodeVisit>> node_visits_;
  absl::flat_hash_map<int, const NodeVisit*> settled_nodes_;

  Stats stats_;
};

}  // namespace geo_algorithms

#endif  // GRAPH_SEARCH_H_
