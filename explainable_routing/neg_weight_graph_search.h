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

#ifndef NEG_WEIGHT_GRAPH_SEARCH_H_
#define NEG_WEIGHT_GRAPH_SEARCH_H_

#include <memory>
#include <queue>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "graph.h"
#include "link_cut_tree.h"

namespace geo_algorithms {

class Search {
 public:
  struct CostModel {
    const absl::flat_hash_map<ArcIndex, int64_t>& default_costs;
    const absl::flat_hash_map<ArcIndex, int64_t> updated_forward_costs;
    const absl::flat_hash_map<ResidualIndex, int64_t> residual_costs;
    const bool nonneg_weights;
  };

  struct Result {
    std::vector<AOrRIndex> path;
    int64_t cost;
  };

  struct Stats {
    int num_edges_traversed = 0;
  };

  // Constructors run Dijkstra slightly modified to account for negative
  // arc weights.
  Search(const Graph& graph, const CostModel& cost_model, int source)
      : graph_(graph),
        cost_model_(cost_model),
        sources_({source}),
        link_cut_tree_(graph.NumNodes()) {
    GenerateSearchTree();
  }

  Search(const Graph& graph, const CostModel& cost_model,
         const absl::flat_hash_set<int> sources)
      : graph_(graph),
        cost_model_(cost_model),
        sources_(sources),
        link_cut_tree_(graph.NumNodes()) {
    GenerateSearchTree();
  }

  bool NegativeCycleExists() const { return negative_cycle_detected_; }

  // Following three methods return an error if a negative cycle exists.
  // Returns cost for shortest path to target in O(1) time.
  absl::StatusOr<int64_t> Cost(int target) const;
  // Returns the closest source to the target in O(1) time.
  absl::StatusOr<int> ClosestSource(int target) const;
  // Returns path to target in time O(length of path returned)
  absl::StatusOr<Result> FindPath(int target) const;

  std::vector<AOrRIndex> SearchTreeArcs() const;

  // Returns a negative cycle if it exists, or an error otherwise.
  absl::StatusOr<std::vector<AOrRIndex>> GetNegativeCycleIfExists() const;

  const Stats& stats() const { return stats_; }

 private:
  struct NodeVisit {
    int id = -1;
    int prev_id = -1;
    int source = -1;
    int64_t cost;
    AOrRIndex incoming_arc;
  };

  void GenerateSearchTree();
  std::vector<AOrRIndex> OutgoingArcs(int node);
  int64_t ActualCost(AOrRIndex arc);
  void CheckForNegativeCycleAndEndIfPresent(const NodeVisit* visit);
  void ProcessQueue();
  void Enqueue(int node, int prev_node, int source, int64_t cost,
               AOrRIndex incoming_arc);

  const Graph& graph_;
  const CostModel& cost_model_;
  const absl::flat_hash_set<int> sources_;
  LinkCutTree link_cut_tree_;

  absl::flat_hash_map<int, absl::flat_hash_set<ResidualIndex>>
      residual_arcs_for_node_;
  std::priority_queue<std::pair<int64_t, const NodeVisit*>> queue_;
  std::vector<std::unique_ptr<NodeVisit>> node_visits_;
  absl::flat_hash_map<int, const NodeVisit*> settled_nodes_;

  Stats stats_;
  bool negative_cycle_detected_;
  std::vector<AOrRIndex> negative_cycle_;
};

}  // namespace geo_algorithms

#endif  // NEG_WEIGHT_GRAPH_SEARCH_H_
