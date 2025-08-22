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

#include "graph_search.h"

#include "absl/log/log.h"
#include "graph.h"
#include "graph_utils.h"
#include "tsv_utils.h"

namespace geo_algorithms {

absl::StatusOr<std::vector<double>> WeightedCostSearch::Cost(int target) const {
  const auto it = settled_nodes_.find(target);
  if (it == settled_nodes_.end()) {
    return absl::NotFoundError("No path to target from any source!");
  }

  const NodeVisit* cur_visit = it->second;
  return cur_visit->cost_vector;
}

absl::StatusOr<int> WeightedCostSearch::ClosestSource(int target) const {
  // LOG(INFO) << "Number of settled nodes: " << settled_nodes_.size() << "";
  const auto it = settled_nodes_.find(target);
  if (it == settled_nodes_.end()) {
    return absl::NotFoundError("No path to target from any source!");
  }

  const NodeVisit* cur_visit = it->second;
  return cur_visit->source;
}

absl::StatusOr<WeightedCostSearch::Result> WeightedCostSearch::Search(
    int target) const {
  const auto it = settled_nodes_.find(target);
  if (it == settled_nodes_.end()) {
    return absl::NotFoundError("No path to target from any source!");
  }

  const NodeVisit* cur_visit = it->second;
  Result result;
  result.cost_vector = cur_visit->cost_vector;
  while (cur_visit->prev_id != -1) {
    result.path.push_back(
        graph_.ArcIndexFor(cur_visit->prev_id, cur_visit->id));
    cur_visit = settled_nodes_.at(cur_visit->prev_id);
  }
  std::reverse(result.path.begin(), result.path.end());
  result.path.push_back(ArcIndex{target, -1});
  return result;
}

std::vector<ArcIndex> WeightedCostSearch::SearchTreeArcs() const {
  std::vector<ArcIndex> ret;
  for (const auto& [dst, visit] : settled_nodes_) {
    if (visit->prev_id == -1) continue;
    ret.push_back(graph_.ArcIndexFor(visit->prev_id, dst));
  }
  return ret;
}

using utils_tsv::TsvRow;
using utils_tsv::TsvSpec;
using utils_tsv::TsvWriter;

void WeightedCostSearch::GenerateSearchTree() {
  for (int source : sources_) {
    Enqueue(source, /*prev_node=*/-1, source, graph_.ZeroCostVector());
  }
  LOG(INFO) << "Before: " << settled_nodes_.size() << "";
  while (!queue_.empty()) {
    ProcessQueue();
  }
  LOG(INFO) << "After: " << settled_nodes_.size() << "";
}

void WeightedCostSearch::ProcessQueue() {
  const NodeVisit* visit = queue_.top().second;
  queue_.pop();

  const auto [it, inserted] = settled_nodes_.insert({visit->id, visit});
  if (!inserted) {
    return;
  }

  for (const MultiCostGraph::Arc& arc : graph_.ArcsForNode(visit->id)) {
    stats_.num_edges_traversed++;
    Enqueue(arc.dst, visit->id, visit->source,
            AddCostVectors(visit->cost_vector, arc.cost_vector));
  }
}

void WeightedCostSearch::Enqueue(int node, int prev_node, int source,
                                 const std::vector<double>& cost_vector) {
  const double cost = WeightedCost(cost_weights_, cost_vector);

  node_visits_.push_back(std::make_unique<NodeVisit>());
  NodeVisit& visit = *node_visits_.back();
  visit.id = node;
  visit.prev_id = prev_node;
  visit.source = source;
  visit.cost_vector = cost_vector;

  queue_.push({-cost, &visit});
}

}  // namespace geo_algorithms
