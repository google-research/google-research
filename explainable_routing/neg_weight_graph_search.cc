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

#include "neg_weight_graph_search.h"

#include "absl/log/log.h"
#include "graph.h"

namespace geo_algorithms {

absl::StatusOr<int64_t> Search::Cost(int target) const {
  if (negative_cycle_detected_) {
    return absl::NotFoundError("Negative cycle detected!");
  }
  const auto it = settled_nodes_.find(target);
  if (it == settled_nodes_.end()) {
    return absl::NotFoundError("No path to target from any source!");
  }

  const NodeVisit* cur_visit = it->second;
  return cur_visit->cost;
}

absl::StatusOr<int> Search::ClosestSource(int target) const {
  if (negative_cycle_detected_) {
    return absl::NotFoundError("Negative cycle detected!");
  }
  const auto it = settled_nodes_.find(target);
  if (it == settled_nodes_.end()) {
    return absl::NotFoundError("No path to target from any source!");
  }

  const NodeVisit* cur_visit = it->second;
  return cur_visit->source;
}

absl::StatusOr<Search::Result> Search::FindPath(int target) const {
  if (negative_cycle_detected_) {
    return absl::NotFoundError("Negative cycle detected!");
  }
  const auto it = settled_nodes_.find(target);
  if (it == settled_nodes_.end()) {
    return absl::NotFoundError("No path to target from any source!");
  }

  const NodeVisit* cur_visit = it->second;
  Result result;
  result.cost = cur_visit->cost;
  while (cur_visit->prev_id != -1) {
    result.path.push_back(cur_visit->incoming_arc);
    cur_visit = settled_nodes_.at(cur_visit->prev_id);
  }
  std::reverse(result.path.begin(), result.path.end());
  result.path.push_back(
      AOrRIndex{.forward = ArcIndex{target, -1}, .is_forward = true});
  return result;
}

std::vector<AOrRIndex> Search::SearchTreeArcs() const {
  std::vector<AOrRIndex> ret;
  for (const auto& [dst, visit] : settled_nodes_) {
    if (visit->prev_id == -1) continue;
    ret.push_back(visit->incoming_arc);
  }
  return ret;
}

absl::StatusOr<std::vector<AOrRIndex>> Search::GetNegativeCycleIfExists()
    const {
  if (!negative_cycle_detected_) {
    return absl::NotFoundError("No negative cycle detected!");
  }
  return negative_cycle_;
}

void Search::GenerateSearchTree() {
  for (const auto& [arc, _] : cost_model_.residual_costs) {
    AOrRIndex idx = AOrRIndex{.residual = arc, .is_forward = false};
    residual_arcs_for_node_[graph_.AOrRIndexSrc(idx)].insert(arc);
  }

  negative_cycle_detected_ = false;
  for (int source : sources_) {
    // note: this AOrRIndex() may not look invalid
    // but only real edges should be used at all in the algorithm
    // so it is ok. Would be better to put null here.
    Enqueue(source, /*prev_node=*/-1, source, 0, AOrRIndex());
  }
  // LOG(INFO) << "Before: " << settled_nodes_.size() << "";
  while (!queue_.empty()) {
    ProcessQueue();
  }
  // LOG(INFO) << "After: " << settled_nodes_.size() << "";
}

std::vector<AOrRIndex> Search::OutgoingArcs(int node) {
  std::vector<AOrRIndex> result;
  absl::flat_hash_set<std::pair<int, int>> pairs_so_far;
  if (residual_arcs_for_node_.contains(node)) {
    for (const ResidualIndex& e : residual_arcs_for_node_.at(node)) {
      AOrRIndex e_full = AOrRIndex{.residual = e, .is_forward = false};
      int dst = graph_.AOrRIndexDst(e_full);
      std::pair<int, int> p = std::make_pair(node, dst);
      CHECK(!pairs_so_far.contains(p));
      pairs_so_far.insert(p);
      result.push_back(e_full);
    }
  }

  for (int i = 0; i < graph_.ArcsForNode(node).size(); i++) {
    AOrRIndex e_full = AOrRIndex{.forward = ArcIndex{.node = node, .index = i},
                                 .is_forward = true};
    int dst = graph_.AOrRIndexDst(e_full);
    std::pair<int, int> p = std::make_pair(node, dst);
    if (!pairs_so_far.contains(p)) {
      result.push_back(e_full);
    }
  }
  return result;
}

int64_t Search::ActualCost(AOrRIndex arc) {
  if (arc.is_forward) {
    if (cost_model_.updated_forward_costs.contains(arc.forward)) {
      return cost_model_.updated_forward_costs.at(arc.forward);
    } else {
      return cost_model_.default_costs.at(arc.forward);
    }
  } else {
    return cost_model_.residual_costs.at(arc.residual);
  }
}

void Search::CheckForNegativeCycleAndEndIfPresent(const NodeVisit* visit) {
  negative_cycle_detected_ =
      (link_cut_tree_.find_root(visit->prev_id) == visit->id);

  if (negative_cycle_detected_) {
    int tip = visit->id;
    int cur = visit->prev_id;
    int64_t possible_cycle_cost = ActualCost(visit->incoming_arc);
    std::vector<AOrRIndex> possible_cycle = {visit->incoming_arc};
    while (cur != tip) {
      const NodeVisit* cur_visit = settled_nodes_.at(cur);
      possible_cycle_cost += ActualCost(cur_visit->incoming_arc);
      possible_cycle.push_back(cur_visit->incoming_arc);
      cur = cur_visit->prev_id;
    }
    if (cur == tip) {
      std::reverse(possible_cycle.begin(), possible_cycle.end());
      CHECK_LT(possible_cycle_cost, 0);
      negative_cycle_ = possible_cycle;
      while (!queue_.empty()) {
        queue_.pop();
      }
    }
  }
}

void Search::ProcessQueue() {
  const NodeVisit* visit = queue_.top().second;
  queue_.pop();

  // rounding introduced due to subtle 0-weight cycles
  // along the main path in explainable routing
  bool replace = !settled_nodes_.contains(visit->id) ||
                 (visit->cost < settled_nodes_.at(visit->id)->cost);

  if (!replace) {
    return;
  }

  if (!cost_model_.nonneg_weights) {
    CHECK_EQ(sources_.size(), 1)
        << " More than one source not supported for negative weights.";
    if (settled_nodes_.contains(visit->id)) {
      if (!sources_.contains(visit->id)) {
        link_cut_tree_.cut(visit->id);
      }
      CHECK_EQ(link_cut_tree_.find_root(visit->id), visit->id);
      CheckForNegativeCycleAndEndIfPresent(visit);
      if (negative_cycle_detected_) {
        return;
      }
    }
    if (visit->prev_id != -1) {
      link_cut_tree_.link(visit->id, visit->prev_id);
    }
  }

  settled_nodes_[visit->id] = visit;

  for (const AOrRIndex& arc : OutgoingArcs(visit->id)) {
    stats_.num_edges_traversed++;
    int64_t arc_cost = ActualCost(arc);
    if (cost_model_.nonneg_weights) {
      CHECK_GE(arc_cost, 0);
    }
    Enqueue(graph_.AOrRIndexDst(arc), visit->id, visit->source,
            visit->cost + arc_cost, arc);
  }
}

void Search::Enqueue(int node, int prev_node, int source, int64_t cost,
                     AOrRIndex incoming_arc) {
  node_visits_.push_back(std::make_unique<NodeVisit>());
  NodeVisit& visit = *node_visits_.back();
  visit.id = node;
  visit.prev_id = prev_node;
  visit.source = source;
  visit.cost = cost;
  visit.incoming_arc = incoming_arc;

  queue_.push({-cost, &visit});
}

}  // namespace geo_algorithms
