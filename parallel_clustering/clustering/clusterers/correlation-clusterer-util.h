// Copyright 2020 The Google Research Authors.
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

#ifndef RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_CORRELATION_CLUSTERER_UTIL_H_
#define RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_CORRELATION_CLUSTERER_UTIL_H_

#include "absl/container/flat_hash_map.h"
#include "clustering/config.pb.h"
#include "clustering/in-memory-clusterer.h"

namespace research_graph::in_memory {

// A helper class that keeps track of the sum of edge weights, accounting
// for missing edges, for best move computations.
class EdgeSum {
 public:
  // The edge weight w should have the edge weight offset subtracted before
  // calling this function.
  void Add(double w) { weight_ += w; }
  // Should be called at most once, after all edges have been Add()ed.
  void RemoveDoubleCounting() { weight_ /= 2; }
  // Retrieve the total weight of all edges seen, correcting for the implicit
  // negative weight of resolution multiplied by the product of the weights of
  // the two nodes incident to each edge.
  double NetWeight(double sum_prod_node_weights,
                   const CorrelationClustererConfig& config) const {
    return weight_ - config.resolution() * sum_prod_node_weights;
  }

 private:
  double weight_ = 0.0;
};

// Computes the best move given certain pre-computed sums of edge weights of the
// following classes of vertices in relation to a fixed set of moving_nodes that
// may change clusters:
//  * Class 0: Neither node is moving.
//  * Class 1: Exactly one node is moving.
//  * Class 2: Both nodes are moving.
// where "moving" means in moving_nodes.
//
// Change in objective if we move all moving nodes to cluster i:
//   class_2_currently_separate + class_1_together_after[i] -
//   class_1_currently_together
// where
//   class_2_currently_separate = Weight of edges in class 2 where the endpoints
//       are in different clusters currently
//   class_1_together_after[i] = Weight of edges in class 1 where the non-moving
//       node is in cluster i
//   class_1_currently_together = Weight of edges in class 1 where the endpoints
//       are in the same cluster currently
//
// Two complications:
//   * We need to avoid double-counting pairs in class 2
//   * We need to account for missing edges, which have weight
//     -resolution. To do so we subtract the number of edges we see in each
//     category from the max possible number of edges (i.e. the number of edges
//     we'd have if the graph was complete).
template <typename ClusterId>
std::pair<absl::optional<ClusterId>, double> BestMoveFromStats(
    const CorrelationClustererConfig& config,
    const std::function<double(ClusterId)>& get_current_cluster_weight,
    double moving_nodes_weight,
    absl::flat_hash_map<ClusterId, double>& cluster_moving_weights,
    EdgeSum class_2_currently_separate, EdgeSum class_1_currently_together,
    const absl::flat_hash_map<ClusterId, EdgeSum>& class_1_together_after) {
  double change_in_objective = 0;

  auto half_square = [](const double x) { return x * x / 2; };
  double max_edges = half_square(moving_nodes_weight);
  for (const auto& [cluster, moving_nodes_weight] : cluster_moving_weights) {
    max_edges -= half_square(moving_nodes_weight);
  }
  change_in_objective +=
      class_2_currently_separate.NetWeight(max_edges, config);

  max_edges = 0;
  for (const auto& [cluster, moving_nodes_weight] : cluster_moving_weights) {
    max_edges += moving_nodes_weight *
                 (get_current_cluster_weight(cluster) - moving_nodes_weight);
  }
  change_in_objective -=
      class_1_currently_together.NetWeight(max_edges, config);

  std::pair<absl::optional<ClusterId>, double> best_move;
  best_move.first = absl::nullopt;
  best_move.second = change_in_objective;
  for (const auto& [cluster, data] : class_1_together_after) {
    max_edges = moving_nodes_weight * (get_current_cluster_weight(cluster) -
                                       cluster_moving_weights[cluster]);
    // Change in objective if we move the moving nodes to cluster i.
    double overall_change_in_objective =
        change_in_objective + data.NetWeight(max_edges, config);
    if (overall_change_in_objective > best_move.second ||
        (overall_change_in_objective == best_move.second &&
         cluster < best_move.first)) {
      best_move.first = cluster;
      best_move.second = overall_change_in_objective;
    }
  }
  return best_move;
}

}  // namespace research_graph::in_memory

#endif  // RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_CORRELATION_CLUSTERER_UTIL_H_
