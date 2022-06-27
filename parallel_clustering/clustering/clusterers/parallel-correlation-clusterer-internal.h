// Copyright 2022 The Google Research Authors.
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

#ifndef RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_PARALLEL_CORRELATION_CLUSTERER_INTERNAL_H_
#define RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_PARALLEL_CORRELATION_CLUSTERER_INTERNAL_H_

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "clustering/config.pb.h"
#include "external/gbbs/gbbs/graph.h"
#include "external/gbbs/gbbs/vertex_subset.h"
#include "clustering/in-memory-clusterer.h"
#include "parallel/parallel-graph-utils.h"

namespace research_graph {
namespace in_memory {

// This class encapsulates the data needed to compute and maintain the
// correlation clustering objective.
class ClusteringHelper {
 public:
  using ClusterId = gbbs::uintE;

  ClusteringHelper(InMemoryClusterer::NodeId num_nodes,
                   const ClustererConfig& clusterer_config,
                   const InMemoryClusterer::Clustering& clustering)
      : num_nodes_(num_nodes),
        cluster_ids_(num_nodes),
        cluster_sizes_(num_nodes, 0),
        clusterer_config_(clusterer_config),
        node_weights_(num_nodes, 1),
        cluster_weights_(num_nodes, 0) {
    SetClustering(clustering);
  }

  ClusteringHelper(InMemoryClusterer::NodeId num_nodes,
                   const ClustererConfig& clusterer_config,
                   std::vector<double> node_weights,
                   const InMemoryClusterer::Clustering& clustering)
      : num_nodes_(num_nodes),
        cluster_ids_(num_nodes),
        cluster_sizes_(num_nodes, 0),
        clusterer_config_(clusterer_config),
        node_weights_(std::move(node_weights)),
        cluster_weights_(num_nodes, 0) {
    SetClustering(clustering);
  }

  // Constructor for testing purposes, to outright set cluster_ids_ and
  // cluster_sizes_.
  ClusteringHelper(size_t num_nodes, std::vector<ClusterId> cluster_ids,
                   std::vector<ClusterId> cluster_sizes,
                   std::vector<double> cluster_weights,
                   const ClustererConfig& clusterer_config,
                   std::vector<double> node_weights)
      : num_nodes_(num_nodes),
        cluster_ids_(std::move(cluster_ids)),
        cluster_sizes_(std::move(cluster_sizes)),
        clusterer_config_(clusterer_config),
        node_weights_(std::move(node_weights)),
        cluster_weights_(cluster_weights) {}

  // Contains objective change, which includes:
  //  * A vector of tuples, indicating the objective change for the
  //    corresponding cluster id if a node is moved to said cluster.
  //  * The objective change of a node moving out of its current cluster
  struct ObjectiveChange {
    std::vector<std::tuple<ClusterId, double>> move_to_change;
    double move_from_change;
  };

  // Moves node i from its current cluster to a new cluster moves[i].
  // If moves[i] == null optional, then the corresponding node will not be
  // moved. A move to the number of nodes in the graph means that a new cluster
  // is created. The size of moves should be equal to num_nodes_.
  // Returns an array where the entry is true if the cluster corresponding to
  // the index was modified, and false if the cluster corresponding to the
  // index was not modified. Nodes may not necessarily move if the best move
  // provided is to stay in their existing cluster.
  std::unique_ptr<bool[]> MoveNodesToCluster(
      const std::vector<absl::optional<ClusterId>>& moves);

  // Returns a tuple of:
  //  * The best cluster to move all of the nodes in moving_nodes to according
  //    to the correlation clustering objective function. An id equal to the
  //    number of nodes in the graph means create a new cluster.
  //  * The change in objective function achieved by that move. May be positive
  //    or negative.
  std::tuple<ClusteringHelper::ClusterId, double> BestMove(
      gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
      const std::vector<gbbs::uintE>& moving_nodes);

  // Returns a tuple of:
  //  * The best cluster to move moving_node to according to the correlation
  //    clustering objective function. An id equal to the number of nodes in the
  //    graph means create a new cluster.
  //  * The change in objective function achieved by that move. May be positive
  //    or negative.
  std::tuple<ClusterId, double> BestMove(
      gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
      InMemoryClusterer::NodeId moving_node);

  // Compute the objective of the current clustering
  double ComputeObjective(
      gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph);

  const std::vector<ClusterId>& ClusterIds() const { return cluster_ids_; }

  const std::vector<ClusterId>& ClusterSizes() const { return cluster_sizes_; }

  const std::vector<double>& ClusterWeights() const { return cluster_weights_; }

  // Returns the weight of the given node, or 1.0 if it has not been set.
  double NodeWeight(InMemoryClusterer::NodeId id) const;

 private:
  std::size_t num_nodes_;
  std::vector<ClusterId> cluster_ids_;
  std::vector<ClusterId> cluster_sizes_;
  ClustererConfig clusterer_config_;
  std::vector<double> node_weights_;
  std::vector<double> cluster_weights_;

  // Initialize cluster_ids_ and cluster_sizes_ given an initial clustering.
  // If clustering is empty, initialize singleton clusters.
  void SetClustering(const InMemoryClusterer::Clustering& clustering);
};

// Given cluster ids and a graph, compress the graph such that the new
// vertices are the cluster ids and the edges are aggregated by sum.
// Self-loops preserve the total weight of the undirected edges in the clusters.
absl::StatusOr<GraphWithWeights> CompressGraph(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& original_graph,
    const std::vector<gbbs::uintE>& cluster_ids, ClusteringHelper* helper);

}  // namespace in_memory
}  // namespace research_graph

#endif  // RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_PARALLEL_CORRELATION_CLUSTERER_INTERNAL_H_
