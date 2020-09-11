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

#include "clustering/clusterers/parallel-correlation-clusterer.h"

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "clustering/clusterers/parallel-correlation-clusterer-internal.h"
#include "clustering/config.pb.h"
#include "clustering/gbbs-graph.h"
#include "clustering/in-memory-clusterer.h"
#include "parallel/parallel-graph-utils.h"
#include "clustering/status_macros.h"

namespace research_graph {
namespace in_memory {

namespace {

// This struct is necessary to perform an edge map with GBBS over a vertex
// set. Essentially, all neighbors are valid in this edge map, and this
// map does not do anything except allow for neighbors to be aggregated
// into the next frontier.
struct CorrelationClustererEdgeMap {
  inline bool cond(gbbs::uintE d) { return true; }
  inline bool update(const gbbs::uintE& s, const gbbs::uintE& d, float wgh) {
    return true;
  }
  inline bool updateAtomic(const gbbs::uintE& s, const gbbs::uintE& d,
                           float wgh) {
    return true;
  }
};

// Given a vertex subset moved_subset, computes best moves for all vertices
// and performs the moves. Returns a vertex subset consisting of all vertices
// adjacent to modified clusters.
std::unique_ptr<gbbs::vertexSubset, void (*)(gbbs::vertexSubset*)>
BestMovesForVertexSubset(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph,
    std::size_t num_nodes, gbbs::vertexSubset* moved_subset,
    ClusteringHelper* helper, const ClustererConfig& clusterer_config) {
  std::vector<absl::optional<ClusteringHelper::ClusterId>> moves(num_nodes,
                                                                 absl::nullopt);

  // Find best moves per vertex in moved_subset
  gbbs::vertexMap(*moved_subset, [&](std::size_t i) {
    std::tuple<ClusteringHelper::ClusterId, double> best_move =
        helper->BestMove(*current_graph, i);
    // If a singleton cluster wishes to move to another singleton cluster,
    // only move if the id of the moving cluster is lower than the id
    // of the cluster it wishes to move to
    auto move_cluster_id = std::get<0>(best_move);
    auto current_cluster_id = helper->ClusterIds()[i];
    if (move_cluster_id < current_graph->n &&
        helper->ClusterSizes()[move_cluster_id] == 1 &&
        helper->ClusterSizes()[current_cluster_id] == 1 &&
        current_cluster_id >= move_cluster_id) {
      best_move = std::make_tuple(current_cluster_id, 0);
    }
    if (std::get<1>(best_move) > 0) moves[i] = std::get<0>(best_move);
  });

  // Compute modified clusters
  auto moved_clusters = helper->MoveNodesToCluster(moves);

  // Perform cluster moves
  if (clusterer_config.correlation_clusterer_config()
          .clustering_moves_method() ==
      CorrelationClustererConfig::DEFAULT_CLUSTER_MOVES) {
    // Reset moves
    pbbs::parallel_for(0, num_nodes,
                       [&](std::size_t i) { moves[i] = absl::nullopt; });

    // Aggregate clusters
    auto get_clusters = [&](gbbs::uintE i) -> gbbs::uintE { return i; };
    std::vector<std::vector<gbbs::uintE>> curr_clustering =
        parallel::OutputIndicesById<ClusteringHelper::ClusterId, gbbs::uintE>(
            helper->ClusterIds(), get_clusters, helper->ClusterIds().size());

    // Compute best move per cluster
    pbbs::parallel_for(0, curr_clustering.size(), [&](std::size_t i) {
      if (!curr_clustering[i].empty()) {
        std::tuple<ClusteringHelper::ClusterId, double> best_move =
            helper->BestMove(*current_graph, curr_clustering[i]);
        // If a cluster wishes to move to another cluster,
        // only move if the id of the moving cluster is lower than the id
        // of the cluster it wishes to move to
        auto move_cluster_id = std::get<0>(best_move);
        auto current_cluster_id =
            helper->ClusterIds()[curr_clustering[i].front()];
        if (move_cluster_id < current_graph->n &&
            current_cluster_id >= move_cluster_id) {
          best_move = std::make_tuple(current_cluster_id, 0);
        }
        if (std::get<1>(best_move) > 0) {
          for (size_t j = 0; j < curr_clustering[i].size(); j++) {
            moves[curr_clustering[i][j]] = std::get<0>(best_move);
          }
        }
      }
    });

    // Compute modified clusters
    auto additional_moved_clusters = helper->MoveNodesToCluster(moves);
    pbbs::parallel_for(0, num_nodes, [&](std::size_t i) {
      moved_clusters[i] |= additional_moved_clusters[i];
    });
  }

  // Mark vertices adjacent to clusters that have moved; these are
  // the vertices whose best moves must be recomputed.
  auto local_moved_subset =
      std::unique_ptr<gbbs::vertexSubset, void (*)(gbbs::vertexSubset*)>(
          new gbbs::vertexSubset(
              num_nodes, num_nodes,
              gbbs::sequence<bool>(
                  num_nodes,
                  [&](std::size_t i) {
                    return moved_clusters[helper->ClusterIds()[i]];
                  })
                  .to_array()),
          [](gbbs::vertexSubset* subset) {
            subset->del();
            delete subset;
          });
  auto edge_map = CorrelationClustererEdgeMap{};
  auto new_moved_subset =
      gbbs::edgeMap(*current_graph, *(local_moved_subset.get()), edge_map);
  return std::unique_ptr<gbbs::vertexSubset, void (*)(gbbs::vertexSubset*)>(
      new gbbs::vertexSubset(std::move(new_moved_subset)),
      [](gbbs::vertexSubset* subset) {
        subset->del();
        delete subset;
      });
}

}  // namespace

absl::Status ParallelCorrelationClusterer::RefineClusters(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering) const {
  const auto& config = clusterer_config.correlation_clusterer_config();

  std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>
      compressed_graph;

  // Set number of iterations based on clustering method
  int num_iterations = 0;
  int num_inner_iterations = 0;
  switch (config.clustering_moves_method()) {
    case CorrelationClustererConfig::DEFAULT_CLUSTER_MOVES:
      num_iterations = 1;
      num_inner_iterations =
          config.num_iterations() > 0 ? config.num_iterations() : 10;
      break;
    case CorrelationClustererConfig::LOUVAIN:
      num_iterations = config.louvain_config().num_iterations() > 0
                           ? config.louvain_config().num_iterations()
                           : 10;
      num_inner_iterations =
          config.louvain_config().num_inner_iterations() > 0
              ? config.louvain_config().num_inner_iterations()
              : 10;
      break;
    default:
      return absl::UnimplementedError(
          "Correlation clustering moves must be DEFAULT_CLUSTER_MOVES or "
          "LOUVAIN.");
  }

  // Initialize clustering helper
  auto helper = absl::make_unique<ClusteringHelper>(
      graph_.Graph()->n, clusterer_config, *initial_clustering);
  // The max objective is the maximum objective given by the inner iterations
  // of best moves rounds
  double max_objective = helper->ComputeObjective(*(graph_.Graph()));

  std::vector<gbbs::uintE> cluster_ids(graph_.Graph()->n);
  std::vector<gbbs::uintE> local_cluster_ids(graph_.Graph()->n);
  pbbs::parallel_for(0, graph_.Graph()->n, [&](std::size_t i) {
    cluster_ids[i] = i;
    local_cluster_ids[i] = helper->ClusterIds()[i];
  });

  for (int iter = 0; iter < num_iterations; ++iter) {
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph =
        (iter == 0) ? graph_.Graph() : compressed_graph.get();
    const auto num_nodes = current_graph->n;
    bool moved = false;
    bool local_moved = true;
    auto moved_subset =
        std::unique_ptr<gbbs::vertexSubset, void (*)(gbbs::vertexSubset*)>(
            new gbbs::vertexSubset(
                num_nodes, num_nodes,
                gbbs::sequence<bool>(num_nodes, true).to_array()),
            [](gbbs::vertexSubset* subset) {
              subset->del();
              delete subset;
            });

    // Iterate over best moves
    for (int local_iter = 0; local_iter < num_inner_iterations && local_moved;
         ++local_iter) {
      auto new_moved_subset =
          BestMovesForVertexSubset(current_graph, num_nodes, moved_subset.get(),
                                   helper.get(), clusterer_config);
      moved_subset.swap(new_moved_subset);
      local_moved = !moved_subset->isEmpty();

      // Compute new objective given by the local moves in this iteration
      double curr_objective = helper->ComputeObjective(*current_graph);

      // Update maximum objective
      if (curr_objective > max_objective) {
        pbbs::parallel_for(0, num_nodes, [&](std::size_t i) {
          local_cluster_ids[i] = helper->ClusterIds()[i];
        });
        max_objective = curr_objective;
        moved |= local_moved;
      }
    }

    // If no moves can be made at all, exit
    if (!moved) break;

    // Compress cluster ids in initial_helper based on helper
    cluster_ids = FlattenClustering(cluster_ids, local_cluster_ids);

    if (iter == num_iterations - 1) break;

    // TODO(jeshi): May want to compress out size 0 clusters when compressing
    // the graph
    GraphWithWeights new_compressed_graph;
    ASSIGN_OR_RETURN(
        new_compressed_graph,
        CompressGraph(*current_graph, local_cluster_ids, helper.get()));
    compressed_graph.swap(new_compressed_graph.graph);
    if (new_compressed_graph.graph) new_compressed_graph.graph->del();

    helper = absl::make_unique<ClusteringHelper>(
        compressed_graph->n, clusterer_config,
        new_compressed_graph.node_weights, InMemoryClusterer::Clustering{});

    // Create new local clusters
    pbbs::parallel_for(0, compressed_graph->n,
                       [&](std::size_t i) { local_cluster_ids[i] = i; });
  }

  if (compressed_graph) compressed_graph->del();

  auto get_clusters = [&](NodeId i) -> NodeId { return i; };

  *initial_clustering = parallel::OutputIndicesById<ClusterId, NodeId>(
      cluster_ids, get_clusters, cluster_ids.size());

  return absl::OkStatus();
}

absl::StatusOr<InMemoryClusterer::Clustering>
ParallelCorrelationClusterer::Cluster(
    const ClustererConfig& clusterer_config) const {
  InMemoryClusterer::Clustering clustering(graph_.Graph()->n);

  // Create all-singletons initial clustering
  pbbs::parallel_for(0, graph_.Graph()->n, [&](std::size_t i) {
    clustering[i] = {static_cast<int32_t>(i)};
  });

  RETURN_IF_ERROR(RefineClusters(clusterer_config, &clustering));

  return clustering;
}

}  // namespace in_memory
}  // namespace research_graph
