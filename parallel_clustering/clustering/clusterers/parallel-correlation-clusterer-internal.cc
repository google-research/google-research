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

#include "clustering/clusterers/parallel-correlation-clusterer-internal.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "clustering/clusterers/correlation-clusterer-util.h"
#include "clustering/config.pb.h"
#include "external/gbbs/gbbs/bridge.h"
#include "external/gbbs/gbbs/gbbs.h"
#include "external/gbbs/gbbs/macros.h"
#include "external/gbbs/pbbslib/random_shuffle.h"
#include "external/gbbs/pbbslib/sample_sort.h"
#include "external/gbbs/pbbslib/seq.h"
#include "external/gbbs/pbbslib/sequence_ops.h"
#include "external/gbbs/pbbslib/utilities.h"
#include "clustering/in-memory-clusterer.h"
#include "parallel/parallel-graph-utils.h"
#include "parallel/parallel-sequence-ops.h"

namespace research_graph {
namespace in_memory {

using NodeId = InMemoryClusterer::NodeId;
using ClusterId = ClusteringHelper::ClusterId;

void ClusteringHelper::SetClustering(
    const InMemoryClusterer::Clustering& clustering) {
  if (clustering.empty()) {
    pbbs::parallel_for(0, num_nodes_, [&](std::size_t i) {
      cluster_sizes_[i] = 1;
      cluster_ids_[i] = i;
      cluster_weights_[i] = node_weights_[i];
    });
  } else {
    pbbs::parallel_for(0, clustering.size(), [&](std::size_t i) {
      cluster_sizes_[i] = clustering[i].size();
      for (auto j : clustering[i]) {
        cluster_ids_[j] = i;
        cluster_weights_[i] += node_weights_[j];
      }
    });
  }
}

double ClusteringHelper::NodeWeight(NodeId id) const {
  return id < node_weights_.size() ? node_weights_[id] : 1.0;
}

double ClusteringHelper::ComputeObjective(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph) {
  const auto& config = clusterer_config_.correlation_clusterer_config();
  std::vector<double> shifted_edge_weight(graph.n);

  // Compute cluster statistics contributions of each vertex
  pbbs::parallel_for(0, graph.n, [&](std::size_t i) {
    gbbs::uintE cluster_id_i = cluster_ids_[i];
    auto add_m = pbbslib::addm<double>();

    auto intra_cluster_sum_map_f = [&](gbbs::uintE u, gbbs::uintE v,
                                       float weight) -> double {
      // This assumes that the graph is undirected, and self-loops are counted
      // as half of the weight.
      if (cluster_id_i == cluster_ids_[v])
        return (weight - config.edge_weight_offset()) / 2;
      return 0;
    };
    shifted_edge_weight[i] = graph.get_vertex(i).reduceOutNgh<double>(
        i, intra_cluster_sum_map_f, add_m);
  });
  double objective =
      parallel::ReduceAdd(absl::Span<const double>(shifted_edge_weight));

  auto resolution_seq = pbbs::delayed_seq<double>(graph.n, [&](std::size_t i) {
    auto cluster_weight = cluster_weights_[cluster_ids_[i]];
    return node_weights_[i] * (cluster_weight - node_weights_[i]);
  });
  objective -= config.resolution() * pbbslib::reduce_add(resolution_seq) / 2;

  return objective;
}

std::unique_ptr<bool[]> ClusteringHelper::MoveNodesToCluster(
    const std::vector<absl::optional<ClusterId>>& moves) {
  auto modified_cluster = absl::make_unique<bool[]>(num_nodes_);
  pbbs::parallel_for(0, num_nodes_,
                     [&](std::size_t i) { modified_cluster[i] = false; });

  // We must update cluster_sizes_ and assign new cluster ids to vertices
  // that want to form a new cluster
  // Obtain all nodes that are moving clusters
  auto get_moving_nodes = [&](size_t i) { return i; };
  auto moving_nodes = pbbs::filter(
      pbbs::delayed_seq<gbbs::uintE>(num_nodes_, get_moving_nodes),
      [&](gbbs::uintE node) -> bool {
        return moves[node].has_value() && moves[node] != cluster_ids_[node];
      },
      pbbs::no_flag);

  if (moving_nodes.empty()) return modified_cluster;

  // Sort moving nodes by original cluster id
  auto sorted_moving_nodes = pbbs::sample_sort(
      moving_nodes,
      [&](gbbs::uintE a, gbbs::uintE b) {
        return cluster_ids_[a] < cluster_ids_[b];
      },
      true);

  // The number of nodes moving out of clusters is given by the boundaries
  // where nodes differ by cluster id
  std::vector<gbbs::uintE> mark_moving_nodes =
      parallel::GetBoundaryIndices<gbbs::uintE>(
          sorted_moving_nodes.size(), [&](std::size_t i, std::size_t j) {
            return cluster_ids_[sorted_moving_nodes[i]] ==
                   cluster_ids_[sorted_moving_nodes[j]];
          });
  std::size_t num_mark_moving_nodes = mark_moving_nodes.size() - 1;

  // Subtract these boundary sizes from cluster_sizes_ in parallel
  pbbs::parallel_for(0, num_mark_moving_nodes, [&](std::size_t i) {
    gbbs::uintE start_id_index = mark_moving_nodes[i];
    gbbs::uintE end_id_index = mark_moving_nodes[i + 1];
    auto prev_id = cluster_ids_[sorted_moving_nodes[start_id_index]];
    cluster_sizes_[prev_id] -= (end_id_index - start_id_index);
    modified_cluster[prev_id] = true;
    for (std::size_t j = start_id_index; j < end_id_index; j++) {
      cluster_weights_[prev_id] -= node_weights_[sorted_moving_nodes[j]];
    }
  });

  // Re-sort moving nodes by new cluster id
  auto resorted_moving_nodes = pbbs::sample_sort(
      moving_nodes,
      [&](gbbs::uintE a, gbbs::uintE b) { return moves[a] < moves[b]; }, true);

  // The number of nodes moving into clusters is given by the boundaries
  // where nodes differ by cluster id
  std::vector<gbbs::uintE> remark_moving_nodes =
      parallel::GetBoundaryIndices<gbbs::uintE>(
          resorted_moving_nodes.size(),
          [&resorted_moving_nodes, &moves](std::size_t i, std::size_t j) {
            return moves[resorted_moving_nodes[i]] ==
                   moves[resorted_moving_nodes[j]];
          });
  std::size_t num_remark_moving_nodes = remark_moving_nodes.size() - 1;

  // Add these boundary sizes to cluster_sizes_ in parallel, excepting
  // those vertices that are forming new clusters
  // Also, excepting those vertices that are forming new clusters, update
  // cluster_ids_
  pbbs::parallel_for(0, num_remark_moving_nodes, [&](std::size_t i) {
    gbbs::uintE start_id_index = remark_moving_nodes[i];
    gbbs::uintE end_id_index = remark_moving_nodes[i + 1];
    auto move_id = moves[resorted_moving_nodes[start_id_index]].value();
    if (move_id != num_nodes_) {
      cluster_sizes_[move_id] += (end_id_index - start_id_index);
      modified_cluster[move_id] = true;
      for (std::size_t j = start_id_index; j < end_id_index; j++) {
        cluster_ids_[resorted_moving_nodes[j]] = move_id;
        cluster_weights_[move_id] += node_weights_[resorted_moving_nodes[j]];
      }
    }
  });

  // If there are vertices forming new clusters
  if (moves[resorted_moving_nodes[moving_nodes.size() - 1]].value() ==
      num_nodes_) {
    // Filter out cluster ids of empty clusters, so that these ids can be
    // reused for vertices forming new clusters. This is an optimization
    // so that cluster ids do not grow arbitrarily large, when assigning
    // new cluster ids.
    auto get_zero_clusters = [&](std::size_t i) { return i; };
    auto seq_zero_clusters =
        pbbs::delayed_seq<gbbs::uintE>(num_nodes_, get_zero_clusters);
    auto zero_clusters = pbbs::filter(
        seq_zero_clusters,
        [&](gbbs::uintE id) -> bool { return cluster_sizes_[id] == 0; },
        pbbs::no_flag);

    // Indexing into these cluster ids gives the new cluster ids for new
    // clusters; update cluster_ids_ and cluster_sizes_ appropriately
    gbbs::uintE start_id_index =
        remark_moving_nodes[num_remark_moving_nodes - 1];
    gbbs::uintE end_id_index = remark_moving_nodes[num_remark_moving_nodes];
    pbbs::parallel_for(start_id_index, end_id_index, [&](std::size_t i) {
      auto cluster_id = zero_clusters[i - start_id_index];
      cluster_ids_[resorted_moving_nodes[i]] = cluster_id;
      cluster_sizes_[cluster_id] = 1;
      modified_cluster[cluster_id] = true;
      cluster_weights_[cluster_id] = node_weights_[resorted_moving_nodes[i]];
    });
  }

  return modified_cluster;
}

std::tuple<ClusteringHelper::ClusterId, double> ClusteringHelper::BestMove(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
    NodeId moving_node) {
  const auto& config = clusterer_config_.correlation_clusterer_config();
  const double offset = config.edge_weight_offset();

  // Weight of nodes in each cluster that are moving.
  absl::flat_hash_map<ClusterId, double> cluster_moving_weights;
  // Class 2 edges where the endpoints are currently in different clusters.
  EdgeSum class_2_currently_separate;
  // Class 1 edges where the endpoints are currently in the same cluster.
  EdgeSum class_1_currently_together;
  // Class 1 edges, grouped by the cluster that the non-moving node is in.
  absl::flat_hash_map<ClusterId, EdgeSum> class_1_together_after;

  double moving_nodes_weight = 0;
  const ClusterId node_cluster = cluster_ids_[moving_node];
  cluster_moving_weights[node_cluster] += node_weights_[moving_node];
  moving_nodes_weight += node_weights_[moving_node];
  auto map_moving_node_neighbors = [&](gbbs::uintE u, gbbs::uintE neighbor,
                                       double weight) {
    weight -= offset;
    const ClusterId neighbor_cluster = cluster_ids_[neighbor];
    if (moving_node == neighbor) {
      // Class 2 edge.
      if (node_cluster != neighbor_cluster) {
        class_2_currently_separate.Add(weight);
      }
    } else {
      // Class 1 edge.
      if (node_cluster == neighbor_cluster) {
        class_1_currently_together.Add(weight);
      }
      class_1_together_after[neighbor_cluster].Add(weight);
    }
  };
  graph.get_vertex(moving_node)
      .mapOutNgh(moving_node, map_moving_node_neighbors, false);
  class_2_currently_separate.RemoveDoubleCounting();
  // Now cluster_moving_weights is correct and class_2_currently_separate,
  // class_1_currently_together, and class_1_by_cluster are ready to call
  // NetWeight().

  std::function<double(ClusterId)> get_cluster_weight = [&](ClusterId cluster) {
    return cluster_weights_[cluster];
  };
  auto best_move =
      BestMoveFromStats(config, get_cluster_weight, moving_nodes_weight,
                        cluster_moving_weights, class_2_currently_separate,
                        class_1_currently_together, class_1_together_after);

  auto move_id =
      best_move.first.has_value() ? best_move.first.value() : graph.n;
  std::tuple<ClusterId, double> best_move_tuple =
      std::make_tuple(move_id, best_move.second);

  return best_move_tuple;
}

std::tuple<ClusteringHelper::ClusterId, double> ClusteringHelper::BestMove(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
    const std::vector<gbbs::uintE>& moving_nodes) {
  const auto& config = clusterer_config_.correlation_clusterer_config();
  const double offset = config.edge_weight_offset();

  std::vector<bool> flat_moving_nodes(graph.n, false);
  for (size_t i = 0; i < moving_nodes.size(); i++) {
    flat_moving_nodes[moving_nodes[i]] = true;
  }

  // Weight of nodes in each cluster that are moving.
  absl::flat_hash_map<ClusterId, double> cluster_moving_weights;
  // Class 2 edges where the endpoints are currently in different clusters.
  EdgeSum class_2_currently_separate;
  // Class 1 edges where the endpoints are currently in the same cluster.
  EdgeSum class_1_currently_together;
  // Class 1 edges, grouped by the cluster that the non-moving node is in.
  absl::flat_hash_map<ClusterId, EdgeSum> class_1_together_after;

  double moving_nodes_weight = 0;
  for (const auto& node : moving_nodes) {
    const ClusterId node_cluster = cluster_ids_[node];
    cluster_moving_weights[node_cluster] += node_weights_[node];
    moving_nodes_weight += node_weights_[node];
    auto map_moving_node_neighbors = [&](gbbs::uintE u, gbbs::uintE neighbor,
                                         float weight) {
      weight -= offset;
      const ClusterId neighbor_cluster = cluster_ids_[neighbor];
      if (flat_moving_nodes[neighbor]) {
        // Class 2 edge.
        if (node_cluster != neighbor_cluster) {
          class_2_currently_separate.Add(weight);
        }
      } else {
        // Class 1 edge.
        if (node_cluster == neighbor_cluster) {
          class_1_currently_together.Add(weight);
        }
        class_1_together_after[neighbor_cluster].Add(weight);
      }
    };
    graph.get_vertex(node).mapOutNgh(node, map_moving_node_neighbors, false);
  }
  class_2_currently_separate.RemoveDoubleCounting();
  // Now cluster_moving_weights is correct and class_2_currently_separate,
  // class_1_currently_together, and class_1_by_cluster are ready to call
  // NetWeight().

  std::function<double(ClusterId)> get_cluster_weight = [&](ClusterId cluster) {
    return cluster_weights_[cluster];
  };
  auto best_move =
      BestMoveFromStats(config, get_cluster_weight, moving_nodes_weight,
                        cluster_moving_weights, class_2_currently_separate,
                        class_1_currently_together, class_1_together_after);

  auto move_id =
      best_move.first.has_value() ? best_move.first.value() : graph.n;
  std::tuple<ClusterId, double> best_move_tuple =
      std::make_tuple(move_id, best_move.second);

  return best_move_tuple;
}

absl::StatusOr<GraphWithWeights> CompressGraph(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& original_graph,
    const std::vector<gbbs::uintE>& cluster_ids, ClusteringHelper* helper) {
  // Obtain the number of vertices in the new graph
  auto get_cluster_ids = [&](size_t i) { return cluster_ids[i]; };
  auto seq_cluster_ids =
      pbbs::delayed_seq<gbbs::uintE>(cluster_ids.size(), get_cluster_ids);
  gbbs::uintE num_compressed_vertices =
      1 + pbbslib::reduce_max(seq_cluster_ids);

  // Compute new inter cluster edges using sorting, allowing self-loops
  auto edge_aggregation_func = [](double w1, double w2) { return w1 + w2; };
  auto is_valid_func = [](ClusteringHelper::ClusterId a,
                          ClusteringHelper::ClusterId b) { return true; };

  OffsetsEdges offsets_edges = ComputeInterClusterEdgesSort(
      original_graph, cluster_ids, num_compressed_vertices,
      edge_aggregation_func, is_valid_func);
  std::vector<gbbs::uintE> offsets = offsets_edges.offsets;
  std::size_t num_edges = offsets_edges.num_edges;
  std::unique_ptr<std::tuple<gbbs::uintE, float>[]> edges =
      std::move(offsets_edges.edges);

  // Obtain cluster ids and node weights of all vertices
  std::vector<std::tuple<ClusterId, double>> node_weights(original_graph.n);
  pbbs::parallel_for(0, original_graph.n, [&](std::size_t i) {
    node_weights[i] = std::make_tuple(cluster_ids[i], helper->NodeWeight(i));
  });

  // Initialize new node weights
  std::vector<double> new_node_weights(num_compressed_vertices, 0);

  // Sort weights of neighbors by cluster id
  auto node_weights_sort = research_graph::parallel::ParallelSampleSort<
      std::tuple<ClusterId, double>>(
      absl::Span<std::tuple<ClusterId, double>>(node_weights.data(),
                                                node_weights.size()),
      [&](std::tuple<ClusterId, double> a, std::tuple<ClusterId, double> b) {
        return std::get<0>(a) < std::get<0>(b);
      });

  // Obtain the boundary indices where cluster ids differ
  std::vector<gbbs::uintE> mark_node_weights =
      parallel::GetBoundaryIndices<gbbs::uintE>(
          node_weights_sort.size(),
          [&node_weights_sort](std::size_t i, std::size_t j) {
            return std::get<0>(node_weights_sort[i]) ==
                   std::get<0>(node_weights_sort[j]);
          });
  std::size_t num_mark_node_weights = mark_node_weights.size() - 1;

  // Reset helper to singleton clusters, with appropriate node weights
  pbbs::parallel_for(0, num_mark_node_weights, [&](std::size_t i) {
    gbbs::uintE start_id_index = mark_node_weights[i];
    gbbs::uintE end_id_index = mark_node_weights[i + 1];
    auto node_weight =
        research_graph::parallel::Reduce<std::tuple<ClusterId, double>>(
            absl::Span<const std::tuple<ClusterId, double>>(
                node_weights_sort.begin() + start_id_index,
                end_id_index - start_id_index),
            [&](std::tuple<ClusterId, double> a,
                std::tuple<ClusterId, double> b) {
              return std::make_tuple(std::get<0>(a),
                                     std::get<1>(a) + std::get<1>(b));
            },
            std::make_tuple(std::get<0>(node_weights[start_id_index]),
                            double{0}));
    new_node_weights[std::get<0>(node_weight)] = std::get<1>(node_weight);
  });

  return GraphWithWeights(MakeGbbsGraph<float>(offsets, num_compressed_vertices,
                                               std::move(edges), num_edges),
                          new_node_weights);
}

}  // namespace in_memory
}  // namespace research_graph
