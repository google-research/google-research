// Copyright 2021 The Google Research Authors.
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

#include "clustering/clusterers/parallel-affinity.h"

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "clustering/clusterers/parallel-affinity-internal.h"
#include "clustering/clusterers/affinity-weight-threshold.h"
#include "clustering/config.pb.h"
#include "clustering/gbbs-graph.h"
#include "clustering/in-memory-clusterer.h"
#include "parallel/parallel-graph-utils.h"
#include "clustering/status_macros.h"

namespace research_graph {
namespace in_memory {

void AddNewClusters(ParallelAffinityClusterer::Clustering new_clusters,
                    ParallelAffinityClusterer::Clustering* clustering) {
  clustering->reserve(clustering->size() + new_clusters.size());
  std::move(new_clusters.begin(), new_clusters.end(),
            std::back_inserter(*clustering));
}


absl::StatusOr<std::vector<ParallelAffinityClusterer::Clustering>>
ParallelAffinityClusterer::HierarchicalCluster(
    const ClustererConfig& config) const {
  const AffinityClustererConfig& affinity_config =
      config.affinity_clusterer_config();

  std::size_t n = graph_.Graph()->n;
  std::vector<gbbs::uintE> cluster_ids(n);
  pbbs::parallel_for(0, n, [&](std::size_t i) { cluster_ids[i] = i; });

  std::vector<ParallelAffinityClusterer::Clustering> result;

  ParallelAffinityClusterer::Clustering clustering;
  std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>
      compressed_graph;
  std::vector<double> node_weights;
  for (int i = 0; i < affinity_config.num_iterations(); ++i) {
    double weight_threshold;
    ASSIGN_OR_RETURN(weight_threshold,
                     AffinityWeightThreshold(affinity_config, i));

    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph =
        (i == 0) ? graph_.Graph() : compressed_graph.get();

    std::vector<gbbs::uintE> compressed_cluster_ids;
    ASSIGN_OR_RETURN(compressed_cluster_ids,
                     NearestNeighborLinkage(*current_graph, weight_threshold));

    cluster_ids = FlattenClustering(cluster_ids, compressed_cluster_ids);

    // TODO(jeshi): Performance can be improved by not finding finished
    // clusters on the last round
    auto new_clusters =
        FindFinishedClusters(*(graph_.Graph()), affinity_config, cluster_ids,
                             compressed_cluster_ids);
    AddNewClusters(std::move(new_clusters), &clustering);

    // Copy current clustering with finished clusters
    result.emplace_back(clustering);

    // TODO(jeshi): Update ComputeClusters to use cluster_ids only, instead of
    // active_vertex
    std::unique_ptr<bool[]> active_vertex(new bool[n]);
    pbbs::parallel_for(0, n, [&](std::size_t i) {
      active_vertex[i] = (cluster_ids[i] != UINT_E_MAX);
    });
    auto current_clusters =
        ComputeClusters(cluster_ids, std::move(active_vertex));
    AddNewClusters(std::move(current_clusters), &(result.back()));

    // Exit if all clusters are finished
    pbbs::sequence<bool> exit_seq = pbbs::sequence<bool>(
        cluster_ids.size(),
        [&](std::size_t i) { return (cluster_ids[i] == UINT_E_MAX); });
    bool to_exit = pbbs::reduce(
        exit_seq,
        pbbs::make_monoid([](bool a, bool b) { return a && b; }, true));
    if (to_exit || i == affinity_config.num_iterations() - 1) break;

    // Compress graph
    GraphWithWeights new_compressed_graph;
    ASSIGN_OR_RETURN(new_compressed_graph,
                     CompressGraph(*current_graph, node_weights,
                                   compressed_cluster_ids, affinity_config));
    compressed_graph.swap(new_compressed_graph.graph);
    if (new_compressed_graph.graph) new_compressed_graph.graph->del();
    node_weights = new_compressed_graph.node_weights;
  }
  if (compressed_graph) compressed_graph->del();

  if (result.empty()) {
    ParallelAffinityClusterer::Clustering trivial_clustering(graph_.Graph()->n);
    pbbs::parallel_for(0, trivial_clustering.size(), [&](NodeId i) {
      trivial_clustering[i] = std::vector<NodeId>{i};
    });
    result.emplace_back(trivial_clustering);
  }

  return result;
}

absl::StatusOr<ParallelAffinityClusterer::Clustering>
ParallelAffinityClusterer::Cluster(const ClustererConfig& config) const {
  std::vector<ParallelAffinityClusterer::Clustering> clustering_hierarchy;
  ASSIGN_OR_RETURN(clustering_hierarchy, HierarchicalCluster(config));

  if (clustering_hierarchy.empty()) {
    ParallelAffinityClusterer::Clustering trivial_clustering(graph_.Graph()->n);
    pbbs::parallel_for(0, trivial_clustering.size(), [&](NodeId i) {
      trivial_clustering[i] = std::vector<NodeId>{i};
    });
    return trivial_clustering;
  } else {
    return clustering_hierarchy.back();
  }
}

}  // namespace in_memory
}  // namespace research_graph
