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

absl::StatusOr<ParallelAffinityClusterer::Clustering>
ParallelAffinityClusterer::Cluster(const ClustererConfig& config) const {
  const AffinityClustererConfig& affinity_config =
      config.affinity_clusterer_config();
  std::size_t n = graph_.Graph()->n;
  std::vector<gbbs::uintE> cluster_ids(n);
  pbbs::parallel_for(0, n, [&](std::size_t i) { cluster_ids[i] = i; });
  ParallelAffinityClusterer::Clustering clustering;
  std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>
      compressed_graph;
  for (int i = 0; i < affinity_config.num_iterations(); ++i) {
    std::vector<gbbs::uintE> compressed_cluster_ids;
    if (i > 0) {
      std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>
          new_compressed_graph;
      ASSIGN_OR_RETURN(
          new_compressed_graph,
          CompressGraph(*(graph_.Graph()), cluster_ids, affinity_config));
      compressed_graph.swap(new_compressed_graph);
      if (new_compressed_graph) new_compressed_graph->del();
      ASSIGN_OR_RETURN(
          compressed_cluster_ids,
          NearestNeighborLinkage(*(compressed_graph.get()),
                                 affinity_config.weight_threshold()));
    } else {
      ASSIGN_OR_RETURN(
          compressed_cluster_ids,
          NearestNeighborLinkage(*(graph_.Graph()),
                                 affinity_config.weight_threshold()));
    }
    cluster_ids = FlattenClustering(cluster_ids, compressed_cluster_ids);

    // TODO(jeshi): Performance can be improved by not finding finished
    // clusters on the last round
    auto new_clusters =
        FindFinishedClusters(*(graph_.Graph()), affinity_config, cluster_ids);
    AddNewClusters(std::move(new_clusters), &clustering);

    // Exit if all clusters are finished
    pbbs::sequence<bool> exit_seq = pbbs::sequence<bool>(
        cluster_ids.size(),
        [&](std::size_t i) { return (cluster_ids[i] == UINT_E_MAX); });
    bool to_exit = pbbs::reduce(
        exit_seq,
        pbbs::make_monoid([](bool a, bool b) { return a && b; }, true));
    if (to_exit) break;
  }
  if (compressed_graph) compressed_graph->del();

  std::unique_ptr<bool[]> finished_vertex(new bool[n]);
  pbbs::parallel_for(0, n, [&](std::size_t i) {
    finished_vertex[i] = (cluster_ids[i] != UINT_E_MAX);
  });
  auto new_clusters = ComputeClusters(cluster_ids, std::move(finished_vertex));
  AddNewClusters(std::move(new_clusters), &clustering);

  return clustering;
}

}  // namespace in_memory
}  // namespace research_graph
