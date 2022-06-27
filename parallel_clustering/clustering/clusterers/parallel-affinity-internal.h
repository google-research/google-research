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

#ifndef RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_PARALLEL_AFFINITY_INTERNAL_H_
#define RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_PARALLEL_AFFINITY_INTERNAL_H_

#include <array>
#include <tuple>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "clustering/config.pb.h"
#include "clustering/in-memory-clusterer.h"
#include "external/gbbs/gbbs/bridge.h"
#include "external/gbbs/gbbs/gbbs.h"
#include "external/gbbs/gbbs/macros.h"
#include "external/gbbs/pbbslib/sample_sort.h"
#include "external/gbbs/pbbslib/seq.h"
#include "external/gbbs/pbbslib/sequence_ops.h"
#include "external/gbbs/pbbslib/utilities.h"
#include "parallel/parallel-sequence-ops.h"
#include "parallel/parallel-graph-utils.h"

namespace research_graph {
namespace in_memory {

// Compute clusters given cluster ids and a boolean array marking vertices
// participating in finished clusters. Note that this array is actually
// a vector of unsigned chars, because elements must at least 1 byte to avoid
// collisions when processing in parallel.
InMemoryClusterer::Clustering ComputeClusters(
    const std::vector<gbbs::uintE>& cluster_ids,
    std::unique_ptr<bool[]> finished_vertex);

// Performs a single round of nearest-neighbor clustering. First, each node
// marks the highest weight incident edge. Then, we compute connected components
// given by the selected edges. For a graph of size n, returns a sequence of
// size n, where 0 <= result[i] < n gives the cluster id of node i. Edges of
// weight smaller than the threshold are ignored. Ties in edge weights are
// broken using edge endpoint ids.
absl::StatusOr<std::vector<gbbs::uintE>> NearestNeighborLinkage(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& G,
    float weight_threshold);

// Compute a compressed graph where vertices are given by cluster ids, and edges
// are aggregated according to affinity_config. A cluster id of UINT_E_MAX
// means that the corresponding vertex has already been clustered into
// a final cluster, by virtue of end conditions given by affinity_config.
// TODO(b/189478197): Switch original_graph to a const reference (which requires
// GBBS to support const get_vertex() calls)
absl::StatusOr<GraphWithWeights> CompressGraph(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& original_graph,
    const std::vector<double>& original_node_weights,
    const std::vector<gbbs::uintE>& cluster_ids,
    const AffinityClustererConfig& affinity_config);

// Determine which clusters, as given by cluster_ids, are "finished", where
// "finished" is defined by AffinityClustererConfig (e.g., a cluster of
// sufficient density or conductance). These clusters are aggregated and
// returned, and the cluster ids and compressed cluster ids for the
// corresponding vertices are updated to be invalid.
InMemoryClusterer::Clustering FindFinishedClusters(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& G,
    const AffinityClustererConfig& affinity_config,
    std::vector<gbbs::uintE>& cluster_ids,
    std::vector<gbbs::uintE>& compressed_cluster_ids);

namespace internal {

struct ClusterStats {
  // Constructor for initialization within parallel_for_bc macro
  ClusterStats(float _density, float _conductance)
      : density(_density), conductance(_conductance) {}
  float density;
  float conductance;
};

std::vector<ClusterStats> ComputeFinishedClusterStats(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& G,
    std::vector<gbbs::uintE>& cluster_ids, gbbs::uintE num_compressed_vertices);

}  // namespace internal

}  // namespace in_memory
}  // namespace research_graph

#endif  // RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_CLUSTERERS_PARALLEL_AFFINITY_INTERNAL_H_
