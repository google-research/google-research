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

#ifndef PARALLEL_CLUSTERING_CLUSTERERS_PARALLEL_CORRELATION_CLUSTERER_H_
#define PARALLEL_CLUSTERING_CLUSTERERS_PARALLEL_CORRELATION_CLUSTERER_H_

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "clusterers/parallel-correlation-clusterer-internal.h"
#include "config.pb.h"  // NOLINT(build/include)
#include "gbbs-graph.h"  // NOLINT(build/include)
#include "in-memory-clusterer.h"  // NOLINT(build/include)
#include "parallel/parallel-graph-utils.h"
#include "status_macros.h"  // NOLINT(build/include)

namespace research_graph {
namespace in_memory {

// A local-search based clusterer optimizing the correlation clustering
// objective. See comment above CorrelationClustererConfig in
// ../config.proto for more. This uses the CorrelationClustererConfig proto.
// Also, note that the input graph is required to be undirected.
class ParallelCorrelationClusterer : public InMemoryClusterer {
 public:
  using ClusterId = gbbs::uintE;

  Graph* MutableGraph() override { return &graph_; }

  absl::StatusOr<Clustering> Cluster(
      const ClustererConfig& config) const override;

  // initial_clustering must include every node in the range
  // [0, MutableGraph().NumNodes()) exactly once.
  absl::Status RefineClusters(const ClustererConfig& clusterer_config,
                              Clustering* initial_clustering) const override;

 private:
  GbbsGraph graph_;
};

}  // namespace in_memory
}  // namespace research_graph

#endif  // PARALLEL_CLUSTERING_CLUSTERERS_PARALLEL_CORRELATION_CLUSTERER_H_
