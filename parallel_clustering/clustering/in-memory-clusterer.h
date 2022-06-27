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

#ifndef RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_IN_MEMORY_CLUSTERER_H_
#define RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_IN_MEMORY_CLUSTERER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "clustering/config.pb.h"

namespace research_graph {
namespace in_memory {

// Interface of an in-memory clustering algorithm. The classes implementing this
// interface maintain a mutable graph, which can be clustered using a given set
// of parameters.
class InMemoryClusterer {
 public:
  // This is a basic interface for building graphs. Note that the interface only
  // specifies how to build a graph, as different clusterers may use different
  // interfaces for accessing it.
  // The node ids are consecutive, 0-based integers. In particular, adding a
  // node of id k to an empty graph creates k+1 nodes 0, ..., k.
  class Graph {
   public:
    using NodeId = int32_t;

    // Represents a weighted node with weighted outgoing edges.
    struct AdjacencyList {
      NodeId id = -1;
      double weight = 1;
      std::vector<std::pair<NodeId, double>> outgoing_edges;
    };

    virtual ~Graph() = default;

    // Adds a weighted node and its weighted out-edges to the graph. Depending
    // on the Graph implementation, the symmetric edges may be added as well,
    // and edge weights may be adjusted for symmetry.
    //
    // Import must be called at most once for each node. If not called for a
    // node, that node defaults to weight 1.
    //
    // IMPLEMENTATIONS MUST ALLOW CONCURRENT CALLS TO Import()!
    virtual absl::Status Import(AdjacencyList adjacency_list) = 0;

    virtual absl::Status FinishImport();
  };

  using NodeId = Graph::NodeId;
  using AdjacencyList = Graph::AdjacencyList;

  // Represents clustering: each element of the vector contains the set of
  // NodeIds in one cluster. We call a clustering non-overlapping if the
  // elements of the clustering are nonempty vectors that together contain each
  // NodeId exactly once.
  using Clustering = std::vector<std::vector<NodeId>>;

  virtual ~InMemoryClusterer() {}

  // Accessor to the maintained graph. Use it to build the graph.
  virtual Graph* MutableGraph() = 0;

  // Clusters the currently maintained graph using the given set of parameters.
  // Returns a clustering, or an error if the algorithm failed to cluster the
  // given graph.
  // Note that the same clustering may have multiple representations, and the
  // function may return any of them.
  virtual absl::StatusOr<Clustering> Cluster(
      const ClustererConfig& config) const = 0;

  // Same as above, except that it returns a sequence of clusterings. The last
  // element of the sequence is the final clustering. This is primarily used for
  // hierarchical clusterings, but callers should NOT assume that there is a
  // strict hierarchy structure (i.e. that clusters in clustering i are obtained
  // by merging clusters from clustering i-1). The default implementation
  // returns a single-element vector with the result of Cluster().
  virtual absl::StatusOr<std::vector<Clustering>> HierarchicalCluster(
      const ClustererConfig& config) const;

  // Refines a list of clusters and redirects the given pointer to new clusters.
  // This function is useful for methods that can refine / operate on an
  // existing clustering. It does not take ownership of clustering. The default
  // implementation does nothing and returns OkStatus.
  virtual absl::Status RefineClusters(const ClustererConfig& config,
                                      Clustering* clustering) const {
    return absl::OkStatus();
  }

  // Provides a pointer to a vector that contains string ids corresponding to
  // the NodeIds. If set, the ids from the provided map are used in the log and
  // error messages. The vector must live during all method calls of this
  // object. This call does *not* take ownership of the pointee. Using this
  // function is not required. If this function is never called, the ids are
  // converted to strings using absl::StrCat.
  void set_node_id_map(const std::vector<std::string>* node_id_map) {
    node_id_map_ = node_id_map;
  }

 protected:
  // Returns the string id corresponding to a given NodeId. If set_node_id_map
  // was called, uses the map to get the ids. Otherwise, returns the string
  // representation of the id.
  std::string StringId(NodeId id) const;

 private:
  // NodeId map set by set_node_id_map(). May be left to nullptr even after
  // initialization.
  const std::vector<std::string>* node_id_map_ = nullptr;
};

}  // namespace in_memory
}  // namespace research_graph

#endif  // RESEARCH_GRAPH_IN_MEMORY_CLUSTERING_IN_MEMORY_CLUSTERER_H_
