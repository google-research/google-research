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

#include "gbbs-graph.h"  // NOLINT(build/include)

#include <algorithm>
#include <memory>

#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "clustering/status_macros.h"
#include "external/gbbs/gbbs/macros.h"

namespace research_graph {
namespace in_memory {

void GbbsGraph::EnsureSize(NodeId id) {
  if (nodes_.size() < id) nodes_.resize(id, gbbs::symmetric_vertex<float>());
}

// TODO(jeshi,laxmand): should adjacency_list be a const&?
absl::Status GbbsGraph::Import(AdjacencyList adjacency_list) {
  using GbbsEdge = std::tuple<gbbs::uintE, float>;
  // TODO(jeshi): Parallelize further by only taking the mutex when
  // nodes_ and edges_ need to double in size, instead of every time
  // a node id is increased.
  auto outgoing_edges_size = adjacency_list.outgoing_edges.size();
  auto out_neighbors = absl::make_unique<GbbsEdge[]>(outgoing_edges_size);
  gbbs::parallel_for(0, outgoing_edges_size, [&](size_t i) {
    out_neighbors[i] = std::make_tuple(
        static_cast<gbbs::uintE>(adjacency_list.outgoing_edges[i].first),
        adjacency_list.outgoing_edges[i].second);
  });
  absl::MutexLock lock(&mutex_);
  EnsureSize(adjacency_list.id + 1);
  nodes_[adjacency_list.id].degree = outgoing_edges_size;
  nodes_[adjacency_list.id].neighbors = out_neighbors.get();
  if (edges_.size() <= adjacency_list.id) edges_.resize(adjacency_list.id + 1);
  edges_[adjacency_list.id] = std::move(out_neighbors);

  return absl::OkStatus();
}

absl::Status GbbsGraph::FinishImport() {
  auto degrees = gbbs::sequence<gbbs::uintE>(
      nodes_.size(), [this](size_t i) { return nodes_[i].getOutDegree(); });
  auto num_edges = pbbslib::reduce_add(degrees.slice());

  auto neighbors = gbbs::sequence<gbbs::uintE>(nodes_.size(), [this](size_t i) {
    if (nodes_[i].getOutDegree() == 0) return gbbs::uintE{0};
    auto max_neighbor =
        std::max_element(nodes_[i].getOutNeighbors(),
                         nodes_[i].getOutNeighbors() + nodes_[i].getOutDegree(),
                         [](const std::tuple<gbbs::uintE, float>& u,
                            const std::tuple<gbbs::uintE, float>& v) {
                           return std::get<0>(u) < std::get<0>(v);
                         });
    return std::get<0>(*max_neighbor);
  });
  auto max_node = pbbslib::reduce_max(neighbors.slice());
  EnsureSize(max_node + 1);

  // The GBBS graph takes no ownership of nodes / edges
  auto g = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>(
      nodes_.size(), num_edges, nodes_.data(), []() {});  // noop deletion_fn
  graph_ = absl::make_unique<
      gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>(g);
  return absl::OkStatus();
}

gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* GbbsGraph::Graph()
    const {
  return graph_.get();
}

absl::Status CopyGraph(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& in_graph,
    InMemoryClusterer::Graph* out_graph) {
  for (gbbs::uintE id = 0; id < in_graph.n; id++) {
    InMemoryClusterer::AdjacencyList adjacency_list;
    adjacency_list.id = id;
    const auto& neighbors = in_graph.get_vertex(id).getOutNeighbors();
    adjacency_list.outgoing_edges.reserve(
        in_graph.get_vertex(id).getOutDegree());
    for (size_t j = 0; j < in_graph.get_vertex(id).getOutDegree(); j++) {
      adjacency_list.outgoing_edges.emplace_back(std::get<0>(neighbors[j]),
                                                 std::get<1>(neighbors[j]));
    }
    RETURN_IF_ERROR(out_graph->Import(std::move(adjacency_list)));
  }
  RETURN_IF_ERROR(out_graph->FinishImport());
  return absl::OkStatus();
}

}  // namespace in_memory
}  // namespace research_graph
