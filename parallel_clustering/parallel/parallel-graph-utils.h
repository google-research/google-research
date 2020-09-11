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

#ifndef RESEARCH_GRAPH_IN_MEMORY_PARALLEL_PARALLEL_GRAPH_UTILS_H_
#define RESEARCH_GRAPH_IN_MEMORY_PARALLEL_PARALLEL_GRAPH_UTILS_H_

#include <cstdio>

#include "external/gbbs/gbbs/gbbs.h"
#include "external/gbbs/gbbs/graph_io.h"
#include "external/gbbs/gbbs/macros.h"
#include "external/gbbs/pbbslib/seq.h"
#include "external/gbbs/pbbslib/sequence_ops.h"
#include "external/gbbs/pbbslib/utilities.h"
#include "parallel/parallel-sequence-ops.h"

namespace research_graph {

struct OffsetsEdges {
  std::vector<gbbs::uintE> offsets;
  std::unique_ptr<std::tuple<gbbs::uintE, float>[]> edges;
  std::size_t num_edges;
};

// Given get_key, which is nondecreasing, defined for 0, ..., num_keys-1, and
// returns an unsigned integer less than n, return an array of length n + 1
// where array[i] := minimum index k such that get_key(k) >= i.
// Note that array[n] = the total number of keys, num_keys.
std::vector<gbbs::uintE> GetOffsets(
    const std::function<gbbs::uintE(std::size_t)>& get_key,
    gbbs::uintE num_keys, std::size_t n);

// Using parallel sorting, compute inter cluster edges given a set of
// cluster_ids that form the vertices of the new graph. Uses aggregate_func
// to combine multiple edges on the same cluster ids. Returns sorted
// edges and offsets array in edges and offsets respectively.
// The number of compressed vertices should be 1 + the maximum cluster id
// in cluster_ids.
OffsetsEdges ComputeInterClusterEdgesSort(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& original_graph,
    const std::vector<gbbs::uintE>& cluster_ids,
    std::size_t num_compressed_vertices,
    const std::function<float(float, float)>& aggregate_func,
    const std::function<bool(gbbs::uintE, gbbs::uintE)>& is_valid_func);

// Given an array of edges (given by a tuple consisting of the second endpoint
// and a weight if the edges are weighted) and the offsets marking the index
// of the first edge corresponding to each vertex (essentially, CSR format),
// return the corresponding graph in GBBS format.
// Note that the returned graph takes ownership of the edges array.
template <typename WeightType>
std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, WeightType>>
MakeGbbsGraph(
    const std::vector<gbbs::uintE>& offsets, std::size_t num_vertices,
    std::unique_ptr<std::tuple<gbbs::uintE, WeightType>[]> edges_pointer,
    std::size_t num_edges) {
  gbbs::symmetric_vertex<WeightType>* vertices =
      new gbbs::symmetric_vertex<WeightType>[num_vertices];
  auto edges = edges_pointer.release();

  pbbs::parallel_for(0, num_vertices, [&](std::size_t i) {
    gbbs::vertex_data vertex_data{offsets[i], offsets[i + 1] - offsets[i]};
    vertices[i] = gbbs::symmetric_vertex<WeightType>(edges, vertex_data);
  });

  return std::make_unique<
      gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, WeightType>>(
      num_vertices, num_edges, vertices, [=]() {
        delete[] vertices;
        delete[] edges;
      });
}

// Given new cluster ids in compressed_cluster_ids, remap the original
// cluster ids. A cluster id of UINT_E_MAX indicates that the vertex
// has already been placed into a finalized cluster, and this is
// preserved in the remapping.
std::vector<gbbs::uintE> FlattenClustering(
    const std::vector<gbbs::uintE>& cluster_ids,
    const std::vector<gbbs::uintE>& compressed_cluster_ids);

}  // namespace research_graph

#endif  // RESEARCH_GRAPH_IN_MEMORY_PARALLEL_PARALLEL_GRAPH_UTILS_H_
