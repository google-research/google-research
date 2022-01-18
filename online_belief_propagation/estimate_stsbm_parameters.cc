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

#include "estimate_stsbm_parameters.h"

#include <assert.h>

#include <algorithm>
#include <vector>

#include "graph.h"

StSBMParameters EstimateStSBMParameters(
    const Graph& graph, const std::vector<int> ground_truth_communities) {
  assert(graph.GetVertexNumber() == ground_truth_communities.size());

  // Compute the number of communities.
  assert(!ground_truth_communities.empty());
  const int k = *std::max_element(ground_truth_communities.begin(),
                                  ground_truth_communities.end()) +
                1;

  // Compute the size of each community.
  std::vector<double> community_sizes(ground_truth_communities.size(), 0.0);
  for (const int ground_truth_community : ground_truth_communities) {
    assert(0 <= ground_truth_community && ground_truth_community < k);
    community_sizes[ground_truth_community] += 1.0;
  }

  // Compute:
  //   - maximum number of potential intra-community edges, i.e., number of
  //   pairs of vertices {u, v} belonging to the same
  //     community; and
  //   - Maximum number of potential intra-community edges, i.e., number of
  //   pairs of vertices {u, v} belonging to distinct
  //     communities.
  double max_potential_intra_community_edges = 0.0;
  double max_potential_inter_community_edges = 0.0;
  for (int i = 0; i < k; ++i) {
    max_potential_intra_community_edges +=
        0.5 * (community_sizes[i]) * (community_sizes[i] - 1.0);
    for (int j = i + 1; j < k; ++j) {
      max_potential_inter_community_edges +=
          community_sizes[i] * community_sizes[j];
    }
  }
  assert(max_potential_intra_community_edges > 0.0);
  assert(max_potential_inter_community_edges > 0.0);

  // Compute the numbers of intra-community and inter-community edges.
  double num_intra_community_edges = 0.0;
  double num_inter_community_edges = 0.0;
  for (int u = 0; u < graph.GetVertexNumber(); ++u) {
    for (const int v : graph.GetNeighborhood(u)) {
      assert(u != v);
      // Skip pairs with u > v, to avoid counting edges twice (since each edge
      // appears in both directions).
      if (u > v) continue;
      if (ground_truth_communities.at(u) == ground_truth_communities.at(v)) {
        num_intra_community_edges += 1.0;
      } else {
        num_inter_community_edges += 1.0;
      }
    }
  }

  return {.n = graph.GetVertexNumber(),
          .k = k,
          .a = graph.GetVertexNumber() * num_intra_community_edges /
               max_potential_intra_community_edges,
          .b = graph.GetVertexNumber() * num_inter_community_edges /
               max_potential_inter_community_edges};
}
