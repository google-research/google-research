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

#include "sbm_graph.h"

#include <assert.h>

#include <random>
#include <set>
#include <tuple>

SBMGraph::SBMGraph(int64_t n, int k, double a, double b, int seed) {
  // Check that input parameters satisfy all requirements.
  assert(n >= 0 && k >= 0 && a >= 0 && b >= 0);
  assert(a >= b);

  std::default_random_engine generator(seed);
  std::uniform_int_distribution<int> random_label(0, k - 1);
  std::uniform_int_distribution<int> random_vertex(0, n - 1);

  // Insert vertices and assign random community labels.
  for (int i = 0; i < n; i++) {
    InsertVertex();
    ground_truth_communities_.push_back(random_label(generator));
  }

  // The goal is to sample each of the n*(n-1)/2 potential edges independently.
  // If the edge is within a community we call it an 'inside edge' and sample it
  // with probability a/n. If the edge is between communities  we call it a
  // 'cross edge' and sample it with probability b/n.
  //
  // For efficiency, instead of sampling each edge individually, we sample the
  // total number of edges from the appropriate binomial distribution, then
  // sample that many edge uniformly at random WITHOUT REPETITION.
  //
  // Since we don't know ahead of time which edges are inside edges and which
  // are cross edges we perform both sampling on the entire collection n*(n-1)/2
  // potential edges and narrow it down later. This results in the sample
  // distribution as sampling each edge independently with the appropriate
  // probability.
  //
  // To enforce that the edge sampling is done without repetition, we keep the
  // candidate edges in a std::set and continue sampling until we reach the
  // desired number of candidates.
  //
  // Sample each potential edge independently with probability a/n.
  std::binomial_distribution<int64_t> random_inside_edge_number(
      (n * (n - 1)) / 2, a / n);
  int64_t inside_edge_number = random_inside_edge_number(generator);
  std::set<std::pair<int, int>> inside_edge_candidates;
  while (inside_edge_candidates.size() < inside_edge_number) {
    // Sample a random potential edge.
    int u = random_vertex(generator);
    int v = random_vertex(generator);
    if (u < v) {
      inside_edge_candidates.emplace(u, v);
    }
    if (v < u) {
      inside_edge_candidates.emplace(v, u);
    }
    // If u = v try again.
  }
  // Narrow down the candidates to those which are inside edges.
  for (const std::pair<int, int>& candidate : inside_edge_candidates) {
    int u = candidate.first;
    int v = candidate.second;
    if (ground_truth_communities_[u] == ground_truth_communities_[v]) {
      InsertEdge(u, v);
    }
  }

  // Sample each potential edge independently with probability b/n.
  std::binomial_distribution<int64_t> random_cross_edge_number(n * (n - 1) / 2,
                                                               b / n);
  int64_t cross_edge_number = random_cross_edge_number(generator);
  std::set<std::pair<int, int>> cross_edge_candidates;
  while (cross_edge_candidates.size() < cross_edge_number) {
    // Sample a random potential edge.
    int u = random_vertex(generator);
    int v = random_vertex(generator);
    if (u < v) {
      cross_edge_candidates.emplace(u, v);
    }
    if (v < u) {
      cross_edge_candidates.emplace(v, u);
    }
    // If u = v try again.
  }
  // Narrow down the candidates to those which are inside edges.
  for (const std::pair<int, int>& candidate : cross_edge_candidates) {
    int u = candidate.first;
    int v = candidate.second;
    if (ground_truth_communities_[u] != ground_truth_communities_[v]) {
      InsertEdge(u, v);
    }
  }
}

const std::vector<int>& SBMGraph::GetGroundTruthCommunities() const {
  return ground_truth_communities_;
}
