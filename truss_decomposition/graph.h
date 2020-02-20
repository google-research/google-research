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

#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <chrono>  // NOLINT
#include <type_traits>
#include <vector>

#include "common.h"          // NOLINT
#include "mmapped_vector.h"  // NOLINT
#include "pkxsort.h"         // NOLINT

enum class GraphFormat { kNDE, kBinary, kBinaryStreamed };
enum class GraphPermutation { kNone, kDegreeOrder, kDegeneracyOrder };

GraphFormat ParseGraphFormat(const std::string &fmt) {
  if (fmt == "nde") return GraphFormat::kNDE;
  if (fmt == "bin") return GraphFormat::kBinary;
  if (fmt == "sbin") return GraphFormat::kBinaryStreamed;
  fprintf(stderr, "Invalid graph format.\n");
  abort();
}

template <typename node_t, typename edge_t>
struct GraphT {
  static_assert(std::is_unsigned<node_t>::value,
                "node_t must be an unsigned type");
  static_assert(std::is_unsigned<edge_t>::value,
                "edge_t must be an unsigned type");

  // Graph fingerprints encode the size of edge and node types, which are
  // integral types of up to 8 bytes each.
  static constexpr size_t kFingerprint = (sizeof(edge_t) << 4) | sizeof(node_t);
  static constexpr size_t kFingerprintStreamed = kFingerprint | 1;

  mmapped_vector<edge_t> adj_start;
  mmapped_vector<node_t> adj;

  size_t N() const { return adj_start.size() - 1; }

  // Reads a graph from `filename` in the format specified by `format`. If
  // `permutation` is not `kNone`, the graph is permuted accordingly. If
  // `forward_only` is true, only edges {a, b} with a < b will be returned.
  // Otherwise, both pairs {a, b} and {b, a} will be present in the graph. Note
  // that `kDegeneracyOrder` is only valid if `forward_only == false`.
  void Read(
      const std::string &filename, GraphFormat format,
      GraphPermutation permutation, bool forward_only,
      std::chrono::high_resolution_clock::time_point *reading_done = nullptr,
      std::chrono::high_resolution_clock::time_point *permutation_computed =
          nullptr) {
    FILE *f = filename.empty() ? stdin : fopen(filename.c_str(), "r");

    mmapped_vector<std::pair<node_t, node_t>> adj_pairs;

    // Temporarily deallocate adj.
    std::string backing_file = adj.BackingFile();
    adj.clear();

    // Read the graph.
    size_t N;
    switch (format) {
      case GraphFormat::kNDE:
        N = ReadNDEGraph(f, forward_only, backing_file, &adj_pairs);
        break;
      case GraphFormat::kBinary:
        N = ReadBinaryGraph(f, forward_only, backing_file, &adj_pairs);
        break;
      case GraphFormat::kBinaryStreamed:
        N = ReadBinaryStreamedGraph(f, forward_only, backing_file, &adj_pairs);
        break;
      default:
        fprintf(stderr, "Invalid graph format enum.\n");
        abort();
    }
    if (!filename.empty()) fclose(f);

    // Clean up duplicate edges.
    // TODO: external memory version
    fprintf(stderr, "Sorting...\n");
    kx::radix_sort(adj_pairs.data(), adj_pairs.data() + adj_pairs.size(),
                   EdgeRadixTraits());
    fprintf(stderr, "Cleaning...\n");
    adj_pairs.resize(
        std::unique(adj_pairs.data(), adj_pairs.data() + adj_pairs.size()) -
        adj_pairs.data());
    fprintf(stderr, "Reading done\n");

    if (reading_done) *reading_done = std::chrono::high_resolution_clock::now();

    // Compute permutation and permute the graph.
    if (permutation != GraphPermutation::kNone) {
      CHECK(!forward_only || permutation == GraphPermutation::kDegreeOrder);
      std::vector<node_t> perm = permutation == GraphPermutation::kDegreeOrder
                                     ? ComputeDegreeOrder(N, &adj_pairs)
                                     : ComputeDegeneracyOrder(N, &adj_pairs);
      fprintf(stderr, "Permutation computed\n");
      if (permutation_computed) {
        *permutation_computed = std::chrono::high_resolution_clock::now();
      }

      std::vector<node_t> reverse_permutation(N);
#pragma omp parallel for
      for (node_t i = 0; i < N; i++) {
        reverse_permutation[perm[i]] = i;
      }
      perm.clear();
#pragma omp parallel for
      for (size_t i = 0; i < adj.size(); i++) {
        std::pair<node_t, node_t> &e = adj_pairs[i];
        e.first = reverse_permutation[e.first];
        e.second = reverse_permutation[e.second];
        if (forward_only && e.first > e.second) std::swap(e.first, e.second);
      }

      fprintf(stderr, "Sorting again...\n");
      kx::radix_sort(adj_pairs.begin(), adj_pairs.end(), EdgeRadixTraits());
      fprintf(stderr, "Permuting done\n");
    }

    // Compute degrees, final adjacency lists and their start position.
    adj_start.resize(N + 1);
    std::fill(adj_start.begin(), adj_start.end(), 0);
    for (size_t i = 0; i < adj_pairs.size(); i++) {
      adj_start[adj_pairs[i].first + 1]++;
    }
    for (size_t i = 0; i < N; i++) {
      adj_start[i + 1] += adj_start[i];
    }

    adj.reinterpret(std::move(adj_pairs));
    // TODO: parallel
    for (edge_t i = 0; 2 * i + 1 < adj.size(); i += 1) {
      adj[i] = adj[2 * i + 1];
    }
    adj.resize(adj.size() / 2);
    adj.shrink();
  }

 private:
  struct EdgeRadixTraits {
    static const int nBytes = 2 * sizeof(node_t);
    size_t Value(const std::pair<node_t, node_t> &x) {
      return ((size_t)x.first << 32) | x.second;
    }
    int kth_byte(const std::pair<node_t, node_t> &x, int k) {
      return (Value(x) >> (8 * k)) & 0xff;
    }
    bool compare(const std::pair<node_t, node_t> &x,
                 const std::pair<node_t, node_t> &y) {
      return Value(x) < Value(y);
    }
  };

  size_t ReadNDEGraph(FILE *f, bool forward_only,
                      const std::string &backing_file,
                      mmapped_vector<std::pair<node_t, node_t>> *adj_pairs) {
    // Number of nodes (first line).
    size_t N = ReadBase10Fast<node_t>(f);
    // Degrees (N lines).
    node_t a, b;
    size_t expected_edges = 0;
    for (node_t i = 0; i < N; i++) {
      a = ReadBase10Fast<node_t>(f);
      b = ReadBase10Fast<node_t>(f);
      expected_edges += b;
    }
    adj_pairs->init(backing_file, expected_edges,
                    /*reserve_only = */ true);
    // Edges (all other lines).
    while (true) {
      a = ReadBase10Fast<node_t>(f);
      b = ReadBase10Fast<node_t>(f);
      if (a == (node_t)EOF || b == (node_t)EOF) break;
      if (a == b) continue;
      if (forward_only && b < a) std::swap(a, b);
      adj_pairs->push_back({a, b});
      if (!forward_only) adj_pairs->push_back({b, a});
    }
    return N;
  }

  size_t ReadBinaryGraph(FILE *f, bool forward_only,
                         const std::string &backing_file,
                         mmapped_vector<std::pair<node_t, node_t>> *adj_pairs) {
    // Fingerprint.
    unsigned long long_t fingerprint = ReadBinaryOrDie<unsigned long long_t>(f);
    CHECK(fingerprint == kFingerprint);
    // Number of nodes.
    size_t N = ReadBinaryOrDie<node_t>(f);
    // Offsets of each adjacency list.
    adj_start.resize(N + 1);
    ReadBinaryOrDie(f, adj_start.data(), N + 1);
    std::vector<node_t> current_adj;
    adj_pairs->init(backing_file, adj_start.back(),
                    /*reserve_only = */ true);
    // Edges.
    for (node_t i = 0; i < N; i++) {
      size_t degree = adj_start[i + 1] - adj_start[i];
      current_adj.reserve(degree);
      ReadBinaryOrDie(f, current_adj.data(), degree);
      for (node_t j = 0; j < adj_start.size(); j++) {
        node_t a = i;
        node_t b = current_adj[j];
        if (a == b) continue;
        if (forward_only && b < a) std::swap(a, b);
        adj_pairs->push_back({a, b});
        if (!forward_only) adj_pairs->push_back({b, a});
      }
    }
    return N;
  }

  size_t ReadBinaryStreamedGraph(
      FILE *f, bool forward_only, const std::string &backing_file,
      mmapped_vector<std::pair<node_t, node_t>> *adj_pairs) {
    // Fingerprint.
    unsigned long long_t fingerprint = ReadBinaryOrDie<unsigned long long_t>(f);
    CHECK(fingerprint == kFingerprintStreamed);
    // Number of nodes.
    size_t N = ReadBinaryOrDie<node_t>(f);
    adj_start.resize(N + 1);
    std::vector<node_t> current_adj;
    // Size is unknown at this point - we guess at least N.
    adj_pairs->init(backing_file, N, /*reserve_only = */ true);
    // Edges.
    for (node_t i = 0; i < N; i++) {
      // Degree.
      size_t degree = ReadBinaryOrDie<node_t>(f);
      current_adj.reserve(degree);
      ReadBinaryOrDie(f, current_adj.data(), degree);
      for (node_t j = 0; j < adj_start.size(); j++) {
        node_t a = i;
        node_t b = current_adj[j];
        if (a == b) continue;
        if (forward_only && b < a) std::swap(a, b);
        adj_pairs->push_back({a, b});
        if (!forward_only) adj_pairs->push_back({b, a});
      }
    }
    return N;
  }

  struct DegreeRadixTraits {
    static const int nBytes = sizeof(node_t);
    int kth_byte(const node_t &x, int k) {
      return (degree[x] >> (8 * k)) & 0xff;
    }
    bool compare(const node_t &x, const node_t &y) {
      return degree[x] < degree[y];
    }
    const std::vector<node_t> &degree;
  };

  // Sort node by increasing order.
  std::vector<node_t> ComputeDegreeOrder(
      size_t N, mmapped_vector<std::pair<node_t, node_t>> *adj_pairs) {
    std::vector<node_t> permutation(N);
    std::vector<node_t> degree(N);
#pragma omp parallel for
    for (node_t i = 0; i < N; i++) permutation[i] = i;
#pragma omp parallel for
    for (size_t i = 0; i < adj.size(); i++) {
      degree[(*adj_pairs)[i].first]++;
      degree[(*adj_pairs)[i].second]++;
    }
    // TODO: external memory version
    kx::radix_sort(permutation.begin(), permutation.end(),
                   DegreeRadixTraits{degree});
    return permutation;
  }

  // Sort nodes in degeneracy order
  // (https://en.wikipedia.org/wiki/Degeneracy_(graph_theory)) by iteratively
  // removing lowest-degree nodes.
  // Here, we assume that the graph is sufficiently small that
  // O(number_of_edges) fits in main memory.
  // TODO: avoid the extra copy of the edges.
  std::vector<node_t> ComputeDegeneracyOrder(
      size_t N, mmapped_vector<std::pair<node_t, node_t>> *adj_pairs) {
    std::vector<std::vector<node_t>> graph(N);
    for (auto edg : *adj_pairs) {
      graph[edg.first].push_back(edg.second);
    }
    std::vector<node_t> permutation;
    std::vector<std::vector<node_t>> nodes_by_degree(N);
    std::vector<node_t> degrees(N);
    std::vector<node_t> positions(N);
    std::vector<bool> used(N);
    std::vector<edge_t> adj_starts;
    for (node_t i = 0; i < N; i++) {
      nodes_by_degree[graph[i].size()].push_back(i);
      degrees[i] = graph[i].size();
      positions[i] = nodes_by_degree[degrees[i]].size() - 1;
    }
    node_t j = 0;
    for (node_t i = 0; i < N; i++) {
      while (nodes_by_degree[j].empty()) j++;
      node_t v = nodes_by_degree[j].back();
      nodes_by_degree[j].pop_back();
      permutation.push_back(v);
      used[v] = true;
      for (auto g : graph[v]) {
        if (used[g]) continue;
        node_t &to_swap = nodes_by_degree[degrees[g]][positions[g]];
        std::swap(to_swap, nodes_by_degree[degrees[g]].back());
        positions[to_swap] = positions[g];
        nodes_by_degree[degrees[g]].pop_back();
        degrees[g]--;
        nodes_by_degree[degrees[g]].push_back(g);
        positions[g] = nodes_by_degree[degrees[g]].size() - 1;
      }
      if (j > 0) j--;
    }
    return permutation;
  }
};

#endif
