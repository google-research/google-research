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

#include <gflags/gflags.h>

#include <chrono>  // NOLINT

#include "common.h"           // NOLINT
#include "graph.h"            // NOLINT
#include "intersect_edges.h"  // NOLINT
#include "mmapped_vector.h"   // NOLINT

DEFINE_string(filename, "", "Input filename. Omit for stdin.");
DEFINE_string(format, "nde",
              "Input file format. Valid values are nde, bin and sbin.");
DEFINE_string(storage_dir, "",
              "Directory for storage of working variables. Leave empty to use "
              "main memory.");

using edge_t = uint32_t;
using node_t = uint32_t;

using Graph = GraphT<node_t, edge_t>;

struct TrussDecompositionState {
  // Graph - node-sized vectors
  mmapped_vector<node_t> actual_degree;
  mmapped_vector<edge_t> adj_list_end;
  mmapped_vector<edge_t> adj_list_fwd_start;

  // Algorithm support structures (node-sized)
  mmapped_vector<node_t> dirty_nodes;
  mmapped_vector<uint8_t> dirty_nodes_bitmap;

  // Graph - edge-sized vectors
  mmapped_vector<edge_t> edge_id;

  // Algorithm support structures (edge-sized)
  mmapped_vector<node_t> edge1;
  mmapped_vector<node_t> edge2;
  mmapped_vector<uint32_t> edge_deleted;
  mmapped_vector<std::atomic<node_t>> edge_triangles_count;

  mmapped_vector<edge_t> edges_to_drop;
  edge_t to_drop_start = 0;
  std::atomic<edge_t> to_drop_end{0};
  edge_t iter_drop_start = 0;
  node_t min_size = 2;

  void SetStorageDir(const std::string &storage_dir) {
    actual_degree.init(FileFromStorageDir(storage_dir, "degrees.bin"));
    adj_list_end.init(FileFromStorageDir(storage_dir, "adj_list_end.bin"));
    adj_list_fwd_start.init(
        FileFromStorageDir(storage_dir, "adj_list_fwd_start.bin"));
    dirty_nodes.init(FileFromStorageDir(storage_dir, "dirty_nodes.bin"));
    dirty_nodes_bitmap.init(
        FileFromStorageDir(storage_dir, "dirty_nodes_bitmap.bin"));
    edge_id.init(FileFromStorageDir(storage_dir, "edge_id.bin"));
    edge1.init(FileFromStorageDir(storage_dir, "edge1.bin"));
    edge2.init(FileFromStorageDir(storage_dir, "edge2.bin"));
    edge_deleted.init(FileFromStorageDir(storage_dir, "edge_deleted.bin"));
    edge_triangles_count.init(
        FileFromStorageDir(storage_dir, "edge_triangles_count.bin"));
    edges_to_drop.init(FileFromStorageDir(storage_dir, "edges_to_drop.bin"));
  }
};

#if defined(__AVX2__)
static __m256i avx2_zero = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0);
static __m256i avx2_one = _mm256_set_epi32(1, 1, 1, 1, 1, 1, 1, 1);
static __m256i bv_offsets_mask =
    _mm256_set_epi32(0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f);

__m256i ComputeCompress256Mask(unsigned int mask) {
  const uint32_t identity_indices = 0x76543210;
  uint32_t wanted_indices = _pext_u32(identity_indices, mask);
  __m256i fourbitvec = _mm256_set1_epi32(wanted_indices);
  __m256i shufmask = _mm256_srlv_epi32(
      fourbitvec, _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0));
  shufmask = _mm256_and_si256(shufmask, _mm256_set1_epi32(0x0F));
  return shufmask;
}
#endif

// Re-compute the triangle count for an edge that has had adjacent edges
// removed.
void UpdateEdgeTriangles(Graph *__restrict__ g,
                         TrussDecompositionState *__restrict__ s, edge_t edg) {
  if (s->edge_triangles_count[edg].load(std::memory_order_relaxed) == 0) return;
  node_t i = s->edge1[edg];
  node_t j = s->edge2[edg];
  edge_t inters = 0;
  IntersectEdges(
      g, g->adj_start[i], s->adj_list_end[i], g->adj_start[j],
      s->adj_list_end[j], [&](edge_t k, edge_t l) {
        edge_t edg2 = s->edge_id[k];
        edge_t edg3 = s->edge_id[l];
        bool edge2_deleted = s->edge_deleted[edg2 >> 5] & (1U << (edg2 & 0x1F));
        bool edge3_deleted = s->edge_deleted[edg3 >> 5] & (1U << (edg3 & 0x1F));
        if ((!edge2_deleted || edg2 > edg) && (!edge3_deleted || edg3 > edg)) {
          inters++;
          if (!edge2_deleted) {
            s->edge_triangles_count[edg2].fetch_sub(1,
                                                    std::memory_order_relaxed);
          }
          if (!edge3_deleted) {
            s->edge_triangles_count[edg3].fetch_sub(1,
                                                    std::memory_order_relaxed);
          }
        }
        return inters <
               s->edge_triangles_count[edg].load(std::memory_order_relaxed);
      });
}

void CountTrianglesFromScratch(Graph *__restrict__ g,
                               TrussDecompositionState *__restrict__ s) {
  auto start = std::chrono::high_resolution_clock::now();
  s->edge_triangles_count.resize(g->adj_start.back() / 2);

#pragma omp parallel for schedule(guided)
  for (node_t i = 0; i < g->N(); i++) {
    if (s->adj_list_end[i] == s->adj_list_fwd_start[i]) continue;
    thread_local std::vector<node_t> local_counts;
    local_counts.clear();
    local_counts.resize(s->adj_list_end[i] - s->adj_list_fwd_start[i]);
    for (edge_t cnt = s->adj_list_fwd_start[i]; cnt < s->adj_list_end[i];
         cnt++) {
      const node_t j = g->adj[cnt];
      IntersectEdges(g, cnt + 1, s->adj_list_end[i], s->adj_list_fwd_start[j],
                     s->adj_list_end[j], [&](edge_t k, edge_t l) {
                       edge_t edg3 = s->edge_id[l];
                       local_counts[cnt - s->adj_list_fwd_start[i]]++;
                       local_counts[k - s->adj_list_fwd_start[i]]++;
                       s->edge_triangles_count[edg3].fetch_add(
                           1, std::memory_order_relaxed);
                       return true;
                     });
    }
    for (edge_t cnt = s->adj_list_fwd_start[i]; cnt < s->adj_list_end[i];
         cnt++) {
      s->edge_triangles_count[s->edge_id[cnt]].fetch_add(
          local_counts[cnt - s->adj_list_fwd_start[i]],
          std::memory_order_relaxed);
    }
  }
  FullFence();
#pragma omp parallel for schedule(guided)
  for (node_t i = 0; i < g->N(); i++) {
    for (edge_t cnt = s->adj_list_fwd_start[i]; cnt < s->adj_list_end[i];
         cnt++) {
      auto edg = s->edge_id[cnt];
      if (s->edge_triangles_count[edg].load(std::memory_order_relaxed) + 2 <
          s->min_size) {
        s->edges_to_drop[s->to_drop_end.fetch_add(
            1, std::memory_order_relaxed)] = edg;
      }
    }
  }
  FullFence();
  auto end = std::chrono::high_resolution_clock::now();
  fprintf(stderr, "time for counting triangles: %lums\n",
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count());
}

// Remove gaps in the adjacency lists after removing edges.
void RecompactifyAdjacencyList(Graph *__restrict__ g,
                               TrussDecompositionState *__restrict__ s,
                               node_t i) {
  if (s->actual_degree[i] == 0) {
    s->adj_list_end[i] = s->adj_list_fwd_start[i] = g->adj_start[i];
    return;
  }
  s->adj_list_fwd_start[i] = g->adj_start[i];
  edge_t j = g->adj_start[i];
  node_t pos = g->adj_start[i];
#if defined(__AVX2__)
  __m256i avx2_i = _mm256_castps_si256(_mm256_broadcast_ss((float *)&i));
  for (; j + 7 < s->adj_list_end[i]; j += 8) {
    __m256i edge_ids = _mm256_loadu_si256((__m256i *)&s->edge_id[j]);
    __m256i vertices = _mm256_loadu_si256((__m256i *)&g->adj[j]);

    __m256i bv_offsets = _mm256_srai_epi32(edge_ids, 5);
    __m256i bv_shifts = _mm256_and_si256(edge_ids, bv_offsets_mask);

    __m256i deleted = _mm256_i32gather_epi32(
        (int *)s->edge_deleted.data(), bv_offsets, sizeof(s->edge_deleted[0]));
    deleted = _mm256_srav_epi32(deleted, bv_shifts);
    deleted = _mm256_and_si256(deleted, avx2_one);
    __m256i present_mask = _mm256_cmpeq_epi32(deleted, avx2_zero);

    __m256i greater_mask = _mm256_cmpgt_epi32(avx2_i, vertices);
    greater_mask = _mm256_and_si256(greater_mask, present_mask);

    uint32_t mask = _mm256_movemask_ps(_mm256_castsi256_ps(present_mask));

    if (!mask) continue;
    __m256i compress256_mask =
        ComputeCompress256Mask(_mm256_movemask_epi8(present_mask));
    edge_ids = _mm256_permutevar8x32_epi32(edge_ids, compress256_mask);
    vertices = _mm256_permutevar8x32_epi32(vertices, compress256_mask);

    _mm256_storeu_si256((__m256i *)&g->adj[pos], vertices);
    _mm256_storeu_si256((__m256i *)&s->edge_id[pos], edge_ids);
    pos += __builtin_popcount(mask);

    uint32_t greater_count = __builtin_popcount(
        _mm256_movemask_ps(_mm256_castsi256_ps(greater_mask)));
    s->adj_list_fwd_start[i] += greater_count;
  }
#endif
  for (; j < s->adj_list_end[i]; j++) {
    auto edg = s->edge_id[j];
    if (s->edge_deleted[edg >> 5] & (1U << (edg & 0x1F))) continue;
    if (g->adj[j] < i) s->adj_list_fwd_start[i]++;
    g->adj[pos] = g->adj[j];
    s->edge_id[pos] = s->edge_id[j];
    pos++;
  }
  s->adj_list_end[i] = pos;
}

// Update the number of triangles for each node and recompactify the adjacency
// list if needed.
void UpdateTriangleCounts(Graph *__restrict__ g,
                          TrussDecompositionState *__restrict__ s) {
#pragma omp parallel for schedule(dynamic)
  for (edge_t i = s->iter_drop_start; i < s->to_drop_start; i++) {
    edge_t edg = s->edges_to_drop[i];
    if (!(s->actual_degree[s->edge1[edg]] + 1 < s->min_size &&
          s->actual_degree[s->edge2[edg]] + 1 < s->min_size)) {
      UpdateEdgeTriangles(g, s, edg);
    }
  }
  FullFence();

  // Mark as to be removed edges whose triangle count became too low.
#pragma omp parallel for schedule(dynamic)
  for (node_t cnt = 0; cnt < s->dirty_nodes.size(); cnt++) {
    node_t i = s->dirty_nodes[cnt];
    if (s->adj_list_end[i] - g->adj_start[i] > s->actual_degree[i]) {
      RecompactifyAdjacencyList(g, s, i);
    }
    for (edge_t j = g->adj_start[i]; j < s->adj_list_end[i]; j++) {
      auto edg = s->edge_id[j];
      auto oth = g->adj[j];
      if (s->edge_triangles_count[edg].load(std::memory_order_relaxed) + 2 <
          s->min_size) {
        if (!s->dirty_nodes_bitmap[oth] || i < oth) {
          s->edges_to_drop[s->to_drop_end.fetch_add(
              1, std::memory_order_relaxed)] = edg;
        }
        continue;
      }
      if (s->actual_degree[i] + 1 < s->min_size) {
        if (!s->dirty_nodes_bitmap[oth] ||
            s->actual_degree[oth] + 1 >= s->min_size || i < oth) {
          s->edges_to_drop[s->to_drop_end.fetch_add(
              1, std::memory_order_relaxed)] = edg;
        }
        continue;
      }
    }
  }
  FullFence();
  for (node_t i : s->dirty_nodes) {
    s->dirty_nodes_bitmap[i] = false;
  }
  s->dirty_nodes.clear();
}

void ComputeTrussDecomposition(Graph *__restrict__ g,
                               TrussDecompositionState *__restrict__ s) {
  // Initialize data structures.
  edge_t dropped_edges = 0;
  node_t first_not_empty = 0;

  edge_t edge_count = 0;
  edge_count = g->adj_start.back();
  s->edge_id.resize(edge_count);
  s->edge1.resize(edge_count / 2);
  s->edge2.resize(edge_count / 2);
  s->edges_to_drop.resize(edge_count);
  s->dirty_nodes_bitmap.resize(g->N());

  s->actual_degree.resize(g->N());
  s->adj_list_end.resize(g->N());
  s->adj_list_fwd_start.resize(g->N());

#pragma omp parallel for
  for (node_t i = 0; i < g->N(); i++) {
    s->actual_degree[i] = g->adj_start[i + 1] - g->adj_start[i];
    s->adj_list_end[i] = g->adj_start[i];
  }

  // We want to assign the same ID to both instances of edge {a, b} [i.e. (a, b)
  // and (b, a)], to be able to keep track of the triangle count of the edge
  // easily.
  edge_t current_id = 0;
  for (node_t i = 0; i < g->N(); i++) {
    for (edge_t j = g->adj_start[i]; j < g->adj_start[i + 1]; j++) {
      node_t to = g->adj[j];
      if (to > i) {
        edge_t my_count = current_id++;

        CHECK(g->adj[s->adj_list_end[i]] == to);
        s->edge_id[s->adj_list_end[i]] = my_count;
        s->adj_list_end[i]++;

        CHECK(g->adj[s->adj_list_end[to]] == i);
        s->edge_id[s->adj_list_end[to]] = my_count;
        s->adj_list_end[to]++;

        s->edge1[my_count] = i;
        s->edge2[my_count] = to;
      }
    }
  }

  // Compute the first edge (a, b) in each adjacency list such that b > a.
#pragma omp parallel for
  for (node_t i = 0; i < g->N(); i++) {
    s->adj_list_fwd_start[i] = g->adj_start[i];
    while (s->adj_list_fwd_start[i] < s->adj_list_end[i] &&
           g->adj[s->adj_list_fwd_start[i]] < i) {
      s->adj_list_fwd_start[i]++;
    }
  }

  // Count (a, b) and (b, a) only once.
  edge_count /= 2;

  // Mark a node for recomputing triangle counts after removing one of its
  // edges.
  auto mark_node = [&](node_t node) {
    if (!s->dirty_nodes_bitmap[node]) {
      s->dirty_nodes.push_back(node);
    }
    s->dirty_nodes_bitmap[node] = true;
  };

  // edge_deleted is a bitvector, so it's composed of ceil(num_edges / 32)
  // uint32_t.
  s->edge_deleted.resize((edge_count + 31) >> 5);

  // Initialize triangle counts.
  CountTrianglesFromScratch(g, s);

  // Remove an edge, marking extremes and decreasing their current degrees.
  auto drop_edge = [&](edge_t e) {
    s->actual_degree[s->edge1[e]]--;
    s->actual_degree[s->edge2[e]]--;
    mark_node(s->edge1[e]);
    mark_node(s->edge2[e]);
  };

  edge_t iter_num = 0;
  while (dropped_edges != edge_count) {
    edge_t iters = 0;
    iter_num++;
    // Remove all the edges marked for deletion and update triangle counts until
    // no edge is removed.
    while (s->to_drop_start != s->to_drop_end) {
      edge_t drop_end = s->to_drop_end;
      iters++;
      for (edge_t i = s->to_drop_start; i < drop_end; i++) {
        edge_t e = s->edges_to_drop[i];
        s->edge_deleted[e >> 5] |= 1U << (e & 0x1F);
        dropped_edges++;
        drop_edge(e);
      }
      s->iter_drop_start = s->to_drop_start;
      s->to_drop_start = s->to_drop_end;
      UpdateTriangleCounts(g, s);
    }
    fprintf(stderr, "Iterations for %4u-truss (%14u g->adj): %4u\n",
            s->min_size, edge_count - dropped_edges, iters);
    s->min_size++;
    s->to_drop_start = s->to_drop_end = s->iter_drop_start = 0;

    // Ignore (now and in future iterations) nodes at the beginning of the graph
    // with degree 0. As the graph is in degeneracy order, nodes at the
    // beginning are much more likely to have degree 0 than other nodes.
    while (first_not_empty < g->N() &&
           g->adj_start[first_not_empty] == s->adj_list_end[first_not_empty]) {
      first_not_empty++;
    }

    // Mark edges to be removed for the next truss size. If the minimum triangle
    // count of the remaining graph is larger than what would allow the
    // existence of a truss of the current size, fast-forward the truss size
    // accordingly.
    edge_t min_triangle_count = g->N();
#pragma omp parallel for schedule(guided) reduction(min : min_triangle_count)
    for (node_t i = first_not_empty; i < g->N(); i++) {
      for (edge_t cnt = s->adj_list_fwd_start[i]; cnt < s->adj_list_end[i];
           cnt++) {
        auto edg = s->edge_id[cnt];
        size_t count =
            s->edge_triangles_count[edg].load(std::memory_order_relaxed);
        if (min_triangle_count > count) min_triangle_count = count;
        if (count + 2 < s->min_size) {
          s->edges_to_drop[s->to_drop_end.fetch_add(
              1, std::memory_order_relaxed)] = edg;
        }
      }
    }
    FullFence();
    if (min_triangle_count != g->N() && min_triangle_count + 2 >= s->min_size) {
      s->min_size = min_triangle_count + 2;
    }
  }
}

void Run() {
  auto start = std::chrono::high_resolution_clock::now();

  Graph g;

  g.adj_start.init(FileFromStorageDir(FLAGS_storage_dir, "adj_start.bin"));
  g.adj.init(FileFromStorageDir(FLAGS_storage_dir, "edges.bin"));

  std::chrono::high_resolution_clock::time_point reading;

  g.Read(FLAGS_filename, ParseGraphFormat(FLAGS_format),
         GraphPermutation::kDegeneracyOrder,
         /*forward_only=*/false, &reading);

  auto permute = std::chrono::high_resolution_clock::now();
  TrussDecompositionState s;
  s.SetStorageDir(FLAGS_storage_dir);
  ComputeTrussDecomposition(&g, &s);
  auto end = std::chrono::high_resolution_clock::now();
  fprintf(stderr, "time for reading: %lums\n",
          std::chrono::duration_cast<std::chrono::milliseconds>(reading - start)
              .count());
  fprintf(
      stderr, "time for permutation: %lums\n",
      std::chrono::duration_cast<std::chrono::milliseconds>(permute - reading)
          .count());
  fprintf(stderr, "time for computing: %lums\n",
          std::chrono::duration_cast<std::chrono::milliseconds>(end - permute)
              .count());
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  Run();
}
