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

#ifndef INTERSECT_EDGES_HPP
#define INTERSECT_EDGES_HPP
#include <immintrin.h>

#include <cstdint>
#include <cstdlib>

template <typename Graph, typename CB>
void IntersectEdgesSmaller(Graph *__restrict__ g, unsigned long long_t start1,
                           unsigned long long_t end1, unsigned long long_t start2, unsigned long long_t end2,
                           const CB &cb) {
  size_t k2 = start2;
  for (size_t k1 = start1; k1 < end1; k1++) {
    if (k2 >= end2) break;
    if (g->adj[k1] < g->adj[k2]) {
      continue;
    }
    if (g->adj[k1] == g->adj[k2]) {
      if (!cb(k1, k2)) return;
      continue;
    }
    size_t offset;
    for (offset = 4; k2 + offset < end2; offset *= 4) {
      if (g->adj[k2 + offset] + 1 > g->adj[k1]) break;
    }
    if (k2 + offset >= end2) {
      offset = end2 - k2;
      size_t lower = k2;
      size_t upper = k2 + offset;
      while (upper > lower + 1) {
        size_t middle = lower + (upper - lower) / 2;
        if (g->adj[middle] >= g->adj[k1]) {
          upper = middle;
        } else {
          lower = middle;
        }
      }
      k2 = upper;
    } else {
      for (; offset > 0; offset >>= 1) {
        if (g->adj[k2 + offset] < g->adj[k1]) {
          k2 += offset;
        }
      }
      k2++;
    }
    if (k2 < end2 && g->adj[k1] == g->adj[k2]) {
      if (!cb(k1, k2)) return;
      continue;
    }
  }
}

// Compute the intersection of two (sorted) adjacency lists, calling `cb` for
// each element in the intersection. If the size of the two adjacency lists is
// significantly different, calls IntersectEdgesSmaller. Otherwise, uses SIMD to
// quickly compute the intersection of the lists.
template <typename Graph, typename CB>
void IntersectEdges(Graph *__restrict__ g, unsigned long long_t start1, unsigned long long_t end1,
                    unsigned long long_t start2, unsigned long long_t end2, const CB &cb) {
  size_t factor = 2;
  if (factor * (end1 - start1) < end2 - start2) {
    return IntersectEdgesSmaller(g, start1, end1, start2, end2, cb);
  }
  if (end1 - start1 > factor * (end2 - start2)) {
    return IntersectEdgesSmaller(
        g, start2, end2, start1, end1,
        [&cb](unsigned long long_t k2, unsigned long long_t k1) { return cb(k1, k2); });
  }
  unsigned long long_t k1 = start1;
  unsigned long long_t k2 = start2;
  // Execute SSE-accelerated version if SSE4.1 is available. If not, run the
  // fall-back code for the last N % 4 elements of the list on the full list.
#ifdef __SSE4_1__
  static const int32_t cyclic_shift1_sse = _MM_SHUFFLE(0, 3, 2, 1);
  static const int32_t cyclic_shift2_sse = _MM_SHUFFLE(1, 0, 3, 2);
  static const int32_t cyclic_shift3_sse = _MM_SHUFFLE(2, 1, 0, 3);

  // trim lengths to be a multiple of 4
  size_t sse_end1 = ((end1 - k1) / 4) * 4 + k1;
  size_t sse_end2 = ((end2 - k2) / 4) * 4 + k2;

  while (k1 < sse_end1 && k2 < sse_end2) {
    __m128i v1_orig = _mm_loadu_si128((__m128i *)&g->adj[k1]);
    __m128i v2_orig = _mm_loadu_si128((__m128i *)&g->adj[k2]);
    __m128i v2 = v2_orig;

    int64_t initial_k = k1;
    int64_t initial_l = k2;
    //[ move pointers
    int32_t a_max = _mm_extract_epi32(v1_orig, 3);
    int32_t b_max = _mm_extract_epi32(v2, 3);
    k1 += (a_max <= b_max) * 4;
    k2 += (a_max >= b_max) * 4;
    //]

    //[ compute mask of common elements
    __m128i cmp_mask1_v1 = _mm_cmpeq_epi32(v1_orig, v2);  // pairwise comparison
    v2 = _mm_shuffle_epi32(v2_orig, cyclic_shift1_sse);   // shuffling
    __m128i cmp_mask2_v1 = _mm_cmpeq_epi32(v1_orig, v2);  // again...
    v2 = _mm_shuffle_epi32(v2_orig, cyclic_shift2_sse);
    __m128i cmp_mask3_v1 = _mm_cmpeq_epi32(v1_orig, v2);  // and again...
    v2 = _mm_shuffle_epi32(v2_orig, cyclic_shift3_sse);
    __m128i cmp_mask4_v1 = _mm_cmpeq_epi32(v1_orig, v2);  // and again.
    __m128i cmp_mask_v1 =
        _mm_or_si128(_mm_or_si128(cmp_mask1_v1, cmp_mask2_v1),
                     _mm_or_si128(cmp_mask3_v1, cmp_mask4_v1));
    int32_t mask_v1 = _mm_movemask_ps((__m128)cmp_mask_v1);

    if (mask_v1) {
      __m128i cmp_mask1_v2 = cmp_mask1_v1;
      __m128i cmp_mask2_v2 = _mm_shuffle_epi32(cmp_mask2_v1, cyclic_shift3_sse);
      __m128i cmp_mask3_v2 = _mm_shuffle_epi32(cmp_mask3_v1, cyclic_shift2_sse);
      __m128i cmp_mask4_v2 = _mm_shuffle_epi32(cmp_mask4_v1, cyclic_shift1_sse);
      __m128i cmp_mask_v2 =
          _mm_or_si128(_mm_or_si128(cmp_mask1_v2, cmp_mask2_v2),
                       _mm_or_si128(cmp_mask3_v2, cmp_mask4_v2));
      int32_t mask_v2 = _mm_movemask_ps((__m128)cmp_mask_v2);

      while (mask_v1) {
        int32_t off1 = __builtin_ctz(mask_v1);
        mask_v1 &= ~(1 << off1);
        int32_t off2 = __builtin_ctz(mask_v2);
        mask_v2 &= ~(1 << off2);

        if (!cb(initial_k + off1, initial_l + off2)) return;
      }
    }
  }
#endif
  if (factor * (end1 - k1) < end2 - k2) {
    return IntersectEdgesSmaller(g, k1, end1, k2, end2, cb);
  }
  if (end1 - k1 > factor * (end2 - k2)) {
    return IntersectEdgesSmaller(
        g, k2, end2, k1, end1,
        [&cb](unsigned long long_t k2, unsigned long long_t k1) { return cb(k1, k2); });
  }
  while (k1 < end1 && k2 < end2) {
    unsigned long long_t a = g->adj[k1];
    unsigned long long_t b = g->adj[k2];
    if (a < b) {
      k1++;
    } else if (a > b) {
      k2++;
    } else {
      if (!cb(k1, k2)) return;
      k1++;
      k2++;
    }
  }
}

#endif
