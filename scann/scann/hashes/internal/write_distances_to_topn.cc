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

#include "scann/hashes/internal/write_distances_to_topn.h"

#include "scann/base/restrict_allowlist.h"
#include "scann/utils/intrinsics/sse4.h"

namespace research_scann {
namespace asymmetric_hashing_internal {

#ifdef __SSE4_1__

namespace {

DatapointIndex CountLessEqual(const int16_t* begin, const int16_t* end,
                              int16_t max_distance) {
  DatapointIndex result = end - begin;
  __m128i max_distance_simd = _mm_set1_epi16(max_distance);

  while (end - begin >= 8) {
    __m128i acc = _mm_setzero_si128();
    DatapointIndex n_simd_iter = std::min<DatapointIndex>(
        numeric_limits<uint16_t>::max(), (end - begin) / 8);
    for (; n_simd_iter != 0; --n_simd_iter, begin += 8) {
      __m128i dists = _mm_loadu_si128(reinterpret_cast<const __m128i*>(begin));

      __m128i to_sub = _mm_cmpgt_epi16(dists, max_distance_simd);
      acc = _mm_sub_epi16(acc, to_sub);
    }

    __m128i lower_extended = _mm_unpacklo_epi16(acc, _mm_setzero_si128());
    __m128i upper_extended = _mm_unpackhi_epi16(acc, _mm_setzero_si128());
    acc = _mm_add_epi32(lower_extended, upper_extended);
    acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 8));
    acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 4));
    result -= _mm_cvtsi128_si32(acc);
  }

  for (; begin < end; ++begin) {
    result -= (*begin > max_distance);
  }

  return result;
}

template <bool restricts_enabled>
void WriteDistancesToTopNImpl(const RestrictAllowlist* whitelist_or_null,
                              int32_t max_distance,
                              ConstSpan<int16_t> distances,
                              TopFixedPointNeighbors* top_n_ptr) {
  DCHECK(top_n_ptr);
  DCHECK((restricts_enabled && whitelist_or_null) ||
         (!restricts_enabled && whitelist_or_null == nullptr));
  const size_t num_neighbors = top_n_ptr->limit();

  RestrictAllowlistConstView whitelist_view;
  if (restricts_enabled) {
    whitelist_view = RestrictAllowlistConstView(whitelist_or_null);
  }

  auto is_whitelisted =
      [&whitelist_view](DatapointIndex dp_index) SCANN_INLINE_LAMBDA {
        return !restricts_enabled || whitelist_view.IsWhitelisted(dp_index);
      };

  const size_t dataset_size = distances.size();

  auto write_all_whitelisted_distances = [&]() -> void {
    vector<pair<DatapointIndex, int32_t>> top_items(dataset_size);
    auto top_items_ptr = top_items.begin();
    for (DatapointIndex i = 0; i < dataset_size; ++i) {
      if (is_whitelisted(i)) {
        top_items_ptr->first = i;
        top_items_ptr->second = distances[i];
        ++top_items_ptr;
      }
    }

    if (restricts_enabled) {
      top_items.resize(top_items_ptr - top_items.begin());
    } else {
      DCHECK(top_items_ptr == top_items.end());
    }
    top_n_ptr->OverwriteContents(
        std::move(top_items),
        std::make_pair(kInvalidDatapointIndex, numeric_limits<int32_t>::max()));
  };

  auto block_contains_whitelisted_8 =
      [whitelist_or_null](DatapointIndex dp_index) -> bool {
    if (!restricts_enabled) return true;
    const uint8_t offset = dp_index % RestrictAllowlist::kBitsPerWord;
    return ((whitelist_or_null->GetWordContainingDatapoint(dp_index) >>
             offset) &
            0xFF) != 0;
  };

  if (max_distance >= numeric_limits<int16_t>::max() &&
      num_neighbors >= dataset_size) {
    return write_all_whitelisted_distances();
  } else if (num_neighbors >= dataset_size) {
    const DatapointIndex n_below_epsilon = CountLessEqual(
        distances.data(), distances.data() + dataset_size, max_distance);
    if (n_below_epsilon == dataset_size) {
      return write_all_whitelisted_distances();
    }

    vector<pair<DatapointIndex, int32_t>> top_items(n_below_epsilon);
    auto top_items_ptr = top_items.begin();
    auto put_result = [&top_items_ptr, max_distance, &is_whitelisted](
                          DatapointIndex i, int16_t d) {
      if (!is_whitelisted(i)) return;
      top_items_ptr->first = i;
      top_items_ptr->second = d;
      ++top_items_ptr;
    };

    __m128i max_distance_simd = _mm_set1_epi16(max_distance);
    DatapointIndex i;
    for (i = 0; i < (dataset_size & ~31); i += 8) {
      if (!block_contains_whitelisted_8(i)) continue;
      __m128i to_push =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(&distances[i]));
      __m128i to_test = _mm_cmpgt_epi16(to_push, max_distance_simd);
      const uint16_t mask = ~static_cast<int16_t>(_mm_movemask_epi8(to_test));

      if (mask) {
        if (mask >> 0 & 1) put_result(i + 0, _mm_extract_epi16(to_push, 0));
        if (mask >> 2 & 1) put_result(i + 1, _mm_extract_epi16(to_push, 1));
        if (mask >> 4 & 1) put_result(i + 2, _mm_extract_epi16(to_push, 2));
        if (mask >> 6 & 1) put_result(i + 3, _mm_extract_epi16(to_push, 3));
        if (mask >> 8 & 1) put_result(i + 4, _mm_extract_epi16(to_push, 4));
        if (mask >> 10 & 1) put_result(i + 5, _mm_extract_epi16(to_push, 5));
        if (mask >> 12 & 1) put_result(i + 6, _mm_extract_epi16(to_push, 6));
        if (mask >> 14 & 1) put_result(i + 7, _mm_extract_epi16(to_push, 7));
      }
    }

    for (; i < dataset_size; ++i) {
      if (distances[i] <= max_distance) put_result(i, distances[i]);
    }

    if (restricts_enabled) {
      top_items.resize(top_items_ptr - top_items.begin());
    } else {
      DCHECK(top_items_ptr == top_items.end());
    }

    top_n_ptr->OverwriteContents(
        std::move(top_items),
        std::make_pair(kInvalidDatapointIndex, numeric_limits<int32_t>::max()));
    return;
  }

  auto top_n = std::move(*top_n_ptr);
  DatapointIndex i;
  for (i = 0; i < (dataset_size & ~31); i += 8) {
    if (!block_contains_whitelisted_8(i)) continue;
    __m128i to_push =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(&distances[i]));
    __m128i to_test = _mm_cmpgt_epi16(to_push, _mm_set1_epi16(max_distance));
    if (~static_cast<int16_t>(_mm_movemask_epi8(to_test))) {
      for (DatapointIndex j = i; j < (i + 8); ++j) {
        if (is_whitelisted(j) && distances[j] <= max_distance) {
          top_n.push(std::make_pair(j, distances[j]));
          if (top_n.full()) {
            max_distance = top_n.approx_bottom().second;
          }
        }
      }
    }
  }
  for (; i < dataset_size; ++i) {
    if (is_whitelisted(i) && distances[i] <= max_distance) {
      top_n.push(std::make_pair(i, distances[i]));
      if (top_n.full()) {
        max_distance = top_n.approx_bottom().second;
      }
    }
  }

  *top_n_ptr = std::move(top_n);
}

template <bool restricts_enabled>
void WriteDistancesToTopNImpl(const RestrictAllowlist* whitelist_or_null,
                              int32_t max_distance,
                              ConstSpan<int32_t> distances,
                              TopFixedPointNeighbors* top_n_ptr) {
  DCHECK((restricts_enabled && whitelist_or_null) ||
         (!restricts_enabled && whitelist_or_null == nullptr));
  const size_t num_neighbors = top_n_ptr->limit();
  const size_t dataset_size = distances.size();

  RestrictAllowlistConstView whitelist_view;
  if (restricts_enabled) {
    whitelist_view = RestrictAllowlistConstView(whitelist_or_null);
  }

  auto is_whitelisted =
      [&whitelist_view](DatapointIndex dp_index) SCANN_INLINE_LAMBDA {
        return !restricts_enabled || whitelist_view.IsWhitelisted(dp_index);
      };

  if (max_distance == numeric_limits<int32_t>::max() &&
      num_neighbors >= dataset_size) {
    vector<pair<DatapointIndex, int32_t>> top_items(dataset_size);
    auto top_items_ptr = top_items.begin();
    for (DatapointIndex i = 0; i < dataset_size; ++i) {
      if (!is_whitelisted(i)) continue;
      top_items_ptr->first = i;
      top_items_ptr->second = distances[i];
      ++top_items_ptr;
    }

    if (restricts_enabled) {
      top_items.resize(top_items_ptr - top_items.begin());
    } else {
      DCHECK(top_items_ptr == top_items.end());
    }
    top_n_ptr->OverwriteContents(
        std::move(top_items),
        std::make_pair(kInvalidDatapointIndex, numeric_limits<int32_t>::max()));
    return;
  }

  auto top_n = std::move(*top_n_ptr);
  for (DatapointIndex i = 0; i < dataset_size; ++i) {
    if (is_whitelisted(i) && distances[i] <= max_distance) {
      top_n.push(std::make_pair(i, distances[i]));
      if (top_n.full()) {
        max_distance = top_n.approx_bottom().second;
      }
    }
  }
  *top_n_ptr = std::move(top_n);
}

}  // namespace

void WriteDistancesToTopN(const RestrictAllowlist* whitelist_or_null,
                          int32_t max_distance, ConstSpan<int16_t> distances,
                          const IdentityPostprocessFunctor&,
                          TopFixedPointNeighbors* top_n) {
  return (!whitelist_or_null)
             ? WriteDistancesToTopNImpl<false>(whitelist_or_null, max_distance,
                                               distances, top_n)
             : WriteDistancesToTopNImpl<true>(whitelist_or_null, max_distance,
                                              distances, top_n);
}

void WriteDistancesToTopN(const RestrictAllowlist* whitelist_or_null,
                          int32_t max_distance, ConstSpan<int32_t> distances,
                          const IdentityPostprocessFunctor&,
                          TopFixedPointNeighbors* top_n) {
  return (!whitelist_or_null)
             ? WriteDistancesToTopNImpl<false>(whitelist_or_null, max_distance,
                                               distances, top_n)
             : WriteDistancesToTopNImpl<true>(whitelist_or_null, max_distance,
                                              distances, top_n);
}

#endif

}  // namespace asymmetric_hashing_internal
}  // namespace research_scann
