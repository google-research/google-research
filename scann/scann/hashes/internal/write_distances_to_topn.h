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

#ifndef SCANN_HASHES_INTERNAL_WRITE_DISTANCES_TO_TOPN_H_
#define SCANN_HASHES_INTERNAL_WRITE_DISTANCES_TO_TOPN_H_

#include <cstdint>

#include "scann/base/restrict_allowlist.h"
#include "scann/hashes/internal/asymmetric_hashing_postprocess.h"
#include "scann/utils/top_n_amortized_constant.h"
#include "scann/utils/types.h"

namespace research_scann {
namespace asymmetric_hashing_internal {

using TopFixedPointNeighbors = TopNeighbors<int32_t>;

template <bool restricts_enabled, typename TopN, typename PostprocessedDistance,
          typename Postprocess, typename Int>
void WriteDistancesToTopNImpl(const RestrictAllowlist* whitelist_or_null,
                              PostprocessedDistance max_distance,
                              ConstSpan<Int> distances,
                              const Postprocess& postprocess, TopN* top_n_ptr);

template <typename TopN, typename PostprocessedDistance, typename Postprocess>
void WriteDistancesToTopN(const RestrictAllowlist* whitelist_or_null,
                          PostprocessedDistance max_distance,
                          ConstSpan<int16_t> distances,
                          const Postprocess& postprocess, TopN* top_n) {
  return (!whitelist_or_null)
             ? WriteDistancesToTopNImpl<false>(whitelist_or_null, max_distance,
                                               distances, postprocess, top_n)
             : WriteDistancesToTopNImpl<true>(whitelist_or_null, max_distance,
                                              distances, postprocess, top_n);
}

template <typename TopN, typename PostprocessedDistance, typename Postprocess>
void WriteDistancesToTopN(const RestrictAllowlist* whitelist_or_null,
                          PostprocessedDistance max_distance,
                          ConstSpan<int32_t> distances,
                          const Postprocess& postprocess, TopN* top_n) {
  return (!whitelist_or_null)
             ? WriteDistancesToTopNImpl<false>(whitelist_or_null, max_distance,
                                               distances, postprocess, top_n)
             : WriteDistancesToTopNImpl<true>(whitelist_or_null, max_distance,
                                              distances, postprocess, top_n);
}

template <bool restricts_enabled, typename TopN, typename PostprocessedDistance,
          typename Postprocess, typename Int>
void WriteDistancesToTopNImpl(const RestrictAllowlist* whitelist_or_null,
                              PostprocessedDistance max_distance,
                              ConstSpan<Int> distances,
                              const Postprocess& postprocess, TopN* top_n_ptr) {
  RestrictAllowlistConstView whitelist_view;
  if (restricts_enabled) {
    whitelist_view = RestrictAllowlistConstView(whitelist_or_null);
  }

  auto is_whitelisted =
      [&whitelist_view](DatapointIndex dp_index) SCANN_INLINE_LAMBDA {
        return !restricts_enabled || whitelist_view.IsWhitelisted(dp_index);
      };

  auto top_n = std::move(*top_n_ptr);
  const DatapointIndex dataset_size = distances.size();
  for (DatapointIndex i = 0; i < dataset_size; ++i) {
    const auto dist = postprocess.Postprocess(distances[i], i);
    if (is_whitelisted(i) && dist <= max_distance) {
      top_n.push(std::make_pair(i, dist));
      if (top_n.full()) max_distance = top_n.approx_bottom().second;
    }
  }
  *top_n_ptr = std::move(top_n);
}

#ifdef __SSE4_1__

void WriteDistancesToTopN(const RestrictAllowlist* whitelist_or_null,
                          int32_t max_distance, ConstSpan<int16_t> distances,
                          const IdentityPostprocessFunctor& postprocess,
                          TopFixedPointNeighbors* top_n);
void WriteDistancesToTopN(const RestrictAllowlist* whitelist_or_null,
                          int32_t max_distance, ConstSpan<int32_t> distances,
                          const IdentityPostprocessFunctor& postprocess,
                          TopFixedPointNeighbors* top_n);

#endif

}  // namespace asymmetric_hashing_internal
}  // namespace research_scann

#endif
