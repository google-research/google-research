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



#ifndef SCANN_HASHES_INTERNAL_ASYMMETRIC_HASHING_LUT16_H_
#define SCANN_HASHES_INTERNAL_ASYMMETRIC_HASHING_LUT16_H_

#include "scann/base/restrict_allowlist.h"
#include "scann/data_format/dataset.h"
#include "scann/hashes/internal/asymmetric_hashing_postprocess.h"
#include "scann/hashes/internal/lut16_interface.h"
#include "scann/hashes/internal/write_distances_to_topn.h"
#include "scann/utils/top_n_amortized_constant.h"
#include "scann/utils/types.h"

namespace research_scann {
namespace asymmetric_hashing_internal {

using TopFixedPointNeighbors = TopNeighbors<int32_t>;

template <typename TopN, typename PostprocessedDistance, typename Postprocess>
void GetNeighborsViaAsymmetricDistanceLUT16WithInt32Accumulator2(
    ConstSpan<uint8_t> lookup, DatapointIndex dataset_size,
    const std::vector<uint8_t>& packed_dataset,
    const RestrictAllowlist* whitelist_or_null,
    PostprocessedDistance max_distance, const Postprocess& postprocess,
    TopN* top_items);

template <typename TopN, typename PostprocessedDistance, typename Postprocess>
void GetNeighborsViaAsymmetricDistanceLUT16WithInt16Accumulator2(
    ConstSpan<uint8_t> lookup, DatapointIndex dataset_size,
    const std::vector<uint8_t>& packed_dataset,
    const RestrictAllowlist* whitelist_or_null,
    PostprocessedDistance max_distance, const Postprocess& postprocess,
    TopN* top_items);

template <size_t kNumQueries, typename TopN, typename PostprocessedDistance,
          typename Postprocess>
void GetNeighborsViaAsymmetricDistanceLUT16WithInt16AccumulatorBatched2(
    array<ConstSpan<int8_t>, kNumQueries> lookups, DatapointIndex dataset_size,
    const std::vector<uint8_t>& packed_dataset,
    array<const RestrictAllowlist*, kNumQueries> restrict_whitelists_or_null,
    array<PostprocessedDistance, kNumQueries> max_distances,
    const Postprocess& postprocess, array<TopN*, kNumQueries> top_items);

template <size_t kNumQueries, typename TopN, typename PostprocessedDistance,
          typename Postprocess>
void GetNeighborsViaAsymmetricDistanceLUT16WithInt32AccumulatorBatched2(
    array<ConstSpan<int8_t>, kNumQueries> lookups, DatapointIndex dataset_size,
    const std::vector<uint8_t>& packed_dataset,
    array<const RestrictAllowlist*, kNumQueries> restrict_whitelists_or_null,
    array<PostprocessedDistance, kNumQueries> max_distances,
    const Postprocess& postprocess, array<TopN*, kNumQueries> top_items);

template <typename TopN, typename PostprocessedDistance, typename Postprocess>
void GetNeighborsViaAsymmetricDistanceLUT16WithInt32Accumulator2(
    ConstSpan<uint8_t> lookup, DatapointIndex dataset_size,
    const std::vector<uint8_t>& packed_dataset,
    const RestrictAllowlist* whitelist_or_null,
    PostprocessedDistance max_distance, const Postprocess& postprocess,
    TopN* top_items) {
  const size_t num_32dp_simd_iters = DivRoundUp(dataset_size, 32);

  unique_ptr<int32_t[]> distances(new int32_t[32 * num_32dp_simd_iters]);
  size_t num_blocks = lookup.size() / 16;

  LUT16Interface::GetDistances(packed_dataset.data(), num_32dp_simd_iters,
                               num_blocks, lookup.data(), distances.get());

  if (std::is_same<Postprocess, IdentityPostprocessFunctor>::value &&
      std::is_same<PostprocessedDistance, int32_t>::value &&
      max_distance >=
          static_cast<int32_t>(numeric_limits<int8_t>::max() * num_blocks)) {
    max_distance = numeric_limits<int32_t>::max();
  }

  WriteDistancesToTopN(whitelist_or_null, max_distance,
                       ConstSpan<int32_t>(distances.get(), dataset_size),
                       postprocess, top_items);
}

template <typename TopN, typename PostprocessedDistance, typename Postprocess>
void GetNeighborsViaAsymmetricDistanceLUT16WithInt16Accumulator2(
    ConstSpan<uint8_t> lookup, DatapointIndex dataset_size,
    const std::vector<uint8_t>& packed_dataset,
    const RestrictAllowlist* whitelist_or_null,
    PostprocessedDistance max_distance, const Postprocess& postprocess,
    TopN* top_items) {
  if (max_distance > numeric_limits<int16_t>::max()) {
    max_distance = numeric_limits<int16_t>::max();
  } else if (max_distance < numeric_limits<int16_t>::min()) {
    return;
  }

  const size_t num_32dp_simd_iters = DivRoundUp(dataset_size, 32);
  unique_ptr<int16_t[]> distances(new int16_t[32 * num_32dp_simd_iters]);
  size_t num_blocks = lookup.size() / 16;

  LUT16Interface::GetDistances(packed_dataset.data(), num_32dp_simd_iters,
                               num_blocks, lookup.data(), distances.get());

  return WriteDistancesToTopN(whitelist_or_null, max_distance,
                              ConstSpan<int16_t>(distances.get(), dataset_size),
                              postprocess, top_items);
}

template <size_t kNumQueries, typename TopN, typename PostprocessedDistance,
          typename DistT, typename Postprocess>
void GetNeighborsViaAsymmetricDistanceLUT16BatchedImpl(
    array<ConstSpan<uint8_t>, kNumQueries> lookups, DatapointIndex dataset_size,
    const std::vector<uint8_t>& packed_dataset,
    array<const RestrictAllowlist*, kNumQueries> restrict_whitelists_or_null,
    array<PostprocessedDistance, kNumQueries> max_distances,
    const Postprocess& postprocess, array<TopN*, kNumQueries> top_items) {
  for (auto& lookup : lookups) {
    DCHECK_EQ(lookup.size(), lookups[0].size());
  }

  bool all_thresholds_too_small = true;
  for (auto& max_dist : max_distances) {
    if (max_dist > numeric_limits<DistT>::max()) {
      max_dist = numeric_limits<DistT>::max();
    }
    if (max_dist >= numeric_limits<DistT>::min()) {
      all_thresholds_too_small = false;
    }
  }
  if (all_thresholds_too_small) return;

  const size_t num_32dp_simd_iters = DivRoundUp(dataset_size, 32);

  array<const uint8_t*, kNumQueries> lookup_ptrs;
  array<DistT*, kNumQueries> distances;
  array<unique_ptr<DistT[]>, kNumQueries> distances_storage;
  for (size_t i = 0; i < kNumQueries; ++i) {
    distances_storage[i].reset(new DistT[32 * num_32dp_simd_iters]);
    distances[i] = distances_storage[i].get();
    lookup_ptrs[i] = lookups[i].data();
  }

  const size_t num_blocks = lookups[0].size() / 16;
  LUT16Interface::GetDistances(packed_dataset.data(), num_32dp_simd_iters,
                               num_blocks, lookup_ptrs, distances);

  for (size_t i = 0; i < kNumQueries; ++i) {
    WriteDistancesToTopN(restrict_whitelists_or_null[i],
                         static_cast<int32_t>(max_distances[i]),
                         ConstSpan<DistT>(distances[i], dataset_size),
                         postprocess, top_items[i]);
  }
}

template <size_t kNumQueries, typename TopN, typename PostprocessedDistance,
          typename Postprocess>
void GetNeighborsViaAsymmetricDistanceLUT16WithInt16AccumulatorBatched2(
    array<ConstSpan<uint8_t>, kNumQueries> lookups, DatapointIndex dataset_size,
    const std::vector<uint8_t>& packed_dataset,
    array<const RestrictAllowlist*, kNumQueries> restrict_whitelists_or_null,
    array<PostprocessedDistance, kNumQueries> max_distances,
    const Postprocess& postprocess, array<TopN*, kNumQueries> top_items) {
  GetNeighborsViaAsymmetricDistanceLUT16BatchedImpl<
      kNumQueries, TopN, PostprocessedDistance, int16_t, Postprocess>(
      lookups, dataset_size, packed_dataset, restrict_whitelists_or_null,
      max_distances, postprocess, top_items);
}

template <size_t kNumQueries, typename TopN, typename PostprocessedDistance,
          typename Postprocess>
void GetNeighborsViaAsymmetricDistanceLUT16WithInt32AccumulatorBatched2(
    array<ConstSpan<uint8_t>, kNumQueries> lookups, DatapointIndex dataset_size,
    const std::vector<uint8_t>& packed_dataset,
    array<const RestrictAllowlist*, kNumQueries> restrict_whitelists_or_null,
    array<PostprocessedDistance, kNumQueries> max_distances,
    const Postprocess& postprocess, array<TopN*, kNumQueries> top_items) {
  GetNeighborsViaAsymmetricDistanceLUT16BatchedImpl<
      kNumQueries, TopN, PostprocessedDistance, int32_t, Postprocess>(
      lookups, dataset_size, packed_dataset, restrict_whitelists_or_null,
      max_distances, postprocess, top_items);
}

}  // namespace asymmetric_hashing_internal
}  // namespace research_scann

#endif
