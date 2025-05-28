// Copyright 2025 The Google Research Authors.
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

#include "scann/tree_x_hybrid/internal/utils.h"

#include "absl/container/btree_map.h"
#include "absl/flags/flag.h"
#include "absl/types/span.h"

#ifdef __x86_64__
#include <x86intrin.h>
#endif

#include <cstdint>

#include "scann/utils/common.h"

namespace research_scann {

#ifdef __x86_64__
SCANN_AVX2_OUTLINE size_t Avx2GatherCreateLeafLocalAllowlist(
    RestrictAllowlistConstView global_allowlist,
    RestrictAllowlistMutableView leaf_view,
    ConstSpan<DatapointIndex> leaf_local_to_global) {
  enum : size_t {
    kDatapointIndicesPerRegister = 256 / (sizeof(DatapointIndex) * 8),
    kSizetBits = sizeof(size_t) * 8
  };
  const size_t num_full_iters =
      leaf_local_to_global.size() / kDatapointIndicesPerRegister;
  for (size_t i : Seq(num_full_iters)) {
    const size_t start_idx = i * kDatapointIndicesPerRegister;
    __m256i dp_idxs = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(&leaf_local_to_global[start_idx]));
    __m256i uint32_idxs = _mm256_srli_epi32(dp_idxs, 5);
    __m256i bit_idxs = _mm256_and_si256(dp_idxs, _mm256_set1_epi32(0x1F));
    __m256i uint32s = _mm256_i32gather_epi32(
        reinterpret_cast<const int*>(global_allowlist.data()), uint32_idxs, 4);
    __m256i shifted = _mm256_srlv_epi32(uint32s, bit_idxs);
    __m256i mask = _mm256_slli_epi32(shifted, 31);
    size_t bitmask_bits = _mm256_movemask_ps(static_cast<__m256>(mask));
    leaf_view.data()[start_idx / kSizetBits] |= bitmask_bits
                                                << (start_idx % kSizetBits);
  }
  return num_full_iters * kDatapointIndicesPerRegister;
}

#endif

StatusOr<bool> ValidateDatapointsByToken(
    absl::Span<const std::vector<DatapointIndex>> datapoints_by_token,
    DatapointIndex num_datapoints) {
  bool is_disjoint = true;

  vector<bool> global_bitmap(num_datapoints, false);
  vector<bool> seen_twice(num_datapoints, false);

  for (const std::vector<DatapointIndex>& dp_list : datapoints_by_token) {
    flat_hash_set<DatapointIndex> duplicates;
    for (DatapointIndex dp_index : dp_list) {
      if (!duplicates.insert(dp_index).second) {
        return InvalidArgumentError(
            absl::StrCat("Duplicate datapoint index within a partition of "
                         "datapoints_by_token:  ",
                         dp_index, "."));
      }
      if (dp_index >= num_datapoints) {
        return OutOfRangeError(
            "Datapoint index in datapoints_by_token is >= number of "
            "datapoints in database (%d vs. %d).",
            dp_index, num_datapoints);
      }
      if (global_bitmap[dp_index]) {
        is_disjoint = false;
        if (seen_twice[dp_index]) {
          return InvalidArgumentError(
              StrCat("Datapoint ", dp_index,
                     " represented more than twice in datapoints_by_token.  "
                     "Only TWO_CENTER_ORTHOGONALITY_AMPLIFIED database "
                     "spilling is supported in tree-X hybrid."));
        } else {
          seen_twice[dp_index] = true;
        }
      } else {
        global_bitmap[dp_index] = true;
      }
    }
  }

  const DatapointIndex num_missing =
      std::count(global_bitmap.begin(), global_bitmap.end(), false);
  if (num_missing > 0) {
    auto false_it =
        std::find(global_bitmap.begin(), global_bitmap.end(), false);
    const size_t first_missing = false_it - global_bitmap.begin();
    return InvalidArgumentError(absl::StrCat(
        "Found ", num_missing,
        " datapoint(s) "
        "that are not represented in any partition.  First missing "
        "datapoint index = ",
        first_missing, "."));
  }

  return is_disjoint;
}

vector<uint32_t> SizeByPartition(
    ConstSpan<std::vector<DatapointIndex>> datapoints_by_token) {
  vector<uint32_t> result(datapoints_by_token.size());
  for (size_t i : IndicesOf(datapoints_by_token)) {
    result[i] = datapoints_by_token[i].size();
  }
  return result;
}

template <typename Container>
void MaybeReserve(Container& c, size_t s) {}

template <>
void MaybeReserve(flat_hash_map<DatapointIndex, float>& c, size_t s) {
  c.reserve(s);
}

template <typename Container>
void DeduplicateDatabaseSpilledResults(NNResultsVector* results,
                                       size_t final_size) {
  DCHECK_GT(final_size, 0);
  DCHECK_LE(results->size() / 2, final_size);
  Container map;
  MaybeReserve(map, results->size());
  for (const auto& neighbor : *results) {
    auto [it, was_inserted] = map.insert(neighbor);
    if (!was_inserted) {
      it->second = 0.5f * it->second + 0.5f * neighbor.second;
    }
  }
  std::copy(map.begin(), map.end(), results->begin());
  results->resize(map.size());
  if (results->size() > final_size) {
    NthElementBranchOptimized(results->begin(),
                              results->begin() + final_size - 1, results->end(),
                              DistanceComparatorBranchOptimized());
    results->resize(final_size);
  }
}

void DeduplicateDatabaseSpilledResults(NNResultsVector* results,
                                       size_t final_size) {
  DeduplicateDatabaseSpilledResults<flat_hash_map<DatapointIndex, float>>(
      results, final_size);
}

}  // namespace research_scann
