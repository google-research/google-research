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

#ifndef SCANN__HASHES_INTERNAL_LUT16_ARGS_H_
#define SCANN__HASHES_INTERNAL_LUT16_ARGS_H_

#include "scann/base/restrict_allowlist.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {
namespace asymmetric_hashing_internal {

template <typename DistT>
struct LUT16ArgsBase {
  SCANN_DECLARE_COPYABLE_CLASS(LUT16ArgsBase);
  static_assert(IsSameAny<DistT, int16_t, int32_t, float>(), "");

  LUT16ArgsBase() {}

  const uint8_t* packed_dataset = nullptr;

  int enable_avx512_codepath = 0;

  size_t num_32dp_simd_iters = 0;

  size_t num_blocks = 0;

  ConstSpan<const uint8_t*> lookups;

  bool should_prefetch = true;
};

template <typename DistT>
struct LUT16Args : public LUT16ArgsBase<DistT> {
  ConstSpan<DistT*> distances;
};

template <typename DistT, typename TopN>
struct LUT16ArgsTopNBase : public LUT16Args<DistT> {
  DatapointIndex first_dp_index = 0;

  DatapointIndex num_datapoints = 0;

  ConstSpan<TopN*> fast_topns;

  ConstSpan<RestrictWhitelistConstView> restrict_whitelists;

  template <size_t kNumQueries>
  SCANN_INLINE array<const uint32_t*, kNumQueries> GetRestrictWhitelistPtrs()
      const {
    array<const uint32_t*, kNumQueries> restrict_whitelist_ptrs;
    if (restrict_whitelists.empty()) {
      std::fill(restrict_whitelist_ptrs.begin(), restrict_whitelist_ptrs.end(),
                nullptr);
      return restrict_whitelist_ptrs;
    }
    DCHECK_EQ(restrict_whitelists.size(), kNumQueries);

    for (size_t i : Seq(kNumQueries)) {
      restrict_whitelist_ptrs[i] =
          reinterpret_cast<const uint32_t*>(restrict_whitelists[i].data());
      if (!restrict_whitelists[i].empty()) {
        DCHECK_EQ(restrict_whitelists[i].size(), num_datapoints);
      }
    }
    return restrict_whitelist_ptrs;
  }
};

template <typename Dist, typename TopN = FastTopNeighbors<Dist>>
struct LUT16ArgsTopN : public LUT16ArgsTopNBase<Dist, TopN> {};

template <typename TopN>
struct LUT16ArgsTopN<float, TopN> : public LUT16ArgsTopNBase<float, TopN> {
  ConstSpan<float> biases;

  ConstSpan<float> fixed_point_multipliers;

  std::function<bool(DatapointIndex)> final_predicate;
};

#define SCANN_INSTANTIATE_CLASS_FOR_LUT16_BATCH_SIZES(EXTERN_KEYWORD, \
                                                      ClassName)      \
  EXTERN_KEYWORD template class ClassName<1, true>;                   \
  EXTERN_KEYWORD template class ClassName<2, true>;                   \
  EXTERN_KEYWORD template class ClassName<3, true>;                   \
  EXTERN_KEYWORD template class ClassName<4, true>;                   \
  EXTERN_KEYWORD template class ClassName<5, true>;                   \
  EXTERN_KEYWORD template class ClassName<6, true>;                   \
  EXTERN_KEYWORD template class ClassName<7, true>;                   \
  EXTERN_KEYWORD template class ClassName<8, true>;                   \
  EXTERN_KEYWORD template class ClassName<9, true>;                   \
  EXTERN_KEYWORD template class ClassName<1, false>;                  \
  EXTERN_KEYWORD template class ClassName<2, false>;                  \
  EXTERN_KEYWORD template class ClassName<3, false>;                  \
  EXTERN_KEYWORD template class ClassName<4, false>;                  \
  EXTERN_KEYWORD template class ClassName<5, false>;                  \
  EXTERN_KEYWORD template class ClassName<6, false>;                  \
  EXTERN_KEYWORD template class ClassName<7, false>;                  \
  EXTERN_KEYWORD template class ClassName<8, false>;                  \
  EXTERN_KEYWORD template class ClassName<9, false>;

}  // namespace asymmetric_hashing_internal
}  // namespace scann_ops
}  // namespace tensorflow

#endif
