// Copyright 2024 The Google Research Authors.
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

#ifndef SCANN_HASHES_INTERNAL_LUT16_INTERFACE_H_
#define SCANN_HASHES_INTERNAL_LUT16_INTERFACE_H_

#include <cstdint>
#include <utility>

#include "scann/hashes/internal/lut16_args.h"
#include "scann/hashes/internal/lut16_avx2.h"
#include "scann/hashes/internal/lut16_avx512.h"
#include "scann/hashes/internal/lut16_highway.h"
#include "scann/hashes/internal/lut16_sse4.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/utils/alignment.h"
#include "scann/utils/common.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann/utils/types.h"

namespace research_scann {
namespace asymmetric_hashing_internal {

class LUT16Interface {
 public:
  template <typename T>
  using Args = LUT16Args<T>;

  template <typename T, typename TopN>
  using ArgsTopN = LUT16ArgsTopN<T, TopN>;

  SCANN_INLINE static void GetDistances(LUT16Args<int16_t> args);

  SCANN_INLINE static void GetDistances(LUT16Args<int32_t> args);

  template <typename TopN>
  SCANN_INLINE static void GetTopDistances(LUT16ArgsTopN<int16_t, TopN> args);
  template <typename TopN>
  SCANN_INLINE static void GetTopFloatDistances(
      LUT16ArgsTopN<float, TopN> args);

  SCANN_INLINE static void GetFloatDistances(
      LUT16Args<float> args, ConstSpan<float> inv_fp_multipliers);

  template <typename DistT>
  SCANN_INLINE static void GetDistances(const uint8_t* packed_dataset,
                                        size_t num_32dp_simd_iters,
                                        size_t num_blocks,
                                        ConstSpan<const uint8_t*> lookups,
                                        ConstSpan<DistT*> distances) {
    LUT16Args<DistT> args;
    args.packed_dataset = packed_dataset;
    args.num_32dp_simd_iters = num_32dp_simd_iters;
    args.num_blocks = num_blocks;
    args.lookups = lookups;
    args.distances = distances;
    GetDistances(std::move(args));
  }

  template <typename DistT>
  SCANN_INLINE static void GetTopDistances(
      const uint8_t* packed_dataset, bool should_prefetch,
      size_t num_32dp_simd_iters, size_t num_blocks,
      ConstSpan<const uint8_t*> lookups, DatapointIndex first_dp_index,
      DatapointIndex num_datapoints,
      ConstSpan<FastTopNeighbors<DistT>*> fast_topns) {
    LUT16ArgsTopN<DistT> args;
    args.packed_dataset = packed_dataset;
    args.prefetch_strategy =
        should_prefetch ? PrefetchStrategy::kSeq : PrefetchStrategy::kOff;
    args.num_32dp_simd_iters = num_32dp_simd_iters;
    args.num_blocks = num_blocks;
    args.lookups = lookups;
    args.first_dp_index = first_dp_index;
    args.num_datapoints = num_datapoints;
    args.fast_topns = fast_topns;
    GetTopDistances(std::move(args));
  }

  template <typename DistT, size_t kNumQueries>
  SCANN_INLINE static void GetDistances(
      const uint8_t* packed_dataset, size_t num_32dp_simd_iters,
      size_t num_blocks, array<const uint8_t*, kNumQueries> lookups,
      array<DistT*, kNumQueries> distances) {
    GetDistances(packed_dataset, num_32dp_simd_iters, num_blocks,
                 MakeConstSpan(lookups), MakeConstSpan(distances));
  }

  template <typename DistT>
  SCANN_INLINE static void GetDistances(const uint8_t* packed_dataset,
                                        size_t num_32dp_simd_iters,
                                        size_t num_blocks,
                                        const uint8_t* lookups1,
                                        DistT* distances1) {
    GetDistances(packed_dataset, num_32dp_simd_iters, num_blocks,
                 ConstSpan<const uint8_t*>(&lookups1, 1),
                 ConstSpan<DistT*>(&distances1, 1));
  }

  template <typename DistT>
  SCANN_INLINE static void GetTopDistances(
      const uint8_t* packed_dataset, bool should_prefetch,
      size_t num_32dp_simd_iters, size_t num_blocks, const uint8_t* lookups1,
      DatapointIndex first_dp_index, DatapointIndex num_datapoints,
      FastTopNeighbors<DistT>* fast_topn) {
    GetTopDistances(packed_dataset, should_prefetch, num_32dp_simd_iters,
                    num_blocks, ConstSpan<const uint8_t*>(&lookups1, 1),
                    first_dp_index, num_datapoints,
                    ConstSpan<FastTopNeighbors<DistT>*>(&fast_topn, 1));
  }

  static AlignedBuffer PlatformSpecificSwizzle(const uint8_t* packed_dataset,
                                               int num_datapoints,
                                               int num_blocks);

  static void PlatformSpecificSwizzleInPlace(uint8_t* packed_dataset,
                                             int num_datapoints,
                                             int num_blocks);
};

#define SCANN_CALL_LUT16_FUNCTION_1(batch_size, kPrefetch, ClassName, \
                                    Function, ...)                    \
  switch (batch_size) {                                               \
    case 1:                                                           \
      return ClassName<1, kPrefetch>::Function(__VA_ARGS__);          \
    case 2:                                                           \
      return ClassName<2, kPrefetch>::Function(__VA_ARGS__);          \
    case 3:                                                           \
      return ClassName<3, kPrefetch>::Function(__VA_ARGS__);          \
    case 4:                                                           \
      return ClassName<4, kPrefetch>::Function(__VA_ARGS__);          \
    case 5:                                                           \
      return ClassName<5, kPrefetch>::Function(__VA_ARGS__);          \
    case 6:                                                           \
      return ClassName<6, kPrefetch>::Function(__VA_ARGS__);          \
    case 7:                                                           \
      return ClassName<7, kPrefetch>::Function(__VA_ARGS__);          \
    case 8:                                                           \
      return ClassName<8, kPrefetch>::Function(__VA_ARGS__);          \
    case 9:                                                           \
      return ClassName<9, kPrefetch>::Function(__VA_ARGS__);          \
    default:                                                          \
      DLOG(FATAL) << "Invalid Batch Size";                            \
  }

#ifdef __x86_64__

#define SCANN_CALL_LUT16_FUNCTION(enable_avx512_codepath, batch_size,   \
                                  prefetch_strategy, Function, ...)     \
  if (prefetch_strategy == PrefetchStrategy::kOff) {                    \
    if (enable_avx512_codepath && RuntimeSupportsAvx512()) {            \
      SCANN_CALL_LUT16_FUNCTION_1(batch_size, PrefetchStrategy::kOff,   \
                                  LUT16Avx512, Function, __VA_ARGS__);  \
    }                                                                   \
    if (RuntimeSupportsAvx2()) {                                        \
      SCANN_CALL_LUT16_FUNCTION_1(batch_size, PrefetchStrategy::kOff,   \
                                  LUT16Avx2, Function, __VA_ARGS__);    \
    } else {                                                            \
      SCANN_CALL_LUT16_FUNCTION_1(batch_size, PrefetchStrategy::kOff,   \
                                  LUT16Sse4, Function, __VA_ARGS__);    \
    }                                                                   \
  } else if (prefetch_strategy == PrefetchStrategy::kSeq) {             \
    if (enable_avx512_codepath && RuntimeSupportsAvx512()) {            \
      SCANN_CALL_LUT16_FUNCTION_1(batch_size, PrefetchStrategy::kSeq,   \
                                  LUT16Avx512, Function, __VA_ARGS__);  \
    }                                                                   \
    if (RuntimeSupportsAvx2()) {                                        \
      SCANN_CALL_LUT16_FUNCTION_1(batch_size, PrefetchStrategy::kSeq,   \
                                  LUT16Avx2, Function, __VA_ARGS__);    \
    } else {                                                            \
      SCANN_CALL_LUT16_FUNCTION_1(batch_size, PrefetchStrategy::kSeq,   \
                                  LUT16Sse4, Function, __VA_ARGS__);    \
    }                                                                   \
  } else {                                                              \
    if (enable_avx512_codepath && RuntimeSupportsAvx512()) {            \
      SCANN_CALL_LUT16_FUNCTION_1(batch_size, PrefetchStrategy::kSmart, \
                                  LUT16Avx512, Function, __VA_ARGS__);  \
    }                                                                   \
    if (RuntimeSupportsAvx2()) {                                        \
      SCANN_CALL_LUT16_FUNCTION_1(batch_size, PrefetchStrategy::kSmart, \
                                  LUT16Avx2, Function, __VA_ARGS__);    \
    } else {                                                            \
      SCANN_CALL_LUT16_FUNCTION_1(batch_size, PrefetchStrategy::kSmart, \
                                  LUT16Sse4, Function, __VA_ARGS__);    \
    }                                                                   \
  }

void LUT16Interface::GetDistances(LUT16Args<int16_t> args) {
  const size_t batch_size = args.lookups.size();
  const auto prefetch_strategy = args.prefetch_strategy;
  const bool enable_avx512_codepath = args.enable_avx512_codepath;
  DCHECK_EQ(batch_size, args.distances.size());
  SCANN_CALL_LUT16_FUNCTION(enable_avx512_codepath, batch_size,
                            prefetch_strategy, GetInt16Distances,
                            std::move(args));
}

void LUT16Interface::GetDistances(LUT16Args<int32_t> args) {
  const size_t batch_size = args.lookups.size();
  const auto prefetch_strategy = args.prefetch_strategy;
  const bool enable_avx512_codepath = args.enable_avx512_codepath;
  DCHECK_EQ(batch_size, args.distances.size());
  SCANN_CALL_LUT16_FUNCTION(enable_avx512_codepath, batch_size,
                            prefetch_strategy, GetInt32Distances,
                            std::move(args));
}

void LUT16Interface::GetFloatDistances(LUT16Args<float> args,
                                       ConstSpan<float> inv_fp_multipliers) {
  const size_t batch_size = args.lookups.size();
  const auto prefetch_strategy = args.prefetch_strategy;
  const bool enable_avx512_codepath = args.enable_avx512_codepath;
  DCHECK_EQ(batch_size, args.distances.size());
  DCHECK_EQ(batch_size, inv_fp_multipliers.size());
  SCANN_CALL_LUT16_FUNCTION(enable_avx512_codepath, batch_size,
                            prefetch_strategy, GetFloatDistances,
                            std::move(args), inv_fp_multipliers);
}

template <typename TopN>
void LUT16Interface::GetTopDistances(LUT16ArgsTopN<int16_t, TopN> args) {
  const size_t batch_size = args.lookups.size();
  const auto prefetch_strategy = args.prefetch_strategy;
  const bool enable_avx512_codepath = args.enable_avx512_codepath;
  DCHECK_EQ(batch_size, args.fast_topns.size());
  SCANN_CALL_LUT16_FUNCTION(enable_avx512_codepath, batch_size,
                            prefetch_strategy, GetTopInt16Distances,
                            std::move(args));
}

template <typename TopN>
void LUT16Interface::GetTopFloatDistances(LUT16ArgsTopN<float, TopN> args) {
  const size_t batch_size = args.lookups.size();
  const auto prefetch_strategy = args.prefetch_strategy;
  const bool enable_avx512_codepath = args.enable_avx512_codepath;
  DCHECK_EQ(batch_size, args.fast_topns.size());
  DCHECK_EQ(batch_size, args.biases.size());
  SCANN_CALL_LUT16_FUNCTION(enable_avx512_codepath, batch_size,
                            prefetch_strategy, GetTopFloatDistances,
                            std::move(args));
}

#else
#define SCANN_CALL_LUT16_FUNCTION(batch_size, prefetch_strategy, Function, \
                                  ...)                                     \
  if (prefetch_strategy == PrefetchStrategy::kOff) {                       \
    SCANN_CALL_LUT16_FUNCTION_1(batch_size, PrefetchStrategy::kOff,        \
                                LUT16Highway, Function, __VA_ARGS__);      \
  } else if (prefetch_strategy == PrefetchStrategy::kSeq) {                \
    SCANN_CALL_LUT16_FUNCTION_1(batch_size, PrefetchStrategy::kSeq,        \
                                LUT16Highway, Function, __VA_ARGS__);      \
  } else {                                                                 \
    SCANN_CALL_LUT16_FUNCTION_1(batch_size, PrefetchStrategy::kSmart,      \
                                LUT16Highway, Function, __VA_ARGS__);      \
  }

void LUT16Interface::GetDistances(LUT16Args<int16_t> args) {
  const size_t batch_size = args.lookups.size();
  const auto prefetch_strategy = args.prefetch_strategy;
  DCHECK_EQ(batch_size, args.distances.size());
  SCANN_CALL_LUT16_FUNCTION(batch_size, prefetch_strategy, GetInt16Distances,
                            std::move(args));
}

void LUT16Interface::GetDistances(LUT16Args<int32_t> args) {
  const size_t batch_size = args.lookups.size();
  const auto prefetch_strategy = args.prefetch_strategy;
  DCHECK_EQ(batch_size, args.distances.size());
  SCANN_CALL_LUT16_FUNCTION(batch_size, prefetch_strategy, GetInt32Distances,
                            std::move(args));
}

void LUT16Interface::GetFloatDistances(LUT16Args<float> args,
                                       ConstSpan<float> inv_fp_multipliers) {
  const size_t batch_size = args.lookups.size();
  const auto prefetch_strategy = args.prefetch_strategy;
  DCHECK_EQ(batch_size, args.distances.size());
  DCHECK_EQ(batch_size, inv_fp_multipliers.size());
  SCANN_CALL_LUT16_FUNCTION(batch_size, prefetch_strategy, GetFloatDistances,
                            std::move(args), inv_fp_multipliers);
}

template <typename TopN>
void LUT16Interface::GetTopDistances(LUT16ArgsTopN<int16_t, TopN> args) {
  const size_t batch_size = args.lookups.size();
  const auto prefetch_strategy = args.prefetch_strategy;
  DCHECK_EQ(batch_size, args.fast_topns.size());
  SCANN_CALL_LUT16_FUNCTION(batch_size, prefetch_strategy, GetTopInt16Distances,
                            std::move(args));
}

template <typename TopN>
void LUT16Interface::GetTopFloatDistances(LUT16ArgsTopN<float, TopN> args) {
  const size_t batch_size = args.lookups.size();
  const auto prefetch_strategy = args.prefetch_strategy;
  DCHECK_EQ(batch_size, args.fast_topns.size());
  DCHECK_EQ(batch_size, args.biases.size());
  SCANN_CALL_LUT16_FUNCTION(batch_size, prefetch_strategy, GetTopFloatDistances,
                            std::move(args));
}

#endif

}  // namespace asymmetric_hashing_internal
}  // namespace research_scann

#endif
