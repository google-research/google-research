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

/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SCANN__HASHES_INTERNAL_LUT16_INTERFACE_H_
#define SCANN__HASHES_INTERNAL_LUT16_INTERFACE_H_

#include "scann/hashes/internal/lut16_args.h"
#include "scann/hashes/internal/lut16_avx2.h"
#include "scann/hashes/internal/lut16_sse4.h"
#include "scann/utils/common.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann/utils/types.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
namespace scann_ops {
namespace asymmetric_hashing_internal {

class LUT16Interface {
 public:
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
    args.should_prefetch = should_prefetch;
    args.num_32dp_simd_iters = num_32dp_simd_iters;
    args.num_blocks = num_blocks;
    args.lookups = lookups;
    args.first_dp_index = first_dp_index;
    args.num_datapoints = num_datapoints;
    args.fast_topns = fast_topns;
    GetTopDistances(std::move(args));
  }

  SCANN_INLINE static void GetTopFloatDistances(
      const uint8_t* packed_dataset, bool should_prefetch,
      size_t num_32dp_simd_iters, size_t num_blocks,
      ConstSpan<const uint8_t*> lookups, DatapointIndex first_dp_index,
      DatapointIndex num_datapoints,
      ConstSpan<FastTopNeighbors<float>*> fast_topns, ConstSpan<float> biases,
      ConstSpan<float> fixed_point_multipliers) {
    LUT16ArgsTopN<float> args;
    args.packed_dataset = packed_dataset;
    args.should_prefetch = should_prefetch;
    args.num_32dp_simd_iters = num_32dp_simd_iters;
    args.num_blocks = num_blocks;
    args.lookups = lookups;
    args.first_dp_index = first_dp_index;
    args.num_datapoints = num_datapoints;
    args.fast_topns = fast_topns;
    args.biases = biases;
    args.fixed_point_multipliers = fixed_point_multipliers;
    GetTopFloatDistances(std::move(args));
  }

  template <typename DistT, size_t kBatchSize>
  SCANN_INLINE static void GetDistances(
      const uint8_t* packed_dataset, size_t num_32dp_simd_iters,
      size_t num_blocks, array<const uint8_t*, kBatchSize> lookups,
      array<DistT*, kBatchSize> distances) {
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

  SCANN_INLINE static void GetTopFloatDistances(
      const uint8_t* packed_dataset, bool should_prefetch,
      size_t num_32dp_simd_iters, size_t num_blocks, const uint8_t* lookups1,
      DatapointIndex first_dp_index, DatapointIndex num_datapoints,
      FastTopNeighbors<float>* fast_topn, float bias,
      float fixed_point_multiplier) {
    GetTopFloatDistances(packed_dataset, should_prefetch, num_32dp_simd_iters,
                         num_blocks, ConstSpan<const uint8_t*>(&lookups1, 1),
                         first_dp_index, num_datapoints,
                         ConstSpan<FastTopNeighbors<float>*>(&fast_topn, 1),
                         ConstSpan<float>(&bias, 1),
                         ConstSpan<float>(&fixed_point_multiplier, 1));
  }
};

#ifdef __x86_64__

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
      LOG(FATAL) << "Invalid Batch Size";                             \
  }

#define SCANN_CALL_LUT16_FUNCTION(batch_size, should_prefetch, Function, ...) \
  if (should_prefetch) {                                                      \
    if (RuntimeSupportsAvx2()) {                                              \
      SCANN_CALL_LUT16_FUNCTION_1(batch_size, true, LUT16Avx2, Function,      \
                                  __VA_ARGS__);                               \
    } else {                                                                  \
      SCANN_CALL_LUT16_FUNCTION_1(batch_size, true, LUT16Sse4, Function,      \
                                  __VA_ARGS__);                               \
    }                                                                         \
  } else {                                                                    \
    if (RuntimeSupportsAvx2()) {                                              \
      SCANN_CALL_LUT16_FUNCTION_1(batch_size, false, LUT16Avx2, Function,     \
                                  __VA_ARGS__);                               \
    } else {                                                                  \
      SCANN_CALL_LUT16_FUNCTION_1(batch_size, false, LUT16Sse4, Function,     \
                                  __VA_ARGS__);                               \
    }                                                                         \
  }

void LUT16Interface::GetDistances(LUT16Args<int16_t> args) {
  const size_t batch_size = args.lookups.size();
  const bool should_prefetch = args.should_prefetch;
  DCHECK_EQ(batch_size, args.distances.size());
  SCANN_CALL_LUT16_FUNCTION(batch_size, should_prefetch, GetInt16Distances,
                            std::move(args));
}

void LUT16Interface::GetDistances(LUT16Args<int32_t> args) {
  const size_t batch_size = args.lookups.size();
  const bool should_prefetch = args.should_prefetch;
  DCHECK_EQ(batch_size, args.distances.size());
  SCANN_CALL_LUT16_FUNCTION(batch_size, should_prefetch, GetInt32Distances,
                            std::move(args));
}

void LUT16Interface::GetFloatDistances(LUT16Args<float> args,
                                       ConstSpan<float> inv_fp_multipliers) {
  const size_t batch_size = args.lookups.size();
  const bool should_prefetch = args.should_prefetch;
  DCHECK_EQ(batch_size, args.distances.size());
  DCHECK_EQ(batch_size, inv_fp_multipliers.size());
  SCANN_CALL_LUT16_FUNCTION(batch_size, should_prefetch, GetFloatDistances,
                            std::move(args), inv_fp_multipliers);
}

template <typename TopN>
void LUT16Interface::GetTopDistances(LUT16ArgsTopN<int16_t, TopN> args) {
  const size_t batch_size = args.lookups.size();
  const bool should_prefetch = args.should_prefetch;
  DCHECK_EQ(batch_size, args.fast_topns.size());
  SCANN_CALL_LUT16_FUNCTION(batch_size, should_prefetch, GetTopInt16Distances,
                            std::move(args));
}

template <typename TopN>
void LUT16Interface::GetTopFloatDistances(LUT16ArgsTopN<float, TopN> args) {
  const size_t batch_size = args.lookups.size();
  const bool should_prefetch = args.should_prefetch;
  DCHECK_EQ(batch_size, args.fast_topns.size());
  DCHECK_EQ(batch_size, args.biases.size());
  SCANN_CALL_LUT16_FUNCTION(batch_size, should_prefetch, GetTopFloatDistances,
                            std::move(args));
}

#else

void LUT16Interface::GetDistances(LUT16Args<int16_t> args) {
  LOG(FATAL) << "LUT16 is only supported on x86!";
}

void LUT16Interface::GetDistances(LUT16Args<int32_t> args) {
  LOG(FATAL) << "LUT16 is only supported on x86!";
}

void LUT16Interface::GetFloatDistances(LUT16Args<float> args,
                                       ConstSpan<float> inv_fp_multipliers) {
  LOG(FATAL) << "LUT16 is only supported on x86!";
}

template <typename TopN>
void LUT16Interface::GetTopDistances(LUT16ArgsTopN<int16_t, TopN> args) {
  LOG(FATAL) << "LUT16 is only supported on x86!";
}

template <typename TopN>
void LUT16Interface::GetTopFloatDistances(LUT16ArgsTopN<float, TopN> args) {
  LOG(FATAL) << "LUT16 is only supported on x86!";
}

#endif

}  // namespace asymmetric_hashing_internal
}  // namespace scann_ops
}  // namespace tensorflow

#endif
