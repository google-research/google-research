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

// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "scann/hashes/internal/lut16_sse4.h"
#include "scann/utils/common.h"

#ifdef __x86_64__

#include "scann/utils/bits.h"
#include "scann/utils/intrinsics/sse4.h"
#include "tensorflow/core/platform/prefetch.h"

namespace tensorflow {
namespace scann_ops {
namespace asymmetric_hashing_internal {
namespace {

struct Accum32Int16s {
  M128_8Xint16 acc00 = _mm_setzero_si128();
  M128_8Xint16 acc08 = _mm_setzero_si128();
  M128_8Xint16 acc16 = _mm_setzero_si128();
  M128_8Xint16 acc24 = _mm_setzero_si128();
};

struct Accum32Int32s {
  M128_4Xint32 acc00 = _mm_setzero_si128();
  M128_4Xint32 acc04 = _mm_setzero_si128();
  M128_4Xint32 acc08 = _mm_setzero_si128();
  M128_4Xint32 acc12 = _mm_setzero_si128();
  M128_4Xint32 acc16 = _mm_setzero_si128();
  M128_4Xint32 acc20 = _mm_setzero_si128();
  M128_4Xint32 acc24 = _mm_setzero_si128();
  M128_4Xint32 acc28 = _mm_setzero_si128();
};

template <size_t size, typename T>
SCANN_INLINE array<T, size> ToLocalArray(ConstSpan<T> span) {
  DCHECK_EQ(span.size(), size);
  array<T, size> result;
  std::copy(span.begin(), span.begin() + size, result.begin());
  return result;
}

template <size_t kBatchSize, bool kPrefetch>
SCANN_SSE4_INLINE array<Accum32Int16s, kBatchSize> Sse4LUT16BottomLoop(
    const uint8_t* data_start, array<const uint8_t*, kBatchSize> lookup_starts,
    DimensionIndex num_blocks) {
  static_assert(kBatchSize <= 3,
                "Register spilling happens when kBatchSize > 3");
  array<Accum32Int16s, kBatchSize> result;
  auto sign7 = M128_16Xuint8::Broadcast(0x0f);
  const auto total_bias =
      M128_8Xint16::Broadcast(static_cast<int16_t>(num_blocks * 128));
  for (; num_blocks != 0; --num_blocks, data_start += 16) {
    if (kPrefetch) {
      ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_NTA>(
          data_start + 768);
    }

    auto mask = M128_16Xuint8::Load(data_start);
    M128_16Xuint8 mask1 =
        M128_16Xuint8((M128_8Xuint16(mask.val()) >> 4).val()) & sign7;
    M128_16Xuint8 mask0 = mask & sign7;
    for (size_t j : Seq(kBatchSize)) {
      const M128_16Xuint8 dict = M128_16Xuint8::Load(lookup_starts[j]);
      lookup_starts[j] += 16;
      const M128_16Xuint8 res0 = dict.Perform16LUT16Lookups(mask0);
      const M128_16Xuint8 res1 = dict.Perform16LUT16Lookups(mask1);

      result[j].acc00 += M128_8Xint16(res0.val());
      result[j].acc08 += res0.ExtractOddAsInt16s();
      result[j].acc16 += M128_8Xint16(res1.val());
      result[j].acc24 += res1.ExtractOddAsInt16s();
    }
  }

  for (size_t j : Seq(kBatchSize)) {
    result[j].acc00 -= result[j].acc08 << 8;
    result[j].acc16 -= result[j].acc24 << 8;
    auto bottom0 = result[j].acc00.InterleaveBottom(result[j].acc08);
    auto top0 = result[j].acc00.InterleaveTop(result[j].acc08);
    bottom0 -= total_bias;
    top0 -= total_bias;
    result[j].acc00 = bottom0;
    result[j].acc08 = top0;
    auto bottom1 = result[j].acc16.InterleaveBottom(result[j].acc24);
    auto top1 = result[j].acc16.InterleaveTop(result[j].acc24);
    bottom1 -= total_bias;
    top1 -= total_bias;
    result[j].acc16 = bottom1;
    result[j].acc24 = top1;
  }

  return result;
}

template <size_t kBottomLevelBatchSize, size_t kBatchSize>
SCANN_SSE4_INLINE array<const uint8_t*, kBottomLevelBatchSize>
MakeBottomLevelBatchLookupArray(
    array<const uint8_t*, kBatchSize> mid_level_lookups, size_t start) {
  DCHECK_LE(start + kBottomLevelBatchSize, kBatchSize);
  array<const uint8_t*, kBottomLevelBatchSize> result;
  for (size_t j : Seq(kBottomLevelBatchSize)) {
    result[j] = mid_level_lookups[start + j];
  }
  return result;
}

template <size_t kBatchSize, bool kPrefetch>
SCANN_SSE4_INLINE array<Accum32Int16s, kBatchSize> Sse4LUT16MiddleLoop(
    const uint8_t* data_start, array<const uint8_t*, kBatchSize> lookup_starts,
    const DimensionIndex num_blocks) {
  constexpr size_t kSizeB = (kBatchSize == 1) ? 1 : 2;
  constexpr size_t kNumBCases[] = {0, 2, 1};
  constexpr size_t kNumB = (kBatchSize == 1) ? 1 : kNumBCases[kBatchSize % 3];

  constexpr size_t kRemaining = kBatchSize - kNumB * kSizeB;
  static_assert(kRemaining % 3 == 0, "");

  constexpr size_t kSizeA = 3;
  constexpr size_t kNumA = kRemaining / 3;

  array<Accum32Int16s, kBatchSize> result;
  for (size_t j : Seq(kNumA)) {
    const size_t start = j * kSizeA;
    auto bottom_level_lookups =
        MakeBottomLevelBatchLookupArray<kSizeA>(lookup_starts, start);
    auto acc = Sse4LUT16BottomLoop<kSizeA, kPrefetch>(
        data_start, bottom_level_lookups, num_blocks);
    for (size_t jj : Seq(kSizeA)) {
      result[start + jj] = acc[jj];
    }
  }

  for (size_t j : Seq(kNumB)) {
    const size_t start = kNumA * kSizeA + j * kSizeB;
    auto bottom_level_lookups =
        MakeBottomLevelBatchLookupArray<kSizeB>(lookup_starts, start);
    auto acc = Sse4LUT16BottomLoop<kSizeB, kPrefetch>(
        data_start, bottom_level_lookups, num_blocks);
    for (size_t jj : Seq(kSizeB)) {
      result[start + jj] = acc[jj];
    }
  }
  return result;
}

template <size_t kBatchSize, bool kPrefetch>
SCANN_SSE4_INLINE array<Accum32Int32s, kBatchSize> Sse4LUT16BottomLoopInt32(
    const uint8_t* data_start, array<const uint8_t*, kBatchSize> lookup_starts,
    DimensionIndex num_blocks) {
  array<Accum32Int32s, kBatchSize> int32_accumulators;
  for (DimensionIndex k = 0; k < num_blocks;) {
    DimensionIndex reaccumulate_limit = std::min(num_blocks - k, unsigned long long_t{256});
    k += reaccumulate_limit;
    auto int16_accumulators = Sse4LUT16MiddleLoop<kBatchSize, kPrefetch>(
        data_start, lookup_starts, reaccumulate_limit);
    data_start += 16 * reaccumulate_limit;
    for (size_t j : Seq(kBatchSize)) {
      lookup_starts[j] += 16 * reaccumulate_limit;
      const auto& int16_accums = int16_accumulators[j];
      auto& int32_accums = int32_accumulators[j];
      int32_accums.acc00 += int16_accums.acc00.ExtractBotAsInt32s();
      int32_accums.acc04 += int16_accums.acc00.ExtractTopAsInt32s();
      int32_accums.acc08 += int16_accums.acc08.ExtractBotAsInt32s();
      int32_accums.acc12 += int16_accums.acc08.ExtractTopAsInt32s();
      int32_accums.acc16 += int16_accums.acc16.ExtractBotAsInt32s();
      int32_accums.acc20 += int16_accums.acc16.ExtractTopAsInt32s();
      int32_accums.acc24 += int16_accums.acc24.ExtractBotAsInt32s();
      int32_accums.acc28 += int16_accums.acc24.ExtractTopAsInt32s();
    }
  }
  return int32_accumulators;
}

}  // namespace

template <size_t kBatchSize, bool kPrefetch>
SCANN_SSE4_OUTLINE void LUT16Sse4<kBatchSize, kPrefetch>::GetInt16Distances(
    LUT16Args<int16_t> args) {
  const uint8_t* packed_dataset = args.packed_dataset;
  const size_t num_32dp_simd_iters = args.num_32dp_simd_iters;
  const size_t num_blocks = args.num_blocks;
  auto lookups = ToLocalArray<kBatchSize>(args.lookups);
  auto distances = ToLocalArray<kBatchSize>(args.distances);
  for (size_t k : Seq(num_32dp_simd_iters)) {
    const size_t dp_idx = k * 32;

    const uint8_t* data_start = packed_dataset + dp_idx * num_blocks / 2;
    auto int16_accumulators = Sse4LUT16MiddleLoop<kBatchSize, kPrefetch>(
        data_start, lookups, num_blocks);
    for (size_t j : Seq(kBatchSize)) {
      auto int16_accums = int16_accumulators[j];
      int16_accums.acc00.Store(distances[j] + dp_idx);
      int16_accums.acc08.Store(distances[j] + dp_idx + 8);
      int16_accums.acc16.Store(distances[j] + dp_idx + 16);
      int16_accums.acc24.Store(distances[j] + dp_idx + 24);
    }
  }
}

template <size_t kBatchSize, bool kPrefetch>
SCANN_SSE4_OUTLINE void LUT16Sse4<kBatchSize, kPrefetch>::GetInt32Distances(
    LUT16Args<int32_t> args) {
  const uint8_t* packed_dataset = args.packed_dataset;
  const size_t num_32dp_simd_iters = args.num_32dp_simd_iters;
  const size_t num_blocks = args.num_blocks;
  auto lookups = ToLocalArray<kBatchSize>(args.lookups);
  auto distances = ToLocalArray<kBatchSize>(args.distances);
  for (DatapointIndex k = 0; k < num_32dp_simd_iters; k++) {
    const uint8_t* data_start = packed_dataset + k * 16 * num_blocks;
    auto int32_accumulators = Sse4LUT16BottomLoopInt32<kBatchSize, kPrefetch>(
        data_start, lookups, num_blocks);
    for (size_t j : Seq(kBatchSize)) {
      const auto& int32_accums = int32_accumulators[j];
      int32_accums.acc00.Store(distances[j] + 32 * k);
      int32_accums.acc04.Store(distances[j] + 32 * k + 4);
      int32_accums.acc08.Store(distances[j] + 32 * k + 8);
      int32_accums.acc12.Store(distances[j] + 32 * k + 12);
      int32_accums.acc16.Store(distances[j] + 32 * k + 16);
      int32_accums.acc20.Store(distances[j] + 32 * k + 20);
      int32_accums.acc24.Store(distances[j] + 32 * k + 24);
      int32_accums.acc28.Store(distances[j] + 32 * k + 28);
    }
  }
}

template <size_t kBatchSize, bool kPrefetch>
SCANN_SSE4_OUTLINE void LUT16Sse4<kBatchSize, kPrefetch>::GetFloatDistances(
    LUT16Args<float> args, ConstSpan<float> inv_fp_multipliers) {
  const uint8_t* packed_dataset = args.packed_dataset;
  const size_t num_32dp_simd_iters = args.num_32dp_simd_iters;
  const size_t num_blocks = args.num_blocks;
  auto lookups = ToLocalArray<kBatchSize>(args.lookups);
  auto distances = ToLocalArray<kBatchSize>(args.distances);
  auto mults = ToLocalArray<kBatchSize>(inv_fp_multipliers);

  for (DatapointIndex k = 0; k < num_32dp_simd_iters; k++) {
    const uint8_t* data_start = packed_dataset + k * 16 * num_blocks;
    auto int32_accumulators = Sse4LUT16BottomLoopInt32<kBatchSize, kPrefetch>(
        data_start, lookups, num_blocks);
    for (size_t j : Seq(kBatchSize)) {
      const auto& int32_accums = int32_accumulators[j];
      float* d = distances[j];
      const M128_4Xfloat mult = M128_4Xfloat::Broadcast(mults[j]);
      (int32_accums.acc00.ConvertToFloat() * mult).Store(d + 32 * k + 0);
      (int32_accums.acc04.ConvertToFloat() * mult).Store(d + 32 * k + 4);
      (int32_accums.acc08.ConvertToFloat() * mult).Store(d + 32 * k + 8);
      (int32_accums.acc12.ConvertToFloat() * mult).Store(d + 32 * k + 12);
      (int32_accums.acc16.ConvertToFloat() * mult).Store(d + 32 * k + 16);
      (int32_accums.acc20.ConvertToFloat() * mult).Store(d + 32 * k + 20);
      (int32_accums.acc24.ConvertToFloat() * mult).Store(d + 32 * k + 24);
      (int32_accums.acc28.ConvertToFloat() * mult).Store(d + 32 * k + 28);
    }
  }
}

namespace {
template <size_t kBatchSize, bool kPrefetch, typename TopN>
SCANN_SSE4_INLINE void GetTopInt16DistancesImpl(
    LUT16ArgsTopN<int16_t, TopN> args) {
  const uint8_t* packed_dataset = args.packed_dataset;
  const size_t num_32dp_simd_iters = args.num_32dp_simd_iters;
  const size_t num_blocks = args.num_blocks;
  auto lookups = ToLocalArray<kBatchSize>(args.lookups);
  const DatapointIndex first_dp_index = args.first_dp_index;
  const uint32_t final_mask = GetFinalMask32(args.num_datapoints);
  DCHECK_EQ(num_32dp_simd_iters, DivRoundUp(args.num_datapoints, 32));

  M128_8Xint16 simd_thresholds[kBatchSize];
  for (size_t j : Seq(kBatchSize)) {
    const int16_t int16_threshold = args.fast_topns[j]->epsilon();
    simd_thresholds[j] = M128_8Xint16::Broadcast(int16_threshold);
  }

  typename TopN::Mutator topn_mutators[kBatchSize];
  for (size_t j : Seq(kBatchSize)) {
    args.fast_topns[j]->AcquireMutator(&topn_mutators[j]);
  }

  int16_t distances_buffer[32];
  auto restrict_whitelist_ptrs =
      args.template GetRestrictWhitelistPtrs<kBatchSize>();
  for (DatapointIndex k : Seq(num_32dp_simd_iters)) {
    const uint8_t* data_start = packed_dataset + k * 16 * num_blocks;
    auto int16_accumulators = Sse4LUT16MiddleLoop<kBatchSize, kPrefetch>(
        data_start, lookups, num_blocks);
    for (size_t j : Seq(kBatchSize)) {
      auto& int16_accums = int16_accumulators[j];

      auto compute_push_mask = [&]() SCANN_INLINE_LAMBDA {
        auto cmp00 = (int16_accums.acc00 < simd_thresholds[j]);
        auto cmp08 = (int16_accums.acc08 < simd_thresholds[j]);
        auto cmp16 = (int16_accums.acc16 < simd_thresholds[j]);
        auto cmp24 = (int16_accums.acc24 < simd_thresholds[j]);
        return MaskFromHighBits(cmp00, cmp08, cmp16, cmp24);
      };
      uint32_t push_mask = compute_push_mask();

      if (!push_mask) continue;

      int16_accums.acc00.Store(distances_buffer + 0);
      int16_accums.acc08.Store(distances_buffer + 8);
      int16_accums.acc16.Store(distances_buffer + 16);
      int16_accums.acc24.Store(distances_buffer + 24);

      if (k == num_32dp_simd_iters - 1) {
        push_mask &= final_mask;
      }
      if (restrict_whitelist_ptrs[j]) {
        push_mask &= restrict_whitelist_ptrs[j][k];
      }

      while (push_mask) {
        const int offset = bits::FindLSBSetNonZero(push_mask);
        push_mask &= (push_mask - 1);
        const DatapointIndex dp_idx = first_dp_index + 32 * k + offset;
        DCHECK(
            !restrict_whitelist_ptrs[j] ||
            args.restrict_whitelists[j].IsWhitelisted(dp_idx - first_dp_index))
            << dp_idx;
        const int16_t distance = distances_buffer[offset];
        const bool needs_collection = topn_mutators[j].Push(dp_idx, distance);
        if (ABSL_PREDICT_FALSE(needs_collection)) {
          topn_mutators[j].GarbageCollect();

          simd_thresholds[j] =
              M128_8Xint16::Broadcast(topn_mutators[j].epsilon());

          push_mask &= compute_push_mask();
        }
      }
    }
  }
}
}  // namespace

template <size_t kBatchSize, bool kPrefetch>
SCANN_SSE4_OUTLINE void LUT16Sse4<kBatchSize, kPrefetch>::GetTopInt16Distances(
    LUT16ArgsTopN<int16_t> args) {
  return GetTopInt16DistancesImpl<kBatchSize, kPrefetch>(std::move(args));
}

SCANN_SSE4_INLINE int16_t GetInt16Threshold(float float_threshold) {
  constexpr float kMaxValue = numeric_limits<int16_t>::max();

  return std::min(float_threshold, kMaxValue);
}

namespace {
template <size_t kBatchSize, bool kPrefetch, typename TopN>
SCANN_SSE4_INLINE void GetTopFloatDistancesImpl(
    LUT16ArgsTopN<float, TopN> args) {
  const uint8_t* packed_dataset = args.packed_dataset;
  const size_t num_32dp_simd_iters = args.num_32dp_simd_iters;
  const size_t num_blocks = args.num_blocks;
  auto lookups = ToLocalArray<kBatchSize>(args.lookups);
  const DatapointIndex first_dp_index = args.first_dp_index;
  const uint32_t final_mask = GetFinalMask32(args.num_datapoints);
  DCHECK_EQ(num_32dp_simd_iters, DivRoundUp(args.num_datapoints, 32));

  auto biases = ToLocalArray<kBatchSize>(args.biases);
  M128_4Xfloat simd_biases[kBatchSize];
  for (size_t j : Seq(kBatchSize)) {
    simd_biases[j] = M128_4Xfloat::Broadcast(biases[j]);
  }

  auto mults = ToLocalArray<kBatchSize>(args.fixed_point_multipliers);
  M128_4Xfloat inv_mults[kBatchSize];
  for (size_t j : Seq(kBatchSize)) {
    inv_mults[j] = M128_4Xfloat::Broadcast(1.0 / mults[j]);
  }

  M128_8Xint16 simd_thresholds[kBatchSize];
  for (size_t j : Seq(kBatchSize)) {
    const float epsilon = args.fast_topns[j]->epsilon();
    const float float_threshold = (epsilon - biases[j]) * mults[j];
    const int16_t int16_threshold = GetInt16Threshold(float_threshold);
    simd_thresholds[j] = M128_8Xint16::Broadcast(int16_threshold);
  }

  typename TopN::Mutator topn_mutators[kBatchSize];
  for (size_t j : Seq(kBatchSize)) {
    args.fast_topns[j]->AcquireMutator(&topn_mutators[j]);
  }

  float distances_buffer[32];
  auto restrict_whitelist_ptrs =
      args.template GetRestrictWhitelistPtrs<kBatchSize>();
  for (DatapointIndex k : Seq(num_32dp_simd_iters)) {
    const uint8_t* data_start = packed_dataset + k * 16 * num_blocks;
    auto int16_accumulators = Sse4LUT16MiddleLoop<kBatchSize, kPrefetch>(
        data_start, lookups, num_blocks);
    for (size_t j : Seq(kBatchSize)) {
      auto& int16_accums = int16_accumulators[j];

      auto compute_push_mask = [&]() SCANN_INLINE_LAMBDA {
        auto cmp00 = (int16_accums.acc00 < simd_thresholds[j]);
        auto cmp08 = (int16_accums.acc08 < simd_thresholds[j]);
        auto cmp16 = (int16_accums.acc16 < simd_thresholds[j]);
        auto cmp24 = (int16_accums.acc24 < simd_thresholds[j]);
        return MaskFromHighBits(cmp00, cmp08, cmp16, cmp24);
      };
      uint32_t push_mask = compute_push_mask();

      if (!push_mask) continue;

      auto fvals00 = int16_accums.acc00.ExtractBotAsInt32s().ConvertToFloat();
      auto fvals04 = int16_accums.acc00.ExtractTopAsInt32s().ConvertToFloat();
      auto fvals08 = int16_accums.acc08.ExtractBotAsInt32s().ConvertToFloat();
      auto fvals12 = int16_accums.acc08.ExtractTopAsInt32s().ConvertToFloat();
      auto fvals16 = int16_accums.acc16.ExtractBotAsInt32s().ConvertToFloat();
      auto fvals20 = int16_accums.acc16.ExtractTopAsInt32s().ConvertToFloat();
      auto fvals24 = int16_accums.acc24.ExtractBotAsInt32s().ConvertToFloat();
      auto fvals28 = int16_accums.acc24.ExtractTopAsInt32s().ConvertToFloat();
      (inv_mults[j] * fvals00 + simd_biases[j]).Store(distances_buffer + 0);
      (inv_mults[j] * fvals04 + simd_biases[j]).Store(distances_buffer + 4);
      (inv_mults[j] * fvals08 + simd_biases[j]).Store(distances_buffer + 8);
      (inv_mults[j] * fvals12 + simd_biases[j]).Store(distances_buffer + 12);
      (inv_mults[j] * fvals16 + simd_biases[j]).Store(distances_buffer + 16);
      (inv_mults[j] * fvals20 + simd_biases[j]).Store(distances_buffer + 20);
      (inv_mults[j] * fvals24 + simd_biases[j]).Store(distances_buffer + 24);
      (inv_mults[j] * fvals28 + simd_biases[j]).Store(distances_buffer + 28);

      if (k == num_32dp_simd_iters - 1) {
        push_mask &= final_mask;
      }
      if (restrict_whitelist_ptrs[j]) {
        push_mask &= restrict_whitelist_ptrs[j][k];
      }

      while (push_mask) {
        const int offset = bits::FindLSBSetNonZero(push_mask);
        push_mask &= (push_mask - 1);
        const DatapointIndex dp_idx = first_dp_index + 32 * k + offset;
        DCHECK(
            !restrict_whitelist_ptrs[j] ||
            args.restrict_whitelists[j].IsWhitelisted(dp_idx - first_dp_index))
            << dp_idx;
        const bool needs_gc =
            topn_mutators[j].Push(dp_idx, distances_buffer[offset]);
        if (ABSL_PREDICT_FALSE(needs_gc)) {
          topn_mutators[j].GarbageCollect();

          const float new_epsilon = topn_mutators[j].epsilon();
          const float float_threshold = (new_epsilon - biases[j]) * mults[j];
          const int16_t int16_threshold = GetInt16Threshold(float_threshold);
          simd_thresholds[j] = M128_8Xint16::Broadcast(int16_threshold);

          push_mask &= compute_push_mask();
        }
      }
    }
  }
}
}  // namespace

template <size_t kBatchSize, bool kPrefetch>
SCANN_SSE4_OUTLINE void LUT16Sse4<kBatchSize, kPrefetch>::GetTopFloatDistances(
    LUT16ArgsTopN<float> args) {
  return GetTopFloatDistancesImpl<kBatchSize, kPrefetch>(std::move(args));
}

SCANN_INSTANTIATE_CLASS_FOR_LUT16_BATCH_SIZES(, LUT16Sse4);

}  // namespace asymmetric_hashing_internal
}  // namespace scann_ops
}  // namespace tensorflow

#endif
