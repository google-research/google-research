#include "scann/hashes/internal/lut16_avx2.h"
#include "scann/oss_wrappers/scann_bits.h"
#include "scann/utils/common.h"

#ifdef __x86_64__

#include "scann/utils/bits.h"
#include "scann/utils/intrinsics/avx2.h"
#include "tensorflow/core/platform/prefetch.h"

namespace research_scann {
namespace asymmetric_hashing_internal {
namespace {

SCANN_AVX2_INLINE Avx2<int16_t> CombineAvxLanes(const Avx2<int16_t>& a,
                                                const Avx2<int16_t>& b) {
  constexpr uint8_t kDestLoEqALo = 0x00;
  constexpr uint8_t kDestLoEqAHi = 0x01;
  constexpr uint8_t kDestHiEqBLo = 0x20;
  constexpr uint8_t kDestHiEqBHi = 0x30;
  constexpr uint8_t t1spec = (kDestLoEqALo + kDestHiEqBHi);
  constexpr uint8_t t2spec = (kDestLoEqAHi + kDestHiEqBLo);
  Avx2<int16_t> term0 = _mm256_permute2x128_si256(*a, *b, t1spec);
  Avx2<int16_t> term1 = _mm256_permute2x128_si256(*a, *b, t2spec);
  return term0 + term1;
}

SCANN_AVX2_INLINE Avx2<int16_t> PostprocessAccumulatorPair(
    const Avx2<int16_t>& even_plus_tag_along_bits, const Avx2<int16_t>& odd) {
  Avx2<int16_t> even = even_plus_tag_along_bits - (odd << 8);

  Avx2<int16_t> lo_per_lane = _mm256_unpacklo_epi16(*even, *odd);
  Avx2<int16_t> hi_per_lane = _mm256_unpackhi_epi16(*even, *odd);

  return CombineAvxLanes(lo_per_lane, hi_per_lane);
}

template <size_t size, typename T>
SCANN_INLINE array<T, size> ToLocalArray(ConstSpan<T> span) {
  DCHECK_EQ(span.size(), size);
  array<T, size> result;
  std::copy(span.begin(), span.begin() + size, result.begin());
  return result;
}

SCANN_INLINE ssize_t ComputeSmartPrefetchIndex(size_t num_blocks,
                                               size_t num_32dp_simd_iters) {
  return std::min(0l, static_cast<ssize_t>(
                          DivRoundUp(kPrefetchBytesAhead / 16, num_blocks)) -
                          static_cast<ssize_t>(num_32dp_simd_iters));
}

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_AVX2_INLINE Avx2<int16_t, kNumQueries, 2> Avx2LUT16BottomLoop(
    const uint8_t* data_start, array<const uint8_t*, kNumQueries> lookup_starts,
    const DimensionIndex num_blocks, const uint8_t* prefetch_start) {
  static_assert(kNumQueries <= 3,
                "Register spilling happens when kNumQueries > 3");
  Avx2<int16_t, kNumQueries, 4> int16_accums = avx2::Zeros();
  const Avx2<uint8_t> sign7 = 0x0F;

  DimensionIndex num_unroll_iter = num_blocks / 2;
  for (; num_unroll_iter != 0; --num_unroll_iter) {
    constexpr uint32_t kPointsPerIter = 32;
    if constexpr (kPrefetch == PrefetchStrategy::kSeq) {
      ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(
          data_start + kPrefetchBytesAhead);
    } else if constexpr (kPrefetch == PrefetchStrategy::kSmart) {
      ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(
          prefetch_start);
      prefetch_start += kPointsPerIter;
    }

    auto mask = Avx2<uint8_t>::Load(data_start);
    data_start += kPointsPerIter;

    Avx2<uint8_t> mask0 = mask & sign7;

    Avx2<uint8_t> mask1 = Avx2<uint8_t>((Avx2<uint16_t>(mask) >> 4)) & sign7;

    for (size_t j : Seq(kNumQueries)) {
      const Avx2<uint8_t> dict = Avx2<uint8_t>::Load(lookup_starts[j]);
      lookup_starts[j] += kPointsPerIter;
      const Avx2<uint8_t> res0 = _mm256_shuffle_epi8(*dict, *mask0);
      const Avx2<uint8_t> res1 = _mm256_shuffle_epi8(*dict, *mask1);

      int16_accums[j][0] += Avx2<int16_t>(Avx2<uint16_t>(res0));
      int16_accums[j][1] += Avx2<int16_t>(Avx2<uint16_t>(res0) >> 8);
      int16_accums[j][2] += Avx2<int16_t>(Avx2<uint16_t>(res1));
      int16_accums[j][3] += Avx2<int16_t>(Avx2<uint16_t>(res1) >> 8);
    }
  }

  Avx2<int16_t, kNumQueries, 2> results;
  for (size_t j : Seq(kNumQueries)) {
    results[j][0] =
        PostprocessAccumulatorPair(int16_accums[j][0], int16_accums[j][1]);
    results[j][1] =
        PostprocessAccumulatorPair(int16_accums[j][2], int16_accums[j][3]);
  }

  const bool has_odd_block = num_blocks & 1;
  if (has_odd_block) {
    const Sse4<uint8_t> sign7 = 0x0F;
    const Sse4<uint8_t> mask = Sse4<uint8_t>::Load(data_start);
    const Sse4<uint8_t> mask0 = mask & sign7;
    const Sse4<uint8_t> mask1 =
        Sse4<uint8_t>((Sse4<uint16_t>(mask) >> 4)) & sign7;

    for (size_t j : Seq(kNumQueries)) {
      auto dict = Sse4<uint8_t>::Load(lookup_starts[j]);
      Sse4<uint8_t> val0 = _mm_shuffle_epi8(*dict, *mask0);
      Sse4<uint8_t> val1 = _mm_shuffle_epi8(*dict, *mask1);
      results[j][0] += Avx2<int16_t>(_mm256_cvtepu8_epi16(*val0));
      results[j][1] += Avx2<int16_t>(_mm256_cvtepu8_epi16(*val1));
    }
  }

  const Avx2<int16_t> total_bias = num_blocks * 128;
  for (size_t j : Seq(kNumQueries)) {
    results[j] -= total_bias;
  }
  return results;
}

template <size_t kBottomLevelBatchSize, size_t kNumQueries>
SCANN_AVX2_INLINE array<const uint8_t*, kBottomLevelBatchSize>
MakeBottomLevelBatchLookupArray(
    array<const uint8_t*, kNumQueries> mid_level_lookups, size_t start) {
  DCHECK_LE(start + kBottomLevelBatchSize, kNumQueries);
  array<const uint8_t*, kBottomLevelBatchSize> result;
  for (size_t j : Seq(kBottomLevelBatchSize)) {
    result[j] = mid_level_lookups[start + j];
  }
  return result;
}

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_AVX2_INLINE Avx2<int16_t, kNumQueries, 2> Avx2LUT16MiddleLoop(
    const uint8_t* data_start, array<const uint8_t*, kNumQueries> lookup_starts,
    const DimensionIndex num_blocks, const uint8_t* prefetch_start) {
  constexpr size_t kSizeB = (kNumQueries == 1) ? 1 : 2;
  constexpr size_t kNumBCases[] = {0, 2, 1};
  constexpr size_t kNumB = (kNumQueries == 1) ? 1 : kNumBCases[kNumQueries % 3];

  constexpr size_t kRemaining = kNumQueries - kNumB * kSizeB;
  static_assert(kRemaining % 3 == 0, "");

  constexpr size_t kSizeA = 3;
  constexpr size_t kNumA = kRemaining / 3;

  Avx2<int16_t, kNumQueries, 2> result;
  for (size_t j : Seq(kNumA)) {
    const size_t start = j * kSizeA;
    auto bottom_level_lookups =
        MakeBottomLevelBatchLookupArray<kSizeA>(lookup_starts, start);
    auto acc = Avx2LUT16BottomLoop<kSizeA, kPrefetch>(
        data_start, bottom_level_lookups, num_blocks, prefetch_start);
    for (size_t jj : Seq(kSizeA)) {
      result[start + jj] = acc[jj];
    }
  }

  for (size_t j : Seq(kNumB)) {
    const size_t start = kNumA * kSizeA + j * kSizeB;
    auto bottom_level_lookups =
        MakeBottomLevelBatchLookupArray<kSizeB>(lookup_starts, start);
    auto acc = Avx2LUT16BottomLoop<kSizeB, kPrefetch>(
        data_start, bottom_level_lookups, num_blocks, prefetch_start);
    for (size_t jj : Seq(kSizeB)) {
      result[start + jj] = acc[jj];
    }
  }
  return result;
}

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_AVX2_INLINE Avx2<int16_t, kNumQueries, 2> PrefetchDispatcher(
    const uint8_t* data_start, array<const uint8_t*, kNumQueries> lookups,
    const DimensionIndex num_blocks, const uint8_t* next_partition,
    ssize_t* __restrict__ next_prefetch_idx) {
  if constexpr (kPrefetch == PrefetchStrategy::kSmart) {
    if ((*next_prefetch_idx) >= 0) {
      const uint8_t* prefetch_start =
          next_partition + (*next_prefetch_idx) * 16 * num_blocks;
      (*next_prefetch_idx)++;
      return Avx2LUT16MiddleLoop<kNumQueries, kPrefetch>(
          data_start, lookups, num_blocks, prefetch_start);
    } else {
      (*next_prefetch_idx)++;
      return Avx2LUT16MiddleLoop<kNumQueries, PrefetchStrategy::kSeq>(
          data_start, lookups, num_blocks, nullptr);
    }
  } else {
    return Avx2LUT16MiddleLoop<kNumQueries, kPrefetch>(data_start, lookups,
                                                       num_blocks, nullptr);
  }
}

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_AVX2_INLINE Avx2<int32_t, kNumQueries, 4> Avx2LUT16MiddleLoopInt32(
    const uint8_t* data_start, array<const uint8_t*, kNumQueries> lookup_starts,
    const DimensionIndex num_blocks, const uint8_t* next_partition,
    ssize_t* __restrict__ next_prefetch_idx) {
  Avx2<int32_t, kNumQueries, 4> int32_accumulators = avx2::Zeros();
  for (DimensionIndex k = 0; k < num_blocks;) {
    DimensionIndex reaccumulate_limit = std::min(num_blocks - k, uint64_t{256});

    auto int16_accums = [&]() SCANN_AVX2_INLINE_LAMBDA {
      if constexpr (kPrefetch == PrefetchStrategy::kSmart) {
        if ((*next_prefetch_idx) >= 0) {
          const uint8_t* prefetch_start =
              next_partition + (*next_prefetch_idx) * 16 * num_blocks + k * 16;
          return Avx2LUT16MiddleLoop<kNumQueries, kPrefetch>(
              data_start, lookup_starts, reaccumulate_limit, prefetch_start);
        } else {
          return Avx2LUT16MiddleLoop<kNumQueries, PrefetchStrategy::kSeq>(
              data_start, lookup_starts, reaccumulate_limit, nullptr);
        }
      } else {
        return Avx2LUT16MiddleLoop<kNumQueries, kPrefetch>(
            data_start, lookup_starts, reaccumulate_limit, nullptr);
      }
    }();
    data_start += 16 * reaccumulate_limit;
    k += reaccumulate_limit;
    for (size_t j : Seq(kNumQueries)) {
      int32_accumulators[j] += int16_accums[j].template ExpandTo<int32_t>();
      lookup_starts[j] += 16 * reaccumulate_limit;
    }
  }
  if constexpr (kPrefetch == PrefetchStrategy::kSmart) (*next_prefetch_idx)++;
  return int32_accumulators;
}

}  // namespace

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_AVX2_OUTLINE void LUT16Avx2<kNumQueries, kPrefetch>::GetInt16Distances(
    LUT16Args<int16_t> args) {
  const uint8_t* packed_dataset = args.packed_dataset;
  const uint8_t* next_partition = args.next_partition;
  const size_t num_32dp_simd_iters = args.num_32dp_simd_iters;
  const size_t num_blocks = args.num_blocks;
  auto lookups = ToLocalArray<kNumQueries>(args.lookups);
  auto distances = ToLocalArray<kNumQueries>(args.distances);

  ssize_t next_prefetch_idx =
      ComputeSmartPrefetchIndex(num_blocks, num_32dp_simd_iters);
  for (DatapointIndex k : Seq(num_32dp_simd_iters)) {
    const uint8_t* data_start = packed_dataset + k * 16 * num_blocks;
    auto int16_accums = PrefetchDispatcher<kNumQueries, kPrefetch>(
        data_start, lookups, num_blocks, next_partition, &next_prefetch_idx);
    for (size_t j : Seq(kNumQueries)) {
      int16_accums[j].Store(distances[j] + 32 * k);
    }
  }
}

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_AVX2_OUTLINE void LUT16Avx2<kNumQueries, kPrefetch>::GetInt32Distances(
    LUT16Args<int32_t> args) {
  const uint8_t* packed_dataset = args.packed_dataset;
  const uint8_t* next_partition = args.next_partition;
  const size_t num_32dp_simd_iters = args.num_32dp_simd_iters;
  const size_t num_blocks = args.num_blocks;
  auto lookups = ToLocalArray<kNumQueries>(args.lookups);
  auto distances = ToLocalArray<kNumQueries>(args.distances);

  ssize_t next_prefetch_idx =
      ComputeSmartPrefetchIndex(num_blocks, num_32dp_simd_iters);
  for (DatapointIndex k : Seq(num_32dp_simd_iters)) {
    const uint8_t* data_start = packed_dataset + k * 16 * num_blocks;
    auto int32_accumulators = Avx2LUT16MiddleLoopInt32<kNumQueries, kPrefetch>(
        data_start, lookups, num_blocks, next_partition, &next_prefetch_idx);
    for (size_t j : Seq(kNumQueries)) {
      int32_accumulators[j].Store(distances[j] + 32 * k);
    }
  }
}

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_AVX2_OUTLINE void LUT16Avx2<kNumQueries, kPrefetch>::GetFloatDistances(
    LUT16Args<float> args, ConstSpan<float> inv_fp_multipliers) {
  const uint8_t* packed_dataset = args.packed_dataset;
  const uint8_t* next_partition = args.next_partition;
  const size_t num_32dp_simd_iters = args.num_32dp_simd_iters;
  const size_t num_blocks = args.num_blocks;
  auto lookups = ToLocalArray<kNumQueries>(args.lookups);
  auto distances = ToLocalArray<kNumQueries>(args.distances);
  auto mults = ToLocalArray<kNumQueries>(inv_fp_multipliers);

  ssize_t next_prefetch_idx =
      ComputeSmartPrefetchIndex(num_blocks, num_32dp_simd_iters);
  for (DatapointIndex k : Seq(num_32dp_simd_iters)) {
    const uint8_t* data_start = packed_dataset + k * 16 * num_blocks;
    Avx2<int32_t, kNumQueries, 4> int32_accums =
        Avx2LUT16MiddleLoopInt32<kNumQueries, kPrefetch>(
            data_start, lookups, num_blocks, next_partition,
            &next_prefetch_idx);
    for (size_t j : Seq(kNumQueries)) {
      (int32_accums[j].template ConvertTo<float>() * Avx2<float>(mults[j]))
          .Store(distances[j] + 32 * k);
    }
  }
}

namespace {
template <size_t kNumQueries, PrefetchStrategy kPrefetch, typename TopN>
SCANN_AVX2_INLINE void GetTopInt16DistancesImpl(
    LUT16ArgsTopN<int16_t, TopN> args) {
  const uint8_t* packed_dataset = args.packed_dataset;
  const uint8_t* next_partition = args.next_partition;
  const size_t num_32dp_simd_iters = args.num_32dp_simd_iters;
  const size_t num_blocks = args.num_blocks;
  auto lookups = ToLocalArray<kNumQueries>(args.lookups);
  const DatapointIndex first_dp_index = args.first_dp_index;
  const uint32_t final_mask = GetFinalMask32(args.num_datapoints);
  DCHECK_EQ(num_32dp_simd_iters, DivRoundUp(args.num_datapoints, 32));

  Avx2<int16_t> simd_thresholds[kNumQueries];
  for (size_t j : Seq(kNumQueries)) {
    const int16_t int16_threshold = args.fast_topns[j]->epsilon();
    simd_thresholds[j] = int16_threshold;
  }

  typename TopN::Mutator topn_mutators[kNumQueries];
  for (size_t j : Seq(kNumQueries)) {
    args.fast_topns[j]->AcquireMutator(&topn_mutators[j]);
  }

  int16_t distances_buffer[32];
  auto restrict_whitelist_ptrs =
      args.template GetRestrictWhitelistPtrs<kNumQueries>();
  ssize_t next_prefetch_idx =
      ComputeSmartPrefetchIndex(num_blocks, num_32dp_simd_iters);
  for (DatapointIndex k : Seq(num_32dp_simd_iters)) {
    const uint8_t* data_start = packed_dataset + k * 16 * num_blocks;
    auto int16_accums = PrefetchDispatcher<kNumQueries, kPrefetch>(
        data_start, lookups, num_blocks, next_partition, &next_prefetch_idx);
    for (size_t j : Seq(kNumQueries)) {
      auto compute_push_mask = [&]() SCANN_AVX2_INLINE_LAMBDA {
        return GetComparisonMask(int16_accums[j] < simd_thresholds[j]);
      };
      uint32_t push_mask = compute_push_mask();

      if (!push_mask) continue;

      int16_accums[j].Store(distances_buffer);

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

          simd_thresholds[j] = topn_mutators[j].epsilon();

          push_mask &= compute_push_mask();
        }
      }
    }
  }
}
}  // namespace

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_AVX2_OUTLINE void LUT16Avx2<kNumQueries, kPrefetch>::GetTopInt16Distances(
    LUT16ArgsTopN<int16_t> args) {
  return GetTopInt16DistancesImpl<kNumQueries, kPrefetch>(std::move(args));
}

SCANN_AVX2_INLINE int16_t GetInt16Threshold(float float_threshold) {
  constexpr float kMaxValue = numeric_limits<int16_t>::max();

  return std::min(float_threshold, kMaxValue);
}

namespace {
template <size_t kNumQueries, PrefetchStrategy kPrefetch,
          bool kHasFinalPredicate, typename TopN>
SCANN_AVX2_INLINE void GetTopFloatDistancesImpl(
    LUT16ArgsTopN<float, TopN> args) {
  const uint8_t* packed_dataset = args.packed_dataset;
  const uint8_t* next_partition = args.next_partition;
  const size_t num_32dp_simd_iters = args.num_32dp_simd_iters;
  const size_t num_blocks = args.num_blocks;
  auto lookups = ToLocalArray<kNumQueries>(args.lookups);
  const DatapointIndex first_dp_index = args.first_dp_index;
  const uint32_t final_mask = GetFinalMask32(args.num_datapoints);
  DCHECK_EQ(num_32dp_simd_iters, DivRoundUp(args.num_datapoints, 32));

  auto biases = ToLocalArray<kNumQueries>(args.biases);
  Avx2<float> simd_biases[kNumQueries];
  for (size_t j : Seq(kNumQueries)) {
    simd_biases[j] = biases[j];
  }

  auto mults = ToLocalArray<kNumQueries>(args.fixed_point_multipliers);
  Avx2<float> inv_mults[kNumQueries];
  for (size_t j : Seq(kNumQueries)) {
    inv_mults[j] = 1.0 / mults[j];
  }

  Avx2<int16_t> simd_thresholds[kNumQueries];
  for (size_t j : Seq(kNumQueries)) {
    const float epsilon = args.fast_topns[j]->epsilon();
    const float float_threshold = (epsilon - biases[j]) * mults[j];
    const int16_t int16_threshold = GetInt16Threshold(float_threshold);
    simd_thresholds[j] = int16_threshold;
  }

  typename TopN::Mutator topn_mutators[kNumQueries];
  for (size_t j : Seq(kNumQueries)) {
    args.fast_topns[j]->AcquireMutator(&topn_mutators[j]);
  }

  float distances_buffer[32];
  auto restrict_whitelist_ptrs =
      args.template GetRestrictWhitelistPtrs<kNumQueries>();
  ssize_t next_prefetch_idx =
      ComputeSmartPrefetchIndex(num_blocks, num_32dp_simd_iters);
  for (DatapointIndex k : Seq(num_32dp_simd_iters)) {
    const uint8_t* data_start = packed_dataset + k * 16 * num_blocks;

    auto int16_accums = PrefetchDispatcher<kNumQueries, kPrefetch>(
        data_start, lookups, num_blocks, next_partition, &next_prefetch_idx);
    for (size_t j : Seq(kNumQueries)) {
      auto compute_push_mask = [&]() SCANN_AVX2_INLINE_LAMBDA {
        return GetComparisonMask(int16_accums[j] < simd_thresholds[j]);
      };
      uint32_t push_mask = compute_push_mask();

      if (!push_mask) continue;

      auto fvals = int16_accums[j]
                       .template ExpandTo<int32_t>()
                       .template ConvertTo<float>();

      (fvals * inv_mults[j] + simd_biases[j]).Store(distances_buffer);

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
        if (kHasFinalPredicate && !args.final_predicate(dp_idx)) continue;

        const bool needs_gc =
            topn_mutators[j].Push(dp_idx, distances_buffer[offset]);
        if (ABSL_PREDICT_FALSE(needs_gc)) {
          topn_mutators[j].GarbageCollect();

          const float new_epsilon = topn_mutators[j].epsilon();
          const float float_threshold = (new_epsilon - biases[j]) * mults[j];
          const int16_t int16_threshold = GetInt16Threshold(float_threshold);
          simd_thresholds[j] = int16_threshold;

          push_mask &= compute_push_mask();
        }
      }
    }
  }
}
}  // namespace

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_AVX2_OUTLINE void LUT16Avx2<kNumQueries, kPrefetch>::GetTopFloatDistances(
    LUT16ArgsTopN<float> args) {
  if (args.final_predicate) {
    return GetTopFloatDistancesImpl<kNumQueries, kPrefetch, true>(
        std::move(args));
  } else {
    return GetTopFloatDistancesImpl<kNumQueries, kPrefetch, false>(
        std::move(args));
  }
}

}  // namespace asymmetric_hashing_internal
}  // namespace research_scann

#endif
