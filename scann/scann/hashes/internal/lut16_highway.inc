#include <cstdint>

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS HWY_ALL_SVE
#endif

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "scann/hashes/internal/lut16_highway.inc"
#include "hwy/foreach_target.h"
#include "scann/hashes/internal/lut16_args.h"

#if !defined(__x86_64__) || defined(SCANN_FORCE_HIGHWAY_LUT16)

#include "absl/base/prefetch.h"
#include "hwy/highway.h"
#include "scann/hashes/internal/lut16_highway.h"
#include "scann/oss_wrappers/scann_bits.h"
#include "scann/utils/bits.h"
#include "scann/utils/common.h"
#include "scann/utils/intrinsics/attributes.h"

HWY_BEFORE_NAMESPACE();

namespace research_scann {
namespace asymmetric_hashing_internal {

namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

SCANN_HIGHWAY_INLINE void ExpandToInt32AccumAndStore(
    const int16_t* HWY_RESTRICT dist16, int32_t* HWY_RESTRICT dist32) {
  const hn::ScalableTag<int16_t> d16;
  const hn::ScalableTag<int32_t> d32;
  const hn::Half<decltype(d16)> d16_half;
  using V16 = decltype(hn::Zero(d16));
  using V32 = decltype(hn::Zero(d32));
  using V8 = decltype(hn::Zero(d16_half));
  const size_t N = hn::Lanes(d16);
  const size_t num_iters = 32 / N;
  for (size_t i = 0; i < num_iters; ++i) {
    V16 val16 = hn::LoadU(d16, dist16 + i * N);

    V8 val16_hi = hn::UpperHalf(d16_half, val16);
    V8 val16_lo = hn::LowerHalf(d16_half, val16);

    V32 val32_hi = hn::PromoteTo(d32, val16_hi);
    V32 val32_lo = hn::PromoteTo(d32, val16_lo);

    V32 dist32_lo = hn::LoadU(d32, dist32 + (i * N));
    V32 dist32_hi = hn::LoadU(d32, dist32 + (i * N) + (N / 2));

    hn::StoreU(hn::Add(dist32_lo, val32_lo), d32, dist32 + (i * N));
    hn::StoreU(hn::Add(dist32_hi, val32_hi), d32, dist32 + (i * N) + (N / 2));
  }
}

SCANN_HIGHWAY_INLINE void ConvertToFloatMultiplyAndStore(const int32_t* dist32,
                                                         const float mult,
                                                         float* distf32) {
  const hn::ScalableTag<int32_t> d32;
  const hn::ScalableTag<float> df;
  using V32 = decltype(hn::Zero(d32));
  using VF = decltype(hn::Zero(df));
  const size_t N = hn::Lanes(d32);
  const size_t num_iters = 32 / N;
  auto mults = hn::Set(df, mult);
  for (size_t i = 0; i < num_iters; ++i) {
    V32 val32 = hn::LoadU(d32, dist32 + (i * N));
    VF valf32 = hn::ConvertTo(df, val32);
    hn::StoreU(hn::Mul(valf32, mults), df, distf32 + (i * N));
  }
}

SCANN_HIGHWAY_INLINE uint32_t
ComputePushMask(const int16_t* HWY_RESTRICT dist16, const int16_t threshold) {
  const hn::ScalableTag<int16_t> d16;
  using V16 = decltype(hn::Zero(d16));
  const size_t N = hn::Lanes(d16);
  const size_t num_iters = 32 / N;

  uint8_t mask_bits[8];
  uint32_t ret = 0;
  size_t total_bytes = 0;

  V16 simd_threshold = hn::Set(d16, threshold);
  for (size_t i = 0; i < num_iters; ++i) {
    auto mask = hn::Lt(hn::LoadU(d16, dist16 + (i * N)), simd_threshold);
    total_bytes += hn::StoreMaskBits(d16, mask, &mask_bits[i * N / 8]);
  }

  DCHECK_EQ(total_bytes, 4);
  for (size_t i = 0; i < total_bytes; ++i) {
    ret |= (uint32_t)mask_bits[i] << (i * 8);
  }
  return ret;
}

SCANN_HIGHWAY_INLINE void ExpandConvertToFloatAndStore(
    const int16_t* HWY_RESTRICT dist16, float* HWY_RESTRICT distf32) {
  const hn::ScalableTag<int16_t> d16;
  const hn::ScalableTag<int32_t> d32;
  const hn::ScalableTag<float> df;
  const hn::Half<decltype(d16)> d16_half;
  using V16 = decltype(hn::Zero(d16));
  using V32 = decltype(hn::Zero(d32));
  using V8 = decltype(hn::Zero(d16_half));
  const size_t N = hn::Lanes(d16);
  const size_t num_iters = 32 / N;
  for (size_t i = 0; i < num_iters; ++i) {
    V16 val16 = hn::LoadU(d16, dist16 + i * N);

    V8 val16_hi = hn::UpperHalf(d16_half, val16);
    V8 val16_lo = hn::LowerHalf(d16_half, val16);

    V32 val32_hi = hn::PromoteTo(d32, val16_hi);
    V32 val32_lo = hn::PromoteTo(d32, val16_lo);

    hn::StoreU(hn::ConvertTo(df, val32_lo), df, distf32 + (i * N));
    hn::StoreU(hn::ConvertTo(df, val32_hi), df, distf32 + (i * N) + (N / 2));
  }
}

SCANN_HIGHWAY_INLINE void MultiplyAddAndStoreFloat(
    const float* HWY_RESTRICT distf32, const float mult, const float biase,
    float* HWY_RESTRICT result) {
  const hn::ScalableTag<float> df;
  using VF = decltype(hn::Zero(df));
  const size_t N = hn::Lanes(df);
  const size_t num_iters = 32 / N;
  VF mults = hn::Set(df, mult);
  VF biases = hn::Set(df, biase);
  for (size_t i = 0; i < num_iters; ++i) {
    VF fval = hn::LoadU(df, distf32 + (i * N));
    hn::StoreU(hn::MulAdd(fval, mults, biases), df, result + (i * N));
  }
}

template <class D, class D16 = hn::Repartition<int16_t, D>>
SCANN_HIGHWAY_INLINE void Accumulate(const D d, hn::Vec<D> res,
                                     hn::Vec<D16>& int16_accums0,
                                     hn::Vec<D16>& int16_accums1) {
  const hn::Repartition<int16_t, D> dto16;
  const hn::Repartition<uint16_t, D> dtou16;
  int16_accums0 =
      hn::Add(int16_accums0, hn::BitCast(dto16, hn::BitCast(dtou16, res)));
  int16_accums1 =
      hn::Add(int16_accums1,
              hn::BitCast(dto16, hn::ShiftRight<8>(hn::BitCast(dtou16, res))));
}

template <PrefetchStrategy kPrefetch, class V16>
SCANN_HIGHWAY_INLINE void HighwayLUT16BottomLoopAccum(
    const uint8_t* HWY_RESTRICT data_start, const uint8_t* lookup_starts,
    const DimensionIndex num_blocks, V16& int16_accums0, V16& int16_accums1,
    V16& int16_accums2, V16& int16_accums3) {
  const hn::ScalableTag<uint8_t> d;
  using V = decltype(hn::Zero(d));
  const size_t N = hn::Lanes(d);
  const size_t num_blocks_per_iter = N / 16;
  const size_t num_iters = num_blocks / num_blocks_per_iter;
  V bitmask = hn::Set(d, 0x0F);

  for (size_t i = 0; i < num_iters; ++i) {
    if constexpr (kPrefetch != PrefetchStrategy::kOff) {
      absl::PrefetchToLocalCacheNta((data_start + i * N) + kPrefetchBytesAhead);
    }

    V dp = hn::LoadU(d, data_start + i * N);
    V dp0 = hn::And(dp, bitmask);
    V dp1 = hn::ShiftRight<4>(dp);
    V lut = hn::LoadU(d, lookup_starts + i * N);
    V res0 = hn::TableLookupBytes(lut, dp0);
    V res1 = hn::TableLookupBytes(lut, dp1);
    Accumulate(d, res0, int16_accums0, int16_accums1);
    Accumulate(d, res1, int16_accums2, int16_accums3);
  }

  const size_t rem = num_blocks % num_blocks_per_iter;
  if (rem > 0) {
    if (N == 32) {
      const hn::Half<decltype(d)> dh;
      V dp =
          hn::ZeroExtendVector(d, hn::LoadU(dh, data_start + (num_iters * N)));
      V dp0 = hn::And(dp, bitmask);
      V dp1 = hn::ShiftRight<4>(dp);
      V lut = hn::ZeroExtendVector(
          d, hn::LoadU(dh, lookup_starts + (num_iters * N)));
      V res0 = hn::TableLookupBytes(lut, dp0);
      V res1 = hn::TableLookupBytes(lut, dp1);
      Accumulate(d, res0, int16_accums0, int16_accums1);
      Accumulate(d, res1, int16_accums2, int16_accums3);
    } else {
      auto mask = hn::FirstN(d, 16 * rem);
      V dp = hn::MaskedLoad(mask, d, data_start + (num_iters * N));
      V dp0 = hn::And(dp, bitmask);
      V dp1 = hn::ShiftRight<4>(dp);
      V lut = MaskedLoad(mask, d, lookup_starts + (num_iters * N));
      V res0 = hn::TableLookupBytes(lut, dp0);
      V res1 = hn::TableLookupBytes(lut, dp1);
      Accumulate(d, res0, int16_accums0, int16_accums1);
      Accumulate(d, res1, int16_accums2, int16_accums3);
    }
  }
}

template <class D16>
SCANN_HIGHWAY_INLINE void HighwayPostProcess128Bits(
    D16 d, const DimensionIndex num_blocks, std::array<int16_t, 32>& dist16,
    hn::Vec<D16>& int16_accums0, hn::Vec<D16>& int16_accums1,
    hn::Vec<D16>& int16_accums2, hn::Vec<D16>& int16_accums3) {
  const hn::ScalableTag<int16_t> d16;
  using V16 = decltype(hn::Zero(d16));
  V16 total_bias = Set(d16, num_blocks * 128);
  V16 v0 = int16_accums0;
  V16 v1 = int16_accums1;

  V16 even = hn::Sub(v0, hn::ShiftLeft<8>(v1));

  v0 = hn::Sub(hn::InterleaveLower(d16, even, v1), total_bias);
  v1 = hn::Sub(hn::InterleaveUpper(d16, even, v1), total_bias);
  hn::StoreU(v0, d16, dist16.data());
  hn::StoreU(v1, d16, dist16.data() + 8);

  v0 = int16_accums2;
  v1 = int16_accums3;
  even = hn::Sub(v0, hn::ShiftLeft<8>(v1));
  v0 = hn::Sub(hn::InterleaveLower(d16, even, v1), total_bias);
  v1 = hn::Sub(hn::InterleaveUpper(d16, even, v1), total_bias);
  hn::StoreU(v0, d16, dist16.data() + 16);
  hn::StoreU(v1, d16, dist16.data() + 24);
}

template <class D16>
SCANN_HIGHWAY_INLINE void HighwayPostProcess256Bits(
    D16 d, const DimensionIndex num_blocks, std::array<int16_t, 32>& dist16,
    hn::Vec<D16>& int16_accums0, hn::Vec<D16>& int16_accums1,
    hn::Vec<D16>& int16_accums2, hn::Vec<D16>& int16_accums3) {
  const hn::ScalableTag<int16_t> d16;
  using V16 = decltype(hn::Zero(d16));

  V16 total_bias = Set(d16, num_blocks * 128);
  V16 v0 = int16_accums0;
  V16 v1 = int16_accums1;
  V16 even = hn::Sub(v0, hn::ShiftLeft<8>(v1));
  v0 = hn::InterleaveLower(d16, even, v1);
  v1 = hn::InterleaveUpper(d16, even, v1);
  V16 lu = hn::ConcatLowerUpper(d16, v1, v0);
  V16 ul = hn::ConcatUpperLower(d16, v1, v0);
  v0 = hn::Sub(hn::Add(lu, ul), total_bias);
  hn::StoreU(v0, d16, dist16.data());
  v0 = int16_accums2;
  v1 = int16_accums3;
  even = hn::Sub(v0, hn::ShiftLeft<8>(v1));
  v0 = hn::InterleaveLower(d16, even, v1);
  v1 = hn::InterleaveUpper(d16, even, v1);
  lu = hn::ConcatLowerUpper(d16, v1, v0);
  ul = hn::ConcatUpperLower(d16, v1, v0);
  v0 = hn::Sub(hn::Add(lu, ul), total_bias);
  hn::StoreU(v0, d16, dist16.data() + 16);
}

template <class D16>
SCANN_HIGHWAY_INLINE void HighwayPostProcess512Bits(
    D16 d, const DimensionIndex num_blocks, std::array<int16_t, 32>& dist16,
    hn::Vec<D16>& int16_accums0, hn::Vec<D16>& int16_accums1,
    hn::Vec<D16>& int16_accums2, hn::Vec<D16>& int16_accums3) {
  const hn::ScalableTag<int16_t> d16;
  const hn::Half<decltype(d16)> d16_half;
  using V16 = decltype(hn::Zero(d16));
  using V8 = decltype(hn::Zero(d16_half));
  V8 total_bias = hn::Set(d16_half, num_blocks * 128);

  V16 v0 = int16_accums0;
  V16 v1 = int16_accums1;
  V16 even = hn::Sub(v0, hn::ShiftLeft<8>(v1));
  v0 = hn::InterleaveLower(d16, even, v1);
  v1 = hn::InterleaveUpper(d16, even, v1);
  V16 lu = hn::ConcatLowerUpper(d16, v1, v0);
  V16 ul = hn::ConcatUpperLower(d16, v1, v0);
  v0 = hn::Add(lu, ul);
  V8 v_lo = hn::LowerHalf(d16_half, v0);
  V8 v_hi = hn::UpperHalf(d16_half, v0);
  V8 luh = hn::ConcatLowerUpper(d16_half, v_hi, v_lo);
  V8 ulh = hn::ConcatUpperLower(d16_half, v_hi, v_lo);
  v_lo = hn::Sub(hn::Add(luh, ulh), total_bias);
  hn::StoreU(v_lo, d16_half, dist16.data());
  v0 = int16_accums2;
  v1 = int16_accums3;
  even = hn::Sub(v0, hn::ShiftLeft<8>(v1));
  v0 = hn::InterleaveLower(d16, even, v1);
  v1 = hn::InterleaveUpper(d16, even, v1);
  lu = hn::ConcatLowerUpper(d16, v1, v0);
  ul = hn::ConcatUpperLower(d16, v1, v0);
  v0 = hn::Add(lu, ul);
  v_lo = hn::LowerHalf(d16_half, v0);
  v_hi = hn::UpperHalf(d16_half, v0);
  luh = hn::ConcatLowerUpper(d16_half, v_hi, v_lo);
  ulh = hn::ConcatUpperLower(d16_half, v_hi, v_lo);
  v_lo = hn::Sub(hn::Add(luh, ulh), total_bias);
  hn::StoreU(v_lo, d16_half, dist16.data() + 16);
}

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_HIGHWAY_INLINE void HighwayLUT16BottomLoop(
    const uint8_t* HWY_RESTRICT data_start,
    array<const uint8_t*, kNumQueries> lookup_starts,
    const DimensionIndex num_blocks,
    array<array<int16_t, 32>, kNumQueries>& dist16) {
  static_assert(kNumQueries <= 3,
                "Register spilling happens when kNumQueries > 3");
  const hn::ScalableTag<uint8_t> d;
  const hn::ScalableTag<int16_t> d16;
  const size_t N = hn::Lanes(d);
  const size_t num_32dp_blocks = N / 16;
  using V16 = decltype(hn::Zero(d16));

  V16 int16_accums0 = hn::Zero(d16);
  V16 int16_accums1 = int16_accums0;
  V16 int16_accums2 = int16_accums0;
  V16 int16_accums3 = int16_accums0;
  V16 int16_accums4 = int16_accums0;
  V16 int16_accums5 = int16_accums0;
  V16 int16_accums6 = int16_accums0;
  V16 int16_accums7 = int16_accums0;
  V16 int16_accums8 = int16_accums0;
  V16 int16_accums9 = int16_accums0;
  V16 int16_accums10 = int16_accums0;
  V16 int16_accums11 = int16_accums0;

  HighwayLUT16BottomLoopAccum<kPrefetch>(
      data_start, lookup_starts[0], num_blocks, int16_accums0, int16_accums1,
      int16_accums2, int16_accums3);
  if constexpr (kNumQueries > 1)
    HighwayLUT16BottomLoopAccum<kPrefetch>(
        data_start, lookup_starts[1], num_blocks, int16_accums4, int16_accums5,
        int16_accums6, int16_accums7);
  if constexpr (kNumQueries > 2)
    HighwayLUT16BottomLoopAccum<kPrefetch>(
        data_start, lookup_starts[2], num_blocks, int16_accums8, int16_accums9,
        int16_accums10, int16_accums11);

  if (num_32dp_blocks == 1) {
    HighwayPostProcess128Bits(d16, num_blocks, dist16[0], int16_accums0,
                              int16_accums1, int16_accums2, int16_accums3);
    if constexpr (kNumQueries > 1)
      HighwayPostProcess128Bits(d16, num_blocks, dist16[1], int16_accums4,
                                int16_accums5, int16_accums6, int16_accums7);
    if constexpr (kNumQueries > 2)
      HighwayPostProcess128Bits(d16, num_blocks, dist16[2], int16_accums8,
                                int16_accums9, int16_accums10, int16_accums11);
  } else if (num_32dp_blocks == 2) {
    HighwayPostProcess256Bits(d16, num_blocks, dist16[0], int16_accums0,
                              int16_accums1, int16_accums2, int16_accums3);
    if constexpr (kNumQueries > 1)
      HighwayPostProcess256Bits(d16, num_blocks, dist16[1], int16_accums4,
                                int16_accums5, int16_accums6, int16_accums7);
    if constexpr (kNumQueries > 2)
      HighwayPostProcess256Bits(d16, num_blocks, dist16[2], int16_accums8,
                                int16_accums9, int16_accums10, int16_accums11);
  } else if (num_32dp_blocks == 4) {
    HighwayPostProcess512Bits(d16, num_blocks, dist16[0], int16_accums0,
                              int16_accums1, int16_accums2, int16_accums3);
    if constexpr (kNumQueries > 1)
      HighwayPostProcess512Bits(d16, num_blocks, dist16[1], int16_accums4,
                                int16_accums5, int16_accums6, int16_accums7);
    if constexpr (kNumQueries > 2)
      HighwayPostProcess512Bits(d16, num_blocks, dist16[2], int16_accums8,
                                int16_accums9, int16_accums10, int16_accums11);
  } else {
    LOG(FATAL) << ("Unimplemented Vector Size\n");
  }
}

template <size_t kBottomLevelBatchSize, size_t kNumQueries>
SCANN_HIGHWAY_INLINE array<const uint8_t*, kBottomLevelBatchSize>
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
SCANN_HIGHWAY_INLINE void HighwayLUT16MiddleLoop(
    const uint8_t* HWY_RESTRICT data_start,
    array<const uint8_t*, kNumQueries>& lookup_starts,
    const DimensionIndex num_blocks,
    array<array<int16_t, 32>, kNumQueries>& result16) {
  constexpr size_t kSizeB = (kNumQueries == 1) ? 1 : 2;
  constexpr size_t kNumBCases[] = {0, 2, 1};
  constexpr size_t kNumB = (kNumQueries == 1) ? 1 : kNumBCases[kNumQueries % 3];

  constexpr size_t kRemaining = kNumQueries - kNumB * kSizeB;
  static_assert(kRemaining % 3 == 0, "");

  constexpr size_t kSizeA = 3;
  constexpr size_t kNumA = kRemaining / 3;

  for (size_t j : Seq(kNumA)) {
    array<array<int16_t, 32>, kSizeA> dist16;
    const size_t start = j * kSizeA;
    auto bottom_level_lookups =
        MakeBottomLevelBatchLookupArray<kSizeA>(lookup_starts, start);
    HighwayLUT16BottomLoop<kSizeA, kPrefetch>(data_start, bottom_level_lookups,
                                              num_blocks, dist16);
    for (size_t jj : Seq(kSizeA)) {
      std::copy(dist16[jj].begin(), dist16[jj].end(),
                result16[start + jj].begin());
    }
  }

  for (size_t j : Seq(kNumB)) {
    array<array<int16_t, 32>, kSizeB> dist16;
    const size_t start = kNumA * kSizeA + j * kSizeB;
    auto bottom_level_lookups =
        MakeBottomLevelBatchLookupArray<kSizeB>(lookup_starts, start);
    HighwayLUT16BottomLoop<kSizeB, kPrefetch>(data_start, bottom_level_lookups,
                                              num_blocks, dist16);
    for (size_t jj : Seq(kSizeB)) {
      std::copy(dist16[jj].begin(), dist16[jj].end(),
                result16[start + jj].begin());
    }
  }
}

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_HIGHWAY_INLINE void HighwayLUT16MiddleLoopInt32(
    const uint8_t* HWY_RESTRICT data_start,
    array<const uint8_t*, kNumQueries> lookup_starts,
    const DimensionIndex num_blocks,
    array<array<int32_t, 32>, kNumQueries>& result32) {
  array<array<int16_t, 32>, kNumQueries> result16;

  for (int i = 0; i < kNumQueries; ++i) {
    result32[i].fill(0);
  }

  for (DimensionIndex k = 0; k < num_blocks;) {
    DimensionIndex reaccumulate_limit = std::min(num_blocks - k, uint64_t{256});
    k += reaccumulate_limit;
    HighwayLUT16MiddleLoop<kNumQueries, kPrefetch>(
        data_start, lookup_starts, reaccumulate_limit, result16);
    data_start += 16 * reaccumulate_limit;
    for (size_t j : Seq(kNumQueries)) {
      ExpandToInt32AccumAndStore(result16[j].data(), result32[j].data());
      lookup_starts[j] += 16 * reaccumulate_limit;
    }
  }
}

template <size_t size, typename T>
SCANN_HIGHWAY_INLINE array<T, size> ToLocalArray(ConstSpan<T> span) {
  DCHECK_EQ(span.size(), size);
  array<T, size> result;
  std::copy(span.begin(), span.begin() + size, result.begin());
  return result;
}

#define SCANN_GET_INT16_DISTANCES_IMPL(batch_size, prefetch_strategy_int)    \
  SCANN_HIGHWAY_INLINE void                                                  \
      GetInt16Distances##batch_size##prefetch_strategy_int(                  \
          LUT16Args<int16_t> args) {                                         \
    const uint8_t* packed_dataset = args.packed_dataset;                     \
    const size_t num_32dp_simd_iters = args.num_32dp_simd_iters;             \
    const size_t num_blocks = args.num_blocks;                               \
    auto lookups = ToLocalArray<batch_size>(args.lookups);                   \
    auto distances = ToLocalArray<batch_size>(args.distances);               \
    array<array<int16_t, 32>, batch_size> result16;                          \
    for (size_t k : Seq(num_32dp_simd_iters)) {                              \
      const size_t dp_idx = k * 32;                                          \
      const uint8_t* data_start = packed_dataset + dp_idx * num_blocks / 2;  \
      HWY_NAMESPACE::HighwayLUT16MiddleLoop<                                 \
          batch_size, static_cast<PrefetchStrategy>(prefetch_strategy_int)>( \
          data_start, lookups, num_blocks, result16);                        \
      for (size_t j : Seq(batch_size)) {                                     \
        std::copy(result16[j].begin(), result16[j].end(),                    \
                  distances[j] + dp_idx);                                    \
      }                                                                      \
    }                                                                        \
  }

#define SCANN_GET_INT16_DISTANCES(batch_size)    \
  SCANN_GET_INT16_DISTANCES_IMPL(batch_size, 0); \
  SCANN_GET_INT16_DISTANCES_IMPL(batch_size, 1); \
  SCANN_GET_INT16_DISTANCES_IMPL(batch_size, 2);

SCANN_GET_INT16_DISTANCES(1);
SCANN_GET_INT16_DISTANCES(2);
SCANN_GET_INT16_DISTANCES(3);
SCANN_GET_INT16_DISTANCES(4);
SCANN_GET_INT16_DISTANCES(5);
SCANN_GET_INT16_DISTANCES(6);
SCANN_GET_INT16_DISTANCES(7);
SCANN_GET_INT16_DISTANCES(8);
SCANN_GET_INT16_DISTANCES(9);
#undef SCANN_GET_INT16_DISTANCES
#undef SCANN_GET_INT16_DISTANCES_IMPL

#define SCANN_GET_INT32_DISTANCES_IMPL(batch_size, prefetch_strategy_int)    \
  SCANN_HIGHWAY_INLINE void                                                  \
      GetInt32Distances##batch_size##prefetch_strategy_int(                  \
          LUT16Args<int32_t> args) {                                         \
    const uint8_t* packed_dataset = args.packed_dataset;                     \
    const size_t num_32dp_simd_iters = args.num_32dp_simd_iters;             \
    const size_t num_blocks = args.num_blocks;                               \
    auto lookups = ToLocalArray<batch_size>(args.lookups);                   \
    auto distances = ToLocalArray<batch_size>(args.distances);               \
    array<array<int32_t, 32>, batch_size> result32;                          \
    for (size_t k : Seq(num_32dp_simd_iters)) {                              \
      const size_t dp_idx = k * 32;                                          \
      const uint8_t* data_start = packed_dataset + dp_idx * num_blocks / 2;  \
      HWY_NAMESPACE::HighwayLUT16MiddleLoopInt32<                            \
          batch_size, static_cast<PrefetchStrategy>(prefetch_strategy_int)>( \
          data_start, lookups, num_blocks, result32);                        \
      for (size_t j : Seq(batch_size)) {                                     \
        std::copy(result32[j].begin(), result32[j].end(),                    \
                  distances[j] + dp_idx);                                    \
      }                                                                      \
    }                                                                        \
  }

#define SCANN_GET_INT32_DISTANCES(batch_size)    \
  SCANN_GET_INT32_DISTANCES_IMPL(batch_size, 0); \
  SCANN_GET_INT32_DISTANCES_IMPL(batch_size, 1); \
  SCANN_GET_INT32_DISTANCES_IMPL(batch_size, 2);

SCANN_GET_INT32_DISTANCES(1);
SCANN_GET_INT32_DISTANCES(2);
SCANN_GET_INT32_DISTANCES(3);
SCANN_GET_INT32_DISTANCES(4);
SCANN_GET_INT32_DISTANCES(5);
SCANN_GET_INT32_DISTANCES(6);
SCANN_GET_INT32_DISTANCES(7);
SCANN_GET_INT32_DISTANCES(8);
SCANN_GET_INT32_DISTANCES(9);
#undef SCANN_GET_INT32_DISTANCES
#undef SCANN_GET_INT32_DISTANCES_IMPL

#define SCANN_GET_FLOAT_DISTANCES_IMPL(batch_size, prefetch_strategy_int)    \
  SCANN_HIGHWAY_INLINE void                                                  \
      GetFloatDistances##batch_size##prefetch_strategy_int(                  \
          LUT16Args<float> args, ConstSpan<float> inv_fp_multipliers) {      \
    const uint8_t* packed_dataset = args.packed_dataset;                     \
    const size_t num_32dp_simd_iters = args.num_32dp_simd_iters;             \
    const size_t num_blocks = args.num_blocks;                               \
    auto lookups = ToLocalArray<batch_size>(args.lookups);                   \
    auto distances = ToLocalArray<batch_size>(args.distances);               \
    auto mults = ToLocalArray<batch_size>(inv_fp_multipliers);               \
    array<array<int32_t, 32>, batch_size> dist32;                            \
    for (size_t k : Seq(num_32dp_simd_iters)) {                              \
      const size_t dp_idx = k * 32;                                          \
      const uint8_t* data_start = packed_dataset + dp_idx * num_blocks / 2;  \
      HWY_NAMESPACE::HighwayLUT16MiddleLoopInt32<                            \
          batch_size, static_cast<PrefetchStrategy>(prefetch_strategy_int)>( \
          data_start, lookups, num_blocks, dist32);                          \
      for (size_t j : Seq(batch_size)) {                                     \
        HWY_NAMESPACE::ConvertToFloatMultiplyAndStore(                       \
            dist32[j].data(), mults[j], distances[j] + dp_idx);              \
      }                                                                      \
    }                                                                        \
  }

#define SCANN_GET_FLOAT_DISTANCES(batch_size)    \
  SCANN_GET_FLOAT_DISTANCES_IMPL(batch_size, 0); \
  SCANN_GET_FLOAT_DISTANCES_IMPL(batch_size, 1); \
  SCANN_GET_FLOAT_DISTANCES_IMPL(batch_size, 2);

SCANN_GET_FLOAT_DISTANCES(1);
SCANN_GET_FLOAT_DISTANCES(2);
SCANN_GET_FLOAT_DISTANCES(3);
SCANN_GET_FLOAT_DISTANCES(4);
SCANN_GET_FLOAT_DISTANCES(5);
SCANN_GET_FLOAT_DISTANCES(6);
SCANN_GET_FLOAT_DISTANCES(7);
SCANN_GET_FLOAT_DISTANCES(8);
SCANN_GET_FLOAT_DISTANCES(9);
#undef SCANN_GET_FLOAT_DISTANCES
#undef SCANN_GET_FLOAT_DISTANCES_IMPL

template <size_t kNumQueries, PrefetchStrategy kPrefetch, typename TopN>
SCANN_HIGHWAY_INLINE void GetTopInt16DistancesImpl(
    LUT16ArgsTopN<int16_t, TopN> args) {
  const uint8_t* packed_dataset = args.packed_dataset;
  const size_t num_32dp_simd_iters = args.num_32dp_simd_iters;
  const size_t num_blocks = args.num_blocks;
  auto lookups = ToLocalArray<kNumQueries>(args.lookups);
  const DatapointIndex first_dp_index = args.first_dp_index;
  const uint32_t final_mask = GetFinalMask32(args.num_datapoints);
  DCHECK_EQ(num_32dp_simd_iters, DivRoundUp(args.num_datapoints, 32));

  array<array<int16_t, 32>, kNumQueries> dist16;
  int16_t distances_buffer[32];

  int16_t simd_thresholds[kNumQueries];
  for (size_t j : Seq(kNumQueries)) {
    const int16_t int16_threshold = args.fast_topns[j]->epsilon();
    simd_thresholds[j] = int16_threshold;
  }

  typename TopN::Mutator topn_mutators[kNumQueries];
  for (size_t j : Seq(kNumQueries)) {
    args.fast_topns[j]->AcquireMutator(&topn_mutators[j]);
  }

  auto restrict_whitelist_ptrs =
      args.template GetRestrictWhitelistPtrs<kNumQueries>();
  for (DatapointIndex k : Seq(num_32dp_simd_iters)) {
    bool can_skip_batch = true;
    for (size_t j : Seq(kNumQueries)) {
      if (restrict_whitelist_ptrs[j] == nullptr ||
          restrict_whitelist_ptrs[j][k] != 0) {
        can_skip_batch = false;
        break;
      }
    }
    if (can_skip_batch) continue;

    const size_t dp_idx = k * 32;
    const uint8_t* data_start = packed_dataset + dp_idx * num_blocks / 2;
    HWY_NAMESPACE::HighwayLUT16MiddleLoop<kNumQueries, kPrefetch>(
        data_start, lookups, num_blocks, dist16);
    for (size_t j : Seq(kNumQueries)) {
      uint32_t push_mask =
          HWY_NAMESPACE::ComputePushMask(dist16[j].data(), simd_thresholds[j]);

      if (!push_mask) continue;

      std::copy(dist16[j].begin(), dist16[j].end(), distances_buffer);

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

          push_mask &= HWY_NAMESPACE::ComputePushMask(dist16[j].data(),
                                                      simd_thresholds[j]);
        }
      }
    }
  }
}

#define SCANN_GET_TOP_INT16_DISTANCES_IMPL(batch_size, prefetch_strategy_int)  \
  SCANN_HIGHWAY_INLINE void                                                    \
      GetTopInt16Distances1_##batch_size##prefetch_strategy_int(               \
          LUT16ArgsTopN<int16_t> args) {                                       \
    return GetTopInt16DistancesImpl<batch_size, static_cast<PrefetchStrategy>( \
                                                    prefetch_strategy_int)>(   \
        std::move(args));                                                      \
  }

#define SCANN_GET_TOP_INT16_DISTANCES(batch_size)    \
  SCANN_GET_TOP_INT16_DISTANCES_IMPL(batch_size, 0); \
  SCANN_GET_TOP_INT16_DISTANCES_IMPL(batch_size, 1); \
  SCANN_GET_TOP_INT16_DISTANCES_IMPL(batch_size, 2);

SCANN_GET_TOP_INT16_DISTANCES(1);
SCANN_GET_TOP_INT16_DISTANCES(2);
SCANN_GET_TOP_INT16_DISTANCES(3);
SCANN_GET_TOP_INT16_DISTANCES(4);
SCANN_GET_TOP_INT16_DISTANCES(5);
SCANN_GET_TOP_INT16_DISTANCES(6);
SCANN_GET_TOP_INT16_DISTANCES(7);
SCANN_GET_TOP_INT16_DISTANCES(8);
SCANN_GET_TOP_INT16_DISTANCES(9);
#undef SCANN_GET_TOP_INT16_DISTANCES
#undef SCANN_GET_TOP_INT16_DISTANCES_IMPL

SCANN_HIGHWAY_INLINE int16_t GetInt16Threshold(float float_threshold) {
  constexpr float kMaxValue = numeric_limits<int16_t>::max();
  return std::min(float_threshold, kMaxValue);
}

template <size_t kNumQueries, PrefetchStrategy kPrefetch, typename TopN>
SCANN_HIGHWAY_INLINE void GetTopFloatDistancesImpl(
    const bool with_spilling, LUT16ArgsTopN<float, TopN> args) {
  const uint8_t* packed_dataset = args.packed_dataset;
  const size_t num_32dp_simd_iters = args.num_32dp_simd_iters;
  const size_t num_blocks = args.num_blocks;
  auto lookups = ToLocalArray<kNumQueries>(args.lookups);
  const DatapointIndex first_dp_index = args.first_dp_index;
  const uint32_t final_mask = GetFinalMask32(args.num_datapoints);
  array<array<int16_t, 32>, kNumQueries> dist16;
  DCHECK_EQ(num_32dp_simd_iters, DivRoundUp(args.num_datapoints, 32));

  auto biases = ToLocalArray<kNumQueries>(args.biases);
  float simd_biases[kNumQueries];
  for (size_t j : Seq(kNumQueries)) {
    simd_biases[j] = biases[j];
  }

  auto mults = ToLocalArray<kNumQueries>(args.fixed_point_multipliers);
  float inv_mults[kNumQueries];
  for (size_t j : Seq(kNumQueries)) {
    inv_mults[j] = 1.0f / mults[j];
  }

  int16_t simd_thresholds[kNumQueries];
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

  float fvals[32];
  float distances_buffer[32];
  auto restrict_whitelist_ptrs =
      args.template GetRestrictWhitelistPtrs<kNumQueries>();
  for (DatapointIndex k : Seq(num_32dp_simd_iters)) {
    bool can_skip_batch = true;
    for (size_t j : Seq(kNumQueries)) {
      if (restrict_whitelist_ptrs[j] == nullptr ||
          restrict_whitelist_ptrs[j][k] != 0) {
        can_skip_batch = false;
        break;
      }
    }
    if (can_skip_batch) continue;

    const uint8_t* data_start = packed_dataset + k * 16 * num_blocks;
    HWY_NAMESPACE::HighwayLUT16MiddleLoop<kNumQueries, kPrefetch>(
        data_start, lookups, num_blocks, dist16);
    for (size_t j : Seq(kNumQueries)) {
      uint32_t push_mask =
          HWY_NAMESPACE::ComputePushMask(dist16[j].data(), simd_thresholds[j]);

      if (!push_mask) continue;

      HWY_NAMESPACE::ExpandConvertToFloatAndStore(dist16[j].data(), fvals);
      HWY_NAMESPACE::MultiplyAddAndStoreFloat(fvals, inv_mults[j],
                                              simd_biases[j], distances_buffer);

      if (k == num_32dp_simd_iters - 1) {
        push_mask &= final_mask;
      }
      if (restrict_whitelist_ptrs[j]) {
        push_mask &= restrict_whitelist_ptrs[j][k];
      }
      if (args.batch_filter_predicate) {
        push_mask =
            args.batch_filter_predicate(first_dp_index + 32 * k, push_mask,
                                        args.datapoint_translation_predicate);
      }

      while (push_mask) {
        const int offset = bits::FindLSBSetNonZero(push_mask);
        push_mask &= (push_mask - 1);
        DatapointIndex dp_idx;
        if (!with_spilling) {
          dp_idx = first_dp_index + 32 * k + offset;
        }

        DCHECK(
            !restrict_whitelist_ptrs[j] ||
            args.restrict_whitelists[j].IsWhitelisted(dp_idx - first_dp_index))
            << dp_idx;
        if (args.datapoint_translation_predicate) {
          dp_idx = args.datapoint_translation_predicate(dp_idx);
        }

        const bool needs_gc =
            topn_mutators[j].Push(dp_idx, distances_buffer[offset]);
        if (ABSL_PREDICT_FALSE(needs_gc)) {
          topn_mutators[j].GarbageCollect();

          const float new_epsilon = topn_mutators[j].epsilon();
          const float float_threshold = (new_epsilon - biases[j]) * mults[j];
          const int16_t int16_threshold = GetInt16Threshold(float_threshold);
          simd_thresholds[j] = int16_threshold;

          push_mask &= HWY_NAMESPACE::ComputePushMask(dist16[j].data(),
                                                      simd_thresholds[j]);
        }
      }
    }
  }
}

#define SCANN_GET_TOP_FLOAT_DISTANCES_IMPL(batch_size, prefetch_strategy_int)  \
  SCANN_HIGHWAY_INLINE void                                                    \
      GetTopFloatDistances1_##batch_size##prefetch_strategy_int(               \
          LUT16ArgsTopN<float> args) {                                         \
    return GetTopFloatDistancesImpl<batch_size, static_cast<PrefetchStrategy>( \
                                                    prefetch_strategy_int)>(   \
        false, std::move(args));                                               \
  }

#define SCANN_GET_TOP_FLOAT_DISTANCES(batch_size)   \
  SCANN_GET_TOP_FLOAT_DISTANCES_IMPL(batch_size, 0) \
  SCANN_GET_TOP_FLOAT_DISTANCES_IMPL(batch_size, 1) \
  SCANN_GET_TOP_FLOAT_DISTANCES_IMPL(batch_size, 2)

SCANN_GET_TOP_FLOAT_DISTANCES(1);
SCANN_GET_TOP_FLOAT_DISTANCES(2);
SCANN_GET_TOP_FLOAT_DISTANCES(3);
SCANN_GET_TOP_FLOAT_DISTANCES(4);
SCANN_GET_TOP_FLOAT_DISTANCES(5);
SCANN_GET_TOP_FLOAT_DISTANCES(6);
SCANN_GET_TOP_FLOAT_DISTANCES(7);
SCANN_GET_TOP_FLOAT_DISTANCES(8);
SCANN_GET_TOP_FLOAT_DISTANCES(9);
#undef SCANN_GET_TOP_FLOAT_DISTANCES
#undef SCANN_GET_TOP_FLOAT_DISTANCES_IMPL

#define SCANN_GET_TOP_FLOAT_DISTANCES_SPILLED_IMPL(batch_size,                 \
                                                   prefetch_strategy_int)      \
  SCANN_HIGHWAY_INLINE void                                                    \
      GetTopFloatDistancesSpilled1_##batch_size##prefetch_strategy_int(        \
          LUT16ArgsTopN<float> args) {                                         \
    return GetTopFloatDistancesImpl<batch_size, static_cast<PrefetchStrategy>( \
                                                    prefetch_strategy_int)>(   \
        true, std::move(args));                                                \
  }

#define SCANN_GET_TOP_FLOAT_DISTANCES_SPILLED(batch_size)   \
  SCANN_GET_TOP_FLOAT_DISTANCES_SPILLED_IMPL(batch_size, 0) \
  SCANN_GET_TOP_FLOAT_DISTANCES_SPILLED_IMPL(batch_size, 1) \
  SCANN_GET_TOP_FLOAT_DISTANCES_SPILLED_IMPL(batch_size, 2)

SCANN_GET_TOP_FLOAT_DISTANCES_SPILLED(1);
SCANN_GET_TOP_FLOAT_DISTANCES_SPILLED(2);
SCANN_GET_TOP_FLOAT_DISTANCES_SPILLED(3);
SCANN_GET_TOP_FLOAT_DISTANCES_SPILLED(4);
SCANN_GET_TOP_FLOAT_DISTANCES_SPILLED(5);
SCANN_GET_TOP_FLOAT_DISTANCES_SPILLED(6);
SCANN_GET_TOP_FLOAT_DISTANCES_SPILLED(7);
SCANN_GET_TOP_FLOAT_DISTANCES_SPILLED(8);
SCANN_GET_TOP_FLOAT_DISTANCES_SPILLED(9);
#undef SCANN_GET_TOP_FLOAT_DISTANCES_SPILLED
#undef SCANN_GET_TOP_FLOAT_DISTANCES_SPILLED_IMPL

}  // namespace HWY_NAMESPACE
}  // namespace asymmetric_hashing_internal
}  // namespace research_scann
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace research_scann {
namespace asymmetric_hashing_internal {

#define SCANN_EXPORT_ALL_PREFETCH_STRATEGIES(base_fn, batch_size) \
  HWY_EXPORT(base_fn##batch_size##0);                             \
  HWY_EXPORT(base_fn##batch_size##1);                             \
  HWY_EXPORT(base_fn##batch_size##2);

#define SCANN_HWY_EXPORT_ALL(batch_size)                                    \
  SCANN_EXPORT_ALL_PREFETCH_STRATEGIES(GetInt16Distances, batch_size);      \
  SCANN_EXPORT_ALL_PREFETCH_STRATEGIES(GetInt32Distances, batch_size);      \
  SCANN_EXPORT_ALL_PREFETCH_STRATEGIES(GetFloatDistances, batch_size);      \
  SCANN_EXPORT_ALL_PREFETCH_STRATEGIES(GetTopInt16Distances1_, batch_size); \
  SCANN_EXPORT_ALL_PREFETCH_STRATEGIES(GetTopFloatDistances1_, batch_size);

SCANN_HWY_EXPORT_ALL(1);
SCANN_HWY_EXPORT_ALL(2);
SCANN_HWY_EXPORT_ALL(3);
SCANN_HWY_EXPORT_ALL(4);
SCANN_HWY_EXPORT_ALL(5);
SCANN_HWY_EXPORT_ALL(6);
SCANN_HWY_EXPORT_ALL(7);
SCANN_HWY_EXPORT_ALL(8);
SCANN_HWY_EXPORT_ALL(9);
#undef SCANN_HWY_EXPORT_ALL
#undef SCANN_EXPORT_ALL_PREFETCH_STRATEGIES

#define SCANN_HWY_DISPATCH_PREFETCH(batch_size, prefetch_strategy, func_name, \
                                    ...)                                      \
  if constexpr (prefetch_strategy == PrefetchStrategy::kOff) {                \
    return HWY_DYNAMIC_DISPATCH(func_name##batch_size##0)(__VA_ARGS__);       \
  } else if constexpr (prefetch_strategy == PrefetchStrategy::kSeq) {         \
    return HWY_DYNAMIC_DISPATCH(func_name##batch_size##1)(__VA_ARGS__);       \
  } else {                                                                    \
    return HWY_DYNAMIC_DISPATCH(func_name##batch_size##2)(__VA_ARGS__);       \
  }

#define SCANN_HWY_DYNAMIC_DISPATCH(batch_size, prefetch_strategy, func_name, \
                                   ...)                                      \
  switch (batch_size) {                                                      \
    case 1:                                                                  \
      SCANN_HWY_DISPATCH_PREFETCH(1, prefetch_strategy, func_name,           \
                                  __VA_ARGS__);                              \
    case 2:                                                                  \
      SCANN_HWY_DISPATCH_PREFETCH(2, prefetch_strategy, func_name,           \
                                  __VA_ARGS__);                              \
    case 3:                                                                  \
      SCANN_HWY_DISPATCH_PREFETCH(3, prefetch_strategy, func_name,           \
                                  __VA_ARGS__);                              \
    case 4:                                                                  \
      SCANN_HWY_DISPATCH_PREFETCH(4, prefetch_strategy, func_name,           \
                                  __VA_ARGS__);                              \
    case 5:                                                                  \
      SCANN_HWY_DISPATCH_PREFETCH(5, prefetch_strategy, func_name,           \
                                  __VA_ARGS__);                              \
    case 6:                                                                  \
      SCANN_HWY_DISPATCH_PREFETCH(6, prefetch_strategy, func_name,           \
                                  __VA_ARGS__);                              \
    case 7:                                                                  \
      SCANN_HWY_DISPATCH_PREFETCH(7, prefetch_strategy, func_name,           \
                                  __VA_ARGS__);                              \
    case 8:                                                                  \
      SCANN_HWY_DISPATCH_PREFETCH(8, prefetch_strategy, func_name,           \
                                  __VA_ARGS__);                              \
    case 9:                                                                  \
      SCANN_HWY_DISPATCH_PREFETCH(9, prefetch_strategy, func_name,           \
                                  __VA_ARGS__);                              \
    default:                                                                 \
      DLOG(FATAL) << "Invalid Batch Size";                                   \
  }

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_HIGHWAY_OUTLINE void
LUT16Highway<kNumQueries, kPrefetch>::GetInt16Distances(
    LUT16Args<int16_t> args) {
  SCANN_HWY_DYNAMIC_DISPATCH(kNumQueries, kPrefetch, GetInt16Distances, args);
}

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_HIGHWAY_OUTLINE void
LUT16Highway<kNumQueries, kPrefetch>::GetInt32Distances(
    LUT16Args<int32_t> args) {
  SCANN_HWY_DYNAMIC_DISPATCH(kNumQueries, kPrefetch, GetInt32Distances, args);
}

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_HIGHWAY_OUTLINE void
LUT16Highway<kNumQueries, kPrefetch>::GetFloatDistances(
    LUT16Args<float> args, ConstSpan<float> inv_fp_multipliers) {
  SCANN_HWY_DYNAMIC_DISPATCH(kNumQueries, kPrefetch, GetFloatDistances, args,
                             inv_fp_multipliers);
}

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_HIGHWAY_OUTLINE void
LUT16Highway<kNumQueries, kPrefetch>::GetTopInt16Distances(
    LUT16ArgsTopN<int16_t> args) {
  SCANN_HWY_DYNAMIC_DISPATCH(kNumQueries, kPrefetch, GetTopInt16Distances1_,
                             args);
}

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_HIGHWAY_OUTLINE void
LUT16Highway<kNumQueries, kPrefetch>::GetTopFloatDistances(
    LUT16ArgsTopN<float> args) {
  SCANN_HWY_DYNAMIC_DISPATCH(kNumQueries, kPrefetch, GetTopFloatDistances1_,
                             args);
}

#undef SCANN_HWY_DYNAMIC_DISPATCH

}  // namespace asymmetric_hashing_internal
}  // namespace research_scann
#endif
#endif
