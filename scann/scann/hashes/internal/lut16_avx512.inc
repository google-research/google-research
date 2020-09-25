

#include "scann/hashes/internal/lut16_avx512.h"

#include "scann/utils/common.h"

#include "scann/oss_wrappers/scann_bits.h"

#ifdef __x86_64__

#include "scann/hashes/internal/lut16_avx512_swizzle.h"
#include "scann/utils/bits.h"
#include "scann/utils/intrinsics/avx512.h"
#include "tensorflow/core/platform/prefetch.h"

namespace tensorflow {
namespace scann_ops {
namespace avx512 {
namespace lut16 {

using asymmetric_hashing_internal::LUT16Args;
using asymmetric_hashing_internal::LUT16ArgsTopN;
using asymmetric_hashing_internal::PrefetchStrategy;

template <PrefetchStrategy kPrefetch0, bool kAlignedData0 = true>
struct LUT16Tuning {
  static constexpr PrefetchStrategy kPrefetch = kPrefetch0;
  static constexpr bool kAlignedData = kAlignedData0;

  using WithoutDataAlignment = LUT16Tuning<kPrefetch, false>;
};

inline bool IsCacheAligned(const uint8_t* data_start) {
  return IsDivisibleBy(reinterpret_cast<uintptr_t>(data_start), 64);
}

template <typename Tuning>
SCANN_AVX512_INLINE Avx512<uint8_t, 2> LoadDatabaseCodes(
    const uint8_t* data_start) {
  if constexpr (Tuning::kAlignedData) {
    DCHECK(IsCacheAligned(data_start));
  }
  if constexpr (Tuning::kPrefetch != PrefetchStrategy::kOff) {
    ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(
        data_start + asymmetric_hashing_internal::kPrefetchBytesAhead);
  }
  Avx512<uint8_t> interleaved_codes =
      Avx512<uint8_t>::Load<Tuning::kAlignedData>(data_start);

  Avx512<uint8_t, 2> codes;

  codes[0] = interleaved_codes & Avx512<uint8_t>(0x0F);

  codes[1] = Avx512<uint8_t>(Avx512<uint16_t>(interleaved_codes) >> 4) &
             Avx512<uint8_t>(0x0F);

  return codes;
}

template <size_t kNumCodesPerIter, typename Tuning>
SCANN_AVX512_INLINE Avx512<uint8_t> LoadLUT(const uint8_t* lookup_ptr) {
  if constexpr (kNumCodesPerIter == 4) {
    return Avx512<uint8_t>::Load(lookup_ptr);
  }
  if constexpr (kNumCodesPerIter == 2) {
    return _mm512_broadcast_i64x4(M256_32Xuint8::Load(lookup_ptr).val());
  }
  if constexpr (kNumCodesPerIter == 1) {
    return _mm512_broadcast_i64x2(M128_16Xuint8::Load(lookup_ptr).val());
  }
}

SCANN_AVX512_INLINE Avx512<int16_t, 4> LUT16Core(Avx512<uint8_t, 2> codes,
                                                 Avx512<uint8_t> lut) {
  const Avx512<uint8_t> vals00 = _mm512_shuffle_epi8(*lut, *codes[0]);
  const Avx512<uint8_t> vals64 = _mm512_shuffle_epi8(*lut, *codes[1]);

  Avx512<int16_t, 4> extr;
  extr[0] = Avx512<int16_t>(Avx512<uint16_t>(vals00));
  extr[1] = Avx512<int16_t>(Avx512<uint16_t>(vals00) >> 8);
  extr[2] = Avx512<int16_t>(Avx512<uint16_t>(vals64));
  extr[3] = Avx512<int16_t>(Avx512<uint16_t>(vals64) >> 8);

  return extr;
}

template <size_t kNumQueries, size_t kNumRegisters>
SCANN_AVX512_INLINE Avx512<int16_t, kNumQueries, kNumRegisters>
SubtractBiasTerm(
    Avx512<int16_t, kNumQueries, kNumRegisters> uint16_accumulators,
    size_t num_codes_per_dp) {
  Avx512<int16_t, kNumQueries, kNumRegisters> signed_results;
  constexpr size_t kBiasTerm = 128;
  for (size_t j : Seq(kNumQueries)) {
    signed_results[j] = Avx512<int16_t, kNumRegisters>(uint16_accumulators[j]);
    signed_results[j] -= Avx512<int16_t>(num_codes_per_dp * kBiasTerm);
  }
  return signed_results;
}

template <size_t kNumQueries, typename Tuning>
SCANN_AVX512_INLINE Avx512<int16_t, kNumQueries, 8> BottomLoop256(
    const uint8_t* data_start, array<const uint8_t*, kNumQueries> lookup_starts,
    const size_t num_codes_per_dp) {
  constexpr size_t kNumDatapoints = 256;
  constexpr size_t kNumCodesPerIter = 1;

  Avx512<int16_t, kNumQueries, 8> int16_accums = avx512::Zeros();
  for (auto _ : Seq(num_codes_per_dp)) {
    Avx512<uint8_t, 2> codes0 = LoadDatabaseCodes<Tuning>(data_start);
    Avx512<uint8_t, 2> codes1 = LoadDatabaseCodes<Tuning>(data_start + 64);
    data_start += 128;

    for (size_t j : Seq(kNumQueries)) {
      Avx512<uint8_t> lut = LoadLUT<kNumCodesPerIter, Tuning>(lookup_starts[j]);
      lookup_starts[j] += 16 * kNumCodesPerIter;

      int16_accums[j] +=
          Avx512Concat(LUT16Core(codes0, lut), LUT16Core(codes1, lut));
    }
  }

  static_assert(Avx512For<int16_t, kNumDatapoints>::kNumRegisters == 8);
  for (size_t j : Seq(kNumQueries)) {
    int16_accums[j][0] -= (int16_accums[j][1] << 8);
    int16_accums[j][2] -= (int16_accums[j][3] << 8);
    int16_accums[j][4] -= (int16_accums[j][5] << 8);
    int16_accums[j][6] -= (int16_accums[j][7] << 8);
  }

  return SubtractBiasTerm(int16_accums, num_codes_per_dp);
}

template <size_t kNumQueries, typename Tuning>
SCANN_AVX512_INLINE Avx512<int16_t, kNumQueries, 4> BottomLoop128(
    const uint8_t* data_start, array<const uint8_t*, kNumQueries> lookup_starts,
    const size_t num_codes_per_dp) {
  constexpr size_t kNumDatapoints = 128;
  constexpr size_t kNumCodesPerIter = 1;

  Avx512<int16_t, kNumQueries, 4> int16_accums = avx512::Zeros();
  for (auto _ : Seq(num_codes_per_dp)) {
    Avx512<uint8_t, 2> codes = LoadDatabaseCodes<Tuning>(data_start);
    data_start += 64;

    for (size_t j : Seq(kNumQueries)) {
      Avx512<uint8_t> lut = LoadLUT<kNumCodesPerIter, Tuning>(lookup_starts[j]);
      lookup_starts[j] += 16 * kNumCodesPerIter;

      int16_accums[j] += LUT16Core(codes, lut);
    }
  }

  static_assert(Avx512For<int16_t, kNumDatapoints>::kNumRegisters == 4);
  for (size_t j : Seq(kNumQueries)) {
    int16_accums[j][0] -= (int16_accums[j][1] << 8);
    int16_accums[j][2] -= (int16_accums[j][3] << 8);
  }

  return SubtractBiasTerm(int16_accums, num_codes_per_dp);
}

template <typename T, size_t kNumRegisters>
SCANN_AVX512_INLINE Avx512<T, kNumRegisters / 2> HorizontalSum2to1(
    Avx512<T, kNumRegisters> int16_accums) {
  static_assert(IsDivisibleBy(kNumRegisters, 2));
  constexpr size_t kAA = 0;
  constexpr size_t kBB = 4;
  Avx512<T, kNumRegisters / 2> reduced;
  for (size_t j : Seq(kNumRegisters / 2)) {
    Avx512<T> shuffled0 = fake_mm512_permutex2var_epi128(
        *int16_accums[2 * j + 0], *int16_accums[2 * j + 1],
        {kAA + 0, kAA + 2, kBB + 0, kBB + 2});
    Avx512<T> shuffled1 = fake_mm512_permutex2var_epi128(
        *int16_accums[2 * j + 0], *int16_accums[2 * j + 1],
        {kAA + 1, kAA + 3, kBB + 1, kBB + 3});
    reduced[j] = shuffled0 + shuffled1;
  }

  return reduced;
}

SCANN_AVX512_INLINE Avx512<int16_t, 1> PostProcess32(
    Avx512<int16_t, 4> int16_accums) {
  Avx512<int16_t, 4> resequenced = Avx512Concat(
      int16_accums[0], int16_accums[2], int16_accums[1], int16_accums[3]);

  return HorizontalSum2to1(HorizontalSum2to1(resequenced));
}

template <size_t kNumQueries, typename OriginalTuning>
SCANN_AVX512_INLINE Avx512<int16_t, kNumQueries, 1> BottomLoop32(
    const uint8_t* data_start, array<const uint8_t*, kNumQueries> lookup_starts,
    const size_t num_codes_per_dp) {
  constexpr size_t kNumDatapoints = 32;
  constexpr size_t kNumCodesPerIter = 4;

  using Tuning = typename OriginalTuning::WithoutDataAlignment;

  Avx512<int16_t, kNumQueries, 4> int16_accums = avx512::Zeros();

  const size_t num_iter = (num_codes_per_dp - 1) / kNumCodesPerIter;
  for (auto _ : Seq((num_iter))) {
    Avx512<uint8_t, 2> codes = LoadDatabaseCodes<Tuning>(data_start);
    data_start += 64;

    for (size_t j : Seq(kNumQueries)) {
      Avx512<uint8_t> lut = LoadLUT<kNumCodesPerIter, Tuning>(lookup_starts[j]);
      lookup_starts[j] += 16 * kNumCodesPerIter;

      int16_accums[j] += LUT16Core(codes, lut);
    }
  }

  {
    constexpr uint16_t kDisableCases[] = {
        0x0000,
        0xFFF0,
        0xFF00,
        0xF000,
    };
    const uint32_t kDisableBits = 0x80'80'80'80;
    const size_t codes_rem = num_codes_per_dp % kNumCodesPerIter;
    const __mmask16 disable_mask = _cvtu32_mask16(kDisableCases[codes_rem]);

    Avx512<uint8_t, 2> codes = LoadDatabaseCodes<Tuning>(data_start);
    codes[0] = _mm512_mask_set1_epi32(*codes[0], disable_mask, kDisableBits);
    codes[1] = _mm512_mask_set1_epi32(*codes[1], disable_mask, kDisableBits);

    for (size_t j : Seq(kNumQueries)) {
      Avx512<uint8_t> lut = LoadLUT<kNumCodesPerIter, Tuning>(lookup_starts[j]);
      lookup_starts[j] += 16 * kNumCodesPerIter;

      int16_accums[j] += LUT16Core(codes, lut);
    }
  }

  Avx512For<int16_t, kNumQueries, kNumDatapoints> results =
      avx512::Uninitialized();
  static_assert(Avx512For<int16_t, kNumDatapoints>::kNumRegisters == 1);
  for (size_t j : Seq(kNumQueries)) {
    int16_accums[j][0] -= (int16_accums[j][1] << 8);
    int16_accums[j][2] -= (int16_accums[j][3] << 8);

    results[j] = PostProcess32(int16_accums[j]);
  }

  return SubtractBiasTerm(results, num_codes_per_dp);
}

template <size_t kNumDatapoints, size_t kNumQueries, typename Tuning>
SCANN_AVX512_INLINE Avx512For<int16_t, kNumQueries, kNumDatapoints> BottomLoop(
    const uint8_t* data_start, array<const uint8_t*, kNumQueries> lookup_starts,
    const size_t num_codes_per_dp) {
  if constexpr (kNumDatapoints == 256) {
    return BottomLoop256<kNumQueries, Tuning>(data_start, lookup_starts,
                                              num_codes_per_dp);
  }
  if constexpr (kNumDatapoints == 128) {
    return BottomLoop128<kNumQueries, Tuning>(data_start, lookup_starts,
                                              num_codes_per_dp);
  }
  if constexpr (kNumDatapoints == 32) {
    return BottomLoop32<kNumQueries, Tuning>(data_start, lookup_starts,
                                             num_codes_per_dp);
  }
  static_assert(kNumDatapoints == 256 || kNumDatapoints == 128 ||
                kNumDatapoints == 32);

  LOG(FATAL) << "Unhandled.";
}

template <size_t kBottomLevelBatchSize, size_t kNumQueries>
SCANN_AVX512_INLINE array<const uint8_t*, kBottomLevelBatchSize>
MakeBottomLevelBatchLookupArray(
    array<const uint8_t*, kNumQueries> mid_level_lookups, size_t start) {
  DCHECK_LE(start + kBottomLevelBatchSize, kNumQueries);
  array<const uint8_t*, kBottomLevelBatchSize> result;
  for (size_t j : Seq(kBottomLevelBatchSize)) {
    result[j] = mid_level_lookups[start + j];
  }
  return result;
}

template <size_t kNumDatapoints, size_t kNumQueries, typename Tuning>
SCANN_AVX512_INLINE Avx512For<int16_t, kNumQueries, kNumDatapoints>
Int16MiddleLoop(const uint8_t* data_start,
                array<const uint8_t*, kNumQueries> lookup_starts,
                const size_t num_codes_per_dp) {
  if constexpr (kNumQueries <= 3) {
    return BottomLoop<kNumDatapoints, kNumQueries, Tuning>(
        data_start, lookup_starts, num_codes_per_dp);
  }

  constexpr size_t kSizeB = (kNumQueries == 1) ? 1 : 2;
  constexpr size_t kNumBCases[] = {0, 2, 1};
  constexpr size_t kNumB = (kNumQueries == 1) ? 1 : kNumBCases[kNumQueries % 3];

  constexpr size_t kRemaining = kNumQueries - kNumB * kSizeB;
  static_assert(kRemaining % 3 == 0, "");

  constexpr size_t kSizeA = 3;
  constexpr size_t kNumA = kRemaining / 3;

  Avx512For<int16_t, kNumQueries, kNumDatapoints> result;
  for (size_t j : Seq(kNumA)) {
    const size_t start = j * kSizeA;

    auto bottom_level_lookups =
        MakeBottomLevelBatchLookupArray<kSizeA>(lookup_starts, start);

    Avx512For<int16_t, kSizeA, kNumDatapoints> acc =
        BottomLoop<kNumDatapoints, kSizeA, Tuning>(
            data_start, bottom_level_lookups, num_codes_per_dp);

    for (size_t jj : Seq(kSizeA)) {
      result[start + jj] = acc[jj];
    }
  }

  for (size_t j : Seq(kNumB)) {
    const size_t start = kNumA * kSizeA + j * kSizeB;
    auto bottom_level_lookups =
        MakeBottomLevelBatchLookupArray<kSizeB>(lookup_starts, start);

    Avx512For<int16_t, kSizeB, kNumDatapoints> acc =
        BottomLoop<kNumDatapoints, kSizeB, Tuning>(
            data_start, bottom_level_lookups, num_codes_per_dp);

    for (size_t jj : Seq(kSizeB)) {
      result[start + jj] = acc[jj];
    }
  }

  return result;
}

template <size_t kNumDatapoints, size_t kNumQueries, typename Tuning>
SCANN_AVX512_INLINE Avx512For<int32_t, kNumQueries, kNumDatapoints>
Int32MiddleLoop(const uint8_t* data_start,
                array<const uint8_t*, kNumQueries> lookup_starts,
                const size_t num_codes_per_dp) {
  Avx512For<int32_t, kNumQueries, kNumDatapoints> int32_accums =
      avx512::Zeros();
  constexpr size_t kMaxCodesPerIter = 256;

  for (size_t k : SeqWithStride<kMaxCodesPerIter>(num_codes_per_dp)) {
    const size_t num_codes_this_iter =
        std::min<size_t>(kMaxCodesPerIter, num_codes_per_dp - k);

    Avx512For<int16_t, kNumQueries, kNumDatapoints> int16_accums =
        Int16MiddleLoop<kNumDatapoints, kNumQueries, Tuning>(
            data_start, lookup_starts, num_codes_this_iter);

    data_start += kNumDatapoints * num_codes_this_iter / 2;
    for (size_t j : Seq(kNumQueries)) {
      lookup_starts[j] += 16 * num_codes_this_iter;
      int32_accums[j] += int16_accums[j].template ExpandTo<int32_t>();
    }
  }
  return int32_accums;
}

template <size_t kNumDatapoints, size_t kNumQueries, typename Tuning>
SCANN_AVX512_INLINE Avx512For<float, kNumQueries, kNumDatapoints>
Float32MiddleLoop(const uint8_t* data_start,
                  array<const uint8_t*, kNumQueries> lookup_starts,
                  const size_t num_codes_per_dp,
                  array<float, kNumQueries> mults) {
  Avx512For<int32_t, kNumQueries, kNumDatapoints> int32_dists =
      Int32MiddleLoop<kNumDatapoints, kNumQueries, Tuning>(
          data_start, lookup_starts, num_codes_per_dp);

  Avx512For<float, kNumQueries, kNumDatapoints> float_dists;
  for (size_t j : Seq(kNumQueries)) {
    float_dists[j] =
        Avx512<float>(mults[j]) * int32_dists[j].template ConvertTo<float>();
  }

  return float_dists;
}

template <typename T, size_t kNumDatapoints, size_t kNumQueries,
          typename Tuning>
SCANN_AVX512_INLINE Avx512For<T, kNumQueries, kNumDatapoints> MiddleLoop(
    const uint8_t* data_start, array<const uint8_t*, kNumQueries> lookup_starts,
    const size_t num_codes_per_dp, array<float, kNumQueries> mults) {
  if constexpr (IsSame<T, int16_t>()) {
    return Int16MiddleLoop<kNumDatapoints, kNumQueries, Tuning>(
        data_start, lookup_starts, num_codes_per_dp);
  } else if constexpr (IsSame<T, int32_t>()) {
    return Int32MiddleLoop<kNumDatapoints, kNumQueries, Tuning>(
        data_start, lookup_starts, num_codes_per_dp);
  } else if constexpr (IsSame<T, float>()) {
    return Float32MiddleLoop<kNumDatapoints, kNumQueries, Tuning>(
        data_start, lookup_starts, num_codes_per_dp, mults);
  } else {
    static_assert(IsSameAny<T, int16_t, int32_t, float>());
    LOG(FATAL) << "Unhandled.";
  }
}

template <size_t size, typename T>
SCANN_INLINE array<T, size> ToLocalArray(ConstSpan<T> span) {
  DCHECK_EQ(span.size(), size);
  array<T, size> result;
  std::copy(span.begin(), span.begin() + size, result.begin());
  return result;
}

template <typename T, size_t kNumQueries, typename Tuning>
SCANN_AVX512_INLINE void GetDistancesImpl(
    LUT16Args<T> args, ConstSpan<float> inv_fp_multipliers = {}) {
  const uint8_t* data_start = args.packed_dataset;
  const size_t num_codes_per_dp = args.num_blocks;
  array<const uint8_t*, kNumQueries> lookups =
      ToLocalArray<kNumQueries>(args.lookups);
  array<T*, kNumQueries> distances_ptrs =
      ToLocalArray<kNumQueries>(args.distances);
  array<float, kNumQueries> mults;
  if constexpr (IsSame<T, float>()) {
    mults = ToLocalArray<kNumQueries>(inv_fp_multipliers);
  }

  DCHECK(data_start);
  DCHECK(IsCacheAligned(data_start));
  for (size_t j : Seq(kNumQueries)) {
    DCHECK(lookups[j]);
    DCHECK(IsCacheAligned(lookups[j]));
  }

  const size_t num_256dp_simd_iters = args.num_32dp_simd_iters / 8;
  for (auto _ : Seq(num_256dp_simd_iters)) {
    Avx512For<T, kNumQueries, 256> dists =
        MiddleLoop<T, 256, kNumQueries, Tuning>(data_start, lookups,
                                                num_codes_per_dp, mults);
    data_start += 256 * num_codes_per_dp / 2;
    for (size_t j : Seq(kNumQueries)) {
      dists[j].Store(distances_ptrs[j]);
      distances_ptrs[j] += 256;
    }
  }
  args.num_32dp_simd_iters %= 8;

  const size_t num_128dp_simd_iters = args.num_32dp_simd_iters / 4;
  for (auto _ : Seq(num_128dp_simd_iters)) {
    Avx512For<T, kNumQueries, 128> dists =
        MiddleLoop<T, 128, kNumQueries, Tuning>(data_start, lookups,
                                                num_codes_per_dp, mults);
    data_start += 128 * num_codes_per_dp / 2;
    for (size_t j : Seq(kNumQueries)) {
      dists[j].Store(distances_ptrs[j]);
      distances_ptrs[j] += 128;
    }
  }
  args.num_32dp_simd_iters %= 4;

  const size_t num_32dp_simd_iters = args.num_32dp_simd_iters;
  for (auto _ : Seq(num_32dp_simd_iters)) {
    Avx512For<T, kNumQueries, 32> dists =
        MiddleLoop<T, 32, kNumQueries, Tuning>(data_start, lookups,
                                               num_codes_per_dp, mults);
    data_start += 32 * num_codes_per_dp / 2;
    for (size_t j : Seq(kNumQueries)) {
      dists[j].Store(distances_ptrs[j]);
      distances_ptrs[j] += 32;
    }
  }
}

}  // namespace lut16
}  // namespace avx512

namespace asymmetric_hashing_internal {

using avx512::lut16::GetDistancesImpl;
using avx512::lut16::LUT16Tuning;

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_AVX512_OUTLINE void
LUT16Avx512<kNumQueries, kPrefetch>::GetInt16Distances(
    LUT16Args<int16_t> args) {
  using Tuning = LUT16Tuning<kPrefetch>;
  GetDistancesImpl<int16_t, kNumQueries, Tuning>(std::move(args), {});
}

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_AVX512_OUTLINE void
LUT16Avx512<kNumQueries, kPrefetch>::GetInt32Distances(
    LUT16Args<int32_t> args) {
  using Tuning = LUT16Tuning<kPrefetch>;
  GetDistancesImpl<int32_t, kNumQueries, Tuning>(std::move(args), {});
}

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
SCANN_AVX512_OUTLINE void
LUT16Avx512<kNumQueries, kPrefetch>::GetFloatDistances(
    LUT16Args<float> args, ConstSpan<float> inv_fp_multipliers) {
  using Tuning = LUT16Tuning<kPrefetch>;
  GetDistancesImpl<float, kNumQueries, Tuning>(std::move(args),
                                               inv_fp_multipliers);
}

}  // namespace asymmetric_hashing_internal

namespace avx512 {
namespace lut16 {

template <size_t kNumRegisters>
SCANN_AVX512_INLINE Avx512<float, 2 * kNumRegisters> FloatConversion(
    Avx512<int16_t, kNumRegisters> int16_dists, Avx512<float> inv_multiplier,
    Avx512<float> bias) {
  return int16_dists.template ExpandTo<int32_t>().template ConvertTo<float>() *
             inv_multiplier +
         bias;
}

SCANN_INLINE constexpr uint64_t GetFinalMask64(size_t num_datapoints) {
  const uint64_t remainder_bits = num_datapoints % 64;
  constexpr uint64_t kOne = 1;
  constexpr uint64_t kZero = 0;
  return (remainder_bits ? (kOne << remainder_bits) : kZero) - kOne;
}
static_assert(GetFinalMask64(8) == 0xFF);

template <typename T, size_t kNumQueries, typename Tuning,
          bool kIgnoreWhitelist = false, int kNumCodes = -1, typename TopN>
SCANN_AVX512_OUTLINE void GetTopDistancesImpl(LUT16ArgsTopN<T, TopN> args) {
  static_assert(IsSameAny<T, int16_t, float>());
  const uint8_t* data_start = args.packed_dataset;
  const size_t num_codes_per_dp = (kNumCodes > 0) ? kNumCodes : args.num_blocks;
  auto lookups = ToLocalArray<kNumQueries>(args.lookups);
  size_t first_dp_index = args.first_dp_index;
  size_t num_32dp_simd_iters = args.num_32dp_simd_iters;
  DCHECK_EQ(num_32dp_simd_iters, DivRoundUp(args.num_datapoints, 32));

  DCHECK(data_start);
  DCHECK(IsCacheAligned(data_start));
  for (size_t j : Seq(kNumQueries)) {
    DCHECK(lookups[j]);
    DCHECK(IsCacheAligned(lookups[j]));
  }

  typename TopN::Mutator topn_mutators[kNumQueries];
  for (size_t j : Seq(kNumQueries)) {
    args.fast_topns[j]->AcquireMutator(&topn_mutators[j]);
  }

  array<const uint64_t*, kNumQueries> whitelist_ptrs;
  if constexpr (!kIgnoreWhitelist) {
    auto whitelists = args.template GetRestrictWhitelistPtrs<kNumQueries>();
    for (size_t j : Seq(kNumQueries)) {
      whitelist_ptrs[j] = reinterpret_cast<const uint64_t*>(whitelists[j]);
    }
  }

  constexpr size_t kNumQueriesFloat = IsSame<T, float>() ? kNumQueries : 0;
  array<float, kNumQueriesFloat> mults;
  array<float, kNumQueriesFloat> biases;
  array<Avx512<float>, kNumQueriesFloat> simd_inv_mults;
  array<Avx512<float>, kNumQueriesFloat> simd_biases;
  if constexpr (IsSame<T, float>()) {
    for (size_t j : Seq(kNumQueriesFloat)) {
      mults[j] = args.fixed_point_multipliers[j];
      biases[j] = args.biases[j];
      simd_inv_mults[j] = (1.0 / args.fixed_point_multipliers[j]);
      simd_biases[j] = args.biases[j];
    }
  }

  Avx512<int16_t> simd_thresholds[kNumQueries];
  auto update_simd_threshold = [&](size_t j) SCANN_AVX512_INLINE_LAMBDA {
    if constexpr (IsSame<T, int16_t>()) {
      return topn_mutators[j].epsilon();
    }
    if constexpr (IsSame<T, float>()) {
      const float new_epsilon = topn_mutators[j].epsilon();
      const float equiv_int16_threshold = (new_epsilon - biases[j]) * mults[j];
      constexpr float kMaxThreshold = numeric_limits<int16_t>::max();
      return static_cast<int16_t>(
          std::min(equiv_int16_threshold, kMaxThreshold));
    }
  };
  for (size_t j : Seq(kNumQueries)) {
    simd_thresholds[j] = update_simd_threshold(j);
  }

  T distances_buffer[256];

  while (num_32dp_simd_iters >= 8) {
    num_32dp_simd_iters -= 8;
    constexpr size_t kNumDatapoints = 256;
    Avx512For<int16_t, kNumQueries, kNumDatapoints> int16_dists =
        Int16MiddleLoop<kNumDatapoints, kNumQueries, Tuning>(
            data_start, lookups, num_codes_per_dp);
    data_start += kNumDatapoints * num_codes_per_dp / 2;

    for (size_t j : Seq(kNumQueries)) {
      if constexpr (IsSame<T, int16_t>()) {
        int16_dists[j].Store(distances_buffer);
      }
      if constexpr (IsSame<T, float>()) {
        Avx512For<float, kNumDatapoints> float_dists =
            FloatConversion(int16_dists[j], simd_inv_mults[j], simd_biases[j]);
        float_dists.Store(distances_buffer);
      }

      constexpr size_t kNumMasks =
          Avx512For<int16_t, kNumDatapoints>::kNumRegisters / 2;

      auto compute_push_mask = [&](size_t mm) SCANN_AVX512_INLINE_LAMBDA {
        array<uint32_t, 2> mask;
        _store_mask32(&mask[0],
                      (int16_dists[j][2 * mm + 0] < simd_thresholds[j])[0]);
        _store_mask32(&mask[1],
                      (int16_dists[j][2 * mm + 1] < simd_thresholds[j])[0]);
        return *reinterpret_cast<uint64_t*>(&mask);
      };

      auto handle_push_mask = [&](size_t mm) SCANN_AVX512_INLINE_LAMBDA {
        uint64_t push_mask = compute_push_mask(mm);
        if constexpr (!kIgnoreWhitelist) {
          if (whitelist_ptrs[j]) {
            push_mask &= *whitelist_ptrs[j];
            whitelist_ptrs[j]++;
          }
        }
        if (mm == kNumMasks - 1) {
          if (!num_32dp_simd_iters) {
            push_mask &= GetFinalMask64(args.num_datapoints);
          }
        }

        while (push_mask) {
          const int offset = bits::FindLSBSetNonZero64(push_mask);
          push_mask &= (push_mask - 1);
          const size_t dp_idx = first_dp_index + 64 * mm + offset;
          const T distance = distances_buffer[64 * mm + offset];
          const bool needs_collection = topn_mutators[j].Push(dp_idx, distance);
          if (ABSL_PREDICT_FALSE(needs_collection)) {
            topn_mutators[j].GarbageCollect();

            simd_thresholds[j] = update_simd_threshold(j);

            push_mask &= compute_push_mask(mm);
          }
        }
      };

      static_assert(kNumMasks == 4);
      handle_push_mask(0);
      handle_push_mask(1);
      handle_push_mask(2);
      handle_push_mask(3);
    }
    first_dp_index += 256;
  }

  while (num_32dp_simd_iters >= 4) {
    num_32dp_simd_iters -= 4;
    constexpr size_t kNumDatapoints = 128;
    Avx512For<int16_t, kNumQueries, kNumDatapoints> int16_dists =
        Int16MiddleLoop<kNumDatapoints, kNumQueries, Tuning>(
            data_start, lookups, num_codes_per_dp);
    data_start += kNumDatapoints * num_codes_per_dp / 2;

    for (size_t j : Seq(kNumQueries)) {
      if constexpr (IsSame<T, int16_t>()) {
        int16_dists[j].Store(distances_buffer);
      }
      if constexpr (IsSame<T, float>()) {
        Avx512For<float, kNumDatapoints> float_dists =
            FloatConversion(int16_dists[j], simd_inv_mults[j], simd_biases[j]);
        float_dists.Store(distances_buffer);
      }

      constexpr size_t kNumMasks =
          Avx512For<int16_t, kNumDatapoints>::kNumRegisters / 2;

      auto compute_push_mask = [&](size_t mm) SCANN_AVX512_INLINE_LAMBDA {
        array<uint32_t, 2> mask;
        _store_mask32(&mask[0],
                      (int16_dists[j][2 * mm + 0] < simd_thresholds[j])[0]);
        _store_mask32(&mask[1],
                      (int16_dists[j][2 * mm + 1] < simd_thresholds[j])[0]);
        return *reinterpret_cast<uint64_t*>(&mask);
      };

      auto handle_push_mask = [&](size_t mm) SCANN_AVX512_INLINE_LAMBDA {
        uint64_t push_mask = compute_push_mask(mm);
        if constexpr (!kIgnoreWhitelist) {
          if (whitelist_ptrs[j]) {
            push_mask &= *whitelist_ptrs[j];
            whitelist_ptrs[j]++;
          }
        }

        if (mm == kNumMasks - 1) {
          if (!num_32dp_simd_iters) {
            push_mask &= GetFinalMask64(args.num_datapoints);
          }
        }

        while (push_mask) {
          const int offset = bits::FindLSBSetNonZero64(push_mask);
          push_mask &= (push_mask - 1);
          const size_t dp_idx = first_dp_index + 64 * mm + offset;
          const T distance = distances_buffer[64 * mm + offset];
          const bool needs_collection = topn_mutators[j].Push(dp_idx, distance);
          if (ABSL_PREDICT_FALSE(needs_collection)) {
            topn_mutators[j].GarbageCollect();

            simd_thresholds[j] = update_simd_threshold(j);

            push_mask &= compute_push_mask(mm);
          }
        }
      };

      static_assert(kNumMasks == 2);
      handle_push_mask(0);
      handle_push_mask(1);
    }
    first_dp_index += 128;
  }

  for (size_t k : Seq(num_32dp_simd_iters)) {
    Avx512For<int16_t, kNumQueries, 32> int16_dists =
        Int16MiddleLoop<32, kNumQueries, Tuning>(data_start, lookups,
                                                 num_codes_per_dp);
    data_start += 32 * num_codes_per_dp / 2;

    for (size_t j : Seq(kNumQueries)) {
      auto compute_push_mask = [&]() SCANN_AVX512_INLINE_LAMBDA {
        static_assert(std::decay_t<decltype(int16_dists[j])>::kNumRegisters ==
                      1);
        return _cvtmask32_u32((int16_dists[j] < simd_thresholds[j])[0]);
      };
      uint32_t push_mask = compute_push_mask();

      if (!push_mask) continue;

      if constexpr (IsSame<T, int16_t>()) {
        int16_dists[j].Store(distances_buffer);
      }
      if constexpr (IsSame<T, float>()) {
        FloatConversion(int16_dists[j], simd_inv_mults[j], simd_biases[j])
            .Store(distances_buffer);
      }

      if constexpr (!kIgnoreWhitelist) {
        if (whitelist_ptrs[j]) {
          push_mask &= reinterpret_cast<const uint32_t*>(whitelist_ptrs[j])[k];
        }
      }

      if (ABSL_PREDICT_FALSE(k == num_32dp_simd_iters - 1)) {
        push_mask &= GetFinalMask32(args.num_datapoints);
      }

      while (push_mask) {
        const int offset = bits::FindLSBSetNonZero(push_mask);
        push_mask &= (push_mask - 1);
        const size_t dp_idx = first_dp_index + 32 * k + offset;
        const T distance = distances_buffer[offset];
        const bool needs_collection = topn_mutators[j].Push(dp_idx, distance);
        if (ABSL_PREDICT_FALSE(needs_collection)) {
          topn_mutators[j].GarbageCollect();

          simd_thresholds[j] = update_simd_threshold(j);

          push_mask &= compute_push_mask();
        }
      }
    }
  }
}

}  // namespace lut16
}  // namespace avx512

namespace asymmetric_hashing_internal {

using avx512::lut16::GetTopDistancesImpl;

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
void LUT16Avx512<kNumQueries, kPrefetch>::GetTopInt16Distances(
    LUT16ArgsTopN<int16_t> args) {
  using Tuning = LUT16Tuning<kPrefetch>;

  if (args.restrict_whitelists.empty()) {
    return GetTopDistancesImpl<int16_t, kNumQueries, Tuning, true>(
        std::move(args));
  } else {
    return GetTopDistancesImpl<int16_t, kNumQueries, Tuning, false>(
        std::move(args));
  }
}

template <size_t kNumQueries, PrefetchStrategy kPrefetch>
void LUT16Avx512<kNumQueries, kPrefetch>::GetTopFloatDistances(
    LUT16ArgsTopN<float> args) {
  using Tuning = LUT16Tuning<kPrefetch>;
  return GetTopDistancesImpl<float, kNumQueries, Tuning>(std::move(args));
}

}  // namespace asymmetric_hashing_internal
}  // namespace scann_ops
}  // namespace tensorflow

#endif
