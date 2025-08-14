

template <int kRegisterIdx, typename T = Simd<float>>
SCANN_SIMD_INLINE auto ExpandToFp32(Sse4<int8_t> int8_vals) {
  if constexpr (IsSame<T, Avx512<float>>()) {
    return Avx512<float>{_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(*int8_vals))};
  }
  if constexpr (IsSame<T, Avx2<float>>()) {
    return Avx2<float>{AvxFunctionsAvx2Fma::Int8ToFloatLower(
        _mm_srli_si128(*int8_vals, 8 * kRegisterIdx))};
  }
  if constexpr (IsSame<T, Avx1<float>>()) {
    return Avx1<float>{AvxFunctionsAvx::Int8ToFloatLower(
        _mm_srli_si128(*int8_vals, 8 * kRegisterIdx))};
  }
  if constexpr (IsSame<T, Sse4<float>>()) {
    return Sse4<float>{_mm_cvtepi32_ps(
        _mm_cvtepi8_epi32(_mm_srli_si128(*int8_vals, 4 * kRegisterIdx)))};
  }
  LOG(FATAL) << "Unhandled: " << SimdName();
}

template <int kRegisterIdx, typename T = Simd<float>>
SCANN_SIMD_INLINE auto ExpandToFp32(Sse4<int16_t> bf16_vals) {
  if constexpr (IsSame<T, Avx2<float>>()) {
    static_assert(kRegisterIdx == 0);
    return Avx2<float>{
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(*bf16_vals), 16)};
  }
  if constexpr (IsSame<T, Avx1<float>>()) {
    static_assert(kRegisterIdx == 0);
    __m128i zeros = _mm_setzero_si128();
    __m128 lo = _mm_castsi128_ps(_mm_unpacklo_epi16(zeros, *bf16_vals));
    __m128 hi = _mm_castsi128_ps(_mm_unpackhi_epi16(zeros, *bf16_vals));
    return Avx1<float>{_mm256_set_m128(hi, lo)};
  }
  if constexpr (IsSame<T, Sse4<float>>()) {
    __m128i zeros = _mm_setzero_si128();
    if constexpr (kRegisterIdx == 0) {
      return Sse4<float>{_mm_unpacklo_epi16(zeros, *bf16_vals)};
    } else {
      static_assert(kRegisterIdx == 1);
      return Sse4<float>{_mm_unpackhi_epi16(zeros, *bf16_vals)};
    }
  }
  LOG(FATAL) << "Unhandled: " << SimdName();
}

template <int kRegisterIdx, typename T = Simd<float>>
SCANN_SIMD_INLINE auto ExpandToFp32(Avx2<int16_t> bf16_vals) {
  if constexpr (IsSame<T, Avx512<float>>()) {
    static_assert(kRegisterIdx == 0);
    return Avx512<float>{
        _mm512_slli_epi32(_mm512_cvtepu16_epi32(*bf16_vals), 16)};
  }
  if constexpr (IsSame<T, Avx2<float>>()) {
    __m256i zeros = _mm256_setzero_si256();
    __m256i permed = _mm256_permute4x64_epi64(*bf16_vals, 0b11'01'10'00);
    if constexpr (kRegisterIdx == 0) {
      return Avx2<float>{_mm256_unpacklo_epi16(zeros, permed)};
    } else {
      static_assert(kRegisterIdx == 1);
      return Avx2<float>{_mm256_unpackhi_epi16(zeros, permed)};
    }
  }
  LOG(FATAL) << "Unhandled: " << SimdName();
}

SCANN_SIMD_INLINE auto SseToSimd(Sse4<float> float_vals) {
  using T = Simd<float>;
  if constexpr (IsSame<T, Avx512<float>>()) {
    return Avx512<float>(
        _mm512_insertf32x4(_mm512_setzero_ps(), *float_vals, 0));
  }
  if constexpr (IsSame<T, Avx2<float>>()) {
    return Avx2<float>(AvxFunctionsAvx2Fma::SseToAvx(*float_vals));
  }
  if constexpr (IsSame<T, Avx1<float>>()) {
    return Avx1<float>(AvxFunctionsAvx::SseToAvx(*float_vals));
  }
  if constexpr (IsSame<T, Sse4<float>>()) {
    return float_vals;
  }
  LOG(FATAL) << "Unhandled: " << SimdName();
}

template <size_t kNumDims>
SCANN_SIMD_INLINE auto LoadXFloats(const float* ptr) {
  if constexpr (kNumDims >= Simd<float>::kNumElements) {
    return SimdFor<float, kNumDims>::Load(ptr);
  } else if constexpr (kNumDims == 8) {
    return Avx512<float>(
        _mm512_insertf32x8(_mm512_setzero_ps(), *Avx2<float>::Load(ptr), 0));
  } else if constexpr (kNumDims == 4) {
    return SseToSimd(Sse4<float>::Load(ptr));
  }
}

template <bool kIsSquaredL2, typename DataT, typename SimdT>
SCANN_SIMD_INLINE SimdT FusedMultiplyOp(SimdT a, SimdT b, SimdT mult,
                                        SimdT accum) {
  if constexpr (!kIsSquaredL2) {
    return FusedMultiplySubtract(a, b, accum);
  } else {
    if constexpr (std::is_same_v<DataT, int8_t>) {
      SimdT diff = (a - b * mult);
      return FusedMultiplyAdd(diff, diff, accum);
    } else {
      SimdT diff = a - b;
      return FusedMultiplyAdd(diff, diff, accum);
    }
  }
}

template <size_t kNumDims, bool kIsSquaredL2, size_t kUnrollBy, typename DataT>
SCANN_SIMD_INLINE Simd<float, kUnrollBy> HandleXDims(
    const float* query, array<const DataT*, kUnrollBy> ptrs,
    const float* inv_multipliers_for_squared_l2, size_t dim,
    Simd<float, kUnrollBy> accums) {
  static_assert(std::is_same_v<DataT, int8_t> ||
                std::is_same_v<DataT, int16_t>);

  auto qq_vals = LoadXFloats<kNumDims>(query + dim);

  static_assert(kNumDims == 4 || kNumDims == 8 || kNumDims == 16);

  static_assert(std::is_same_v<DataT, int8_t> || kNumDims <= 8 ||
                Simd<int16_t>::kRegisterBits >= 256);

  std::conditional_t<kNumDims == 16 && std::is_same_v<DataT, int16_t>,
                     Avx2<DataT, kUnrollBy>, Sse4<DataT, kUnrollBy>>
      db_vals;
  for (size_t jj : Seq(kUnrollBy)) {
    if constexpr (std::is_same_v<DataT, int8_t>) {
      if constexpr (kNumDims == 16) {
        db_vals[jj] = Sse4<int8_t>::Load(ptrs[jj] + dim);
      }
      if constexpr (kNumDims == 8) {
        db_vals[jj] =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptrs[jj] + dim));
      }
      if constexpr (kNumDims == 4) {
        db_vals[jj] =
            _mm_cvtsi32_si128(ABSL_INTERNAL_UNALIGNED_LOAD32(ptrs[jj] + dim));
      }
    } else {
      if constexpr (kNumDims == 16) {
        db_vals[jj] = Avx2<int16_t>::Load(ptrs[jj] + dim);
      }
      if constexpr (kNumDims == 8) {
        db_vals[jj] = Sse4<int16_t>::Load(ptrs[jj] + dim);
      }
      if constexpr (kNumDims == 4) {
        db_vals[jj] =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptrs[jj] + dim));
      }
    }
  }

  decltype(qq_vals) mult;
  if constexpr (kIsSquaredL2 && std::is_same_v<DataT, int8_t>) {
    mult = LoadXFloats<kNumDims>(inv_multipliers_for_squared_l2 + dim);
  }

  asm("" ::: "memory");

  if constexpr (kNumDims == 4) {
    for (size_t jj : Seq(kUnrollBy)) {
      Simd<float> db_vals_float =
          SseToSimd(ExpandToFp32<0, Sse4<float>>(db_vals[jj]));
      accums[jj] = FusedMultiplyOp<kIsSquaredL2, DataT>(qq_vals, db_vals_float,
                                                        mult, accums[jj]);
    }
    return accums;
  }

  if constexpr (decltype(qq_vals)::kNumRegisters >= 1) {
    for (size_t jj : Seq(kUnrollBy)) {
      accums[jj] = FusedMultiplyOp<kIsSquaredL2, DataT>(
          qq_vals[0], ExpandToFp32<0>(db_vals[jj]), mult[0], accums[jj]);
    }
  }
  if constexpr (decltype(qq_vals)::kNumRegisters >= 2) {
    for (size_t jj : Seq(kUnrollBy)) {
      accums[jj] = FusedMultiplyOp<kIsSquaredL2, DataT>(
          qq_vals[1], ExpandToFp32<1>(db_vals[jj]), mult[1], accums[jj]);
    }
  }
  if constexpr (decltype(qq_vals)::kNumRegisters >= 3) {
    for (size_t jj : Seq(kUnrollBy)) {
      accums[jj] = FusedMultiplyOp<kIsSquaredL2, DataT>(
          qq_vals[2], ExpandToFp32<2>(db_vals[jj]), mult[2], accums[jj]);
    }
  }
  if constexpr (decltype(qq_vals)::kNumRegisters >= 4) {
    for (size_t jj : Seq(kUnrollBy)) {
      accums[jj] = FusedMultiplyOp<kIsSquaredL2, DataT>(
          qq_vals[3], ExpandToFp32<3>(db_vals[jj]), mult[3], accums[jj]);
    }
  }
  static_assert(decltype(qq_vals)::kNumRegisters <= 4);

  return accums;
}

SCANN_SIMD_INLINE double StaticallyInvokeOneToOneDenseDotProduct(
    const DatapointPtr<float>& qq, const DatapointPtr<int8_t>& db) {
  using T = Simd<float>;
  if constexpr (IsSame<T, Avx512<float>>()) {
    return ::research_scann::dp_internal::DenseDotProductAvx2(db, qq);
  }
  if constexpr (IsSame<T, Avx2<float>>()) {
    return ::research_scann::dp_internal::DenseDotProductAvx2(db, qq);
  }
  if constexpr (IsSame<T, Avx1<float>>()) {
    return ::research_scann::dp_internal::DenseDotProductAvx1(db, qq);
  }
  if constexpr (IsSame<T, Sse4<float>>()) {
    return ::research_scann::dp_internal::DenseDotProductSse4(db, qq);
  }
  LOG(FATAL) << "Unhandled: " << SimdName();
}

template <size_t kDimensionality, bool kIsSquaredL2>
SCANN_SIMD_INLINE float ComputeOneToOneScore(
    const float* __restrict__ query, const int8_t* ptr,
    const float* __restrict__ inv_multipliers_for_squared_l2,
    size_t dimensionality) {
  if constexpr (kIsSquaredL2) {
    array<const int8_t*, 1> ptrs = {ptr};
    Simd<float, 1> accums = Zeros();
    size_t dim = 0;
    for (; dim + 16 <= dimensionality; dim += 16) {
      accums = HandleXDims<16, kIsSquaredL2>(
          query, ptrs, inv_multipliers_for_squared_l2, dim, accums);
    }

    float dist = HorizontalSum(accums[0]);

    for (; dim < dimensionality; dim++) {
      const float mult = inv_multipliers_for_squared_l2[dim];
      dist = FusedMultiplyOp<kIsSquaredL2, int8_t>(
          query[dim], static_cast<float>(ptr[dim]), mult, dist);
    }
    return dist;
  } else {
    DatapointPtr<float> qq_dptr(nullptr, query, dimensionality, dimensionality);
    DatapointPtr<int8_t> db_dptr(nullptr, ptr, dimensionality, dimensionality);
    return -StaticallyInvokeOneToOneDenseDotProduct(qq_dptr, db_dptr);
  }
}

template <size_t kDimensionality, bool kIsSquaredL2>
SCANN_SIMD_INLINE float ComputeOneToOneScore(
    const float* __restrict__ query, const int16_t* ptr,
    const float* __restrict__ inv_multipliers_for_squared_l2,
    size_t dimensionality) {
  array<const int16_t*, 1> ptrs = {ptr};
  Simd<float, 1> accums = Zeros();
  size_t dim = 0;
  if constexpr (Simd<int16_t>::kRegisterBits >= 256) {
    for (; dim + 16 <= dimensionality; dim += 16) {
      accums = HandleXDims<16, kIsSquaredL2>(
          query, ptrs, inv_multipliers_for_squared_l2, dim, accums);
    }
  } else {
    for (; dim + 8 <= dimensionality; dim += 8) {
      accums = HandleXDims<8, kIsSquaredL2>(
          query, ptrs, inv_multipliers_for_squared_l2, dim, accums);
    }
  }

  float dist = HorizontalSum(accums[0]);

  for (; dim < dimensionality; dim++) {
    dist = FusedMultiplyOp<kIsSquaredL2, int16_t>(
        query[dim], Bfloat16Decompress(ptr[dim]), 0.0f, dist);
  }
  return dist;
}

template <bool kIsSquaredL2, bool kShouldPrefetch, size_t kUnrollBy,
          typename DataT>
SCANN_SIMD_INLINE void ComputeOneToManyScores(
    const float* __restrict__ query, array<const DataT*, kUnrollBy> ptrs,
    array<const DataT*, kUnrollBy> prefetch_ptrs,
    const float* __restrict__ inv_multipliers_for_squared_l2,
    size_t dimensionality, size_t num_outer_iters,
    array<float, kUnrollBy>& results) {
  Simd<float, kUnrollBy> accums = Zeros();

  size_t dim = 0;
  if constexpr (std::is_same_v<DataT, int8_t> ||
                Simd<int16_t>::kRegisterBits >= 256) {
    for (; dim + 16 <= dimensionality; dim += 16) {
      if constexpr (kShouldPrefetch) {
        for (size_t jj : Seq(kUnrollBy)) {
          absl::PrefetchToLocalCacheNta(prefetch_ptrs[jj] + dim);
        }
      }
      accums = HandleXDims<16, kIsSquaredL2>(
          query, ptrs, inv_multipliers_for_squared_l2, dim, accums);
    }

    if (dim + 8 <= dimensionality) {
      accums = HandleXDims<8, kIsSquaredL2>(
          query, ptrs, inv_multipliers_for_squared_l2, dim, accums);
      dim += 8;
    }
  } else {
    for (; dim + 8 <= dimensionality; dim += 8) {
      if constexpr (kShouldPrefetch) {
        for (size_t jj : Seq(kUnrollBy)) {
          absl::PrefetchToLocalCacheNta(prefetch_ptrs[jj] + dim);
        }
      }
      accums = HandleXDims<8, kIsSquaredL2>(
          query, ptrs, inv_multipliers_for_squared_l2, dim, accums);
    }
  }

  if (dim + 4 <= dimensionality) {
    accums = HandleXDims<4, kIsSquaredL2>(
        query, ptrs, inv_multipliers_for_squared_l2, dim, accums);
    dim += 4;
  }

  if constexpr (kUnrollBy == 4) {
    HorizontalSum4X(accums[0], accums[1], accums[2], accums[3], &results[0],
                    &results[1], &results[2], &results[3]);
  } else if constexpr (kUnrollBy == 3) {
    HorizontalSum3X(accums[0], accums[1], accums[2], &results[0], &results[1],
                    &results[2]);
  } else if constexpr (kUnrollBy == 2) {
    HorizontalSum2X(accums[0], accums[1], &results[0], &results[1]);
  } else {
    for (size_t jj : Seq(kUnrollBy)) {
      results[jj] = HorizontalSum(accums[jj]);
    }
  }

  for (; dim < dimensionality; ++dim) {
    for (size_t jj : Seq(kUnrollBy)) {
      float mult;
      if constexpr (kIsSquaredL2 && std::is_same_v<DataT, int8_t>) {
        mult = inv_multipliers_for_squared_l2[dim];
      } else {
        mult = 0.0;
      }
      float decompressed;
      if constexpr (std::is_same_v<DataT, int8_t>) {
        decompressed = static_cast<float>(ptrs[jj][dim]);
      } else {
        decompressed = Bfloat16Decompress(ptrs[jj][dim]);
      }
      results[jj] = FusedMultiplyOp<kIsSquaredL2, DataT>(
          query[dim], decompressed, mult, results[jj]);
    }
  }
}

SCANN_SIMD_INLINE Avx2<int16_t> MultiplyAndAddPacked(Avx2<uint8_t> a,
                                                     Avx2<int8_t> b) {
  return _mm256_maddubs_epi16(*a, *b);
}
SCANN_SIMD_INLINE Avx2<int32_t> MultiplyAndAddPacked(Avx2<int16_t> a,
                                                     Avx2<int16_t> b) {
  return _mm256_madd_epi16(*a, *b);
}

SCANN_SIMD_INLINE Sse4<int16_t> MultiplyAndAddPacked(Sse4<uint8_t> a,
                                                     Sse4<int8_t> b) {
  return _mm_maddubs_epi16(*a, *b);
}
SCANN_SIMD_INLINE Sse4<int32_t> MultiplyAndAddPacked(Sse4<int16_t> a,
                                                     Sse4<int16_t> b) {
  return _mm_madd_epi16(*a, *b);
}

SCANN_SIMD_INLINE Simd<int16_t> MultiplyAndAddPacked(Avx1<uint8_t> a,
                                                     Avx1<int8_t> b) {
  LOG(FATAL) << "Unhandled " << SimdName();
}
SCANN_SIMD_INLINE Simd<int32_t> MultiplyAndAddPacked(Avx1<int16_t> a,
                                                     Avx1<int16_t> b) {
  LOG(FATAL) << "Unhandled " << SimdName();
}

template <bool kIsSquaredL2, bool kShouldPrefetch, size_t kUnrollBy>
SCANN_SIMD_INLINE void ComputeOneToManyScores(
    const int8_t* __restrict__ query, array<const uint8_t*, kUnrollBy> ptrs,
    array<const uint8_t*, kUnrollBy> prefetch_ptrs,
    const float* __restrict__ inv_multipliers_for_squared_l2,
    size_t dimensionality, size_t num_outer_iters,
    array<int32_t, kUnrollBy>& results) {
  static_assert(!kIsSquaredL2, "Uint4Int8 doesn't support squared l2.");
  static_assert(kUnrollBy == 1, "Uint4Int8 doesn't support unrolling.");
  const uint8_t* ptr = ptrs[0];
  const uint8_t* prefetch_ptr = prefetch_ptrs[0];

  Simd<int16_t> result_16 = Zeros();
  Simd<int32_t> result_32 = Zeros();
  const Simd<uint8_t> kDMask(0x0f);
  const Simd<int16_t> kOnes16(1);
  constexpr size_t kNumElements = Simd<int8_t>::kNumElements;
  for (int i : Seq(DivRoundUp(dimensionality, kNumElements * 2))) {
    if constexpr (kShouldPrefetch) {
      absl::PrefetchToLocalCacheNta(prefetch_ptr);
      prefetch_ptr += kNumElements;
    }
    if (i % 4 == 0) {
      result_32 -= MultiplyAndAddPacked(kOnes16, result_16);
      result_16 = Zeros();
    }

    Simd<int8_t> q_low = Simd<int8_t>::Load(query);
    query += kNumElements;
    Simd<int8_t> q_high = Simd<int8_t>::Load(query);
    query += kNumElements;

    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(ptr, kNumElements);
    Simd<uint8_t> d_both = Simd<uint8_t>::Load(ptr);
    ptr += kNumElements;
    Simd<uint8_t> d_low = d_both & kDMask;
    Simd<uint8_t> d_high =
        Simd<uint8_t>((Simd<uint16_t>(d_both) >> 4)) & kDMask;
    result_16 += MultiplyAndAddPacked(d_low, q_low) +
                 MultiplyAndAddPacked(d_high, q_high);
  }
  result_32 -= MultiplyAndAddPacked(kOnes16, result_16);

  int32_t result = 0;

  for (int32_t v : result_32.Store()) {
    result += v;
  }
  results[0] = result;
}

SCANN_SIMD_INLINE void PermuteQueryForUint4Int8(const int8_t* query,
                                                int8_t* permuted_query,
                                                size_t dimensionality) {
  DCHECK_EQ(dimensionality % 64, 0);

  if constexpr (std::is_same_v<Simd<int8_t>, Avx2<int8_t>>) {
    const __m256i* q = reinterpret_cast<const __m256i*>(query);
    __m256i* permuted_q = reinterpret_cast<__m256i*>(permuted_query);
    __m128i kShuffle128 =
        _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    __m256i kShuffle256 = _mm256_set_m128i(kShuffle128, kShuffle128);
    for (int i = 0; i < dimensionality / 32; i += 2) {
      __m256i q_low = _mm256_permute4x64_epi64(
          _mm256_shuffle_epi8(_mm256_loadu_si256(q + i), kShuffle256),
          0b11011000);
      __m256i q_high = _mm256_permute4x64_epi64(
          _mm256_shuffle_epi8(_mm256_loadu_si256(q + i + 1), kShuffle256),
          0b10001101);
      _mm256_storeu_si256(permuted_q + i,
                          _mm256_blend_epi32(q_low, q_high, 0b11110000));
      _mm256_storeu_si256(permuted_q + i + 1,
                          _mm256_permute2x128_si256(q_low, q_high, 0b00100001));
    }
    return;
  } else if constexpr (std::is_same_v<Simd<int8_t>, Sse4<int8_t>>) {
    const __m128i* q = reinterpret_cast<const __m128i*>(query);
    __m128i* permuted_q = reinterpret_cast<__m128i*>(permuted_query);
    __m128i kShuffleLow =
        _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    __m128i kShuffleHigh =
        _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14);
    for (int i = 0; i < dimensionality / 16; i += 2) {
      __m128i q_low = _mm_shuffle_epi8(_mm_loadu_si128(q + i), kShuffleLow);
      __m128i q_high =
          _mm_shuffle_epi8(_mm_loadu_si128(q + i + 1), kShuffleHigh);
      _mm_storeu_si128(permuted_q + i,
                       _mm_blend_epi16(q_low, q_high, 0b11110000));
      _mm_storeu_si128(
          permuted_q + i + 1,
          _mm_shuffle_epi32(_mm_blend_epi16(q_low, q_high, 0b00001111),
                            0b01001110));
    }
    return;
  }
  LOG(FATAL) << "Unhandled " << SimdName();
}

template <typename CallbackT, typename ResultT, typename DataT>
SCANN_SIMD_INLINE void InvokeCallbackSimd(const CallbackT& callback,
                                          size_t result_idx, ResultT val,
                                          size_t datapoint_bytes,
                                          const DataT* ptr) {
  if constexpr (one_to_many_low_level::NeedsBottomBitsSideData<
                    CallbackT>::value) {
    if (datapoint_bytes % 32 == 0) {
      auto simd_data = SimdFor<DataT, 32>::Zeros();
      for (size_t i = 0; i < datapoint_bytes; i += 32) {
        simd_data ^= SimdFor<DataT, 32>::Load(ptr + i);
      }

      {
        SimdFor<uint16_t, 16> simd_data_16(simd_data);
        if constexpr (std::is_same_v<DataT, uint8_t>) {
          simd_data_16 ^= simd_data_16 >> 4;
        }

        simd_data_16 <<= 7;
        simd_data = SimdFor<DataT, 32>(simd_data_16);
      }

      uint32_t bottom_bits = 0;
      for (size_t i = 0; i < simd_data.kNumRegisters; ++i) {
        bottom_bits |= (simd_data[i].MaskFromHighBits())
                       << (i * simd_data.kRegisterBytes);
      }

      if constexpr (std::is_same_v<DataT, int8_t>) {
        DCHECK_EQ(bottom_bits, DecodeBottomBitsDataFromInt8(
                                   MakeConstSpan(ptr, datapoint_bytes)));
      } else {
        static_assert(std::is_same_v<DataT, uint8_t>);
        DCHECK_EQ(bottom_bits, DecodeBottomBitsDataFromPackedInt4(
                                   MakeConstSpan(ptr, datapoint_bytes)));
      }
      callback.invoke(result_idx, val, bottom_bits);
      return;
    }
  }
  InvokeCallback(callback, result_idx, val, datapoint_bytes, ptr);
}

template <int kDimensionality, size_t kUnrollBy, bool kHasIndices,
          bool kIsSquaredL2, bool kShouldPrefetch, typename DataT,
          typename DatasetViewT, typename IndexT, typename ResultElemT,
          typename CallbackT, typename QueryT>
SCANN_SIMD_INLINE void OneToManyAsymmetricTemplate(
    const QueryT* __restrict__ query, DatasetViewT dataset_view,
    const float* __restrict__ inv_multipliers_for_squared_l2,
    const IndexT* indices, MutableSpan<ResultElemT> result,
    CallbackT callback) {
  const size_t dimensionality =
      kDimensionality > 0 ? kDimensionality : dataset_view.dimensionality();

  const size_t num_datapoints = result.size();
  if (num_datapoints == 0 || dimensionality == 0) return;

  constexpr size_t kMinPrefetchAheadBytes = 2304;

  constexpr size_t kCacheLine = 64;
  const size_t datapoint_bytes = std::is_same_v<DataT, uint8_t>
                                     ? DivRoundUp(dimensionality, 2)
                                     : sizeof(DataT) * dimensionality;
  const size_t cache_lines_per_datapoint =
      DivRoundUp(datapoint_bytes, kCacheLine);
  size_t num_prefetch_datapoints;
  if (kShouldPrefetch) {
    num_prefetch_datapoints = std::max<size_t>(
        1, kMinPrefetchAheadBytes /
               (kUnrollBy * cache_lines_per_datapoint * kCacheLine));
  }

  auto get_db_ptr = [indices, &dataset_view, result, callback](size_t i)
                        SCANN_INLINE_LAMBDA -> const DataT* {
    using ::research_scann::one_to_many_low_level::GetDatapointIndex;
    const size_t idx = kHasIndices ? indices[i] : GetDatapointIndex(result, i);
    callback.prefetch(idx);
    return dataset_view.GetPtr(idx);
  };

  const size_t num_outer_iters = num_datapoints / kUnrollBy;

  if constexpr (kShouldPrefetch) {
    for (size_t j = num_datapoints / kUnrollBy * kUnrollBy; j < num_datapoints;
         j++) {
      const DataT* prefetch_ptr = get_db_ptr(j);
      for (size_t n : Seq(cache_lines_per_datapoint)) {
        absl::PrefetchToLocalCacheNta(prefetch_ptr +
                                      n * kCacheLine / sizeof(DataT));
      }
    }

    for (size_t j : Seq(std::min(num_prefetch_datapoints, num_outer_iters))) {
      array<const DataT*, kUnrollBy> prefetch_ptrs;
      for (size_t jj : Seq(kUnrollBy)) {
        prefetch_ptrs[jj] = get_db_ptr(j + jj * num_outer_iters);
      }

      for (size_t n : Seq(cache_lines_per_datapoint)) {
        for (size_t jj : Seq(kUnrollBy)) {
          absl::PrefetchToLocalCacheNta(prefetch_ptrs[jj] +
                                        n * kCacheLine / sizeof(DataT));
        }
      }
    }
  }

  std::array<QueryT, kDimensionality> query_stack_storage;
  std::unique_ptr<QueryT[]> permuted_query_heap_storage;
  if constexpr (std::is_same_v<DataT, uint8_t>) {
    QueryT* permuted_query;

    const size_t round_up_dims = NextMultipleOf(dimensionality, 64);
    if constexpr (kDimensionality > 0) {
      DCHECK_EQ(round_up_dims, kDimensionality);
      permuted_query = query_stack_storage.data();
    } else {
      permuted_query_heap_storage = std::make_unique<QueryT[]>(round_up_dims);
      permuted_query = permuted_query_heap_storage.get();
    }
    std::unique_ptr<QueryT[]> query_heap_storage;
    if (dimensionality != round_up_dims) {
      query_heap_storage = std::make_unique<QueryT[]>(round_up_dims);
      std::copy(query, query + dimensionality, query_heap_storage.get());
      std::fill(query_heap_storage.get() + dimensionality,
                query_heap_storage.get() + round_up_dims, 0);
      query = query_heap_storage.get();
    }

    PermuteQueryForUint4Int8(query, permuted_query, round_up_dims);
    query = permuted_query;
  } else {
    if constexpr (kDimensionality > 0) {
      DCHECK_EQ(dimensionality, kDimensionality);

      std::copy(query, query + kDimensionality, query_stack_storage.data());
      query = query_stack_storage.data();
    }
  }

  if constexpr (kUnrollBy > 1) {
    for (size_t j = num_datapoints / kUnrollBy * kUnrollBy; j < num_datapoints;
         j++) {
      const DataT* ptr = get_db_ptr(j);
      InvokeCallbackSimd(
          callback, j,
          ComputeOneToOneScore<0, kIsSquaredL2>(
              query, ptr, inv_multipliers_for_squared_l2, dimensionality),
          datapoint_bytes, ptr);
    }
  }

  array<const DataT*, kUnrollBy> prefetch_ptrs;
  for (size_t j : Seq(num_outer_iters)) {
    if constexpr (kShouldPrefetch) {
      if (j + num_prefetch_datapoints < num_outer_iters) {
        const size_t prefetch_j = j + num_prefetch_datapoints;
        for (size_t jj : Seq(kUnrollBy)) {
          prefetch_ptrs[jj] = get_db_ptr(prefetch_j + jj * num_outer_iters);
        }
      } else {
        std::fill(prefetch_ptrs.begin(), prefetch_ptrs.end(), get_db_ptr(0));
      }
    }

    array<const DataT*, kUnrollBy> ptrs;
    for (size_t jj : Seq(kUnrollBy)) {
      ptrs[jj] = get_db_ptr(j + jj * num_outer_iters);
    }

    using ResultT =
        std::conditional_t<std::is_same_v<QueryT, int8_t>, int32_t, float>;
    array<ResultT, kUnrollBy> results;
    ComputeOneToManyScores<kIsSquaredL2, kShouldPrefetch>(
        query, ptrs, prefetch_ptrs, inv_multipliers_for_squared_l2,
        dimensionality, num_outer_iters, results);

    for (size_t jj : Seq(kUnrollBy)) {
      InvokeCallbackSimd(callback, j + jj * num_outer_iters, results[jj],
                         datapoint_bytes, ptrs[jj]);
    }
  }
}

template <bool kHasIndices, bool kIsSquaredL2, typename DatasetViewT,
          typename IndexT, typename ResultElemT, typename CallbackT>
SCANN_SIMD_OUTLINE void OneToManyInt8FloatImpl(
    const float* __restrict__ query, DatasetViewT dataset_view,
    const float* __restrict__ inv_multipliers_for_squared_l2,
    const IndexT* indices, MutableSpan<ResultElemT> result,
    CallbackT callback) {
  const size_t dims = dataset_view.dimensionality();
  if (dims == 128) {
    OneToManyAsymmetricTemplate<128, 3, kHasIndices, kIsSquaredL2, true,
                                int8_t>(query, std::move(dataset_view),
                                        inv_multipliers_for_squared_l2, indices,
                                        result, std::move(callback));
  } else if (dims == 64) {
    OneToManyAsymmetricTemplate<64, 3, kHasIndices, kIsSquaredL2, true, int8_t>(
        query, std::move(dataset_view), inv_multipliers_for_squared_l2, indices,
        result, std::move(callback));
  } else {
    OneToManyAsymmetricTemplate<0, 3, kHasIndices, kIsSquaredL2, true, int8_t>(
        query, std::move(dataset_view), inv_multipliers_for_squared_l2, indices,
        result, std::move(callback));
  }
}

template <bool kHasIndices, bool kIsSquaredL2, typename DatasetViewT,
          typename IndexT, typename ResultElemT, typename CallbackT>
SCANN_SIMD_OUTLINE void OneToManyBf16FloatImpl(const float* __restrict__ query,
                                               DatasetViewT dataset_view,
                                               const IndexT* indices,
                                               MutableSpan<ResultElemT> result,
                                               CallbackT callback) {
  constexpr const float* kNoMultipliersForBfloat16 = nullptr;
  const size_t dims = dataset_view.dimensionality();
  if (dims == 128) {
    OneToManyAsymmetricTemplate<128, 3, kHasIndices, kIsSquaredL2, true,
                                int16_t>(query, std::move(dataset_view),
                                         kNoMultipliersForBfloat16, indices,
                                         result, std::move(callback));
  } else if (dims == 64) {
    OneToManyAsymmetricTemplate<64, 3, kHasIndices, kIsSquaredL2, true,
                                int16_t>(query, std::move(dataset_view),
                                         kNoMultipliersForBfloat16, indices,
                                         result, std::move(callback));
  } else {
    OneToManyAsymmetricTemplate<0, 3, kHasIndices, kIsSquaredL2, true, int16_t>(
        query, std::move(dataset_view), kNoMultipliersForBfloat16, indices,
        result, std::move(callback));
  }
}

template <bool kHasIndices, typename DatasetViewT, typename IndexT,
          typename ResultElemT, typename CallbackT>
SCANN_SIMD_OUTLINE void OneToManyUint4Int8Impl(const int8_t* __restrict__ query,
                                               DatasetViewT dataset_view,
                                               const IndexT* indices,
                                               MutableSpan<ResultElemT> result,
                                               CallbackT callback) {
  const size_t dims = dataset_view.dimensionality();
  if (dims == 256) {
    OneToManyAsymmetricTemplate<256, 1, kHasIndices, false, true, uint8_t>(
        query, std::move(dataset_view), nullptr, indices, result,
        std::move(callback));
  } else if (dims == 128) {
    OneToManyAsymmetricTemplate<128, 1, kHasIndices, false, true, uint8_t>(
        query, std::move(dataset_view), nullptr, indices, result,
        std::move(callback));
  } else if (dims == 64) {
    OneToManyAsymmetricTemplate<64, 1, kHasIndices, false, true, uint8_t>(
        query, std::move(dataset_view), nullptr, indices, result,
        std::move(callback));
  } else {
    OneToManyAsymmetricTemplate<0, 1, kHasIndices, false, true, uint8_t>(
        query, std::move(dataset_view), nullptr, indices, result,
        std::move(callback));
  }
}
