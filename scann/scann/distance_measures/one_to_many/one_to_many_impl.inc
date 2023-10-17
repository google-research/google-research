

template <int kRegisterIdx, typename T = Simd<float>>
SCANN_SIMD_INLINE auto Int8ToFloat(Sse4<int8_t> int8_vals) {
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

template <bool kIsSquaredL2, typename SimdT>
SCANN_SIMD_INLINE SimdT FusedMultiplyOp(SimdT a, SimdT b, SimdT mult,
                                        SimdT accum) {
  if constexpr (kIsSquaredL2) {
    SimdT diff = (a - b * mult);
    return FusedMultiplyAdd(diff, diff, accum);
  } else {
    return FusedMultiplySubtract(a, b, accum);
  }
}

template <size_t kNumDims, bool kIsSquaredL2, size_t kUnrollBy>
SCANN_SIMD_INLINE Simd<float, kUnrollBy> HandleXDims(
    const float* query, array<const int8_t*, kUnrollBy> ptrs,
    const float* inv_multipliers_for_squared_l2, size_t dim,
    Simd<float, kUnrollBy> accums) {
  auto qq_vals = LoadXFloats<kNumDims>(query + dim);

  Sse4<int8_t, kUnrollBy> db_vals_int8;
  for (size_t jj : Seq(kUnrollBy)) {
    if constexpr (kNumDims == 16) {
      db_vals_int8[jj] = Sse4<int8_t>::Load(ptrs[jj] + dim);
    }
    if constexpr (kNumDims == 8) {
      db_vals_int8[jj] =
          _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptrs[jj] + dim));
    }
    if constexpr (kNumDims == 4) {
      db_vals_int8[jj] =
          _mm_cvtsi32_si128(ABSL_INTERNAL_UNALIGNED_LOAD32(ptrs[jj] + dim));
    }
  }

  decltype(qq_vals) mult;
  if constexpr (kIsSquaredL2) {
    mult = LoadXFloats<kNumDims>(inv_multipliers_for_squared_l2 + dim);
  }

  asm("" ::: "memory");

  if constexpr (kNumDims == 4) {
    for (size_t jj : Seq(kUnrollBy)) {
      Simd<float> db_vals_float =
          SseToSimd(Int8ToFloat<0, Sse4<float>>(db_vals_int8[jj]));
      accums[jj] = FusedMultiplyOp<kIsSquaredL2>(qq_vals, db_vals_float, mult,
                                                 accums[jj]);
    }
    return accums;
  }

  if constexpr (decltype(qq_vals)::kNumRegisters >= 1) {
    for (size_t jj : Seq(kUnrollBy)) {
      accums[jj] = FusedMultiplyOp<kIsSquaredL2>(
          qq_vals[0], Int8ToFloat<0>(db_vals_int8[jj]), mult[0], accums[jj]);
    }
  }
  if constexpr (decltype(qq_vals)::kNumRegisters >= 2) {
    for (size_t jj : Seq(kUnrollBy)) {
      accums[jj] = FusedMultiplyOp<kIsSquaredL2>(
          qq_vals[1], Int8ToFloat<1>(db_vals_int8[jj]), mult[1], accums[jj]);
    }
  }
  if constexpr (decltype(qq_vals)::kNumRegisters >= 3) {
    for (size_t jj : Seq(kUnrollBy)) {
      accums[jj] = FusedMultiplyOp<kIsSquaredL2>(
          qq_vals[2], Int8ToFloat<2>(db_vals_int8[jj]), mult[2], accums[jj]);
    }
  }
  if constexpr (decltype(qq_vals)::kNumRegisters >= 4) {
    for (size_t jj : Seq(kUnrollBy)) {
      accums[jj] = FusedMultiplyOp<kIsSquaredL2>(
          qq_vals[3], Int8ToFloat<3>(db_vals_int8[jj]), mult[3], accums[jj]);
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
      dist = FusedMultiplyOp<kIsSquaredL2>(
          query[dim], static_cast<float>(ptr[dim]), mult, dist);
    }
    return dist;
  } else {
    DatapointPtr<float> qq_dptr(nullptr, query, dimensionality, dimensionality);
    DatapointPtr<int8_t> db_dptr(nullptr, ptr, dimensionality, dimensionality);
    return -StaticallyInvokeOneToOneDenseDotProduct(qq_dptr, db_dptr);
  }
}

template <int kDimensionality, size_t kUnrollBy, bool kHasIndices,
          bool kIsSquaredL2, bool kShouldPrefetch, typename DatasetViewT,
          typename IndexT, typename ResultElemT, typename CallbackT>
SCANN_SIMD_INLINE void OneToManyInt8FloatTemplate(
    const float* __restrict__ query, DatasetViewT dataset_view,
    const float* __restrict__ inv_multipliers_for_squared_l2,
    const IndexT* indices, MutableSpan<ResultElemT> result,
    CallbackT callback) {
  const size_t dimensionality =
      kDimensionality > 0 ? kDimensionality : dataset_view.dimensionality();

  const size_t num_datapoints = result.size();
  if (num_datapoints == 0 || dimensionality == 0) return;

  constexpr size_t kMinPrefetchAheadBytes = 2304;

  constexpr size_t kCacheLine = 64;
  const size_t cache_lines_per_datapoint =
      DivRoundUp(dimensionality, kCacheLine);
  size_t num_prefetch_datapoints;
  if (kShouldPrefetch) {
    num_prefetch_datapoints = std::max<size_t>(
        1, kMinPrefetchAheadBytes /
               (kUnrollBy * cache_lines_per_datapoint * kCacheLine));
  }

  auto get_db_ptr = [indices, &dataset_view, result, callback](size_t i)
                        SCANN_INLINE_LAMBDA -> const int8_t* {
    using ::research_scann::one_to_many_low_level::GetDatapointIndex;
    const size_t idx = kHasIndices ? indices[i] : GetDatapointIndex(result, i);
    callback.prefetch(idx);
    return dataset_view.GetPtr(idx);
  };

  const size_t num_outer_iters = num_datapoints / kUnrollBy;

  if constexpr (kShouldPrefetch) {
    for (size_t j = num_datapoints / kUnrollBy * kUnrollBy; j < num_datapoints;
         j++) {
      const int8_t* prefetch_ptr = get_db_ptr(j);
      for (size_t n : Seq(cache_lines_per_datapoint)) {
        ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_NTA>(
            prefetch_ptr + n * kCacheLine);
      }
    }

    for (size_t j : Seq(std::min(num_prefetch_datapoints, num_outer_iters))) {
      array<const int8_t*, kUnrollBy> prefetch_ptrs;
      for (size_t jj : Seq(kUnrollBy)) {
        prefetch_ptrs[jj] = get_db_ptr(j + jj * num_outer_iters);
      }

      for (size_t n : Seq(cache_lines_per_datapoint)) {
        for (size_t jj : Seq(kUnrollBy)) {
          ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_NTA>(
              prefetch_ptrs[jj] + n * kCacheLine);
        }
      }
    }
  }

  std::array<float, kDimensionality> query_storage;
  if constexpr (kDimensionality > 0) {
    DCHECK_EQ(dimensionality, kDimensionality);

    std::copy(query, query + kDimensionality, query_storage.data());
    query = query_storage.data();
  }

  for (size_t j = num_datapoints / kUnrollBy * kUnrollBy; j < num_datapoints;
       j++) {
    const int8_t* ptr = get_db_ptr(j);
    callback.invoke(
        j, ComputeOneToOneScore<0, kIsSquaredL2>(
               query, ptr, inv_multipliers_for_squared_l2, dimensionality));
  }

  for (size_t j : Seq(num_outer_iters)) {
    if constexpr (kShouldPrefetch) {
      if (j + num_prefetch_datapoints < num_outer_iters) {
        const size_t prefetch_j = j + num_prefetch_datapoints;

        array<const int8_t*, kUnrollBy> prefetch_ptrs;
        for (size_t jj : Seq(kUnrollBy)) {
          prefetch_ptrs[jj] = get_db_ptr(prefetch_j + jj * num_outer_iters);
        }

        for (size_t n : Seq(cache_lines_per_datapoint)) {
          for (size_t jj : Seq(kUnrollBy)) {
            ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_NTA>(
                prefetch_ptrs[jj] + n * kCacheLine);
          }
        }
      }
    }

    array<const int8_t*, kUnrollBy> ptrs;
    for (size_t jj : Seq(kUnrollBy)) {
      ptrs[jj] = get_db_ptr(j + jj * num_outer_iters);
    }

    Simd<float, kUnrollBy> accums = Zeros();

    size_t dim = 0;
    for (; dim + 16 <= dimensionality; dim += 16) {
      accums = HandleXDims<16, kIsSquaredL2>(
          query, ptrs, inv_multipliers_for_squared_l2, dim, accums);
    }

    if (dim + 8 <= dimensionality) {
      accums = HandleXDims<8, kIsSquaredL2>(
          query, ptrs, inv_multipliers_for_squared_l2, dim, accums);
      dim += 8;
    }

    if (dim + 4 <= dimensionality) {
      accums = HandleXDims<4, kIsSquaredL2>(
          query, ptrs, inv_multipliers_for_squared_l2, dim, accums);
      dim += 4;
    }

    array<float, kUnrollBy> results;
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
        float mult = 0.0;
        if constexpr (kIsSquaredL2) {
          mult = inv_multipliers_for_squared_l2[dim];
        } else {
          mult = 0.0;
        }
        results[jj] = FusedMultiplyOp<kIsSquaredL2>(
            query[dim], static_cast<float>(ptrs[jj][dim]), mult, results[jj]);
      }
    }

    for (size_t jj : Seq(kUnrollBy)) {
      callback.invoke(j + jj * num_outer_iters, results[jj]);
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
    OneToManyInt8FloatTemplate<128, 3, kHasIndices, kIsSquaredL2, true>(
        query, std::move(dataset_view), inv_multipliers_for_squared_l2, indices,
        result, std::move(callback));
  } else if (dims == 64) {
    OneToManyInt8FloatTemplate<64, 3, kHasIndices, kIsSquaredL2, true>(
        query, std::move(dataset_view), inv_multipliers_for_squared_l2, indices,
        result, std::move(callback));
  } else {
    OneToManyInt8FloatTemplate<0, 3, kHasIndices, kIsSquaredL2, true>(
        query, std::move(dataset_view), inv_multipliers_for_squared_l2, indices,
        result, std::move(callback));
  }
}
