

#include <cstdint>
template <typename DatasetView, bool kShouldPrefetch, bool kHasIndices,
          typename IndexT, typename ResultElemT, typename CallbackLambda>
SCANN_SIMD_INLINE void DenseDotProductDistanceOneToManyInt8FloatImpl(
    const float* query, const DatasetView* __restrict__ dataset_view,
    const IndexT* indices, MutableSpan<ResultElemT> result,
    CallbackLambda* __restrict__ callback) {
  const size_t dims = dataset_view->dimensionality();
  const size_t num_datapoints = result.size();
  if (num_datapoints == 0 || dims == 0) return;
  constexpr size_t kMinPrefetchAheadBytes =
      (IsFloatingType<ResultElemT>()) ? 2048 : 1024;
  size_t num_prefetch_datapoints;
  if (kShouldPrefetch) {
    num_prefetch_datapoints =
        (kMinPrefetchAheadBytes > dims) ? (kMinPrefetchAheadBytes / dims) : 1;
  }

  auto get_db_ptr = [indices, &dataset_view, result, callback](size_t i)
                        SCANN_INLINE_LAMBDA -> const int8_t* {
    const size_t idx =
        kHasIndices
            ? indices[i]
            : ::research_scann::one_to_many_low_level::GetDatapointIndex(result,
                                                                         i);
    callback->prefetch(idx);
    return dataset_view->GetPtr(idx);
  };

  const size_t num_outer_iters = num_datapoints / 3;
  for (size_t i = 0; i < num_outer_iters; ++i) {
    const int8_t* i0 = get_db_ptr(i);
    const int8_t* i1 = get_db_ptr(i + num_outer_iters);
    const int8_t* i2 = get_db_ptr(i + 2 * num_outer_iters);

    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps();
    size_t j = 0;

    if (j + 16 <= dims) {
      const int8_t *p0 = nullptr, *p1 = nullptr, *p2 = nullptr;

      if (kShouldPrefetch && i + num_prefetch_datapoints < num_outer_iters) {
        p0 = get_db_ptr(i + num_prefetch_datapoints);
        p1 = get_db_ptr(i + num_outer_iters + num_prefetch_datapoints);
        p2 = get_db_ptr(i + 2 * num_outer_iters + num_prefetch_datapoints);
      }

      for (; j + 16 <= dims; j += 16) {
        __m256 q = _mm256_loadu_ps(query + j);
        __m256 q2 = _mm256_loadu_ps(query + j + 8);
        __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(i0 + j));
        __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(i1 + j));
        __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(i2 + j));

        if (kShouldPrefetch && p0) {
          ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(
              p0 + j);
          ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(
              p1 + j);
          ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(
              p2 + j);
        }

        a0 = AvxFuncs::MultiplySub(q, AvxFuncs::Int8ToFloatLower(v0), a0);
        a1 = AvxFuncs::MultiplySub(q, AvxFuncs::Int8ToFloatLower(v1), a1);
        a2 = AvxFuncs::MultiplySub(q, AvxFuncs::Int8ToFloatLower(v2), a2);
        a0 = AvxFuncs::MultiplySub(q2, AvxFuncs::Int8ToFloatUpper(v0), a0);
        a1 = AvxFuncs::MultiplySub(q2, AvxFuncs::Int8ToFloatUpper(v1), a1);
        a2 = AvxFuncs::MultiplySub(q2, AvxFuncs::Int8ToFloatUpper(v2), a2);
      }
    }

    if (j + 8 <= dims) {
      __m256 q = _mm256_loadu_ps(query + j);
      __m128i v0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(i0 + j));
      __m128i v1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(i1 + j));
      __m128i v2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(i2 + j));
      a0 = AvxFuncs::MultiplySub(q, AvxFuncs::Int8ToFloatLower(v0), a0);
      a1 = AvxFuncs::MultiplySub(q, AvxFuncs::Int8ToFloatLower(v1), a1);
      a2 = AvxFuncs::MultiplySub(q, AvxFuncs::Int8ToFloatLower(v2), a2);
      j += 8;
    }

    if (j + 4 <= dims) {
      __m128 q = _mm_loadu_ps(query + j);
      __m128i v0 = _mm_cvtepi8_epi32(
          _mm_cvtsi32_si128(*reinterpret_cast<const int*>(i0 + j)));
      __m128i v1 = _mm_cvtepi8_epi32(
          _mm_cvtsi32_si128(*reinterpret_cast<const int*>(i1 + j)));
      __m128i v2 = _mm_cvtepi8_epi32(
          _mm_cvtsi32_si128(*reinterpret_cast<const int*>(i2 + j)));
      a0 = _mm256_sub_ps(
          a0, AvxFuncs::SseToAvx(_mm_mul_ps(q, _mm_cvtepi32_ps(v0))));
      a1 = _mm256_sub_ps(
          a1, AvxFuncs::SseToAvx(_mm_mul_ps(q, _mm_cvtepi32_ps(v1))));
      a2 = _mm256_sub_ps(
          a2, AvxFuncs::SseToAvx(_mm_mul_ps(q, _mm_cvtepi32_ps(v2))));
      j += 4;
    }

    float result0;
    float result1;
    float result2;
    ::research_scann::avx1::HorizontalSum3X({a0}, {a1}, {a2}, &result0,
                                            &result1, &result2);
    for (; j < dims; ++j) {
      result0 -= query[j] * i0[j];
      result1 -= query[j] * i1[j];
      result2 -= query[j] * i2[j];
    }

    callback->invoke(i, result0);
    callback->invoke(i + num_outer_iters, result1);
    callback->invoke(i + 2 * num_outer_iters, result2);
  }
}

template <typename DatasetView, bool kHasIndices, typename IndexT,
          typename ResultElemT, typename CallbackLambda>
SCANN_SIMD_OUTLINE void DenseDotProductDistanceOneToManyInt8Float(
    const float* query, const DatasetView* __restrict__ dataset_view,
    const IndexT* indices, MutableSpan<ResultElemT> result,
    CallbackLambda* __restrict__ callback) {
  constexpr size_t kMinPrefetchAheadBytes =
      (IsFloatingType<ResultElemT>()) ? 32 : 16;
  constexpr size_t kMaxPrefetchAheadBytes = 2048;
  const size_t dims = dataset_view->dimensionality();
  if (dims == 128) {
    class CompileTimeDimensionalityWrapper {
     public:
      explicit CompileTimeDimensionalityWrapper(
          const DatasetView* __restrict__ base)
          : base_(base) {}
      const int8_t* GetPtr(size_t i) const { return base_->GetPtr(i); }
      constexpr size_t dimensionality() const { return 128; }

     private:
      const DatasetView* __restrict__ base_;
    };

    CompileTimeDimensionalityWrapper wrapped_view(dataset_view);

    DenseDotProductDistanceOneToManyInt8FloatImpl<
        CompileTimeDimensionalityWrapper, true, kHasIndices, IndexT>(
        query, &wrapped_view, indices, result, callback);
  } else if (dims >= kMinPrefetchAheadBytes && dims <= kMaxPrefetchAheadBytes) {
    DenseDotProductDistanceOneToManyInt8FloatImpl<DatasetView, true,
                                                  kHasIndices, IndexT>(
        query, dataset_view, indices, result, callback);
  } else {
    DenseDotProductDistanceOneToManyInt8FloatImpl<DatasetView, false,
                                                  kHasIndices, IndexT>(
        query, dataset_view, indices, result, callback);
  }
}
