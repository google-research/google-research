// Copyright 2025 The Google Research Authors.
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



#ifndef SCANN_DISTANCE_MEASURES_ONE_TO_MANY_ONE_TO_MANY_SYMMETRIC_H_
#define SCANN_DISTANCE_MEASURES_ONE_TO_MANY_ONE_TO_MANY_SYMMETRIC_H_

#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS HWY_ALL_SVE
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "absl/base/optimization.h"
#include "absl/base/prefetch.h"
#include "absl/synchronization/mutex.h"
#include "hwy/highway.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measures.h"
#include "scann/distance_measures/one_to_many/one_to_many_helpers.h"
#include "scann/distance_measures/one_to_many/one_to_many_symmetric.h"
#include "scann/utils/common.h"
#include "scann/utils/internal/avx2_funcs.h"
#include "scann/utils/internal/avx_funcs.h"
#include "scann/utils/intrinsics/fma.h"
#include "scann/utils/intrinsics/highway.h"
#include "scann/utils/intrinsics/horizontal_sum.h"
#include "scann/utils/intrinsics/simd.h"
#include "scann/utils/types.h"

namespace research_scann {

#ifdef __SSE3__

namespace one_to_many_low_level {
constexpr bool kX64Simd = true;
}

#else

namespace one_to_many_low_level {
constexpr bool kX64Simd = false;
}

#endif

template <typename T, typename ResultElem>
void DenseDistanceOneToMany(const DistanceMeasure& dist,
                            const DatapointPtr<T>& query,
                            const DenseDataset<T>& database,
                            MutableSpan<ResultElem> result,
                            ThreadPool* pool = nullptr);

template <typename T, typename U, typename ResultElem>
std::pair<DatapointIndex, U> DenseDistanceOneToManyTop1(
    const DistanceMeasure& dist, const DatapointPtr<T>& query,
    const DenseDataset<T>& database, MutableSpan<ResultElem> result,
    ThreadPool* pool = nullptr);

template <typename T, typename ResultElem, typename DatasetView>
void DenseDistanceOneToMany(const DistanceMeasure& dist,
                            const DatapointPtr<T>& query,
                            const DatasetView* __restrict__ database,
                            MutableSpan<ResultElem> result,
                            ThreadPool* pool = nullptr);

template <typename T, typename ResultElem>
void DenseDotProductDistanceOneToMany(const DatapointPtr<T>& query,
                                      const DenseDataset<T>& database,
                                      MutableSpan<ResultElem> result,
                                      ThreadPool* pool = nullptr);
template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
void DenseDotProductDistanceOneToMany(const DatapointPtr<T>& query,
                                      const DatasetView* __restrict__ database,
                                      MutableSpan<ResultElem> result,
                                      CallbackFunctor* __restrict__ callback,
                                      ThreadPool* pool = nullptr);

template <typename T, typename ResultElem>
void DenseAbsDotProductDistanceOneToMany(const DatapointPtr<T>& query,
                                         const DenseDataset<T>& database,
                                         MutableSpan<ResultElem> result,
                                         ThreadPool* pool = nullptr);
template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
void DenseAbsDotProductDistanceOneToMany(
    const DatapointPtr<T>& query, const DatasetView* __restrict__ database,
    MutableSpan<ResultElem> result, CallbackFunctor* __restrict__ callback,
    ThreadPool* pool = nullptr);

template <typename T, typename ResultElem>
void DenseCosineDistanceOneToMany(const DatapointPtr<T>& query,
                                  const DenseDataset<T>& database,
                                  MutableSpan<ResultElem> result,
                                  ThreadPool* pool = nullptr);
template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
void DenseCosineDistanceOneToMany(const DatapointPtr<T>& query,
                                  const DatasetView* __restrict__ database,
                                  MutableSpan<ResultElem> result,
                                  CallbackFunctor* __restrict__ callback,
                                  ThreadPool* pool = nullptr);

template <typename T, typename ResultElem>
void DenseSquaredL2DistanceOneToMany(const DatapointPtr<T>& query,
                                     const DenseDataset<T>& database,
                                     MutableSpan<ResultElem> result,
                                     ThreadPool* pool = nullptr);
template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
void DenseSquaredL2DistanceOneToMany(const DatapointPtr<T>& query,
                                     const DatasetView* __restrict__ database,
                                     MutableSpan<ResultElem> result,
                                     CallbackFunctor* __restrict__ callback,
                                     ThreadPool* pool = nullptr);

template <typename T, typename ResultElem>
void DenseL2DistanceOneToMany(const DatapointPtr<T>& query,
                              const DenseDataset<T>& database,
                              MutableSpan<ResultElem> result,
                              ThreadPool* pool = nullptr);
template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
void DenseL2DistanceOneToMany(const DatapointPtr<T>& query,
                              const DatasetView* __restrict__ database,
                              MutableSpan<ResultElem> result,
                              CallbackFunctor* __restrict__ callback,
                              ThreadPool* pool = nullptr);

template <typename T, typename ResultElem>
void DenseL1DistanceOneToMany(const DatapointPtr<T>& query,
                              const DenseDataset<T>& database,
                              MutableSpan<ResultElem> result,
                              ThreadPool* pool = nullptr);
template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
void DenseL1DistanceOneToMany(const DatapointPtr<T>& query,
                              const DatasetView* __restrict__ database,
                              MutableSpan<ResultElem> result,
                              CallbackFunctor* __restrict__ callback,
                              ThreadPool* pool = nullptr);

template <typename T, typename ResultElem>
void DenseLimitedInnerProductDistanceOneToMany(const DatapointPtr<T>& query,
                                               const DenseDataset<T>& database,
                                               MutableSpan<ResultElem> result,
                                               ThreadPool* pool = nullptr);
template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
void DenseLimitedInnerProductDistanceOneToMany(
    const DatapointPtr<T>& query, const DatasetView* __restrict__ database,
    MutableSpan<ResultElem> result, CallbackFunctor* __restrict__ callback,
    ThreadPool* pool = nullptr);

template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
enable_if_t<IsIntegerType<T>() && sizeof(T) == 4 &&
                one_to_many_low_level::kX64Simd,
            void>
DenseGeneralHammingDistanceOneToMany(const DatapointPtr<T>& query,
                                     const DatasetView* __restrict__ database,
                                     MutableSpan<ResultElem> result,
                                     CallbackFunctor* __restrict__ callback,
                                     ThreadPool* pool);
template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
enable_if_t<!IsIntegerType<T>() || sizeof(T) != 4 ||
                !one_to_many_low_level::kX64Simd,
            void>
DenseGeneralHammingDistanceOneToMany(const DatapointPtr<T>& query,
                                     const DatasetView* __restrict__ database,
                                     MutableSpan<ResultElem> result,
                                     CallbackFunctor* __restrict__ callback,
                                     ThreadPool* pool);
template <typename T, typename ResultElem>
void DenseGeneralHammingDistanceOneToMany(const DatapointPtr<T>& query,
                                          const DenseDataset<T>& database,
                                          MutableSpan<ResultElem> result,
                                          ThreadPool* pool);

namespace one_to_many_low_level {

template <typename T, typename DatasetView, typename Lambdas,
          typename ResultElem, typename CallbackFunctor>
enable_if_t<IsIntegerType<T>() || (std::is_same_v<T, double> && !kX64Simd),
            void>
DenseAccumulatingDistanceMeasureOneToMany(
    const DatapointPtr<T>& query, const DatasetView* __restrict__ database,
    const Lambdas& lambdas, MutableSpan<ResultElem> result,
    CallbackFunctor* __restrict__ callback, ThreadPool* pool) {
  for (size_t i = 0; i < result.size(); ++i) {
    callback->invoke(
        i, lambdas.VectorVector(
               query,
               MakeDatapointPtr(database->GetPtr(GetDatapointIndex(result, i)),
                                database->dimensionality())));
  }
}

template <typename T, typename DatasetView, typename Lambdas,
          typename ResultElem, typename CallbackFunctor>
enable_if_t<IsIntegerType<T>(), void>
DenseAccumulatingDistanceMeasureOneToManyNoFma(
    const DatapointPtr<T>& query, const DatasetView* __restrict__ database,
    const Lambdas& lambdas, MutableSpan<ResultElem> result,
    CallbackFunctor* __restrict__ callback, ThreadPool* pool) {
  return DenseAccumulatingDistanceMeasureOneToMany<T, DatasetView, Lambdas,
                                                   ResultElem>(
      query, database, lambdas, result, callback, pool);
}

#ifdef __SSE3__

SCANN_AVX1_INLINE __m128 SumTopBottomAvx(__m256 x) {
  const __m128 upper = _mm256_extractf128_ps(x, 1);
  const __m128 lower = _mm256_castps256_ps128(x);
  return _mm_add_ps(upper, lower);
}

template <typename T, typename DatasetView, typename Lambdas,
          typename ResultElem, bool kShouldPrefetch, typename CallbackFunctor>
enable_if_t<std::is_same<T, float>::value, void> SCANN_AVX1_OUTLINE
DenseAccumulatingDistanceMeasureOneToManyInternalAvx1(
    const DatapointPtr<T>& query, const DatasetView* __restrict__ database,
    const Lambdas& lambdas, MutableSpan<ResultElem> result,
    CallbackFunctor* __restrict__ callback, ThreadPool* pool) {
  if (result.empty()) return;
  DCHECK(!pool || !kShouldPrefetch);
  const size_t dims = query.dimensionality();

  DCHECK_EQ(dims, database->dimensionality());
  DCHECK_GE(dims, 8);
  DCHECK_GT(dims, 0);

  Lambdas lambdas_vec[4] = {lambdas, lambdas, lambdas, lambdas};
  const size_t num_outer_iters = result.size() / 3;
  const size_t parallel_end = num_outer_iters * 3;

  constexpr size_t kMinPrefetchAheadDims =
      (IsFloatingType<ResultElem>()) ? 512 : 256;
  size_t num_prefetch_datapoints;
  if (kShouldPrefetch) {
    num_prefetch_datapoints = std::max<size_t>(1, kMinPrefetchAheadDims / dims);
  }

  auto get_db_ptr = [&database, result, callback](size_t i)
                        SCANN_INLINE_LAMBDA -> const float* {
    auto idx = GetDatapointIndex(result, i);
    callback->prefetch(idx);
    return database->GetPtr(idx);
  };

  auto sum4 = [](__m128 x) -> float {
    x = _mm_add_ps(x, _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(x), 8)));
    return x[0] + x[1];
  };

  ParallelFor<32>(Seq(num_outer_iters), pool, [&](size_t i) SCANN_AVX1 {
    const float* f0 = get_db_ptr(i);
    const float* f1 = get_db_ptr(i + num_outer_iters);
    const float* f2 = get_db_ptr(i + 2 * num_outer_iters);
    const float *p0 = nullptr, *p1 = nullptr, *p2 = nullptr;

    if (kShouldPrefetch && i + num_prefetch_datapoints < num_outer_iters) {
      p0 = get_db_ptr(i + num_prefetch_datapoints);
      p1 = get_db_ptr(i + num_outer_iters + num_prefetch_datapoints);
      p2 = get_db_ptr(i + 2 * num_outer_iters + num_prefetch_datapoints);
    }

    __m256 a0_256 = _mm256_setzero_ps();
    __m256 a1_256 = _mm256_setzero_ps();
    __m256 a2_256 = _mm256_setzero_ps();
    size_t j = 0;

    for (; j + 8 <= dims; j += 8) {
      __m256 q = _mm256_loadu_ps(query.values() + j);
      __m256 v0 = _mm256_loadu_ps(f0 + j);
      __m256 v1 = _mm256_loadu_ps(f1 + j);
      __m256 v2 = _mm256_loadu_ps(f2 + j);

      if (kShouldPrefetch && p0) {
        absl::PrefetchToLocalCache(p0 + j);
        absl::PrefetchToLocalCache(p1 + j);
        absl::PrefetchToLocalCache(p2 + j);
      }

      a0_256 = lambdas_vec[0].AccTerm(a0_256, q, v0);
      a1_256 = lambdas_vec[1].AccTerm(a1_256, q, v1);
      a2_256 = lambdas_vec[2].AccTerm(a2_256, q, v2);
    }

    __m128 a0 = SumTopBottomAvx(a0_256);
    __m128 a1 = SumTopBottomAvx(a1_256);
    __m128 a2 = SumTopBottomAvx(a2_256);

    if (j + 4 <= dims) {
      __m128 q = _mm_loadu_ps(query.values() + j);
      __m128 v0 = _mm_loadu_ps(f0 + j);
      __m128 v1 = _mm_loadu_ps(f1 + j);
      __m128 v2 = _mm_loadu_ps(f2 + j);

      if (kShouldPrefetch && p0) {
        absl::PrefetchToLocalCache(p0 + j);
        absl::PrefetchToLocalCache(p1 + j);
        absl::PrefetchToLocalCache(p2 + j);
      }

      a0 = lambdas_vec[0].AccTerm(a0, q, v0);
      a1 = lambdas_vec[1].AccTerm(a1, q, v1);
      a2 = lambdas_vec[2].AccTerm(a2, q, v2);
      j += 4;
    }

    if (j + 2 <= dims) {
      __m128 q = _mm_setzero_ps();
      __m128 v0 = _mm_setzero_ps();
      __m128 v1 = _mm_setzero_ps();
      __m128 v2 = _mm_setzero_ps();
      q = _mm_loadh_pi(q, reinterpret_cast<const __m64*>(query.values() + j));
      v0 = _mm_loadh_pi(v0, reinterpret_cast<const __m64*>(f0 + j));
      v1 = _mm_loadh_pi(v1, reinterpret_cast<const __m64*>(f1 + j));
      v2 = _mm_loadh_pi(v2, reinterpret_cast<const __m64*>(f2 + j));
      a0 = lambdas_vec[0].AccTerm(a0, q, v0);
      a1 = lambdas_vec[1].AccTerm(a1, q, v1);
      a2 = lambdas_vec[2].AccTerm(a2, q, v2);
      j += 2;
    }

    float result0 = sum4(a0);
    float result1 = sum4(a1);
    float result2 = sum4(a2);

    if (j < dims) {
      DCHECK_EQ(j + 1, dims);
      result0 = lambdas_vec[0].AccTerm(result0, query.values()[j], f0[j]);
      result1 = lambdas_vec[1].AccTerm(result1, query.values()[j], f1[j]);
      result2 = lambdas_vec[2].AccTerm(result2, query.values()[j], f2[j]);
    }

    callback->invoke(i, lambdas_vec[0].Postprocess(result0));
    callback->invoke(i + num_outer_iters, lambdas_vec[1].Postprocess(result1));
    callback->invoke(i + 2 * num_outer_iters,
                     lambdas_vec[2].Postprocess(result2));
  });

  size_t i = parallel_end;
  for (; i < result.size(); ++i) {
    const DatapointPtr<float> f0 =
        MakeDatapointPtr(database->GetPtr(GetDatapointIndex(result, i)), dims);
    callback->invoke(i, lambdas.VectorVector(query, f0));
  }
}

template <typename T, typename DatasetView, typename Lambdas,
          typename ResultElem, bool kShouldPrefetch, typename CallbackFunctor>
enable_if_t<std::is_same<T, float>::value, void> SCANN_AVX2_OUTLINE
DenseAccumulatingDistanceMeasureOneToManyInternalAvx2(
    const DatapointPtr<T>& query, const DatasetView* __restrict__ database,
    const Lambdas& lambdas, MutableSpan<ResultElem> result,
    CallbackFunctor* __restrict__ callback, ThreadPool* pool) {
  if (result.empty()) return;
  DCHECK(!pool || !kShouldPrefetch);
  const size_t dims = query.dimensionality();
  DCHECK_GE(dims, 8);

  DCHECK_GT(dims, 0);
  Lambdas lambdas_vec[4] = {lambdas, lambdas, lambdas, lambdas};
  const size_t num_outer_iters = result.size() / 3;
  const size_t parallel_end = num_outer_iters * 3;

  constexpr size_t kMinPrefetchAheadDims =
      (IsFloatingType<ResultElem>()) ? 512 : 256;
  size_t num_prefetch_datapoints;
  if (kShouldPrefetch) {
    num_prefetch_datapoints = std::max<size_t>(1, kMinPrefetchAheadDims / dims);
  }

  auto get_db_ptr = [&database, result, callback](size_t i)
                        SCANN_INLINE_LAMBDA -> const float* {
    auto idx = GetDatapointIndex(result, i);
    callback->prefetch(idx);
    return database->GetPtr(idx);
  };

  auto sum4 = [](__m128 x) -> float {
    x = _mm_add_ps(x, _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(x), 8)));
    return x[0] + x[1];
  };

  ParallelFor<32>(Seq(num_outer_iters), pool, [&](size_t i) SCANN_AVX2 {
    const float* f0 = get_db_ptr(i);
    const float* f1 = get_db_ptr(i + num_outer_iters);
    const float* f2 = get_db_ptr(i + 2 * num_outer_iters);
    const float *p0 = nullptr, *p1 = nullptr, *p2 = nullptr;

    if (kShouldPrefetch && i + num_prefetch_datapoints < num_outer_iters) {
      p0 = get_db_ptr(i + num_prefetch_datapoints);
      p1 = get_db_ptr(i + num_outer_iters + num_prefetch_datapoints);
      p2 = get_db_ptr(i + 2 * num_outer_iters + num_prefetch_datapoints);
    }

    __m256 a0_256 = _mm256_setzero_ps();
    __m256 a1_256 = _mm256_setzero_ps();
    __m256 a2_256 = _mm256_setzero_ps();
    size_t j = 0;

    for (; j + 8 <= dims; j += 8) {
      __m256 q = _mm256_loadu_ps(query.values() + j);
      __m256 v0 = _mm256_loadu_ps(f0 + j);
      __m256 v1 = _mm256_loadu_ps(f1 + j);
      __m256 v2 = _mm256_loadu_ps(f2 + j);

      if (kShouldPrefetch && p0) {
        absl::PrefetchToLocalCache(p0 + j);
        absl::PrefetchToLocalCache(p1 + j);
        absl::PrefetchToLocalCache(p2 + j);
      }

      a0_256 = lambdas_vec[0].FmaTerm(a0_256, q, v0);
      a1_256 = lambdas_vec[1].FmaTerm(a1_256, q, v1);
      a2_256 = lambdas_vec[2].FmaTerm(a2_256, q, v2);
    }

    __m128 a0 = SumTopBottomAvx(a0_256);
    __m128 a1 = SumTopBottomAvx(a1_256);
    __m128 a2 = SumTopBottomAvx(a2_256);

    if (j + 4 <= dims) {
      __m128 q = _mm_loadu_ps(query.values() + j);
      __m128 v0 = _mm_loadu_ps(f0 + j);
      __m128 v1 = _mm_loadu_ps(f1 + j);
      __m128 v2 = _mm_loadu_ps(f2 + j);

      if (kShouldPrefetch && p0) {
        absl::PrefetchToLocalCache(p0 + j);
        absl::PrefetchToLocalCache(p1 + j);
        absl::PrefetchToLocalCache(p2 + j);
      }

      a0 = lambdas_vec[0].FmaTerm(a0, q, v0);
      a1 = lambdas_vec[1].FmaTerm(a1, q, v1);
      a2 = lambdas_vec[2].FmaTerm(a2, q, v2);
      j += 4;
    }

    if (j + 2 <= dims) {
      __m128 q = _mm_setzero_ps();
      __m128 v0 = _mm_setzero_ps();
      __m128 v1 = _mm_setzero_ps();
      __m128 v2 = _mm_setzero_ps();
      q = _mm_loadh_pi(q, reinterpret_cast<const __m64*>(query.values() + j));
      v0 = _mm_loadh_pi(v0, reinterpret_cast<const __m64*>(f0 + j));
      v1 = _mm_loadh_pi(v1, reinterpret_cast<const __m64*>(f1 + j));
      v2 = _mm_loadh_pi(v2, reinterpret_cast<const __m64*>(f2 + j));
      a0 = lambdas_vec[0].FmaTerm(a0, q, v0);
      a1 = lambdas_vec[1].FmaTerm(a1, q, v1);
      a2 = lambdas_vec[2].FmaTerm(a2, q, v2);
      j += 2;
    }

    float result0 = sum4(a0);
    float result1 = sum4(a1);
    float result2 = sum4(a2);

    if (j < dims) {
      DCHECK_EQ(j + 1, dims);
      result0 = lambdas_vec[0].AccTerm(result0, query.values()[j], f0[j]);
      result1 = lambdas_vec[1].AccTerm(result1, query.values()[j], f1[j]);
      result2 = lambdas_vec[2].AccTerm(result2, query.values()[j], f2[j]);
    }

    callback->invoke(i, lambdas_vec[0].Postprocess(result0));
    callback->invoke(i + num_outer_iters, lambdas_vec[1].Postprocess(result1));
    callback->invoke(i + 2 * num_outer_iters,
                     lambdas_vec[2].Postprocess(result2));
  });

  size_t i = parallel_end;
  for (; i < result.size(); ++i) {
    const DatapointPtr<float> f0 =
        MakeDatapointPtr(database->GetPtr(GetDatapointIndex(result, i)), dims);
    callback->invoke(i, lambdas.VectorVector(query, f0));
  }
}

template <typename T, typename DatasetView, typename ResultElem,
          bool kShouldPrefetch, typename CallbackFunctor>
void DenseGeneralHammingDistanceMeasureOneToManyInternal(
    const DatapointPtr<T>& query, const DatasetView* __restrict__ database,
    MutableSpan<ResultElem> result, CallbackFunctor* __restrict__ callback,
    ThreadPool* pool) {
  DCHECK(!pool || !kShouldPrefetch);
  const size_t dims = query.dimensionality();

  DCHECK_EQ(database->dimensionality(), dims);
  DCHECK_GT(dims, 0);
  const size_t num_outer_iters = result.size() / 3;
  const size_t parallel_end = num_outer_iters * 3;

  constexpr size_t kMinPrefetchAheadDims =
      (IsFloatingType<ResultElem>()) ? 512 : 256;
  size_t num_prefetch_datapoints;
  if (kShouldPrefetch) {
    num_prefetch_datapoints = std::max<size_t>(1, kMinPrefetchAheadDims / dims);
  }

  auto get_db_ptr = [&database, result, callback](size_t i)
                        SCANN_INLINE_LAMBDA -> const T* {
    auto idx = GetDatapointIndex(result, i);
    callback->prefetch(idx);
    return database->GetPtr(idx);
  };

  auto sum4 = [](__m128i x) -> uint32_t {
    x = _mm_add_epi32(x, _mm_srli_si128(x, 8));
    x = _mm_add_epi32(x, _mm_srli_si128(x, 4));
    return _mm_cvtsi128_si32(x);
  };

  ParallelFor<32>(
      Seq(num_outer_iters), pool, [&](size_t i) ABSL_ATTRIBUTE_ALWAYS_INLINE {
        const T* i0 = get_db_ptr(i);
        const T* i1 = get_db_ptr(i + num_outer_iters);
        const T* i2 = get_db_ptr(i + 2 * num_outer_iters);
        const T *p0 = nullptr, *p1 = nullptr, *p2 = nullptr;

        if (kShouldPrefetch && i + num_prefetch_datapoints < num_outer_iters) {
          p0 = get_db_ptr(i + num_prefetch_datapoints);
          p1 = get_db_ptr(i + num_outer_iters + num_prefetch_datapoints);
          p2 = get_db_ptr(i + 2 * num_outer_iters + num_prefetch_datapoints);
        }

        __m128i a0 = _mm_setzero_si128();
        __m128i a1 = _mm_setzero_si128();
        __m128i a2 = _mm_setzero_si128();
        size_t j = 0;

        for (; j + 4 <= dims; j += 4) {
          __m128i q = _mm_loadu_si128(
              reinterpret_cast<const __m128i*>(query.values() + j));
          __m128i v0 =
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(i0 + j));
          __m128i v1 =
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(i1 + j));
          __m128i v2 =
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(i2 + j));

          if (kShouldPrefetch && p0) {
            absl::PrefetchToLocalCache(p0 + j);
            absl::PrefetchToLocalCache(p1 + j);
            absl::PrefetchToLocalCache(p2 + j);
          }

          a0 = _mm_sub_epi32(a0, _mm_cmpeq_epi32(q, v0));
          a1 = _mm_sub_epi32(a1, _mm_cmpeq_epi32(q, v1));
          a2 = _mm_sub_epi32(a2, _mm_cmpeq_epi32(q, v2));
        }

        uint32_t result0 = sum4(a0);
        uint32_t result1 = sum4(a1);
        uint32_t result2 = sum4(a2);

        auto do_single_dim = [&]() {
          result0 += query.values()[j] == i0[j];
          result1 += query.values()[j] == i1[j];
          result2 += query.values()[j] == i2[j];
          ++j;
        };

        switch (dims - j) {
          case 3:
            do_single_dim();
            ABSL_FALLTHROUGH_INTENDED;
          case 2:
            do_single_dim();
            ABSL_FALLTHROUGH_INTENDED;
          case 1:
            do_single_dim();
            ABSL_FALLTHROUGH_INTENDED;
          case 0:
          default:
            DCHECK_EQ(dims, j);
        }

        callback->invoke(i, dims - result0);
        callback->invoke(i + num_outer_iters, dims - result1);
        callback->invoke(i + 2 * num_outer_iters, dims - result2);
      });

  size_t i = parallel_end;
  GeneralHammingDistance dist;
  for (; i < result.size(); ++i) {
    const DatapointPtr<T> i0 = MakeDatapointPtr<T>(
        database->GetPtr(GetDatapointIndex(result, i)), dims);
    callback->invoke(i, dist.GetDistanceDense(query, i0));
  }
}

#endif

template <typename T, typename DatasetView, typename Lambdas,
          typename ResultElem, typename CallbackFunctor>
enable_if_t<std::is_same<T, float>::value, void>
DenseAccumulatingDistanceMeasureOneToManyNoFma(
    const DatapointPtr<T>& query, const DatasetView* __restrict__ database,
    const Lambdas& lambdas, MutableSpan<ResultElem> result,
    CallbackFunctor* __restrict__ callback, ThreadPool* pool) {
  constexpr size_t kMinPrefetchAheadDims =
      (IsFloatingType<ResultElem>()) ? 8 : 4;
  constexpr size_t kMaxPrefetchAheadDims = 512;
  const DimensionIndex dims = query.nonzero_entries();

  if (dims < 8 || !kX64Simd || !RuntimeSupportsAvx1()) {
    if (!pool && database->dimensionality() <= kMaxPrefetchAheadDims &&
        database->dimensionality() >= kMinPrefetchAheadDims) {
      return DenseAccumulatingDistanceMeasureOneToManyInternal<
          T, DatasetView, Lambdas, ResultElem, true>(query, database, lambdas,
                                                     result, callback, nullptr);
    } else {
      return DenseAccumulatingDistanceMeasureOneToManyInternal<
          T, DatasetView, Lambdas, ResultElem, false>(query, database, lambdas,
                                                      result, callback, pool);
    }
  }

#ifdef __x86_64__
  if (!pool && database->dimensionality() <= kMaxPrefetchAheadDims &&
      database->dimensionality() >= kMinPrefetchAheadDims) {
    return DenseAccumulatingDistanceMeasureOneToManyInternalAvx1<
        T, DatasetView, Lambdas, ResultElem, true>(query, database, lambdas,
                                                   result, callback, nullptr);
  } else {
    return DenseAccumulatingDistanceMeasureOneToManyInternalAvx1<
        T, DatasetView, Lambdas, ResultElem, false>(query, database, lambdas,
                                                    result, callback, pool);
  }
#endif
}

template <typename T, typename DatasetView, typename Lambdas,
          typename ResultElem, typename CallbackFunctor>
enable_if_t<std::is_same<T, float>::value, void>
DenseAccumulatingDistanceMeasureOneToMany(
    const DatapointPtr<T>& query, const DatasetView* __restrict__ database,
    const Lambdas& lambdas, MutableSpan<ResultElem> result,
    CallbackFunctor* __restrict__ callback, ThreadPool* pool) {
  const DimensionIndex dims = query.nonzero_entries();

  if (dims < 8 || !kX64Simd || !RuntimeSupportsAvx2()) {
    return DenseAccumulatingDistanceMeasureOneToManyNoFma<T, DatasetView,
                                                          Lambdas, ResultElem>(
        query, database, lambdas, result, callback, pool);
  }

#ifdef __x86_64__
  constexpr size_t kMinPrefetchAheadDims =
      (IsFloatingType<ResultElem>()) ? 8 : 4;
  constexpr size_t kMaxPrefetchAheadDims = 512;
  if (!pool && database->dimensionality() <= kMaxPrefetchAheadDims &&
      database->dimensionality() >= kMinPrefetchAheadDims) {
    return DenseAccumulatingDistanceMeasureOneToManyInternalAvx2<
        T, DatasetView, Lambdas, ResultElem, true>(query, database, lambdas,
                                                   result, callback, nullptr);
  } else {
    return DenseAccumulatingDistanceMeasureOneToManyInternalAvx2<
        T, DatasetView, Lambdas, ResultElem, false>(query, database, lambdas,
                                                    result, callback, pool);
  }
#endif
}

template <typename T, typename DatasetView, typename Lambdas,
          typename ResultElem, bool kShouldPrefetch, typename CallbackFunctor>
enable_if_t<std::is_same_v<T, float>, void> SCANN_OUTLINE
DenseAccumulatingDistanceMeasureOneToManyInternal(
    const DatapointPtr<T>& query, const DatasetView* __restrict__ database,
    const Lambdas& lambdas, MutableSpan<ResultElem> result,
    CallbackFunctor* __restrict__ callback, ThreadPool* pool) {
  if (result.empty()) return;
  DCHECK(!pool || !kShouldPrefetch);
  const size_t dims = query.dimensionality();
  DCHECK_EQ(dims, database->dimensionality());

  DCHECK_GT(dims, 0);
  Lambdas lambdas_vec[4] = {lambdas, lambdas, lambdas, lambdas};
  const size_t num_outer_iters = result.size() / 3;
  const size_t parallel_end = num_outer_iters * 3;

  constexpr size_t kMinPrefetchAheadDims =
      (IsFloatingType<ResultElem>()) ? 512 : 256;
  size_t num_prefetch_datapoints;
  if (kShouldPrefetch) {
    num_prefetch_datapoints = std::max<size_t>(1, kMinPrefetchAheadDims / dims);
  }

  auto get_db_ptr = [&database, result, callback](size_t i)
                        SCANN_INLINE_LAMBDA -> const float* {
    auto idx = GetDatapointIndex(result, i);
    callback->prefetch(idx);
    return database->GetPtr(idx);
  };

  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  const D d;
  using D2 = hn::ScalableTag<float, -1>;
  const D2 d2;

  ParallelFor<32>(
      Seq(num_outer_iters), pool, [&](size_t i) ABSL_ATTRIBUTE_ALWAYS_INLINE {
        const float* f0 = get_db_ptr(i);
        const float* f1 = get_db_ptr(i + num_outer_iters);
        const float* f2 = get_db_ptr(i + 2 * num_outer_iters);
        const float *p0 = nullptr, *p1 = nullptr, *p2 = nullptr;

        if (kShouldPrefetch && i + num_prefetch_datapoints < num_outer_iters) {
          p0 = get_db_ptr(i + num_prefetch_datapoints);
          p1 = get_db_ptr(i + num_outer_iters + num_prefetch_datapoints);
          p2 = get_db_ptr(i + 2 * num_outer_iters + num_prefetch_datapoints);
        }

        hn::Vec<D> a0 = hn::Zero(d);
        hn::Vec<D> a1 = hn::Zero(d);
        hn::Vec<D> a2 = hn::Zero(d);
        size_t j = 0;

        for (; j + hn::Lanes(d) <= dims; j += hn::Lanes(d)) {
          auto q = hn::LoadU(d, query.values() + j);
          auto v0 = hn::LoadU(d, f0 + j);
          auto v1 = hn::LoadU(d, f1 + j);
          auto v2 = hn::LoadU(d, f2 + j);

          if (kShouldPrefetch && p0) {
            absl::PrefetchToLocalCache(p0 + j);
            absl::PrefetchToLocalCache(p1 + j);
            absl::PrefetchToLocalCache(p2 + j);
          }

          a0 = lambdas_vec[0].template AccTerm<D>(a0, q, v0);
          a1 = lambdas_vec[1].template AccTerm<D>(a1, q, v1);
          a2 = lambdas_vec[2].template AccTerm<D>(a2, q, v2);
        }

        if (j + hn::Lanes(d2) <= dims) {
          auto q = hn::ZeroExtendVector(d, hn::LoadU(d2, query.values() + j));
          auto v0 = hn::ZeroExtendVector(d, hn::LoadU(d2, f0 + j));
          auto v1 = hn::ZeroExtendVector(d, hn::LoadU(d2, f1 + j));
          auto v2 = hn::ZeroExtendVector(d, hn::LoadU(d2, f2 + j));
          a0 = lambdas_vec[0].template AccTerm<D>(a0, q, v0);
          a1 = lambdas_vec[1].template AccTerm<D>(a1, q, v1);
          a2 = lambdas_vec[2].template AccTerm<D>(a2, q, v2);
          j += hn::Lanes(d2);
        }

        float result0 = hn::ReduceSum(d, a0);
        float result1 = hn::ReduceSum(d, a1);
        float result2 = hn::ReduceSum(d, a2);

        for (; j < dims; ++j) {
          result0 = lambdas_vec[0].AccTerm(result0, query.values()[j], f0[j]);
          result1 = lambdas_vec[1].AccTerm(result1, query.values()[j], f1[j]);
          result2 = lambdas_vec[2].AccTerm(result2, query.values()[j], f2[j]);

          if (hn::Lanes(d) <= 4) break;
        }

        callback->invoke(i, lambdas_vec[0].Postprocess(result0));
        callback->invoke(i + num_outer_iters,
                         lambdas_vec[1].Postprocess(result1));
        callback->invoke(i + 2 * num_outer_iters,
                         lambdas_vec[2].Postprocess(result2));
      });

  size_t i = parallel_end;
  for (; i < result.size(); ++i) {
    const DatapointPtr<float> f0 = MakeDatapointPtr<T>(
        database->GetPtr(GetDatapointIndex(result, i)), dims);
    callback->invoke(i, lambdas.VectorVector(query, f0));
  }
}

template <typename T, typename DatasetView, typename Lambdas,
          typename ResultElem, bool kShouldPrefetch, typename CallbackFunctor>
enable_if_t<std::is_same_v<T, double>, void>
DenseAccumulatingDistanceMeasureOneToManyInternal(
    const DatapointPtr<T>& query, const DatasetView* __restrict__ database,
    const Lambdas& lambdas, MutableSpan<ResultElem> result,
    CallbackFunctor* __restrict__ callback, ThreadPool* pool) {
  if (result.empty()) return;
  const size_t dims = query.dimensionality();

  DCHECK_EQ(database->dimensionality(), dims);
  DCHECK_GT(dims, 0);
  Lambdas lambdas_vec[3] = {lambdas, lambdas, lambdas};
  const size_t num_outer_iters = result.size() / 3;
  const size_t parallel_end = num_outer_iters * 3;

  constexpr size_t kMinPrefetchAheadDims =
      (IsFloatingType<ResultElem>()) ? 256 : 128;
  size_t num_prefetch_datapoints;
  if (kShouldPrefetch) {
    num_prefetch_datapoints = std::max<size_t>(1, kMinPrefetchAheadDims / dims);
  }

  auto get_db_ptr = [&database, result, callback](size_t i)
                        SCANN_INLINE_LAMBDA -> const double* {
    auto idx = GetDatapointIndex(result, i);
    callback->prefetch(idx);
    return database->GetPtr(idx);
  };

  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<double>;
  const D d;

  ParallelFor<32>(
      Seq(num_outer_iters), pool, [&](size_t i) ABSL_ATTRIBUTE_ALWAYS_INLINE {
        const double* f0 = get_db_ptr(i);
        const double* f1 = get_db_ptr(i + num_outer_iters);
        const double* f2 = get_db_ptr(i + 2 * num_outer_iters);
        const double *p0 = nullptr, *p1 = nullptr, *p2 = nullptr;

        if (kShouldPrefetch && i + num_prefetch_datapoints < num_outer_iters) {
          p0 = get_db_ptr(i + num_prefetch_datapoints);
          p1 = get_db_ptr(i + num_outer_iters + num_prefetch_datapoints);
          p2 = get_db_ptr(i + 2 * num_outer_iters + num_prefetch_datapoints);
        }

        auto a0 = hn::Zero(d);
        auto a1 = hn::Zero(d);
        auto a2 = hn::Zero(d);
        size_t j = 0;

        const size_t kLanes = hn::Lanes(d);
        for (; j + kLanes <= dims; j += kLanes) {
          auto q = hn::LoadU(d, query.values() + j);
          auto v0 = hn::LoadU(d, f0 + j);
          auto v1 = hn::LoadU(d, f1 + j);
          auto v2 = hn::LoadU(d, f2 + j);

          if (kShouldPrefetch && p0) {
            absl::PrefetchToLocalCache(p0 + j);
            absl::PrefetchToLocalCache(p1 + j);
            absl::PrefetchToLocalCache(p2 + j);
          }

          a0 = lambdas_vec[0].template AccTerm<D>(a0, q, v0);
          a1 = lambdas_vec[1].template AccTerm<D>(a1, q, v1);
          a2 = lambdas_vec[2].template AccTerm<D>(a2, q, v2);
        }

        double result0 = hn::ReduceSum(d, a0);
        double result1 = hn::ReduceSum(d, a1);
        double result2 = hn::ReduceSum(d, a2);

        for (; j < dims; ++j) {
          result0 = lambdas_vec[0].AccTerm(result0, query.values()[j], f0[j]);
          result1 = lambdas_vec[1].AccTerm(result1, query.values()[j], f1[j]);
          result2 = lambdas_vec[2].AccTerm(result2, query.values()[j], f2[j]);

          if (kLanes <= 2) break;
        }

        callback->invoke(i, lambdas_vec[0].Postprocess(result0));
        callback->invoke(i + num_outer_iters,
                         lambdas_vec[1].Postprocess(result1));
        callback->invoke(i + 2 * num_outer_iters,
                         lambdas_vec[2].Postprocess(result2));
      });

  size_t i = parallel_end;
  for (; i < result.size(); ++i) {
    const DatapointPtr<double> f0 = MakeDatapointPtr<double>(
        database->GetPtr(GetDatapointIndex(result, i)), dims);
    callback->invoke(i, lambdas.VectorVector(query, f0));
  }
}

template <typename T, typename DatasetView, typename Lambdas,
          typename ResultElem, typename CallbackFunctor>
enable_if_t<std::is_same<T, double>::value && kX64Simd, void>
DenseAccumulatingDistanceMeasureOneToMany(
    const DatapointPtr<T>& query, const DatasetView* __restrict__ database,
    const Lambdas& lambdas, MutableSpan<ResultElem> result,
    CallbackFunctor* __restrict__ callback, ThreadPool* pool) {
  constexpr size_t kMinPrefetchAheadDims =
      (IsFloatingType<ResultElem>()) ? 4 : 2;
  constexpr size_t kMaxPrefetchAheadDims = 256;
  if (!pool && database->dimensionality() <= kMaxPrefetchAheadDims &&
      database->dimensionality() >= kMinPrefetchAheadDims) {
    return DenseAccumulatingDistanceMeasureOneToManyInternal<
        T, DatasetView, Lambdas, ResultElem, true>(query, database, lambdas,
                                                   result, callback, nullptr);
  } else {
    return DenseAccumulatingDistanceMeasureOneToManyInternal<
        T, DatasetView, Lambdas, ResultElem, false>(query, database, lambdas,
                                                    result, callback, pool);
  }
}

template <typename T, typename DatasetView, typename Lambdas,
          typename ResultElem, typename CallbackFunctor>
SCANN_INLINE enable_if_t<std::is_same<T, double>::value, void>
DenseAccumulatingDistanceMeasureOneToManyNoFma(
    const DatapointPtr<T>& query, const DatasetView* __restrict__ database,
    const Lambdas& lambdas, MutableSpan<ResultElem> result,
    CallbackFunctor* __restrict__ callback, ThreadPool* pool) {
  return DenseAccumulatingDistanceMeasureOneToMany<T, DatasetView, Lambdas,
                                                   ResultElem>(
      query, database, lambdas, result, callback, pool);
}

template <typename T, typename Lambdas>
void DenseAccumulatingDistanceMeasureOneToManyBlockTransposed(
    const DatapointPtr<T>& query, ConstSpan<T> database, const Lambdas& lambdas,
    MutableSpan<T> result) {
  const size_t lanes =
      hwy::HWY_NAMESPACE::Lanes(hwy::HWY_NAMESPACE::ScalableTag<T>());
  const size_t dims = query.dimensionality();
  DCHECK_EQ(database.size() % (lanes * dims), 0);
  DCHECK_EQ(result.size() % lanes, 0);
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<T>;
  const D d;
  const size_t num_simd_blocks = database.size() / (lanes * dims);

  size_t dst_idx = 0;
  const ConstSpan<T> qspan = query.values_span();
  for (size_t simd_block_idx = 0; simd_block_idx + 1 < num_simd_blocks;
       simd_block_idx += 2) {
    hn::Vec<D> a0 = hn::Zero(d);
    hn::Vec<D> a1 = hn::Zero(d);
    const T* d0_ptr = database.data() + simd_block_idx * dims * lanes;
    const T* d1_ptr = d0_ptr + dims * lanes;
    for (DimensionIndex dim_idx : Seq(dims)) {
      auto q = hn::Set(d, qspan[dim_idx]);
      auto d0 = hn::LoadU(d, d0_ptr);
      auto d1 = hn::LoadU(d, d1_ptr);
      a0 = lambdas.template AccTerm<D>(a0, q, d0);
      a1 = lambdas.template AccTerm<D>(a1, q, d1);
      d0_ptr += lanes;
      d1_ptr += lanes;
    }
    hn::StoreU(a0, d, result.data() + dst_idx);
    dst_idx += lanes;
    hn::StoreU(a1, d, result.data() + dst_idx);
    dst_idx += lanes;
  }

  if (num_simd_blocks & 1) {
    hn::Vec<D> a0 = hn::Zero(d);
    const size_t simd_block_idx = num_simd_blocks - 1;
    const T* d0_ptr = database.data() + simd_block_idx * dims * lanes;
    for (DimensionIndex dim_idx : Seq(dims)) {
      auto q = hn::Set(d, qspan[dim_idx]);
      auto d0 = hn::LoadU(d, d0_ptr);
      a0 = lambdas.template AccTerm<D>(a0, q, d0);
      d0_ptr += lanes;
    }
    hn::StoreU(a0, d, result.data() + dst_idx);
  }
}

template <typename T>
class DotProductDistanceLambdas {
 public:
#ifdef __SSE3__
  static __m128 AccTerm(__m128 acc, __m128 a, __m128 b) { return acc - a * b; }

  static SCANN_AVX1_INLINE __m256 AccTerm(__m256 acc, __m256 a, __m256 b) {
    return acc - a * b;
  }

  static SCANN_AVX2_INLINE __m256 FmaTerm(__m256 acc, __m256 a, __m256 b) {
    return _mm256_fnmadd_ps(a, b, acc);
  }

  static SCANN_AVX2_INLINE __m128 FmaTerm(__m128 acc, __m128 a, __m128 b) {
    return _mm_fnmadd_ps(a, b, acc);
  }

  static __m128d AccTerm(__m128d acc, __m128d a, __m128d b) {
    return acc - a * b;
  }
#endif

  template <typename D>
  static hwy::HWY_NAMESPACE::Vec<D> AccTerm(hwy::HWY_NAMESPACE::Vec<D> acc,
                                            hwy::HWY_NAMESPACE::Vec<D> a,
                                            hwy::HWY_NAMESPACE::Vec<D> b) {
    return hwy::HWY_NAMESPACE::NegMulAdd(a, b, acc);
  }

  static float AccTerm(float acc, float a, float b) { return acc - a * b; }
  static double AccTerm(double acc, double a, double b) { return acc - a * b; }
  static float Postprocess(float val) { return val; }
  static double Postprocess(double val) { return val; }

  double VectorVector(const DatapointPtr<T>& a,
                      const DatapointPtr<T>& b) const {
    return dist_.GetDistanceDense(a, b);
  }

 private:
  DotProductDistance dist_;
};

template <typename T>
class SquaredL2DistanceLambdas {
 public:
#ifdef __SSE3__
  static __m128 AccTerm(__m128 acc, __m128 a, __m128 b) {
    __m128 tmp = a - b;
    return acc + tmp * tmp;
  }

  SCANN_AVX1_INLINE static __m256 AccTerm(__m256 acc, __m256 a, __m256 b) {
    __m256 tmp = a - b;
    return acc + tmp * tmp;
  }

  static SCANN_AVX2_INLINE __m256 FmaTerm(__m256 acc, __m256 a, __m256 b) {
    __m256 tmp = _mm256_sub_ps(a, b);
    return _mm256_fmadd_ps(tmp, tmp, acc);
  }

  static SCANN_AVX2_INLINE __m128 FmaTerm(__m128 acc, __m128 a, __m128 b) {
    __m128 tmp = _mm_sub_ps(a, b);
    return _mm_fmadd_ps(tmp, tmp, acc);
  }

  static __m128d AccTerm(__m128d acc, __m128d a, __m128d b) {
    __m128d tmp = a - b;
    return acc + tmp * tmp;
  }
#endif

  template <typename D>
  static hwy::HWY_NAMESPACE::Vec<D> AccTerm(hwy::HWY_NAMESPACE::Vec<D> acc,
                                            hwy::HWY_NAMESPACE::Vec<D> a,
                                            hwy::HWY_NAMESPACE::Vec<D> b) {
    auto diff = a - b;
    return hwy::HWY_NAMESPACE::MulAdd(diff, diff, acc);
  }

  static float AccTerm(float acc, float a, float b) {
    const float tmp = a - b;
    return acc + tmp * tmp;
  }

  static double AccTerm(double acc, double a, double b) {
    const double tmp = a - b;
    return acc + tmp * tmp;
  }

  static float Postprocess(float val) { return val; }
  static double Postprocess(double val) { return val; }

  double VectorVector(const DatapointPtr<T>& a,
                      const DatapointPtr<T>& b) const {
    return dist_.GetDistanceDense(a, b);
  }

 private:
  SquaredL2Distance dist_;
};

}  // namespace one_to_many_low_level

template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
void DenseDotProductDistanceOneToMany(const DatapointPtr<T>& query,
                                      const DatasetView* __restrict__ database,
                                      MutableSpan<ResultElem> result,
                                      CallbackFunctor* __restrict__ callback,
                                      ThreadPool* pool) {
  one_to_many_low_level::DotProductDistanceLambdas<T> lambdas;
  return one_to_many_low_level::DenseAccumulatingDistanceMeasureOneToMany(
      query, database, lambdas, result, callback, pool);
}

template <typename T, typename ResultElem>
void DenseDotProductDistanceOneToMany(const DatapointPtr<T>& query,
                                      const DenseDataset<T>& database,
                                      MutableSpan<ResultElem> result,
                                      ThreadPool* pool) {
  one_to_many_low_level::DotProductDistanceLambdas<T> lambdas;
  auto view = DefaultDenseDatasetView<T>(database);
  one_to_many_low_level::SetDistanceFunctor<ResultElem> set_distance_functor(
      result);
  return one_to_many_low_level::DenseAccumulatingDistanceMeasureOneToMany(
      query, &view, lambdas, result, &set_distance_functor, pool);
}

namespace one_to_many_low_level {
template <typename T>
class AbsDotProductDistanceLambdas {
 public:
#ifdef __SSE3__
  static __m128 AccTerm(__m128 acc, __m128 a, __m128 b) { return acc + a * b; }

  static SCANN_AVX1_INLINE __m256 AccTerm(__m256 acc, __m256 a, __m256 b) {
    return acc + a * b;
  }

  static SCANN_AVX2_INLINE __m256 FmaTerm(__m256 acc, __m256 a, __m256 b) {
    return _mm256_fmadd_ps(a, b, acc);
  }

  static SCANN_AVX2_INLINE __m128 FmaTerm(__m128 acc, __m128 a, __m128 b) {
    return _mm_fmadd_ps(a, b, acc);
  }

  static __m128d AccTerm(__m128d acc, __m128d a, __m128d b) {
    return acc + a * b;
  }
#endif

  template <typename D>
  static hwy::HWY_NAMESPACE::Vec<D> AccTerm(hwy::HWY_NAMESPACE::Vec<D> acc,
                                            hwy::HWY_NAMESPACE::Vec<D> a,
                                            hwy::HWY_NAMESPACE::Vec<D> b) {
    return hwy::HWY_NAMESPACE::MulAdd(a, b, acc);
  }

  static float AccTerm(double acc, float a, float b) { return acc + a * b; }
  static double AccTerm(double acc, double a, double b) { return acc + a * b; }
  static float Postprocess(float val) { return -std::abs(val); }
  static double Postprocess(double val) { return -std::abs(val); }

  double VectorVector(const DatapointPtr<T>& a,
                      const DatapointPtr<T>& b) const {
    return dist_.GetDistanceDense(a, b);
  }

 private:
  AbsDotProductDistance dist_;
};
}  // namespace one_to_many_low_level

template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
void DenseAbsDotProductDistanceOneToMany(
    const DatapointPtr<T>& query, const DatasetView* __restrict__ database,
    MutableSpan<ResultElem> result, CallbackFunctor* __restrict__ callback,
    ThreadPool* pool) {
  one_to_many_low_level::AbsDotProductDistanceLambdas<T> lambdas;
  return one_to_many_low_level::DenseAccumulatingDistanceMeasureOneToMany(
      query, database, lambdas, result, callback, pool);
}

template <typename T, typename ResultElem>
void DenseAbsDotProductDistanceOneToMany(const DatapointPtr<T>& query,
                                         const DenseDataset<T>& database,
                                         MutableSpan<ResultElem> result,
                                         ThreadPool* pool) {
  auto view = DefaultDenseDatasetView<T>(database);
  one_to_many_low_level::SetDistanceFunctor<ResultElem> set_distance_functor(
      result);
  DenseAbsDotProductDistanceOneToMany(query, &view, result,
                                      &set_distance_functor, pool);
}

namespace one_to_many_low_level {
template <typename T>
class CosineDistanceLambdas {
 public:
#ifdef __SSE3__
  static __m128 AccTerm(__m128 acc, __m128 a, __m128 b) { return acc + a * b; }

  static SCANN_AVX1_INLINE __m256 AccTerm(__m256 acc, __m256 a, __m256 b) {
    return acc + a * b;
  }

  static SCANN_AVX2_INLINE __m256 FmaTerm(__m256 acc, __m256 a, __m256 b) {
    return _mm256_fmadd_ps(a, b, acc);
  }

  static SCANN_AVX2_INLINE __m128 FmaTerm(__m128 acc, __m128 a, __m128 b) {
    return _mm_fmadd_ps(a, b, acc);
  }

  static __m128d AccTerm(__m128d acc, __m128d a, __m128d b) {
    return acc + a * b;
  }
#endif

  template <typename D>
  static hwy::HWY_NAMESPACE::Vec<D> AccTerm(hwy::HWY_NAMESPACE::Vec<D> acc,
                                            hwy::HWY_NAMESPACE::Vec<D> a,
                                            hwy::HWY_NAMESPACE::Vec<D> b) {
    return hwy::HWY_NAMESPACE::MulAdd(a, b, acc);
  }

  static float AccTerm(float acc, float a, float b) { return acc + a * b; }
  static double AccTerm(double acc, double a, double b) { return acc + a * b; }
  static float Postprocess(float val) { return 1.0f - val; }
  static double Postprocess(double val) { return 1.0 - val; }

  double VectorVector(const DatapointPtr<T>& a,
                      const DatapointPtr<T>& b) const {
    return dist_.GetDistanceDense(a, b);
  }

 private:
  CosineDistance dist_;
};
}  // namespace one_to_many_low_level

template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
void DenseCosineDistanceOneToMany(const DatapointPtr<T>& query,
                                  const DatasetView* __restrict__ database,
                                  MutableSpan<ResultElem> result,
                                  CallbackFunctor* __restrict__ callback,
                                  ThreadPool* pool) {
  one_to_many_low_level::CosineDistanceLambdas<T> lambdas;
  return one_to_many_low_level::DenseAccumulatingDistanceMeasureOneToMany(
      query, database, lambdas, result, callback, pool);
}

template <typename T, typename ResultElem>
void DenseCosineDistanceOneToMany(const DatapointPtr<T>& query,
                                  const DenseDataset<T>& database,
                                  MutableSpan<ResultElem> result,
                                  ThreadPool* pool) {
  auto view = DefaultDenseDatasetView<T>(database);
  one_to_many_low_level::SetDistanceFunctor<ResultElem> set_distance_functor(
      result);
  DenseCosineDistanceOneToMany(query, &view, result, &set_distance_functor,
                               pool);
}

template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
void DenseSquaredL2DistanceOneToMany(const DatapointPtr<T>& query,
                                     const DatasetView* __restrict__ database,
                                     MutableSpan<ResultElem> result,
                                     CallbackFunctor* __restrict__ callback,
                                     ThreadPool* pool) {
  one_to_many_low_level::SquaredL2DistanceLambdas<T> lambdas;
  return one_to_many_low_level::DenseAccumulatingDistanceMeasureOneToMany(
      query, database, lambdas, result, callback, pool);
}

template <typename T, typename ResultElem>
void DenseSquaredL2DistanceOneToMany(const DatapointPtr<T>& query,
                                     const DenseDataset<T>& database,
                                     MutableSpan<ResultElem> result,
                                     ThreadPool* pool) {
  auto view = DefaultDenseDatasetView<T>(database);
  one_to_many_low_level::SetDistanceFunctor<ResultElem> set_distance_functor(
      result);
  DenseSquaredL2DistanceOneToMany(query, &view, result, &set_distance_functor,
                                  pool);
}

namespace one_to_many_low_level {
template <typename T>
class L2DistanceLambdas {
 public:
#ifdef __SSE3__
  static __m128 AccTerm(__m128 acc, __m128 a, __m128 b) {
    __m128 tmp = _mm_sub_ps(a, b);
    return acc + tmp * tmp;
  }

  SCANN_AVX1_INLINE static __m256 AccTerm(__m256 acc, __m256 a, __m256 b) {
    __m256 tmp = _mm256_sub_ps(a, b);
    return acc + tmp * tmp;
  }

  static SCANN_AVX2_INLINE __m256 FmaTerm(__m256 acc, __m256 a, __m256 b) {
    __m256 tmp = _mm256_sub_ps(a, b);
    return _mm256_fmadd_ps(tmp, tmp, acc);
  }

  static SCANN_AVX2_INLINE __m128 FmaTerm(__m128 acc, __m128 a, __m128 b) {
    __m128 tmp = _mm_sub_ps(a, b);
    return _mm_fmadd_ps(tmp, tmp, acc);
  }

  static __m128d AccTerm(__m128d acc, __m128d a, __m128d b) {
    __m128d tmp = _mm_sub_pd(a, b);
    return acc + tmp * tmp;
  }
#endif

  template <typename D>
  static hwy::HWY_NAMESPACE::Vec<D> AccTerm(hwy::HWY_NAMESPACE::Vec<D> acc,
                                            hwy::HWY_NAMESPACE::Vec<D> a,
                                            hwy::HWY_NAMESPACE::Vec<D> b) {
    auto diff = a - b;
    return hwy::HWY_NAMESPACE::MulAdd(diff, diff, acc);
  }

  static float AccTerm(float acc, float a, float b) {
    const float tmp = a - b;
    return acc + tmp * tmp;
  }

  static double AccTerm(double acc, double a, double b) {
    const double tmp = a - b;
    return acc + tmp * tmp;
  }

  static float Postprocess(float val) { return std::sqrt(val); }
  static double Postprocess(double val) { return std::sqrt(val); }

  double VectorVector(const DatapointPtr<T>& a,
                      const DatapointPtr<T>& b) const {
    return dist_.GetDistanceDense(a, b);
  }

 private:
  L2Distance dist_;
};
}  // namespace one_to_many_low_level

template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
void DenseL2DistanceOneToMany(const DatapointPtr<T>& query,
                              const DatasetView* __restrict__ database,
                              MutableSpan<ResultElem> result,
                              CallbackFunctor* __restrict__ callback,
                              ThreadPool* pool) {
  one_to_many_low_level::L2DistanceLambdas<T> lambdas;
  return one_to_many_low_level::DenseAccumulatingDistanceMeasureOneToMany(
      query, database, lambdas, result, callback, pool);
}

template <typename T, typename ResultElem>
void DenseL2DistanceOneToMany(const DatapointPtr<T>& query,
                              const DenseDataset<T>& database,
                              MutableSpan<ResultElem> result,
                              ThreadPool* pool) {
  auto view = DefaultDenseDatasetView<T>(database);
  one_to_many_low_level::SetDistanceFunctor<ResultElem> set_distance_functor(
      result);
  DenseL2DistanceOneToMany(query, &view, result, &set_distance_functor, pool);
}

namespace one_to_many_low_level {
#ifdef __SSE3__
constexpr int32_t kAbsMaskScalar = 0x7FFFFFFF;
#endif

template <typename T>
class L1DistanceLambdas {
 public:
#ifdef __SSE3__
  static __m128 AccTerm(__m128 acc, __m128 a, __m128 b) {
    return acc + AbsPs(_mm_sub_ps(a, b));
  }

  SCANN_AVX1_INLINE static __m256 AccTerm(__m256 acc, __m256 a, __m256 b) {
    return acc + AbsPs(_mm256_sub_ps(a, b));
  }

  static __m128d AccTerm(__m128d acc, __m128d a, __m128d b) {
    return acc + AbsPd(_mm_sub_pd(a, b));
  }
#endif

  template <typename D>
  static hwy::HWY_NAMESPACE::Vec<D> AccTerm(hwy::HWY_NAMESPACE::Vec<D> acc,
                                            hwy::HWY_NAMESPACE::Vec<D> a,
                                            hwy::HWY_NAMESPACE::Vec<D> b) {
    return acc + hwy::HWY_NAMESPACE::Abs(a - b);
  }

  static float AccTerm(double acc, float a, float b) {
    return acc + std::abs(a - b);
  }

  static double AccTerm(double acc, double a, double b) {
    return acc + std::abs(a - b);
  }

  static float Postprocess(float val) { return val; }
  static double Postprocess(double val) { return val; }

  double VectorVector(const DatapointPtr<T>& a,
                      const DatapointPtr<T>& b) const {
    return dist_.GetDistanceDense(a, b);
  }

 private:
  L1Distance dist_;

#ifdef __SSE3__

  static const __m128 kAbsMaskVectorFloat;

  static const __m128d kAbsMaskVectorDouble;

  static __m128 AbsPs(__m128 x) { return _mm_and_ps(x, kAbsMaskVectorFloat); }
  static SCANN_AVX1_INLINE __m256 AbsPs(__m256 x) {
    static const __m256 kAbsMaskVectorFloat256 =
        _mm256_castsi256_ps(_mm256_set_epi32(
            kAbsMaskScalar, kAbsMaskScalar, kAbsMaskScalar, kAbsMaskScalar,
            kAbsMaskScalar, kAbsMaskScalar, kAbsMaskScalar, kAbsMaskScalar));
    return _mm256_and_ps(x, kAbsMaskVectorFloat256);
  }

  static __m128d AbsPd(__m128d x) {
    return _mm_and_pd(x, kAbsMaskVectorDouble);
  }
#endif
};

#ifdef __SSE3__
constexpr int32_t kAllSet = 0xFFFFFFFF;

template <typename T>
const __m128 L1DistanceLambdas<T>::kAbsMaskVectorFloat =
    _mm_castsi128_ps(_mm_set_epi32(kAbsMaskScalar, kAbsMaskScalar,
                                   kAbsMaskScalar, kAbsMaskScalar));

template <typename T>
const __m128d L1DistanceLambdas<T>::kAbsMaskVectorDouble = _mm_castsi128_pd(
    _mm_set_epi32(kAbsMaskScalar, kAllSet, kAbsMaskScalar, kAllSet));
#endif

}  // namespace one_to_many_low_level

template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
void DenseL1DistanceOneToMany(const DatapointPtr<T>& query,
                              const DatasetView* __restrict__ database,
                              MutableSpan<ResultElem> result,
                              CallbackFunctor* __restrict__ callback,
                              ThreadPool* pool) {
  one_to_many_low_level::L1DistanceLambdas<T> lambdas;
  return one_to_many_low_level::DenseAccumulatingDistanceMeasureOneToManyNoFma(
      query, database, lambdas, result, callback, pool);
}

template <typename T, typename ResultElem>
void DenseL1DistanceOneToMany(const DatapointPtr<T>& query,
                              const DenseDataset<T>& database,
                              MutableSpan<ResultElem> result,
                              ThreadPool* pool) {
  auto view = DefaultDenseDatasetView<T>(database);
  one_to_many_low_level::SetDistanceFunctor<ResultElem> set_distance_functor(
      result);
  DenseL1DistanceOneToMany(query, &view, result, &set_distance_functor, pool);
}

namespace limited_inner_internal {

namespace hn = hwy::HWY_NAMESPACE;
using HwyFloats = decltype(hn::Zero(hn::ScalableTag<float>()));
using HwyDoubles = decltype(hn::Zero(hn::ScalableTag<double>()));

template <typename T>
class LimitedInnerProductDistanceLambdas {
 public:
  explicit LimitedInnerProductDistanceLambdas(double norm_query2)
      : norm_a2_(norm_query2) {}

  template <typename D>
  hn::Vec<D> AccTerm(hn::Vec<D> acc, hn::Vec<D> a, hn::Vec<D> b) {
    return AccTermImpl(acc, a, b);
  }

  float AccTerm(float acc, float a, float b) {
    norm_b2_s_ += b * b;
    return acc + a * b;
  }

  double AccTerm(double acc, double a, double b) {
    norm_b2_d_ += b * b;
    return acc + a * b;
  }

  float Postprocess(float val) {
    norm_b2_s_ += hn::ReduceSum(hn::ScalableTag<float>(), norm_b2_ps_);

    norm_b2_ps_ = hn::Zero(hn::ScalableTag<float>());

    const float denom = std::sqrt(
        norm_a2_ * std::max(static_cast<float>(norm_a2_), norm_b2_s_));

    norm_b2_s_ = 0;
    if (denom == 0.0f) {
      return 0.0f;
    }
    return -val / denom;
  }

  double Postprocess(double val) {
    norm_b2_d_ += hn::ReduceSum(hn::ScalableTag<double>(), norm_b2_pd_);

    norm_b2_pd_ = hn::Zero(hn::ScalableTag<double>());

    const double denom = std::sqrt(norm_a2_ * std::max(norm_a2_, norm_b2_d_));

    norm_b2_d_ = 0;
    if (denom == 0.0) {
      return 0.0;
    }
    return -val / denom;
  }

  double VectorVector(const DatapointPtr<T>& a,
                      const DatapointPtr<T>& b) const {
    return dist_.GetDistanceDense(a, b);
  }

 private:
  HwyFloats AccTermImpl(HwyFloats acc, HwyFloats a, HwyFloats b) {
    norm_b2_ps_ = hn::MulAdd(b, b, norm_b2_ps_);
    return hn::MulAdd(a, b, acc);
  }

  HwyDoubles AccTermImpl(HwyDoubles acc, HwyDoubles a, HwyDoubles b) {
    norm_b2_pd_ = hn::MulAdd(b, b, norm_b2_pd_);
    return hn::MulAdd(a, b, acc);
  }

  LimitedInnerProductDistance dist_;
  double norm_a2_;

  HwyFloats norm_b2_ps_ = hn::Zero(hn::ScalableTag<float>());
  HwyDoubles norm_b2_pd_ = hn::Zero(hn::ScalableTag<double>());

  float norm_b2_s_ = 0;
  double norm_b2_d_ = 0;
};

}  // namespace limited_inner_internal

template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
void DenseLimitedInnerProductDistanceOneToMany(
    const DatapointPtr<T>& query, const DatasetView* __restrict__ database,
    MutableSpan<ResultElem> result, CallbackFunctor* __restrict__ callback,
    ThreadPool* pool) {
  using Lambdas = limited_inner_internal::LimitedInnerProductDistanceLambdas<T>;
  Lambdas lambdas(SquaredL2Norm(query));
  if constexpr (std::is_floating_point_v<T>) {
    constexpr size_t kMinPrefetchAheadDims =
        (IsFloatingType<ResultElem>()) ? (32 / sizeof(T)) : (16 / sizeof(T));
    constexpr size_t kMaxPrefetchAheadDims = 2048 / sizeof(T);

    if (database->dimensionality() <= kMaxPrefetchAheadDims &&
        database->dimensionality() >= kMinPrefetchAheadDims) {
      return DenseAccumulatingDistanceMeasureOneToManyInternal<
          T, DatasetView, Lambdas, ResultElem, true>(query, database, lambdas,
                                                     result, callback, nullptr);
    } else {
      return DenseAccumulatingDistanceMeasureOneToManyInternal<
          T, DatasetView, Lambdas, ResultElem, false>(
          query, database, lambdas, result, callback, nullptr);
    }
  } else {
    return one_to_many_low_level::
        DenseAccumulatingDistanceMeasureOneToManyNoFma(query, database, lambdas,
                                                       result, callback, pool);
  }
}

template <typename T, typename ResultElem>
void DenseLimitedInnerProductDistanceOneToMany(const DatapointPtr<T>& query,
                                               const DenseDataset<T>& database,
                                               MutableSpan<ResultElem> result,
                                               ThreadPool* pool) {
  auto view = DefaultDenseDatasetView<T>(database);
  one_to_many_low_level::SetDistanceFunctor<ResultElem> set_distance_functor(
      result);
  DenseLimitedInnerProductDistanceOneToMany(query, &view, result,
                                            &set_distance_functor, pool);
}

#ifdef __SSE3__

template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
enable_if_t<IsIntegerType<T>() && sizeof(T) == 4 &&
                one_to_many_low_level::kX64Simd,
            void>
DenseGeneralHammingDistanceOneToMany(const DatapointPtr<T>& query,
                                     const DatasetView* __restrict__ database,
                                     MutableSpan<ResultElem> result,
                                     CallbackFunctor* __restrict__ callback,
                                     ThreadPool* pool) {
  constexpr size_t kMinPrefetchAheadDims =
      (IsFloatingType<ResultElem>()) ? 8 : 4;
  constexpr size_t kMaxPrefetchAheadDims = 512;
  if (!pool && database->dimensionality() <= kMaxPrefetchAheadDims &&
      database->dimensionality() >= kMinPrefetchAheadDims) {
    return one_to_many_low_level::
        DenseGeneralHammingDistanceMeasureOneToManyInternal<T, DatasetView,
                                                            ResultElem, true>(
            query, database, result, callback, nullptr);
  } else {
    return one_to_many_low_level::
        DenseGeneralHammingDistanceMeasureOneToManyInternal<T, DatasetView,
                                                            ResultElem, false>(
            query, database, result, callback, pool);
  }
}

#endif

template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
enable_if_t<!IsIntegerType<T>() || sizeof(T) != 4 ||
                !one_to_many_low_level::kX64Simd,
            void>
DenseGeneralHammingDistanceOneToMany(const DatapointPtr<T>& query,
                                     const DatasetView* __restrict__ database,
                                     MutableSpan<ResultElem> result,
                                     CallbackFunctor* __restrict__ callback,
                                     ThreadPool* pool) {
  GeneralHammingDistance dist;
  const size_t dim = database->dimensionality();
  ParallelFor<1>(Seq(result.size()), pool, [&](size_t i) {
    const DatapointPtr<T> i0 = MakeDatapointPtr(
        database->GetPtr(one_to_many_low_level::GetDatapointIndex(result, i)),
        dim);
    callback->invoke(i, dist.GetDistanceDense(query, i0));
  });
}

template <typename T, typename ResultElem>
void DenseGeneralHammingDistanceOneToMany(const DatapointPtr<T>& query,
                                          const DenseDataset<T>& database,
                                          MutableSpan<ResultElem> result,
                                          ThreadPool* pool) {
  auto view = DefaultDenseDatasetView<T>(database);
  one_to_many_low_level::SetDistanceFunctor<ResultElem> set_distance_functor(
      result);
  DenseGeneralHammingDistanceOneToMany(query, &view, result,
                                       &set_distance_functor, pool);
}

template <typename T, typename ResultElem>
SCANN_INLINE void DenseDistanceOneToMany(const DistanceMeasure& dist,
                                         const DatapointPtr<T>& query,
                                         const DenseDataset<T>& database,
                                         MutableSpan<ResultElem> result,
                                         ThreadPool* pool) {
  auto view = DefaultDenseDatasetView<T>(database);
  one_to_many_low_level::SetDistanceFunctor<ResultElem> set_distance_functor(
      result);
  return DenseDistanceOneToMany(dist, query, &view, result,
                                &set_distance_functor, pool);
}

template <typename T, typename U, typename ResultElem>
std::pair<DatapointIndex, U> DenseDistanceOneToManyTop1(
    const DistanceMeasure& dist, const DatapointPtr<T>& query,
    const DenseDataset<T>& database, MutableSpan<ResultElem> result,
    ThreadPool* pool) {
  auto view = DefaultDenseDatasetView<T>(database);
  one_to_many_low_level::SetTop1Functor<ResultElem, U> set_top1_functor;
  DenseDistanceOneToMany(dist, query, &view, result, &set_top1_functor, pool);
  return set_top1_functor.Top1Pair(result);
}

template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
void DenseDistanceOneToMany(const DistanceMeasure& dist,
                            const DatapointPtr<T>& query,
                            const DatasetView* __restrict__ database,
                            MutableSpan<ResultElem> result,
                            CallbackFunctor* __restrict__ callback,
                            ThreadPool* pool) {
  switch (dist.specially_optimized_distance_tag()) {
    case DistanceMeasure::L1:
      return DenseL1DistanceOneToMany<T>(query, database, result, callback,
                                         pool);
    case DistanceMeasure::SQUARED_L2:
      return DenseSquaredL2DistanceOneToMany<T>(query, database, result,
                                                callback, pool);
    case DistanceMeasure::L2:
      return DenseL2DistanceOneToMany<T>(query, database, result, callback,
                                         pool);
    case DistanceMeasure::DOT_PRODUCT:
      return DenseDotProductDistanceOneToMany<T>(query, database, result,
                                                 callback, pool);
    case DistanceMeasure::ABS_DOT_PRODUCT:
      return DenseAbsDotProductDistanceOneToMany<T>(query, database, result,
                                                    callback, pool);
    case DistanceMeasure::COSINE:
      return DenseCosineDistanceOneToMany<T>(query, database, result, callback,
                                             pool);
    case DistanceMeasure::LIMITED_INNER_PRODUCT:
      return DenseLimitedInnerProductDistanceOneToMany<T>(
          query, database, result, callback, pool);
    case DistanceMeasure::GENERAL_HAMMING:
      return DenseGeneralHammingDistanceOneToMany<T, ResultElem>(
          query, database, result, callback, pool);
    default:
      const size_t dim = database->dimensionality();
      ParallelFor<1>(Seq(result.size()), pool, [&](size_t i) {
        callback->invoke(
            i, dist.GetDistanceDense(
                   query,
                   MakeDatapointPtr(
                       database->GetPtr(
                           one_to_many_low_level::GetDatapointIndex(result, i)),
                       dim)));
      });
  }
}

template <typename T, typename ResultElem, typename DatasetView>
void DenseDistanceOneToMany(const DistanceMeasure& dist,
                            const DatapointPtr<T>& query,
                            const DatasetView* __restrict__ database,
                            MutableSpan<ResultElem> result, ThreadPool* pool) {
  one_to_many_low_level::SetDistanceFunctor<ResultElem> set_distance_functor(
      result);
  return DenseDistanceOneToMany<
      T, ResultElem, DatasetView,
      one_to_many_low_level::SetDistanceFunctor<ResultElem>>(
      dist, query, database, result, &set_distance_functor, pool);
}

template <typename T>
void DenseDistanceOneToManyBlockTransposed(
    DistanceMeasure::SpeciallyOptimizedDistanceTag dist_tag,
    const DatapointPtr<T>& query, ConstSpan<T> block_transposed_db,
    MutableSpan<float> result) {
  switch (dist_tag) {
    case DistanceMeasure::L1:
      return one_to_many_low_level::
          DenseAccumulatingDistanceMeasureOneToManyBlockTransposed<T>(
              query, block_transposed_db,
              one_to_many_low_level::L1DistanceLambdas<T>(), result);
    case DistanceMeasure::SQUARED_L2:
      return one_to_many_low_level::
          DenseAccumulatingDistanceMeasureOneToManyBlockTransposed<T>(
              query, block_transposed_db,
              one_to_many_low_level::SquaredL2DistanceLambdas<T>(), result);
    case DistanceMeasure::DOT_PRODUCT:
      return one_to_many_low_level::
          DenseAccumulatingDistanceMeasureOneToManyBlockTransposed<T>(
              query, block_transposed_db,
              one_to_many_low_level::DotProductDistanceLambdas<T>(), result);
    case DistanceMeasure::COSINE:
      return one_to_many_low_level::
          DenseAccumulatingDistanceMeasureOneToManyBlockTransposed<T>(
              query, block_transposed_db,
              one_to_many_low_level::CosineDistanceLambdas<T>(), result);
    default:
      LOG(FATAL) << "Unsupported distance measure for block-transposed "
                    "database.  Only DotProductDistance, SquaredL2Distance, "
                    "CosineDistance and L1Distance are supported.";
  }
}

#define SCANN_INSTANTIATE_ONE_TO_MANY_NO_AVX(                                  \
    EXTERN_KEYWORD, TYPE, DATASET_VIEW, LAMBDAS, RESULT_ELEM, SHOULD_PREFETCH, \
    CALLBACK_FUNCTOR)                                                          \
  EXTERN_KEYWORD template void                                                 \
  one_to_many_low_level::DenseAccumulatingDistanceMeasureOneToManyInternal<    \
      TYPE, DATASET_VIEW, LAMBDAS, RESULT_ELEM, SHOULD_PREFETCH,               \
      CALLBACK_FUNCTOR>(                                                       \
      const DatapointPtr<TYPE>& query,                                         \
      const DATASET_VIEW* __restrict__ database, const LAMBDAS& lambdas,       \
      MutableSpan<RESULT_ELEM> result,                                         \
      CALLBACK_FUNCTOR* __restrict__ callback, ThreadPool* pool);

#ifdef __x86_64__
#define SCANN_INSTANTIATE_ONE_TO_MANY(EXTERN_KEYWORD, TYPE, DATASET_VIEW,     \
                                      LAMBDAS, RESULT_ELEM, SHOULD_PREFETCH,  \
                                      CALLBACK_FUNCTOR)                       \
  SCANN_INSTANTIATE_ONE_TO_MANY_NO_AVX(EXTERN_KEYWORD, TYPE, DATASET_VIEW,    \
                                       LAMBDAS, RESULT_ELEM, SHOULD_PREFETCH, \
                                       CALLBACK_FUNCTOR)                      \
  EXTERN_KEYWORD template void one_to_many_low_level::                        \
      DenseAccumulatingDistanceMeasureOneToManyInternalAvx1<                  \
          TYPE, DATASET_VIEW, LAMBDAS, RESULT_ELEM, SHOULD_PREFETCH,          \
          CALLBACK_FUNCTOR>(                                                  \
          const DatapointPtr<TYPE>& query,                                    \
          const DATASET_VIEW* __restrict__ database, const LAMBDAS& lambdas,  \
          MutableSpan<RESULT_ELEM> result,                                    \
          CALLBACK_FUNCTOR* __restrict__ callback, ThreadPool* pool);         \
  EXTERN_KEYWORD template void one_to_many_low_level::                        \
      DenseAccumulatingDistanceMeasureOneToManyInternalAvx2<                  \
          TYPE, DATASET_VIEW, LAMBDAS, RESULT_ELEM, SHOULD_PREFETCH,          \
          CALLBACK_FUNCTOR>(                                                  \
          const DatapointPtr<TYPE>& query,                                    \
          const DATASET_VIEW* __restrict__ database, const LAMBDAS& lambdas,  \
          MutableSpan<RESULT_ELEM> result,                                    \
          CALLBACK_FUNCTOR* __restrict__ callback, ThreadPool* pool);

#else
#define SCANN_INSTANTIATE_ONE_TO_MANY SCANN_INSTANTIATE_ONE_TO_MANY_NO_AVX
#endif

#define SCANN_INSTANTIATE_COMMON_ONE_TO_MANY_FOR_DISTANCE(EXTERN_KEYWORD, \
                                                          TYPE, LAMBDAS)  \
  SCANN_INSTANTIATE_ONE_TO_MANY(                                          \
      EXTERN_KEYWORD, TYPE, DefaultDenseDatasetView<TYPE>, LAMBDAS, TYPE, \
      false, one_to_many_low_level::SetDistanceFunctor<TYPE>)             \
  SCANN_INSTANTIATE_ONE_TO_MANY(                                          \
      EXTERN_KEYWORD, TYPE, DefaultDenseDatasetView<TYPE>, LAMBDAS, TYPE, \
      true, one_to_many_low_level::SetDistanceFunctor<TYPE>)

#define SCANN_INSTANTIATE_COMMON_ONE_TO_MANY_FOR_DISTANCE_NO_AVX(         \
    EXTERN_KEYWORD, TYPE, LAMBDAS)                                        \
  SCANN_INSTANTIATE_ONE_TO_MANY_NO_AVX(                                   \
      EXTERN_KEYWORD, TYPE, DefaultDenseDatasetView<TYPE>, LAMBDAS, TYPE, \
      false, one_to_many_low_level::SetDistanceFunctor<TYPE>)             \
  SCANN_INSTANTIATE_ONE_TO_MANY_NO_AVX(                                   \
      EXTERN_KEYWORD, TYPE, DefaultDenseDatasetView<TYPE>, LAMBDAS, TYPE, \
      true, one_to_many_low_level::SetDistanceFunctor<TYPE>)

#define SCANN_INSTANTIATE_COMMON_FLOAT_ONE_TO_MANY(EXTERN_KEYWORD)            \
  SCANN_INSTANTIATE_COMMON_ONE_TO_MANY_FOR_DISTANCE(                          \
      EXTERN_KEYWORD, float,                                                  \
      one_to_many_low_level::DotProductDistanceLambdas<float>)                \
  SCANN_INSTANTIATE_COMMON_ONE_TO_MANY_FOR_DISTANCE(                          \
      EXTERN_KEYWORD, float,                                                  \
      one_to_many_low_level::SquaredL2DistanceLambdas<float>)                 \
  SCANN_INSTANTIATE_COMMON_ONE_TO_MANY_FOR_DISTANCE(                          \
      EXTERN_KEYWORD, float,                                                  \
      one_to_many_low_level::CosineDistanceLambdas<float>)                    \
  SCANN_INSTANTIATE_COMMON_ONE_TO_MANY_FOR_DISTANCE(                          \
      EXTERN_KEYWORD, float,                                                  \
      one_to_many_low_level::AbsDotProductDistanceLambdas<float>)             \
  SCANN_INSTANTIATE_COMMON_ONE_TO_MANY_FOR_DISTANCE(                          \
      EXTERN_KEYWORD, float, one_to_many_low_level::L2DistanceLambdas<float>) \
  SCANN_INSTANTIATE_COMMON_ONE_TO_MANY_FOR_DISTANCE_NO_AVX(                   \
      EXTERN_KEYWORD, float,                                                  \
      limited_inner_internal::LimitedInnerProductDistanceLambdas<float>)      \
  SCANN_INSTANTIATE_COMMON_ONE_TO_MANY_FOR_DISTANCE_NO_AVX(                   \
      EXTERN_KEYWORD, float, one_to_many_low_level::L1DistanceLambdas<float>)

#define SCANN_INSTANTIATE_COMMON_DOUBLE_ONE_TO_MANY(EXTERN_KEYWORD)       \
  SCANN_INSTANTIATE_COMMON_ONE_TO_MANY_FOR_DISTANCE_NO_AVX(               \
      EXTERN_KEYWORD, double,                                             \
      one_to_many_low_level::DotProductDistanceLambdas<double>)           \
  SCANN_INSTANTIATE_COMMON_ONE_TO_MANY_FOR_DISTANCE_NO_AVX(               \
      EXTERN_KEYWORD, double,                                             \
      one_to_many_low_level::SquaredL2DistanceLambdas<double>)            \
  SCANN_INSTANTIATE_COMMON_ONE_TO_MANY_FOR_DISTANCE_NO_AVX(               \
      EXTERN_KEYWORD, double,                                             \
      one_to_many_low_level::CosineDistanceLambdas<double>)               \
  SCANN_INSTANTIATE_COMMON_ONE_TO_MANY_FOR_DISTANCE_NO_AVX(               \
      EXTERN_KEYWORD, double,                                             \
      one_to_many_low_level::AbsDotProductDistanceLambdas<double>)        \
  SCANN_INSTANTIATE_COMMON_ONE_TO_MANY_FOR_DISTANCE_NO_AVX(               \
      EXTERN_KEYWORD, double,                                             \
      one_to_many_low_level::L2DistanceLambdas<double>)                   \
  SCANN_INSTANTIATE_COMMON_ONE_TO_MANY_FOR_DISTANCE_NO_AVX(               \
      EXTERN_KEYWORD, double,                                             \
      limited_inner_internal::LimitedInnerProductDistanceLambdas<double>) \
  SCANN_INSTANTIATE_COMMON_ONE_TO_MANY_FOR_DISTANCE_NO_AVX(               \
      EXTERN_KEYWORD, double,                                             \
      one_to_many_low_level::L1DistanceLambdas<double>)

SCANN_INSTANTIATE_COMMON_FLOAT_ONE_TO_MANY(extern)
SCANN_INSTANTIATE_COMMON_DOUBLE_ONE_TO_MANY(extern)

}  // namespace research_scann

#endif
