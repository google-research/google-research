// Copyright 2021 The Google Research Authors.
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



#ifndef SCANN_DISTANCE_MEASURES_ONE_TO_MANY_ONE_TO_MANY_H_
#define SCANN_DISTANCE_MEASURES_ONE_TO_MANY_ONE_TO_MANY_H_

#include <atomic>
#include <cmath>
#include <limits>
#include <type_traits>

#include "absl/base/optimization.h"
#include "absl/synchronization/mutex.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measures.h"
#include "scann/utils/common.h"
#include "scann/utils/intrinsics/simd.h"
#include "scann/utils/types.h"
#include "tensorflow/core/platform/prefetch.h"

namespace research_scann {

#ifdef __SSE3__

namespace one_to_many_low_level {
constexpr bool kSimd = true;
}

#else

namespace one_to_many_low_level {
constexpr bool kSimd = false;
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

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, const DenseDataset<int8_t>& database,
    MutableSpan<float> result);

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, const DenseDataset<int8_t>& database,
    MutableSpan<double> result);

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, const DenseDataset<int8_t>& database,
    MutableSpan<pair<DatapointIndex, float>> result);

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, const DenseDataset<int8_t>& database,
    MutableSpan<pair<uint64_t, float>> result);

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, const DenseDataset<int8_t>& database,
    MutableSpan<pair<DatapointIndex, double>> result);

void DenseDotProductDistanceOneToManyInt8Float(
    const DatapointPtr<float>& query, const DenseDataset<int8_t>& database,
    ConstSpan<DatapointIndex> indices, MutableSpan<float> result);

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
enable_if_t<
    IsIntegerType<T>() && sizeof(T) == 4 && one_to_many_low_level::kSimd, void>
DenseGeneralHammingDistanceOneToMany(const DatapointPtr<T>& query,
                                     const DatasetView* __restrict__ database,
                                     MutableSpan<ResultElem> result,
                                     CallbackFunctor* __restrict__ callback,
                                     ThreadPool* pool);
template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
enable_if_t<!IsIntegerType<T>() || sizeof(T) != 4 ||
                !one_to_many_low_level::kSimd,
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

template <typename ValueT>
inline size_t GetDatapointIndex(ValueT* result, size_t index) {
  return index;
}
template <typename ValueT>
inline size_t GetDatapointIndex(MutableSpan<ValueT> result, size_t index) {
  return index;
}

template <typename IndexT, typename ValueT>
inline size_t GetDatapointIndex(pair<IndexT, ValueT>* result, size_t index) {
  return result[index].first;
}
template <typename IndexT, typename ValueT>
inline size_t GetDatapointIndex(MutableSpan<pair<IndexT, ValueT>> result,
                                size_t index) {
  return result[index].first;
}

template <typename ValueT>
SCANN_INLINE void SetDistance(ValueT* result, size_t index, float val) {
  result[index] = val;
}
template <typename ValueT1, typename ValueT2>
inline void SetDistance(MutableSpan<ValueT1> result, size_t index,
                        ValueT2 val) {
  result[index] = static_cast<ValueT1>(val);
}

template <typename IndexT, typename ValueT1, typename ValueT2>
inline void SetDistance(MutableSpan<pair<IndexT, ValueT1>> result, size_t index,
                        ValueT2 val) {
  result[index].second = static_cast<ValueT1>(val);
}
template <typename IndexT, typename ValueT1, typename ValueT2>
SCANN_INLINE void SetDistance(pair<IndexT, ValueT1>* result, size_t index,
                              ValueT2 val) {
  result[index].second = static_cast<ValueT1>(val);
}

template <typename ResultElem>
class SetDistanceFunctor {
 public:
  explicit SetDistanceFunctor(MutableSpan<ResultElem> result_span)
      : result_(result_span) {}

  template <typename ValueT>
  SCANN_INLINE void invoke(size_t index, ValueT val) {
    SetDistance(result_, index, val);
  }

  SCANN_INLINE void prefetch(size_t index) {}

 private:
  MutableSpan<ResultElem> result_;
};

template <typename ResultElem, typename ValueT>
class SetTop1Functor {
  static_assert(std::is_arithmetic<ValueT>(),
                "Value must be a arithmetic type");

 public:
  SCANN_INLINE void invoke(size_t index, ValueT val) {
    if (val > smallest_.load(std::memory_order_relaxed)) return;
    absl::MutexLock lock(&mutex_);
    if (!is_smaller(index, val)) return;
    smallest_.store(val, std::memory_order_relaxed);
    index_ = index;
  }

  SCANN_INLINE void prefetch(size_t index) {}

  std::pair<DatapointIndex, ValueT> Top1Pair(
      MutableSpan<ResultElem> result_span) {
    if (result_span.empty()) {
      return std::make_pair(kInvalidDatapointIndex,
                            std::numeric_limits<ValueT>::max());
    }
    return std::make_pair(GetDatapointIndex(result_span, index_),
                          smallest_.load(std::memory_order_relaxed));
  }

 private:
  SCANN_INLINE bool is_smaller(size_t index, ValueT val) {
    ValueT smallest = smallest_.load(std::memory_order_relaxed);
    const bool is_eq_or_nan =
        smallest == val ||
        (IsFloatingType<ValueT>() && std::isunordered(smallest, val));
    if (ABSL_PREDICT_FALSE(is_eq_or_nan)) {
      return index < index_;
    }
    return val < smallest;
  }

  mutable absl::Mutex mutex_;
  std::atomic<ValueT> smallest_{std::numeric_limits<ValueT>::max()};
  DatapointIndex index_ = kInvalidDatapointIndex;
};

template <typename T, typename DatasetView, typename Lambdas,
          typename ResultElem, typename CallbackFunctor>
enable_if_t<IsIntegerType<T>() || !kSimd, void>
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
enable_if_t<IsIntegerType<T>() || !kSimd, void>
DenseAccumulatingDistanceMeasureOneToManyNoFma(
    const DatapointPtr<T>& query, const DatasetView* __restrict__ database,
    const Lambdas& lambdas, MutableSpan<ResultElem> result,
    CallbackFunctor* __restrict__ callback, ThreadPool* pool) {
  return DenseAccumulatingDistanceMeasureOneToMany<T, DatasetView, Lambdas,
                                                   ResultElem>(
      query, database, lambdas, result, callback, pool);
}

#ifdef __SSE3__

template <typename T, typename DatasetView, typename Lambdas,
          typename ResultElem, bool kShouldPrefetch, typename CallbackFunctor>
enable_if_t<std::is_same<T, float>::value, void> SCANN_OUTLINE
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

  auto sum4 = [](__m128 x) -> float {
    x = _mm_add_ps(x, _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(x), 8)));
    return x[0] + x[1];
  };

  ParallelFor<8>(
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

        __m128 a0 = _mm_setzero_ps();
        __m128 a1 = _mm_setzero_ps();
        __m128 a2 = _mm_setzero_ps();
        size_t j = 0;

        for (; j + 4 <= dims; j += 4) {
          __m128 q = _mm_loadu_ps(query.values() + j);
          __m128 v0 = _mm_loadu_ps(f0 + j);
          __m128 v1 = _mm_loadu_ps(f1 + j);
          __m128 v2 = _mm_loadu_ps(f2 + j);

          if (kShouldPrefetch) {
            ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(
                p0 + j);
            ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(
                p1 + j);
            ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(
                p2 + j);
          }

          a0 = _mm_add_ps(a0, lambdas_vec[0].GetTerm(q, v0));
          a1 = _mm_add_ps(a1, lambdas_vec[1].GetTerm(q, v1));
          a2 = _mm_add_ps(a2, lambdas_vec[2].GetTerm(q, v2));
        }

        if (j + 2 <= dims) {
          __m128 q = _mm_setzero_ps();
          __m128 v0 = _mm_setzero_ps();
          __m128 v1 = _mm_setzero_ps();
          __m128 v2 = _mm_setzero_ps();
          q = _mm_loadh_pi(q,
                           reinterpret_cast<const __m64*>(query.values() + j));
          v0 = _mm_loadh_pi(v0, reinterpret_cast<const __m64*>(f0 + j));
          v1 = _mm_loadh_pi(v1, reinterpret_cast<const __m64*>(f1 + j));
          v2 = _mm_loadh_pi(v2, reinterpret_cast<const __m64*>(f2 + j));
          a0 = _mm_add_ps(a0, lambdas_vec[0].GetTerm(q, v0));
          a1 = _mm_add_ps(a1, lambdas_vec[1].GetTerm(q, v1));
          a2 = _mm_add_ps(a2, lambdas_vec[2].GetTerm(q, v2));
          j += 2;
        }

        float result0 = sum4(a0);
        float result1 = sum4(a1);
        float result2 = sum4(a2);

        if (j < dims) {
          DCHECK_EQ(j + 1, dims);
          result0 += lambdas_vec[0].GetTerm(query.values()[j], f0[j]);
          result1 += lambdas_vec[1].GetTerm(query.values()[j], f1[j]);
          result2 += lambdas_vec[2].GetTerm(query.values()[j], f2[j]);
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

  ParallelFor<8>(Seq(num_outer_iters), pool, [&](size_t i) SCANN_AVX1 {
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

      if (kShouldPrefetch) {
        ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(p0 +
                                                                           j);
        ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(p1 +
                                                                           j);
        ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(p2 +
                                                                           j);
      }

      a0_256 = _mm256_add_ps(a0_256, lambdas_vec[0].GetTerm(q, v0));
      a1_256 = _mm256_add_ps(a1_256, lambdas_vec[1].GetTerm(q, v1));
      a2_256 = _mm256_add_ps(a2_256, lambdas_vec[2].GetTerm(q, v2));
    }

    __m128 a0 = SumTopBottomAvx(a0_256);
    __m128 a1 = SumTopBottomAvx(a1_256);
    __m128 a2 = SumTopBottomAvx(a2_256);

    if (j + 4 <= dims) {
      __m128 q = _mm_loadu_ps(query.values() + j);
      __m128 v0 = _mm_loadu_ps(f0 + j);
      __m128 v1 = _mm_loadu_ps(f1 + j);
      __m128 v2 = _mm_loadu_ps(f2 + j);

      if (kShouldPrefetch) {
        ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(p0 +
                                                                           j);
        ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(p1 +
                                                                           j);
        ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(p2 +
                                                                           j);
      }

      a0 = _mm_add_ps(a0, lambdas_vec[0].GetTerm(q, v0));
      a1 = _mm_add_ps(a1, lambdas_vec[1].GetTerm(q, v1));
      a2 = _mm_add_ps(a2, lambdas_vec[2].GetTerm(q, v2));
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
      a0 = _mm_add_ps(a0, lambdas_vec[0].GetTerm(q, v0));
      a1 = _mm_add_ps(a1, lambdas_vec[1].GetTerm(q, v1));
      a2 = _mm_add_ps(a2, lambdas_vec[2].GetTerm(q, v2));
      j += 2;
    }

    float result0 = sum4(a0);
    float result1 = sum4(a1);
    float result2 = sum4(a2);

    if (j < dims) {
      DCHECK_EQ(j + 1, dims);
      result0 += lambdas_vec[0].GetTerm(query.values()[j], f0[j]);
      result1 += lambdas_vec[1].GetTerm(query.values()[j], f1[j]);
      result2 += lambdas_vec[2].GetTerm(query.values()[j], f2[j]);
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

  ParallelFor<8>(Seq(num_outer_iters), pool, [&](size_t i) SCANN_AVX2 {
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

      if (kShouldPrefetch) {
        ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(p0 +
                                                                           j);
        ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(p1 +
                                                                           j);
        ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(p2 +
                                                                           j);
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

      if (kShouldPrefetch) {
        ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(p0 +
                                                                           j);
        ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(p1 +
                                                                           j);
        ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(p2 +
                                                                           j);
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
      result0 += lambdas_vec[0].GetTerm(query.values()[j], f0[j]);
      result1 += lambdas_vec[1].GetTerm(query.values()[j], f1[j]);
      result2 += lambdas_vec[2].GetTerm(query.values()[j], f2[j]);
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

  if (dims < 8 || !RuntimeSupportsAvx1()) {
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
}

template <typename T, typename DatasetView, typename Lambdas,
          typename ResultElem, typename CallbackFunctor>
enable_if_t<std::is_same<T, float>::value, void>
DenseAccumulatingDistanceMeasureOneToMany(
    const DatapointPtr<T>& query, const DatasetView* __restrict__ database,
    const Lambdas& lambdas, MutableSpan<ResultElem> result,
    CallbackFunctor* __restrict__ callback, ThreadPool* pool) {
  constexpr size_t kMinPrefetchAheadDims =
      (IsFloatingType<ResultElem>()) ? 8 : 4;
  constexpr size_t kMaxPrefetchAheadDims = 512;
  const DimensionIndex dims = query.nonzero_entries();

  if (dims < 8 || !RuntimeSupportsAvx2()) {
    return DenseAccumulatingDistanceMeasureOneToManyNoFma<T, DatasetView,
                                                          Lambdas, ResultElem>(
        query, database, lambdas, result, callback, pool);
  }

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
}

template <typename T, typename DatasetView, typename Lambdas,
          typename ResultElem, bool kShouldPrefetch, typename CallbackFunctor>
enable_if_t<std::is_same<T, double>::value, void>
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

  ParallelFor<8>(
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

        __m128d a0 = _mm_setzero_pd();
        __m128d a1 = _mm_setzero_pd();
        __m128d a2 = _mm_setzero_pd();
        size_t j = 0;
        for (; j + 2 <= dims; j += 2) {
          __m128d q = _mm_loadu_pd(query.values() + j);
          __m128d v0 = _mm_loadu_pd(f0 + j);
          __m128d v1 = _mm_loadu_pd(f1 + j);
          __m128d v2 = _mm_loadu_pd(f2 + j);

          if (kShouldPrefetch) {
            ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(
                p0 + j);
            ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(
                p1 + j);
            ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(
                p2 + j);
          }

          a0 = _mm_add_pd(a0, lambdas_vec[0].GetTerm(q, v0));
          a1 = _mm_add_pd(a1, lambdas_vec[1].GetTerm(q, v1));
          a2 = _mm_add_pd(a2, lambdas_vec[2].GetTerm(q, v2));
        }

        double result0 = a0[0] + a0[1];
        double result1 = a1[0] + a1[1];
        double result2 = a2[0] + a2[1];

        if (j < dims) {
          DCHECK_EQ(j + 1, dims);
          result0 += lambdas_vec[0].GetTerm(query.values()[j], f0[j]);
          result1 += lambdas_vec[1].GetTerm(query.values()[j], f1[j]);
          result2 += lambdas_vec[2].GetTerm(query.values()[j], f2[j]);
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
enable_if_t<std::is_same<T, double>::value, void>
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

  ParallelFor<8>(
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

          if (kShouldPrefetch) {
            ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(
                p0 + j);
            ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(
                p1 + j);
            ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(
                p2 + j);
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

template <typename T>
class DotProductDistanceLambdas {
 public:
#ifdef __SSE3__
  static __m128 GetTerm(__m128 a, __m128 b) { return -_mm_mul_ps(a, b); }

  static SCANN_AVX1_INLINE __m256 GetTerm(__m256 a, __m256 b) {
    return -_mm256_mul_ps(a, b);
  }

  static SCANN_AVX2_INLINE __m256 FmaTerm(__m256 acc, __m256 a, __m256 b) {
    return _mm256_fnmadd_ps(a, b, acc);
  }

  static SCANN_AVX2_INLINE __m128 FmaTerm(__m128 acc, __m128 a, __m128 b) {
    return _mm_fnmadd_ps(a, b, acc);
  }

  static __m128d GetTerm(__m128d a, __m128d b) { return -_mm_mul_pd(a, b); }
#endif

  static float GetTerm(float a, float b) { return -a * b; }
  static double GetTerm(double a, double b) { return -a * b; }
  static float Postprocess(float val) { return val; }
  static double Postprocess(double val) { return val; }

  double VectorVector(const DatapointPtr<T>& a,
                      const DatapointPtr<T>& b) const {
    return dist_.GetDistanceDense(a, b);
  }

 private:
  DotProductDistance dist_;
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

template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
void DenseAbsDotProductDistanceOneToMany(
    const DatapointPtr<T>& query, const DatasetView* __restrict__ database,
    MutableSpan<ResultElem> result, CallbackFunctor* __restrict__ callback,
    ThreadPool* pool) {
  class DotProductDistanceLambdas {
   public:
#ifdef __SSE3__
    static __m128 GetTerm(__m128 a, __m128 b) { return _mm_mul_ps(a, b); }

    static SCANN_AVX1_INLINE __m256 GetTerm(__m256 a, __m256 b) {
      return _mm256_mul_ps(a, b);
    }

    static SCANN_AVX2_INLINE __m256 FmaTerm(__m256 acc, __m256 a, __m256 b) {
      return _mm256_fmadd_ps(a, b, acc);
    }

    static SCANN_AVX2_INLINE __m128 FmaTerm(__m128 acc, __m128 a, __m128 b) {
      return _mm_fmadd_ps(a, b, acc);
    }

    static __m128d GetTerm(__m128d a, __m128d b) { return _mm_mul_pd(a, b); }
#endif

    static float GetTerm(float a, float b) { return a * b; }
    static double GetTerm(double a, double b) { return a * b; }
    static float Postprocess(float val) { return -std::abs(val); }
    static double Postprocess(double val) { return -std::abs(val); }

    double VectorVector(const DatapointPtr<T>& a,
                        const DatapointPtr<T>& b) const {
      return dist_.GetDistanceDense(a, b);
    }

   private:
    AbsDotProductDistance dist_;
  };

  DotProductDistanceLambdas lambdas;
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

template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
void DenseCosineDistanceOneToMany(const DatapointPtr<T>& query,
                                  const DatasetView* __restrict__ database,
                                  MutableSpan<ResultElem> result,
                                  CallbackFunctor* __restrict__ callback,
                                  ThreadPool* pool) {
  class CosineDistanceLambdas {
   public:
#ifdef __SSE3__
    static __m128 GetTerm(__m128 a, __m128 b) { return _mm_mul_ps(a, b); }

    static SCANN_AVX1_INLINE __m256 GetTerm(__m256 a, __m256 b) {
      return _mm256_mul_ps(a, b);
    }

    static SCANN_AVX2_INLINE __m256 FmaTerm(__m256 acc, __m256 a, __m256 b) {
      return _mm256_fmadd_ps(a, b, acc);
    }

    static SCANN_AVX2_INLINE __m128 FmaTerm(__m128 acc, __m128 a, __m128 b) {
      return _mm_fmadd_ps(a, b, acc);
    }

    static __m128d GetTerm(__m128d a, __m128d b) { return _mm_mul_pd(a, b); }
#endif

    static float GetTerm(float a, float b) { return a * b; }
    static double GetTerm(double a, double b) { return a * b; }
    static float Postprocess(float val) { return 1.0f - val; }
    static double Postprocess(double val) { return 1.0 - val; }

    double VectorVector(const DatapointPtr<T>& a,
                        const DatapointPtr<T>& b) const {
      return dist_.GetDistanceDense(a, b);
    }

   private:
    CosineDistance dist_;
  };

  CosineDistanceLambdas lambdas;
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
  class SquaredL2DistanceLambdas {
   public:
#ifdef __SSE3__
    static __m128 GetTerm(__m128 a, __m128 b) {
      __m128 tmp = _mm_sub_ps(a, b);
      return _mm_mul_ps(tmp, tmp);
    }

    SCANN_AVX1_INLINE static __m256 GetTerm(__m256 a, __m256 b) {
      __m256 tmp = _mm256_sub_ps(a, b);
      return _mm256_mul_ps(tmp, tmp);
    }

    static SCANN_AVX2_INLINE __m256 FmaTerm(__m256 acc, __m256 a, __m256 b) {
      __m256 tmp = _mm256_sub_ps(a, b);
      return _mm256_fmadd_ps(tmp, tmp, acc);
    }

    static SCANN_AVX2_INLINE __m128 FmaTerm(__m128 acc, __m128 a, __m128 b) {
      __m128 tmp = _mm_sub_ps(a, b);
      return _mm_fmadd_ps(tmp, tmp, acc);
    }

    static __m128d GetTerm(__m128d a, __m128d b) {
      __m128d tmp = _mm_sub_pd(a, b);
      return _mm_mul_pd(tmp, tmp);
    }
#endif

    static float GetTerm(float a, float b) {
      const float tmp = a - b;
      return tmp * tmp;
    }

    static double GetTerm(double a, double b) {
      const double tmp = a - b;
      return tmp * tmp;
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

  SquaredL2DistanceLambdas lambdas;
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

template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
void DenseL2DistanceOneToMany(const DatapointPtr<T>& query,
                              const DatasetView* __restrict__ database,
                              MutableSpan<ResultElem> result,
                              CallbackFunctor* __restrict__ callback,
                              ThreadPool* pool) {
  class L2DistanceLambdas {
   public:
#ifdef __SSE3__
    static __m128 GetTerm(__m128 a, __m128 b) {
      __m128 tmp = _mm_sub_ps(a, b);
      return _mm_mul_ps(tmp, tmp);
    }

    SCANN_AVX1_INLINE static __m256 GetTerm(__m256 a, __m256 b) {
      __m256 tmp = _mm256_sub_ps(a, b);
      return _mm256_mul_ps(tmp, tmp);
    }

    static SCANN_AVX2_INLINE __m256 FmaTerm(__m256 acc, __m256 a, __m256 b) {
      __m256 tmp = _mm256_sub_ps(a, b);
      return _mm256_fmadd_ps(tmp, tmp, acc);
    }

    static SCANN_AVX2_INLINE __m128 FmaTerm(__m128 acc, __m128 a, __m128 b) {
      __m128 tmp = _mm_sub_ps(a, b);
      return _mm_fmadd_ps(tmp, tmp, acc);
    }

    static __m128d GetTerm(__m128d a, __m128d b) {
      __m128d tmp = _mm_sub_pd(a, b);
      return _mm_mul_pd(tmp, tmp);
    }
#endif

    static float GetTerm(float a, float b) {
      const float tmp = a - b;
      return tmp * tmp;
    }

    static double GetTerm(double a, double b) {
      const double tmp = a - b;
      return tmp * tmp;
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

  L2DistanceLambdas lambdas;
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

template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
void DenseL1DistanceOneToMany(const DatapointPtr<T>& query,
                              const DatasetView* __restrict__ database,
                              MutableSpan<ResultElem> result,
                              CallbackFunctor* __restrict__ callback,
                              ThreadPool* pool) {
#ifdef __SSE3__

  static constexpr int32_t kAbsMaskScalar = 0x7FFFFFFF;
  static const __m128 kAbsMaskVectorFloat = _mm_castsi128_ps(_mm_set_epi32(
      kAbsMaskScalar, kAbsMaskScalar, kAbsMaskScalar, kAbsMaskScalar));

  static constexpr int32_t kAllSet = 0xFFFFFFFF;
  static const __m128d kAbsMaskVectorDouble = _mm_castsi128_pd(
      _mm_set_epi32(kAbsMaskScalar, kAllSet, kAbsMaskScalar, kAllSet));
#endif

  class L1DistanceLambdas {
   public:
#ifdef __SSE3__
    static __m128 GetTerm(__m128 a, __m128 b) {
      return AbsPs(_mm_sub_ps(a, b));
    }

    SCANN_AVX1_INLINE static __m256 GetTerm(__m256 a, __m256 b) {
      return AbsPs(_mm256_sub_ps(a, b));
    }

    static __m128d GetTerm(__m128d a, __m128d b) {
      return AbsPd(_mm_sub_pd(a, b));
    }
#endif

    static float GetTerm(float a, float b) { return std::abs(a - b); }

    static double GetTerm(double a, double b) { return std::abs(a - b); }

    static float Postprocess(float val) { return val; }
    static double Postprocess(double val) { return val; }

    double VectorVector(const DatapointPtr<T>& a,
                        const DatapointPtr<T>& b) const {
      return dist_.GetDistanceDense(a, b);
    }

   private:
    L1Distance dist_;

#ifdef __SSE3__
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

  L1DistanceLambdas lambdas;
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

template <typename T, typename ResultElem, typename DatasetView,
          typename CallbackFunctor>
void DenseLimitedInnerProductDistanceOneToMany(
    const DatapointPtr<T>& query, const DatasetView* __restrict__ database,
    MutableSpan<ResultElem> result, CallbackFunctor* __restrict__ callback,
    ThreadPool* pool) {
  class LimitedInnerProductDistanceLambdas {
   public:
    explicit LimitedInnerProductDistanceLambdas(double norm_query2)
        : norm_a2_(norm_query2) {}

#ifdef __SSE3__
    __m128 GetTerm(__m128 a, __m128 b) {
      norm_b2_ps_ = _mm_add_ps(norm_b2_ps_, _mm_mul_ps(b, b));
      return _mm_mul_ps(a, b);
    }

    SCANN_AVX1_INLINE __m256 GetTerm(__m256 a, __m256 b) {
      norm_b2_ps_ = _mm_add_ps(
          norm_b2_ps_,
          one_to_many_low_level::SumTopBottomAvx(_mm256_mul_ps(b, b)));
      return _mm256_mul_ps(a, b);
    }

    __m128d GetTerm(__m128d a, __m128d b) {
      norm_b2_pd_ = _mm_add_pd(norm_b2_pd_, _mm_mul_pd(b, b));
      return _mm_mul_pd(a, b);
    }
#endif

    float GetTerm(float a, float b) {
      norm_b2_s_ += b * b;
      return a * b;
    }

    double GetTerm(double a, double b) {
      norm_b2_d_ += b * b;
      return a * b;
    }

    float Postprocess(float val) {
#ifdef __SSE3__
      norm_b2_ps_ = _mm_hadd_ps(norm_b2_ps_, norm_b2_ps_);
      norm_b2_ps_ = _mm_hadd_ps(norm_b2_ps_, norm_b2_ps_);

      norm_b2_s_ += norm_b2_ps_[0];

      norm_b2_ps_ = _mm_setzero_ps();
#endif
      const float denom = std::sqrt(
          norm_a2_ * std::max(static_cast<float>(norm_a2_), norm_b2_s_));

      norm_b2_s_ = 0;
      if (denom == 0.0f) {
        return 0.0f;
      }
      return -val / denom;
    }

    double Postprocess(double val) {
#ifdef __SSE3__
      norm_b2_pd_ = _mm_hadd_pd(norm_b2_pd_, norm_b2_pd_);

      norm_b2_d_ += norm_b2_pd_[0];

      norm_b2_pd_ = _mm_setzero_pd();
#endif
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
    LimitedInnerProductDistance dist_;
    double norm_a2_;

#ifdef __SSE3__
    __m128 norm_b2_ps_ = _mm_setzero_ps();
    __m128d norm_b2_pd_ = _mm_setzero_pd();
#endif
    float norm_b2_s_ = 0;
    double norm_b2_d_ = 0;
  };

  LimitedInnerProductDistanceLambdas lambdas(SquaredL2Norm(query));
  return one_to_many_low_level::DenseAccumulatingDistanceMeasureOneToManyNoFma(
      query, database, lambdas, result, callback, pool);
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
enable_if_t<
    IsIntegerType<T>() && sizeof(T) == 4 && one_to_many_low_level::kSimd, void>
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
                !one_to_many_low_level::kSimd,
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

}  // namespace research_scann

#endif
