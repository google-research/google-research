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

#ifndef SCANN_DISTANCE_MEASURES_MANY_TO_MANY_MANY_TO_MANY_TEMPLATES_H_
#define SCANN_DISTANCE_MEASURES_MANY_TO_MANY_MANY_TO_MANY_TEMPLATES_H_

#include <cstdint>

#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/many_to_many/fp8_transposed.h"
#include "scann/distance_measures/many_to_many/many_to_many.h"
#include "scann/distance_measures/one_to_many/one_to_many.h"
#include "scann/distance_measures/one_to_one/dot_product.h"
#include "scann/utils/intrinsics/horizontal_sum.h"
#include "scann/utils/intrinsics/simd.h"
#include "scann/utils/types.h"

#define SCANN_CALL_FUNCTION_BY_MM_BATCH_SIZE_5(batch_size, function, ...) \
  switch (batch_size) {                                                   \
    case 0:                                                               \
      break;                                                              \
    case 1:                                                               \
      function<1>(__VA_ARGS__);                                           \
      break;                                                              \
    case 2:                                                               \
      function<2>(__VA_ARGS__);                                           \
      break;                                                              \
    case 3:                                                               \
      function<3>(__VA_ARGS__);                                           \
      break;                                                              \
    case 4:                                                               \
      function<4>(__VA_ARGS__);                                           \
      break;                                                              \
    case 5:                                                               \
      function<5>(__VA_ARGS__);                                           \
      break;                                                              \
    default:                                                              \
      LOG(FATAL) << "Invalid Batch Size";                                 \
  }
#define SCANN_CALL_FUNCTION_BY_MM_BATCH_SIZE_6(batch_size, function, ...) \
  switch (batch_size) {                                                   \
    case 0:                                                               \
      break;                                                              \
    case 1:                                                               \
      function<1>(__VA_ARGS__);                                           \
      break;                                                              \
    case 2:                                                               \
      function<2>(__VA_ARGS__);                                           \
      break;                                                              \
    case 3:                                                               \
      function<3>(__VA_ARGS__);                                           \
      break;                                                              \
    case 4:                                                               \
      function<4>(__VA_ARGS__);                                           \
      break;                                                              \
    case 5:                                                               \
      function<5>(__VA_ARGS__);                                           \
      break;                                                              \
    case 6:                                                               \
      function<6>(__VA_ARGS__);                                           \
      break;                                                              \
    default:                                                              \
      LOG(FATAL) << "Invalid Batch Size";                                 \
  }

namespace research_scann {

#ifdef __x86_64__

namespace sse4 {
#define SCANN_SIMD_ATTRIBUTE SCANN_SSE4
#include "scann/distance_measures/many_to_many/many_to_many_impl.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace sse4

namespace avx1 {
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX1
#include "scann/distance_measures/many_to_many/many_to_many_impl.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace avx1

namespace avx2 {
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX2
#include "scann/distance_measures/many_to_many/many_to_many_impl.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace avx2

namespace avx512 {
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX512
#include "scann/distance_measures/many_to_many/many_to_many_impl.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace avx512

#else

namespace fallback {
#define SCANN_SIMD_ATTRIBUTE
#include "scann/distance_measures/many_to_many/many_to_many_impl.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace fallback

#endif

namespace mm_internal {

inline bool IsSupportedDistanceMeasure(const DistanceMeasure& dist) {
  switch (dist.specially_optimized_distance_tag()) {
    case DistanceMeasure::DOT_PRODUCT:
    case DistanceMeasure::SQUARED_L2:
    case DistanceMeasure::COSINE:
      return true;
    default:
      return false;
  }
}

template <typename FloatT, typename CallbackT>
void CallOneToManyDistance(const DistanceMeasure& dist,
                           const DenseDataset<FloatT>& queries,
                           const DenseDataset<FloatT>& database,
                           ThreadPool* pool, CallbackT callback) {
  auto one_query_results_storage = make_unique<FloatT[]>(database.size());
  MutableSpan<FloatT> one_query_results(one_query_results_storage.get(),
                                        database.size());
  for (size_t query_idx : IndicesOf(queries)) {
    DenseDistanceOneToMany(dist, queries[query_idx], database,
                           one_query_results, pool);
    callback(one_query_results, 0, query_idx);
  }
}

template <typename FloatT, typename CallbackT>
SCANN_INLINE void DenseDistanceManyToManyImpl2(
    const DistanceMeasure& dist, const DenseDataset<FloatT>& queries,
    const DenseDataset<FloatT>& database, ThreadPool* pool,
    CallbackT callback) {
  static_assert(IsSameAny<FloatT, float, double>(),
                "DenseDistanceManyToMany only works with float/double.");
  DCHECK_GE(queries.size(), 2);
  DCHECK(IsSupportedDistanceMeasure(dist));
  DCHECK_NE(dist.specially_optimized_distance_tag(), DistanceMeasure::COSINE);

#ifdef __x86_64__
  if (RuntimeSupportsAvx512()) {
    return avx512::DenseDistanceManyToManyImpl(dist, queries, database, pool,
                                               callback);
  } else if (RuntimeSupportsAvx2()) {
    return avx2::DenseDistanceManyToManyImpl(dist, queries, database, pool,
                                             std::move(callback));
  } else if (RuntimeSupportsAvx1()) {
    return avx1::DenseDistanceManyToManyImpl(dist, queries, database, pool,
                                             std::move(callback));
  } else if (RuntimeSupportsSse4()) {
    return sse4::DenseDistanceManyToManyImpl(dist, queries, database, pool,
                                             std::move(callback));
  } else {
    LOG(FATAL) << "Pre-SSE4 hardware is not supported on x64.";
  }

#else
  return fallback::DenseDistanceManyToManyImpl(dist, queries, database, pool,
                                               std::move(callback));
#endif
}

template <typename CallbackT>
SCANN_INLINE void DenseDistanceManyToManyFP8PretransposedImpl2(
    const DistanceMeasure& dist, const DenseDataset<float>& queries,
    const FP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    CallbackT callback) {
  DCHECK_GE(queries.size(), 1);
  DCHECK(IsSupportedDistanceMeasure(dist));
  DCHECK_NE(dist.specially_optimized_distance_tag(), DistanceMeasure::COSINE);

#ifdef __x86_64__
  if (RuntimeSupportsAvx512()) {
    return avx512::DenseManyToManyFP8PretransposedImpl(dist, queries, database,
                                                       pool, callback);
  } else if (RuntimeSupportsAvx2()) {
    return avx2::DenseManyToManyFP8PretransposedImpl(dist, queries, database,
                                                     pool, std::move(callback));
  } else if (RuntimeSupportsAvx1()) {
    return avx1::DenseManyToManyFP8PretransposedImpl(dist, queries, database,
                                                     pool, std::move(callback));
  } else if (RuntimeSupportsSse4()) {
    return sse4::DenseManyToManyFP8PretransposedImpl(dist, queries, database,
                                                     pool, std::move(callback));
  } else {
    LOG(FATAL) << "Pre-SSE4 hardware is not supported on x64.";
  }

#else
  return fallback::DenseManyToManyFP8PretransposedImpl(
      dist, queries, database, pool, std::move(callback));
#endif
}

template <typename FloatT, typename CallbackT>
void DenseDistanceManyToManyImpl(const DistanceMeasure& dist,
                                 const DenseDataset<FloatT>& queries,
                                 const DenseDataset<FloatT>& database,
                                 ThreadPool* pool, CallbackT callback) {
  static_assert(IsSameAny<FloatT, float, double>(),
                "DenseDistanceManyToMany only works with float/double.");

  if (database.empty() || queries.empty()) return;

  if (queries.size() == 1 || !IsSupportedDistanceMeasure(dist)) {
    return CallOneToManyDistance(dist, queries, database, pool,
                                 std::move(callback));
  }

  if (dist.specially_optimized_distance_tag() == DistanceMeasure::COSINE) {
    auto dot_to_cosine_wrapper =
        [&callback](MutableSpan<FloatT> block_distances,
                    DatapointIndex base_dp_idx, DatapointIndex query_idx) {
          for (auto& elem : block_distances) {
            elem += static_cast<FloatT>(1.0);
          }
          callback(block_distances, base_dp_idx, query_idx);
        };
    return DenseDistanceManyToManyImpl2<FloatT>(
        DotProductDistance(), queries, database, pool,
        std::move(dot_to_cosine_wrapper));
  } else {
    return DenseDistanceManyToManyImpl2<FloatT, CallbackT>(
        dist, queries, database, pool, std::move(callback));
  }
}

template <typename CallbackT>
Status DenseDistanceManyToManyFP8PretransposedImpl(
    const DistanceMeasure& dist, const DenseDataset<float>& queries,
    const FP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    CallbackT callback) {
  if (queries.empty()) return OkStatus();

  if (!IsSupportedDistanceMeasure(dist)) {
    return InvalidArgumentError(
        "DenseDistanceManyToManyFP8Pretransposed only supports dot product, "
        "cosine and squared L2 distance.");
  }

  if (dist.specially_optimized_distance_tag() == DistanceMeasure::COSINE) {
    auto dot_to_cosine_wrapper = [&callback](MutableSpan<float> block_distances,
                                             DatapointIndex base_dp_idx,
                                             DatapointIndex query_idx) {
      for (auto& elem : block_distances) {
        elem += static_cast<float>(1.0);
      }
      callback(block_distances, base_dp_idx, query_idx);
    };
    DenseDistanceManyToManyFP8PretransposedImpl2(
        DotProductDistance(), queries, database, pool,
        std::move(dot_to_cosine_wrapper));
  } else {
    DenseDistanceManyToManyFP8PretransposedImpl2<CallbackT>(
        dist, queries, database, pool, std::move(callback));
  }
  return OkStatus();
}

#undef SCANN_CALL_FUNCTION_BY_MM_BATCH_SIZE_5
#undef SCANN_CALL_FUNCTION_BY_MM_BATCH_SIZE_6

}  // namespace mm_internal
}  // namespace research_scann

#endif
