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



#ifndef SCANN_DISTANCE_MEASURES_MANY_TO_MANY_MANY_TO_MANY_H_
#define SCANN_DISTANCE_MEASURES_MANY_TO_MANY_MANY_TO_MANY_H_

#include <atomic>
#include <cstdint>
#include <limits>
#include <utility>

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measures.h"
#include "scann/distance_measures/many_to_many/fp8_transposed.h"
#include "scann/distance_measures/many_to_many/many_to_many_common.h"
#include "scann/distance_measures/many_to_many/many_to_many_floating_point.h"
#include "scann/distance_measures/many_to_many/sfp8_transposed.h"
#include "scann/utils/common.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/types.h"

ABSL_DECLARE_FLAG(bool, enable_scann_brute_force_determinism);

namespace research_scann {

namespace mm_internal {

template <typename CallbackT>
Status DenseDistanceManyToManyFP8PretransposedImpl(
    const DistanceMeasure& dist, const DenseDataset<float>& queries,
    const FP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    CallbackT callback);

SCANN_INSTANTIATE_MANY_TO_MANY_FP8(extern,
                                   DenseDistanceManyToManyFP8PretransposedImpl);

template <typename CallbackT>
Status DenseDistanceManyToManySFP8PretransposedImpl(
    const DistanceMeasure& dist, const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    CallbackT callback);

extern template Status DenseDistanceManyToManySFP8PretransposedImpl(
    const DistanceMeasure& dist, const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    ManyToManyResultsCallback<float> callback);
extern template Status DenseDistanceManyToManySFP8PretransposedImpl(
    const DistanceMeasure& dist, const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    EpsilonFilteringCallback<float> callback);

template <typename DatabaseT, typename CallbackT>
void DenseManyToManyOrthogonalityAmplifiedImpl(
    const DenseDataset<float>& queries,
    const DenseDataset<float>& normalized_residuals, float lambda,
    const DatabaseT& database, ThreadPool* pool, CallbackT callback);

extern template void DenseManyToManyOrthogonalityAmplifiedImpl(
    const DenseDataset<float>& queries,
    const DenseDataset<float>& normalized_residuals, float lambda,
    const FP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    EpsilonFilteringOffsetWrapper<float> callback);
extern template void DenseManyToManyOrthogonalityAmplifiedImpl(
    const DenseDataset<float>& queries,
    const DenseDataset<float>& normalized_residuals, float lambda,
    const FP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    ManyToManyResultsCallback<float> callback);
extern template void DenseManyToManyOrthogonalityAmplifiedImpl(
    const DenseDataset<float>& queries,
    const DenseDataset<float>& normalized_residuals, float lambda,
    const DenseDataset<float>& database, ThreadPool* pool,
    EpsilonFilteringCallback<float> callback);

}  // namespace mm_internal

inline void DenseManyToManyOrthogonalityAmplified(
    const DenseDataset<float>& queries,
    const DenseDataset<float>& normalized_residuals, float lambda,
    const FP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    EpsilonFilteringOffsetWrapper<float> callback) {
  mm_internal::DenseManyToManyOrthogonalityAmplifiedImpl(
      queries, normalized_residuals, lambda, database, pool, callback);
}
inline void DenseManyToManyOrthogonalityAmplified(
    const DenseDataset<float>& queries,
    const DenseDataset<float>& normalized_residuals, float lambda,
    const FP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    ManyToManyResultsCallback<float> callback) {
  mm_internal::DenseManyToManyOrthogonalityAmplifiedImpl(
      queries, normalized_residuals, lambda, database, pool, callback);
}
inline void DenseManyToManyOrthogonalityAmplified(
    const DenseDataset<float>& queries,
    const DenseDataset<float>& normalized_residuals, float lambda,
    const DenseDataset<float>& database, ThreadPool* pool,
    EpsilonFilteringCallback<float> callback) {
  mm_internal::DenseManyToManyOrthogonalityAmplifiedImpl<
      DenseDataset<float>, EpsilonFilteringCallback<float>>(
      queries, normalized_residuals, lambda, database, pool, callback);
}

inline Status DenseDistanceManyToManyFP8Pretransposed(
    const DistanceMeasure& dist, const DenseDataset<float>& queries,
    const FP8SimdBlockTransposedDatabase& database,
    ManyToManyResultsCallback<float> callback) {
  return mm_internal::DenseDistanceManyToManyFP8PretransposedImpl(
      dist, queries, database, nullptr, std::move(callback));
}
inline Status DenseDistanceManyToManyFP8Pretransposed(
    const DistanceMeasure& dist, const DenseDataset<float>& queries,
    const FP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    ManyToManyResultsCallback<float> callback) {
  return mm_internal::DenseDistanceManyToManyFP8PretransposedImpl(
      dist, queries, database, pool, std::move(callback));
}
inline Status DenseDistanceManyToManyFP8Pretransposed(
    const DistanceMeasure& dist, const DenseDataset<float>& queries,
    const FP8SimdBlockTransposedDatabase& database,
    EpsilonFilteringOffsetWrapper<float> callback) {
  return mm_internal::DenseDistanceManyToManyFP8PretransposedImpl(
      dist, queries, database, nullptr, std::move(callback));
}
inline Status DenseDistanceManyToManyFP8Pretransposed(
    const DistanceMeasure& dist, const DenseDataset<float>& queries,
    const FP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    EpsilonFilteringOffsetWrapper<float> callback) {
  return mm_internal::DenseDistanceManyToManyFP8PretransposedImpl(
      dist, queries, database, pool, std::move(callback));
}
inline Status DenseDistanceManyToManyFP8Pretransposed(
    const DistanceMeasure& dist, const DenseDataset<float>& queries,
    const FP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    EpsilonFilteringCallback<float> callback) {
  return mm_internal::DenseDistanceManyToManyFP8PretransposedImpl(
      dist, queries, database, pool, std::move(callback));
}
inline Status DenseDistanceManyToManyFP8PretransposedTopK(
    const DistanceMeasure& dist, const DenseDataset<float>& queries,
    const FP8SimdBlockTransposedDatabase& database,
    MutableSpan<FastTopNeighbors<float>> topns, ThreadPool* pool = nullptr) {
  SCANN_RET_CHECK_EQ(queries.size(), topns.size());
  ManyToManyTopKCallback topk_callback(topns, pool);
  EpsilonFilteringCallback<float> eps_callback(topk_callback.epsilons(),
                                               topk_callback);
  return DenseDistanceManyToManyFP8Pretransposed(dist, queries, database, pool,
                                                 std::move(eps_callback));
}
inline StatusOr<vector<pair<DatapointIndex, float>>>
DenseDistanceManyToManyFP8PretransposedTop1(
    const DistanceMeasure& dist, const DenseDataset<float>& queries,
    const FP8SimdBlockTransposedDatabase& database,
    ThreadPool* pool = nullptr) {
  vector<pair<DatapointIndex, float>> result(
      queries.size(),
      std::make_pair(kInvalidDatapointIndex, numeric_limits<float>::max()));
  ManyToManyTop1Callback<float> top1_callback(MakeMutableSpan(result), pool);
  EpsilonFilteringCallback<float> eps_callback(top1_callback.epsilons(),
                                               top1_callback);
  SCANN_RETURN_IF_ERROR(DenseDistanceManyToManyFP8Pretransposed(
      dist, queries, database, pool, std::move(eps_callback)));
  return result;
}

inline Status DenseDistanceManyToManySFP8Pretransposed(
    const DistanceMeasure& dist, const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& database,
    ManyToManyResultsCallback<float> callback) {
  return mm_internal::DenseDistanceManyToManySFP8PretransposedImpl(
      dist, queries, database, nullptr, std::move(callback));
}
inline Status DenseDistanceManyToManySFP8Pretransposed(
    const DistanceMeasure& dist, const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    ManyToManyResultsCallback<float> callback) {
  return mm_internal::DenseDistanceManyToManySFP8PretransposedImpl(
      dist, queries, database, pool, std::move(callback));
}
inline Status DenseDistanceManyToManySFP8Pretransposed(
    const DistanceMeasure& dist, const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    EpsilonFilteringCallback<float> callback) {
  return mm_internal::DenseDistanceManyToManySFP8PretransposedImpl(
      dist, queries, database, pool, std::move(callback));
}
inline Status DenseDistanceManyToManySFP8PretransposedTopK(
    const DistanceMeasure& dist, const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& database,
    MutableSpan<FastTopNeighbors<float>> topns, ThreadPool* pool = nullptr) {
  SCANN_RET_CHECK_EQ(queries.size(), topns.size());
  ManyToManyTopKCallback topk_callback(topns, pool);
  EpsilonFilteringCallback<float> eps_callback(topk_callback.epsilons(),
                                               topk_callback);
  return DenseDistanceManyToManySFP8Pretransposed(dist, queries, database, pool,
                                                  std::move(eps_callback));
}
inline StatusOr<vector<pair<DatapointIndex, float>>>
DenseDistanceManyToManySFP8PretransposedTop1(
    const DistanceMeasure& dist, const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& database,
    ThreadPool* pool = nullptr) {
  vector<pair<DatapointIndex, float>> result(
      queries.size(),
      std::make_pair(kInvalidDatapointIndex, numeric_limits<float>::max()));
  ManyToManyTop1Callback<float> top1_callback(MakeMutableSpan(result), pool);
  EpsilonFilteringCallback<float> eps_callback(top1_callback.epsilons(),
                                               top1_callback);
  SCANN_RETURN_IF_ERROR(DenseDistanceManyToManySFP8Pretransposed(
      dist, queries, database, pool, std::move(eps_callback)));
  return result;
}

}  // namespace research_scann

#endif
