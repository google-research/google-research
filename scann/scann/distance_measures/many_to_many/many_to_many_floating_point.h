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



#ifndef SCANN_DISTANCE_MEASURES_MANY_TO_MANY_MANY_TO_MANY_FLOATING_POINT_H_
#define SCANN_DISTANCE_MEASURES_MANY_TO_MANY_MANY_TO_MANY_FLOATING_POINT_H_

#include <atomic>
#include <cstdint>
#include <limits>
#include <utility>

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measures.h"
#include "scann/distance_measures/many_to_many/many_to_many_common.h"
#include "scann/utils/common.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/types.h"

ABSL_DECLARE_FLAG(bool, enable_scann_brute_force_determinism);

namespace research_scann {

namespace mm_internal {

template <typename FloatT, typename CallbackT>
void DenseDistanceManyToManyImpl(const DistanceMeasure& dist,
                                 DefaultDenseDatasetView<FloatT> queries,
                                 const DenseDataset<FloatT>& database,
                                 ThreadPool* pool, CallbackT callback);

SCANN_INSTANTIATE_MANY_TO_MANY(extern, DenseDistanceManyToManyImpl);

template <typename FloatT, typename CallbackT>
SCANN_INLINE void DenseDistanceManyToManyImpl(
    const DistanceMeasure& dist, const DenseDataset<FloatT>& queries,
    const DenseDataset<FloatT>& database, ThreadPool* pool,
    CallbackT callback) {
  DenseDistanceManyToManyImpl(dist, DefaultDenseDatasetView<FloatT>(queries),
                              database, pool, callback);
}

}  // namespace mm_internal

template <typename FloatT>
void DenseDistanceManyToMany(const DistanceMeasure& dist,
                             const DenseDataset<FloatT>& queries,
                             const DenseDataset<FloatT>& database,
                             ManyToManyResultsCallback<FloatT> callback) {
  mm_internal::DenseDistanceManyToManyImpl(dist, queries, database, nullptr,
                                           std::move(callback));
}

template <typename FloatT>
void DenseDistanceManyToMany(const DistanceMeasure& dist,
                             const DenseDataset<FloatT>& queries,
                             const DenseDataset<FloatT>& database,
                             ThreadPool* pool,
                             ManyToManyResultsCallback<FloatT> callback) {
  mm_internal::DenseDistanceManyToManyImpl(dist, queries, database, pool,
                                           std::move(callback));
}
template <typename FloatT>
void DenseDistanceManyToMany(const DistanceMeasure& dist,
                             const DenseDataset<FloatT>& queries,
                             const DenseDataset<FloatT>& database,
                             EpsilonFilteringCallback<FloatT> callback) {
  mm_internal::DenseDistanceManyToManyImpl(dist, queries, database, nullptr,
                                           std::move(callback));
}
template <typename FloatT>
void DenseDistanceManyToMany(const DistanceMeasure& dist,
                             const DenseDataset<FloatT>& queries,
                             const DenseDataset<FloatT>& database,
                             ThreadPool* pool,
                             EpsilonFilteringCallback<FloatT> callback) {
  mm_internal::DenseDistanceManyToManyImpl(dist, queries, database, pool,
                                           std::move(callback));
}

template <typename FloatT>
vector<pair<DatapointIndex, FloatT>> DenseDistanceManyToManyTop1(
    const DistanceMeasure& dist, DefaultDenseDatasetView<FloatT> queries,
    const DenseDataset<FloatT>& database, ThreadPool* pool = nullptr) {
  static_assert(IsSameAny<FloatT, float, double>(),
                "DenseDistanceManyToMany only works with float/double.");
  vector<pair<DatapointIndex, FloatT>> result(
      queries.size(),
      std::make_pair(kInvalidDatapointIndex, numeric_limits<FloatT>::max()));
  ManyToManyTop1Callback<FloatT> top1_callback(MakeMutableSpan(result));
  EpsilonFilteringCallback<FloatT> eps_callback(top1_callback.epsilons(),
                                                top1_callback);
  mm_internal::DenseDistanceManyToManyImpl(dist, queries, database, pool,
                                           eps_callback);
  return result;
}

template <typename FloatT>
vector<pair<DatapointIndex, FloatT>> DenseDistanceManyToManyTop1(
    const DistanceMeasure& dist, const DenseDataset<FloatT>& queries,
    const DenseDataset<FloatT>& database, ThreadPool* pool = nullptr) {
  return DenseDistanceManyToManyTop1(
      dist, DefaultDenseDatasetView<FloatT>(queries), database, pool);
}

inline void DenseDistanceManyToManyTopK(
    const DistanceMeasure& dist, const DenseDataset<float>& queries,
    const DenseDataset<float>& database,
    MutableSpan<FastTopNeighbors<float>> topns, ThreadPool* pool = nullptr) {
  DCHECK_EQ(queries.size(), topns.size());
  ManyToManyTopKCallback topk_callback(topns, pool);
  EpsilonFilteringCallback<float> eps_callback(topk_callback.epsilons(),
                                               topk_callback);
  mm_internal::DenseDistanceManyToManyImpl(dist, queries, database, pool,
                                           eps_callback);
}

inline void DenseDistanceManyToManyTopKRemapped(
    const DistanceMeasure& dist, const DenseDataset<float>& queries,
    const DenseDataset<float>& database,
    MutableSpan<FastTopNeighbors<float>*> topns,
    ConstSpan<DatapointIndex> datapoint_index_mapping,
    ThreadPool* pool = nullptr) {
  DCHECK_EQ(queries.size(), topns.size());
  ManyToManyTopKCallbackRemapped topk_callback(topns, datapoint_index_mapping,
                                               pool);
  EpsilonFilteringCallback<float> eps_callback(topk_callback.epsilons(),
                                               topk_callback);
  mm_internal::DenseDistanceManyToManyImpl(dist, queries, database, pool,
                                           eps_callback);
}

}  // namespace research_scann

#endif
