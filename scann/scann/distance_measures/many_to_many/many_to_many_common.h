// Copyright 2020 The Google Research Authors.
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

/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SCANN__DISTANCE_MEASURES_MANY_TO_MANY_MANY_TO_MANY_COMMON_H_
#define SCANN__DISTANCE_MEASURES_MANY_TO_MANY_MANY_TO_MANY_COMMON_H_

#include <array>

#include "absl/base/internal/spinlock.h"
#include "absl/synchronization/mutex.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measures.h"
#include "scann/utils/types.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
namespace scann_ops {

template <typename FloatT>
using ManyToManyResultsCallback =
    std::function<void(MutableSpan<FloatT> block_distances,
                       DatapointIndex first_dp_idx, DatapointIndex query_idx)>;

template <typename FloatT>
class ManyToManyTop1Callback {
 public:
  SCANN_DECLARE_COPYABLE_CLASS(ManyToManyTop1Callback);

  static constexpr int kNumSpinLocks = 128;
  static_assert((kNumSpinLocks & (kNumSpinLocks - 1)) == 0,
                "kNumSpinLocks has to be power of 2");

  explicit ManyToManyTop1Callback(
      pair<DatapointIndex,
           std::atomic<FloatT>>* __restrict__ top1_result_by_query)
      : top1_result_by_query_(top1_result_by_query),
        mutexes_(make_shared<
                 std::array<absl::base_internal::SpinLock, kNumSpinLocks>>()) {}

  SCANN_INLINE void operator()(MutableSpan<FloatT> block,
                               DatapointIndex first_dp_idx,
                               DatapointIndex query_idx) {
    constexpr auto kMemoryOrder = std::memory_order_relaxed;
    auto& top1 = top1_result_by_query_[query_idx];
    FloatT best_dist = top1.second.load(kMemoryOrder);

    bool update_needed = false;
    for (size_t j : Seq(block.size())) {
      if (block[j] < best_dist) update_needed = true;
    }
    if (ABSL_PREDICT_TRUE(!update_needed)) return;

    best_dist = block[0];
    DCHECK(!std::isnan(best_dist)) << "NAN at DP idx 0";
    size_t best_j = 0;
    for (size_t j : Seq(1, block.size())) {
      const FloatT dist = block[j];
      DCHECK(!std::isnan(dist)) << "NAN at DP idx " << first_dp_idx + j;
      if (dist < best_dist) {
        best_j = j;
      }
      best_dist = std::min(best_dist, dist);
    }

    DatapointIndex mutex_idx = query_idx & (kNumSpinLocks - 1);
    absl::base_internal::SpinLockHolder lock(&(*mutexes_)[mutex_idx]);
    if (best_dist < top1.second.load(kMemoryOrder)) {
      top1.first = first_dp_idx + best_j;
      top1.second.store(best_dist, kMemoryOrder);
    }
  }

 private:
  pair<DatapointIndex, std::atomic<FloatT>>* __restrict__ top1_result_by_query_;

  shared_ptr<std::array<absl::base_internal::SpinLock, kNumSpinLocks>> mutexes_;
};

template <typename FloatT, typename QueryNorm = FloatT,
          typename DbNorm = FloatT>
class ManyToManyTop1SquaredL2Callback {
 public:
  SCANN_DECLARE_COPYABLE_CLASS(ManyToManyTop1SquaredL2Callback);

  ManyToManyTop1SquaredL2Callback(
      pair<DatapointIndex,
           std::atomic<FloatT>>* __restrict__ top1_result_by_query,
      ConstSpan<QueryNorm> squared_query_norms,
      ConstSpan<DbNorm> squared_db_norms)
      : impl_(top1_result_by_query),
        squared_query_norms_(squared_query_norms),
        squared_db_norms_(squared_db_norms) {}

  SCANN_INLINE void operator()(MutableSpan<FloatT> block,
                               DatapointIndex first_dp_idx,
                               DatapointIndex query_idx) {
    {
      const auto squared_query_norm = squared_query_norms_[query_idx];
      FloatT* __restrict__ dists_ptr = block.data();
      const DbNorm* __restrict__ db_norm_ptr =
          squared_db_norms_.data() + first_dp_idx;
      for (size_t i : IndicesOf(block)) {
        dists_ptr[i] = squared_query_norm + db_norm_ptr[i] +
                       static_cast<FloatT>(2.0) * dists_ptr[i];
      }
    }
    return impl_(block, first_dp_idx, query_idx);
  }

 private:
  ManyToManyTop1Callback<FloatT> impl_;
  ConstSpan<QueryNorm> squared_query_norms_;
  ConstSpan<DbNorm> squared_db_norms_;
};

template <typename FloatT>
struct ManyToManyArgs {
  size_t dimensionality;

  const FloatT* queries;
  const FloatT* query_norms = nullptr;
  size_t num_queries;

  const FloatT* database;
  const FloatT* database_norms = nullptr;
  size_t num_datapoints;

  thread::ThreadPool* pool = nullptr;
};

#define SCANN_INSTANTIATE_MANY_TO_MANY_2(EXTERN_OR_NOTHING, METHOD_NAME, T, \
                                         CALLBACK)                          \
  EXTERN_OR_NOTHING template void METHOD_NAME(                              \
      const DistanceMeasure& dist, const DenseDataset<T>& queries,          \
      const DenseDataset<T>& database, thread::ThreadPool* pool,            \
      CALLBACK callback);

#define SCANN_INSTANTIATE_MANY_TO_MANY_1(EXTERN_OR_NOTHING, METHOD_NAME, T) \
  SCANN_INSTANTIATE_MANY_TO_MANY_2(EXTERN_OR_NOTHING, METHOD_NAME, T,       \
                                   ManyToManyResultsCallback<T>);           \
  SCANN_INSTANTIATE_MANY_TO_MANY_2(EXTERN_OR_NOTHING, METHOD_NAME, T,       \
                                   ManyToManyTop1Callback<T>);              \
  SCANN_INSTANTIATE_MANY_TO_MANY_2(                                         \
      EXTERN_OR_NOTHING, METHOD_NAME, T,                                    \
      ManyToManyTop1SquaredL2Callback<T SCANN_COMMA T SCANN_COMMA T>);

#define SCANN_INSTANTIATE_MANY_TO_MANY(EXTERN_OR_NOTHING, METHOD_NAME)     \
  SCANN_INSTANTIATE_MANY_TO_MANY_1(EXTERN_OR_NOTHING, METHOD_NAME, float); \
  SCANN_INSTANTIATE_MANY_TO_MANY_1(EXTERN_OR_NOTHING, METHOD_NAME, double);

#define SCANN_INSTANTIATE_MANY_TO_MANY_FP8_1(EXTERN_OR_NOTHING, METHOD_NAME, \
                                             CALLBACK)                       \
  EXTERN_OR_NOTHING template Status METHOD_NAME(                             \
      const DistanceMeasure& dist, const DenseDataset<float>& queries,       \
      const FP8SimdBlockTransposedDatabase& database,                        \
      thread::ThreadPool* pool, CALLBACK callback);

#define SCANN_INSTANTIATE_MANY_TO_MANY_FP8(EXTERN_OR_NOTHING, METHOD_NAME) \
  SCANN_INSTANTIATE_MANY_TO_MANY_FP8_1(EXTERN_OR_NOTHING, METHOD_NAME,     \
                                       ManyToManyResultsCallback<float>);  \
  SCANN_INSTANTIATE_MANY_TO_MANY_FP8_1(EXTERN_OR_NOTHING, METHOD_NAME,     \
                                       ManyToManyTop1Callback<float>);

}  // namespace scann_ops
}  // namespace tensorflow

#endif
