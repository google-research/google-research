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

#ifndef SCANN_DISTANCE_MEASURES_MANY_TO_MANY_MANY_TO_MANY_COMMON_H_
#define SCANN_DISTANCE_MEASURES_MANY_TO_MANY_MANY_TO_MANY_COMMON_H_

#include <array>
#include <atomic>

#include "absl/base/internal/spinlock.h"
#include "absl/synchronization/mutex.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measures.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/utils/common.h"
#include "scann/utils/intrinsics/simd.h"
#include "scann/utils/types.h"

namespace research_scann {

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

#ifdef __x86_64__

  SCANN_AVX512_INLINE void InvokeOptimized(Avx512<float, 2> simd_dists,
                                           size_t first_dp_idx,
                                           size_t query_idx) {
    auto& top1 = top1_result_by_query_[query_idx];
    float best_dist = top1.second.load(std::memory_order_relaxed);

    auto cmp = (simd_dists < Avx512<float>::Broadcast(best_dist));
    if (ABSL_PREDICT_TRUE(_kortestz_mask16_u8(cmp[0], cmp[1]))) return;

    auto dists = simd_dists.Store();
    InvokeSlowPath(MakeMutableSpan(dists), first_dp_idx, query_idx);
  }

  SCANN_AVX1_INLINE void InvokeOptimized(Avx1<float, 2> simd_dists,
                                         size_t first_dp_idx,
                                         size_t query_idx) {
    auto& top1 = top1_result_by_query_[query_idx];
    float best_dist = top1.second.load(std::memory_order_relaxed);

    auto cmp = (simd_dists < Avx1<float>::Broadcast(best_dist));
    if (ABSL_PREDICT_TRUE((cmp[0] | cmp[1]).MaskFromHighBits() == 0)) return;

    auto dists = simd_dists.Store();
    InvokeSlowPath(MakeMutableSpan(dists), first_dp_idx, query_idx);
  }

  SCANN_SSE4_INLINE void InvokeOptimized(Sse4<float, 2> simd_dists,
                                         size_t first_dp_idx,
                                         size_t query_idx) {
    auto& top1 = top1_result_by_query_[query_idx];
    float best_dist = top1.second.load(std::memory_order_relaxed);

    auto cmp = (simd_dists < Sse4<float>::Broadcast(best_dist));
    if (ABSL_PREDICT_TRUE((cmp[0] | cmp[1]).MaskFromHighBits() == 0)) return;

    auto dists = simd_dists.Store();
    InvokeSlowPath(MakeMutableSpan(dists), first_dp_idx, query_idx);
  }

#endif

  SCANN_INLINE void InvokeOptimized(fallback::Simd<float, 2> dists,
                                    size_t first_dp_idx, size_t query_idx) {
    FloatT candidates[] = {dists[0].Unwrap(), dists[1].Unwrap()};
    Invoke(MakeMutableSpan(candidates, 2), first_dp_idx, query_idx);
  }

  SCANN_INLINE void operator()(MutableSpan<FloatT> block, size_t first_dp_idx,
                               size_t query_idx) {
    Invoke(block, first_dp_idx, query_idx);
  }

  SCANN_INLINE void Invoke(MutableSpan<FloatT> block, size_t first_dp_idx,
                           size_t query_idx) {
    auto& top1 = top1_result_by_query_[query_idx];
    FloatT best_dist = top1.second.load(std::memory_order_relaxed);

    bool update_needed = false;
    for (size_t j : Seq(block.size())) {
      if (block[j] < best_dist) update_needed = true;
    }
    if (ABSL_PREDICT_TRUE(!update_needed)) return;

    InvokeSlowPath(block, first_dp_idx, query_idx);
  }

 private:
  SCANN_OUTLINE void InvokeSlowPath(MutableSpan<FloatT> block,
                                    size_t first_dp_idx, size_t query_idx) {
    FloatT best_dist = block[0];
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

    size_t mutex_idx = query_idx & (kNumSpinLocks - 1);
    auto& top1 = top1_result_by_query_[query_idx];
    absl::base_internal::SpinLockHolder lock(&(*mutexes_)[mutex_idx]);
    if (best_dist < top1.second.load(std::memory_order_relaxed)) {
      top1.first = first_dp_idx + best_j;
      top1.second.store(best_dist, std::memory_order_relaxed);
    }
  }

  pair<DatapointIndex, std::atomic<FloatT>>* __restrict__ top1_result_by_query_;

  shared_ptr<std::array<absl::base_internal::SpinLock, kNumSpinLocks>> mutexes_;
};

template <typename FloatT>
class ManyToManyTop1OffsetWrapper {
 public:
  SCANN_DECLARE_COPYABLE_CLASS(ManyToManyTop1OffsetWrapper);

  ManyToManyTop1OffsetWrapper(ManyToManyTop1Callback<FloatT> base,
                              size_t dp_idx_offset,
                              ConstSpan<DatapointIndex> query_idx_table)
      : base_(std::move(base)),
        dp_idx_offset_(dp_idx_offset),
        query_idx_table_(query_idx_table) {}

#ifdef __x86_64__

  SCANN_AVX512_INLINE void InvokeOptimized(Avx512<float, 2> simd_dists,
                                           size_t first_dp_idx,
                                           size_t query_idx) {
    base_.InvokeOptimized(simd_dists, first_dp_idx + dp_idx_offset_,
                          query_idx_table_[query_idx]);
  }

  SCANN_AVX1_INLINE void InvokeOptimized(Avx1<float, 2> simd_dists,
                                         size_t first_dp_idx,
                                         size_t query_idx) {
    base_.InvokeOptimized(simd_dists, first_dp_idx + dp_idx_offset_,
                          query_idx_table_[query_idx]);
  }

  SCANN_SSE4_INLINE void InvokeOptimized(Sse4<float, 2> simd_dists,
                                         size_t first_dp_idx,
                                         size_t query_idx) {
    base_.InvokeOptimized(simd_dists, first_dp_idx + dp_idx_offset_,
                          query_idx_table_[query_idx]);
  }

#endif

  SCANN_INLINE void InvokeOptimized(fallback::Simd<float, 2> dists,
                                    size_t first_dp_idx, size_t query_idx) {
    base_.InvokeOptimized(dists, first_dp_idx + dp_idx_offset_,
                          query_idx_table_[query_idx]);
  }

  SCANN_INLINE void operator()(MutableSpan<FloatT> block, size_t first_dp_idx,
                               size_t query_idx) {
    Invoke(block, first_dp_idx, query_idx);
  }

  SCANN_INLINE void Invoke(MutableSpan<FloatT> block, size_t first_dp_idx,
                           size_t query_idx) {
    base_.Invoke(block, first_dp_idx + dp_idx_offset_,
                 query_idx_table_[query_idx]);
  }

 private:
  ManyToManyTop1Callback<FloatT> base_;
  const size_t dp_idx_offset_;
  const ConstSpan<DatapointIndex> query_idx_table_;
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

  ThreadPool* pool = nullptr;
};

template <typename CallbackT>
struct IsOptimizedCallback {
  static constexpr bool value = false;
};

template <typename FloatT>
struct IsOptimizedCallback<ManyToManyTop1Callback<FloatT>> {
  static constexpr bool value = std::is_same_v<FloatT, float>;
};

#define SCANN_INSTANTIATE_MANY_TO_MANY_2(EXTERN_OR_NOTHING, METHOD_NAME, T, \
                                         CALLBACK)                          \
  EXTERN_OR_NOTHING template void METHOD_NAME(                              \
      const DistanceMeasure& dist, const DenseDataset<T>& queries,          \
      const DenseDataset<T>& database, ThreadPool* pool, CALLBACK callback);

#define SCANN_INSTANTIATE_MANY_TO_MANY_1(EXTERN_OR_NOTHING, METHOD_NAME, T) \
  SCANN_INSTANTIATE_MANY_TO_MANY_2(EXTERN_OR_NOTHING, METHOD_NAME, T,       \
                                   ManyToManyResultsCallback<T>);           \
  SCANN_INSTANTIATE_MANY_TO_MANY_2(EXTERN_OR_NOTHING, METHOD_NAME, T,       \
                                   ManyToManyTop1Callback<T>);

#define SCANN_INSTANTIATE_MANY_TO_MANY(EXTERN_OR_NOTHING, METHOD_NAME)     \
  SCANN_INSTANTIATE_MANY_TO_MANY_1(EXTERN_OR_NOTHING, METHOD_NAME, float); \
  SCANN_INSTANTIATE_MANY_TO_MANY_1(EXTERN_OR_NOTHING, METHOD_NAME, double);

#define SCANN_INSTANTIATE_MANY_TO_MANY_FP8_1(EXTERN_OR_NOTHING, METHOD_NAME, \
                                             CALLBACK)                       \
  EXTERN_OR_NOTHING template Status METHOD_NAME(                             \
      const DistanceMeasure& dist, const DenseDataset<float>& queries,       \
      const FP8SimdBlockTransposedDatabase& database, ThreadPool* pool,      \
      CALLBACK callback);

#define SCANN_INSTANTIATE_MANY_TO_MANY_FP8(EXTERN_OR_NOTHING, METHOD_NAME) \
  SCANN_INSTANTIATE_MANY_TO_MANY_FP8_1(EXTERN_OR_NOTHING, METHOD_NAME,     \
                                       ManyToManyResultsCallback<float>);  \
  SCANN_INSTANTIATE_MANY_TO_MANY_FP8_1(EXTERN_OR_NOTHING, METHOD_NAME,     \
                                       ManyToManyTop1OffsetWrapper<float>);

}  // namespace research_scann

#endif
