// Copyright 2022 The Google Research Authors.
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
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/intrinsics/simd.h"
#include "scann/utils/types.h"
#include "tensorflow/core/platform/prefetch.h"

namespace research_scann {

template <typename FloatT>
using ManyToManyResultsCallback =
    std::function<void(MutableSpan<FloatT> block_distances,
                       DatapointIndex first_dp_idx, DatapointIndex query_idx)>;

template <typename FloatT>
class EpsilonFilteringCallback {
 public:
  EpsilonFilteringCallback(std::atomic<FloatT>* epsilons,
                           ManyToManyResultsCallback<FloatT> slow_path_fn)
      : epsilons_(epsilons), slow_path_fn_(std::move(slow_path_fn)) {}

#ifdef __x86_64__

  SCANN_AVX512_INLINE void InvokeOptimized(Avx512<float, 2> simd_dists,
                                           size_t first_dp_idx,
                                           size_t query_idx) {
    float best_dist = epsilons_[query_idx].load(std::memory_order_relaxed);

    auto cmp = (simd_dists < Avx512<float>::Broadcast(best_dist));
    if (ABSL_PREDICT_TRUE(_kortestz_mask16_u8(cmp[0], cmp[1]))) return;

    auto dists = simd_dists.Store();
    slow_path_fn_(MakeMutableSpan(dists), first_dp_idx, query_idx);
  }

  SCANN_AVX1_INLINE void InvokeOptimized(Avx1<float, 2> simd_dists,
                                         size_t first_dp_idx,
                                         size_t query_idx) {
    float best_dist = epsilons_[query_idx].load(std::memory_order_relaxed);

    auto cmp = (simd_dists < Avx1<float>::Broadcast(best_dist));
    if (ABSL_PREDICT_TRUE((cmp[0] | cmp[1]).MaskFromHighBits() == 0)) return;

    auto dists = simd_dists.Store();
    slow_path_fn_(MakeMutableSpan(dists), first_dp_idx, query_idx);
  }

  SCANN_SSE4_INLINE void InvokeOptimized(Sse4<float, 2> simd_dists,
                                         size_t first_dp_idx,
                                         size_t query_idx) {
    float best_dist = epsilons_[query_idx].load(std::memory_order_relaxed);

    auto cmp = (simd_dists < Sse4<float>::Broadcast(best_dist));
    if (ABSL_PREDICT_TRUE((cmp[0] | cmp[1]).MaskFromHighBits() == 0)) return;

    auto dists = simd_dists.Store();
    slow_path_fn_(MakeMutableSpan(dists), first_dp_idx, query_idx);
  }

#else

  SCANN_INLINE void InvokeOptimized(fallback::Simd<float, 2> dists,
                                    size_t first_dp_idx, size_t query_idx) {
    FloatT candidates[] = {dists[0].Unwrap(), dists[1].Unwrap()};
    Invoke(MakeMutableSpan(candidates, 2), first_dp_idx, query_idx);
  }

#endif

  SCANN_INLINE void operator()(MutableSpan<FloatT> block, size_t first_dp_idx,
                               size_t query_idx) {
    Invoke(block, first_dp_idx, query_idx);
  }

  SCANN_INLINE void Invoke(MutableSpan<FloatT> block, size_t first_dp_idx,
                           size_t query_idx) {
    FloatT best_dist = epsilons_[query_idx].load(std::memory_order_relaxed);

    bool update_needed = false;
    for (size_t j : Seq(block.size())) {
      if (block[j] < best_dist) update_needed = true;
    }
    if (ABSL_PREDICT_TRUE(!update_needed)) return;

    slow_path_fn_(block, first_dp_idx, query_idx);
  }

 private:
  std::atomic<FloatT>* epsilons_;
  ManyToManyResultsCallback<FloatT> slow_path_fn_;
};

template <typename FloatT>
class ManyToManyTop1Callback {
 public:
  SCANN_DECLARE_COPYABLE_CLASS(ManyToManyTop1Callback);

  static constexpr int kNumSpinLocks = 128;
  static_assert((kNumSpinLocks & (kNumSpinLocks - 1)) == 0,
                "kNumSpinLocks has to be power of 2");

  explicit ManyToManyTop1Callback(
      MutableSpan<pair<DatapointIndex, FloatT>> top1_result_by_query)
      : top1_result_by_query_(top1_result_by_query.data()),
        epsilons_(
            make_unique<std::atomic<FloatT>[]>(top1_result_by_query.size())),
        mutexes_(make_shared<
                 std::array<absl::base_internal::SpinLock, kNumSpinLocks>>()) {
    for (size_t i : IndicesOf(top1_result_by_query)) {
      epsilons_[i].store(top1_result_by_query[i].second,
                         std::memory_order_relaxed);
    }
  }

  void operator()(MutableSpan<FloatT> block, size_t first_dp_idx,
                  size_t query_idx) {
    auto& top1 = top1_result_by_query_[query_idx];
    const size_t mutex_idx = query_idx & (kNumSpinLocks - 1);
    auto mutex_ptr = &(*mutexes_)[mutex_idx];
    ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(&top1);
    ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(
        mutex_ptr);

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

    absl::base_internal::SpinLockHolder lock(mutex_ptr);
    if (ABSL_PREDICT_TRUE(best_dist < top1.second)) {
      top1.first = first_dp_idx + best_j;
      top1.second = best_dist;
      epsilons_[query_idx].store(best_dist, std::memory_order_relaxed);
    }
  }

  std::atomic<FloatT>* epsilons() const { return epsilons_.get(); }

 private:
  pair<DatapointIndex, FloatT>* top1_result_by_query_;

  shared_ptr<std::atomic<FloatT>[]> epsilons_;
  shared_ptr<std::array<absl::base_internal::SpinLock, kNumSpinLocks>> mutexes_;
};

class ManyToManyTopKCallback {
 public:
  SCANN_DECLARE_COPYABLE_CLASS(ManyToManyTopKCallback);

  static constexpr int kNumMutexes = 512;
  static_assert((kNumMutexes & (kNumMutexes - 1)) == 0,
                "kNumMutexes has to be power of 2");

  explicit ManyToManyTopKCallback(MutableSpan<FastTopNeighbors<float>> topns)
      : topns_(topns.data()),
        epsilons_(make_unique<std::atomic<float>[]>(topns.size())),
        mutexes_(make_shared<std::array<absl::Mutex, kNumMutexes>>()) {
    for (size_t i : IndicesOf(topns)) {
      epsilons_[i].store(topns[i].epsilon(), std::memory_order_relaxed);
    }
  }

  void operator()(MutableSpan<float> block, size_t first_dp_idx,
                  size_t query_idx) {
    auto& topn = topns_[query_idx];
    const size_t mutex_idx = query_idx & (kNumMutexes - 1);
    absl::MutexLock lock(&(*mutexes_)[mutex_idx]);
    topn.PushBlock(block, first_dp_idx);
    epsilons_[query_idx].store(topns_[query_idx].epsilon(),
                               std::memory_order_relaxed);
  }

  std::atomic<float>* epsilons() const { return epsilons_.get(); }

 private:
  FastTopNeighbors<float>* topns_;

  shared_ptr<std::atomic<float>[]> epsilons_;
  shared_ptr<std::array<absl::Mutex, kNumMutexes>> mutexes_;
};

template <typename FloatT>
class EpsilonFilteringOffsetWrapper {
 public:
  SCANN_DECLARE_COPYABLE_CLASS(EpsilonFilteringOffsetWrapper);

  EpsilonFilteringOffsetWrapper(EpsilonFilteringCallback<FloatT> base,
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
  EpsilonFilteringCallback<FloatT> base_;
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
struct IsOptimizedCallback<EpsilonFilteringCallback<FloatT>> {
  static constexpr bool value = std::is_same_v<FloatT, float>;
};
template <typename FloatT>
struct IsOptimizedCallback<EpsilonFilteringOffsetWrapper<FloatT>> {
  static constexpr bool value = std::is_same_v<FloatT, float>;
};
template <>
struct IsOptimizedCallback<ManyToManyTopKCallback> {
  static constexpr bool value = true;
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
                                   EpsilonFilteringCallback<T>);

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
                                       EpsilonFilteringOffsetWrapper<float>);

}  // namespace research_scann

#endif
