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

#ifndef SCANN_UTILS_PARALLEL_FOR_H_
#define SCANN_UTILS_PARALLEL_FOR_H_

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/functional/function_ref.h"
#include "absl/synchronization/mutex.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/utils/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace research_scann {

enum : size_t {
  kDynamicBatchSize = numeric_limits<size_t>::max(),
};

struct ParallelForOptions {
  size_t max_parallelism = numeric_limits<size_t>::max();
};

template <size_t kItersPerBatch = kDynamicBatchSize, typename SeqT,
          typename Function>
SCANN_INLINE void ParallelFor(SeqT seq, ThreadPool* pool, Function func,
                              ParallelForOptions opts = ParallelForOptions());

template <size_t kItersPerBatch = kDynamicBatchSize, typename SeqT,
          typename Function>
SCANN_INLINE Status
ParallelForWithStatus(SeqT seq, ThreadPool* pool, Function Func,
                      ParallelForOptions opts = ParallelForOptions()) {
  Status finite_check_status = OkStatus();

  std::atomic_bool is_ok_status{true};
  absl::Mutex mutex;
  ParallelFor(
      seq, pool,
      [&](size_t idx) {
        if (!is_ok_status.load(std::memory_order_relaxed)) {
          return;
        }
        Status status = Func(idx);
        if (!status.ok()) {
          absl::MutexLock lock(&mutex);
          finite_check_status = status;
          is_ok_status.store(false, std::memory_order_relaxed);
        }
      },
      opts);
  return finite_check_status;
}

namespace parallel_for_internal {

template <size_t kItersPerBatch, typename SeqT, typename Function>
class ParallelForClosure : public std::function<void()> {
 public:
  static constexpr bool kIsDynamicBatch = (kItersPerBatch == kDynamicBatchSize);
  ParallelForClosure(SeqT seq, Function func)
      : func_(func),
        index_(*seq.begin()),
        range_end_(*seq.end()),
        reference_count_(1) {}

  SCANN_INLINE void RunParallel(ThreadPool* pool, size_t desired_threads) {
    DCHECK(pool);

    size_t n_threads =
        std::min<size_t>(desired_threads - 1, pool->NumThreads());

    if (kIsDynamicBatch) {
      batch_size_ =
          SeqT::Stride() * std::max(1ul, desired_threads / 4 / n_threads);
    }

    reference_count_ += n_threads;
    while (n_threads--) {
      pool->Schedule([this]() { Run(); });
    }

    DoWork();

    termination_mutex_.WriterLock();
    termination_mutex_.WriterUnlock();

    if (--reference_count_ == 0) delete this;
  }

  void Run() {
    termination_mutex_.ReaderLock();
    DoWork();
    termination_mutex_.ReaderUnlock();

    if (--reference_count_ == 0) delete this;
  }

  SCANN_INLINE void DoWork() {
    const size_t range_end = range_end_;

    constexpr size_t kStaticBatchSize = SeqT::Stride() * kItersPerBatch;
    const size_t batch_size = kIsDynamicBatch ? batch_size_ : kStaticBatchSize;
    DCHECK_NE(batch_size, kDynamicBatchSize);
    DCHECK_EQ(batch_size % SeqT::Stride(), 0);

    for (;;) {
      const size_t batch_begin = index_.fetch_add(batch_size);

      const size_t batch_end = std::min(batch_begin + batch_size, range_end);
      if (ABSL_PREDICT_FALSE(batch_begin >= range_end)) break;
      for (size_t idx : SeqWithStride<SeqT::Stride()>(batch_begin, batch_end)) {
        func_(idx);
      }
    }
  }

 private:
  Function func_;

  std::atomic<size_t> index_;

  const size_t range_end_;

  absl::Mutex termination_mutex_;

  std::atomic<uint32_t> reference_count_;

  size_t batch_size_ = kItersPerBatch;
};

}  // namespace parallel_for_internal

template <size_t kItersPerBatch, typename SeqT, typename Function>
SCANN_INLINE void ParallelFor(SeqT seq, ThreadPool* pool, Function func,
                              ParallelForOptions opts) {
  constexpr size_t kMinItersPerBatch =
      kItersPerBatch == kDynamicBatchSize ? 1 : kItersPerBatch;
  const size_t desired_threads = std::min(
      opts.max_parallelism, DivRoundUp(*seq.end() - *seq.begin(),
                                       SeqT::Stride() * kMinItersPerBatch));

  if (!pool || desired_threads <= 1) {
    for (size_t idx : seq) {
      func(idx);
    }
    return;
  }

  using parallel_for_internal::ParallelForClosure;
  auto closure =
      new ParallelForClosure<kItersPerBatch, SeqT, Function>(seq, func);
  closure->RunParallel(pool, desired_threads);
}

}  // namespace research_scann

#endif
