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

#ifndef SCANN_UTILS_THREADS_H_
#define SCANN_UTILS_THREADS_H_

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/utils/types.h"

namespace research_scann {

unique_ptr<ThreadPool> StartThreadPool(const std::string& pool_name,
                                       ssize_t num_threads);

class ThreadPoolInterface {
 public:
  ThreadPoolInterface(std::nullptr_t) : ThreadPoolInterface() {}

  ThreadPoolInterface(ThreadPool* pool)
      : thread_pool_(pool), executor_(thread_pool_) {}

  ThreadPoolInterface() = default;
  ThreadPoolInterface(const ThreadPoolInterface&) = default;
  ThreadPoolInterface(ThreadPoolInterface&&) = default;
  ThreadPoolInterface& operator=(const ThreadPoolInterface&) = default;
  ThreadPoolInterface& operator=(ThreadPoolInterface&&) = default;

  bool operator==(std::nullptr_t) const { return executor_ == nullptr; }
  bool operator!=(std::nullptr_t) const { return executor_ != nullptr; }

  operator bool() const { return executor_ != nullptr; }
  bool operator!() const { return executor_ == nullptr; }

  template <typename Callable>
  bool TrySchedule(Callable&& callable) {
    DCHECK(*this);
    executor_->Schedule(callable);
    return true;
  }

  std::optional<size_t> num_threads() const {
    DCHECK(*this);
    return thread_pool_->NumThreads();
  }

  int num_pending_closures() {
    DCHECK(*this);
    return 0;
  }

 private:
  ThreadPool* thread_pool_ = nullptr;
  ThreadPool* executor_ = nullptr;
};

}  // namespace research_scann

#endif
