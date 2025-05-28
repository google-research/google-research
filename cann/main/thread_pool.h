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

#ifndef MAIN_THREAD_POOL_H_
#define MAIN_THREAD_POOL_H_

// NOLINTBEGIN

#include <queue>
#include <thread>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/synchronization/mutex.h"

// A simple ThreadPool based on
// https://github.com/abseil/abseil-cpp/blob/master/absl/synchronization/internal/thread_pool.h

class ThreadPool {
 public:
  // Use num_threads=-1 to select a number of threads equal to
  // hardware_concurrency.
  explicit ThreadPool(int num_threads);
  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;
  ~ThreadPool();

  // Schedule a function to be run on a ThreadPool thread.
  void Schedule(absl::AnyInvocable<void()> func);

  void AwaitEmptyQueue();

 private:
  bool WorkAvailable() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return !queue_.empty();
  }
  bool NoWorkWaiting() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return queue_.empty();
  }

  void WorkLoop();

  absl::Mutex mu_;
  std::queue<absl::AnyInvocable<void()>> queue_ ABSL_GUARDED_BY(mu_);
  std::vector<std::thread> threads_;
};

// NOLINTEND

#endif  // MAIN_THREAD_POOL_H_
