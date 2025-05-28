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

#include "main/thread_pool.h"

#include <cassert>

#include "absl/functional/any_invocable.h"

ThreadPool::ThreadPool(int num_threads) {
  if (num_threads < 0) {
    num_threads = std::thread::hardware_concurrency();
  }
  threads_.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    threads_.push_back(std::thread(&ThreadPool::WorkLoop, this));
  }
}

ThreadPool::~ThreadPool() {
  {
    absl::MutexLock l(&mu_);
    for (size_t i = 0; i < threads_.size(); i++) {
      queue_.push(nullptr);  // Shutdown signal.
    }
  }
  for (auto &t : threads_) {
    t.join();
  }
}

void ThreadPool::Schedule(absl::AnyInvocable<void()> func) {
  assert(func != nullptr);
  absl::MutexLock l(&mu_);
  queue_.push(std::move(func));
}

void ThreadPool::AwaitEmptyQueue() {
  absl::MutexLock l(&mu_);
  mu_.Await(absl::Condition(this, &ThreadPool::NoWorkWaiting));
}

void ThreadPool::WorkLoop() {
  while (true) {
    absl::AnyInvocable<void()> func;
    {
      absl::MutexLock l(&mu_);
      mu_.Await(absl::Condition(this, &ThreadPool::WorkAvailable));
      func = std::move(queue_.front());
      queue_.pop();
    }
    if (func == nullptr) {  // Shutdown signal.
      break;
    }
    func();
  }
}
