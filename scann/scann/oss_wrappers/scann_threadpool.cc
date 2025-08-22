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

#include "scann/oss_wrappers/scann_threadpool.h"

#include <functional>
#include <memory>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"

namespace research_scann {

ThreadPool::ThreadPool(absl::string_view name, int num_threads) {
  eigen_threadpool_ = std::make_unique<Eigen::ThreadPool>(num_threads, true);
}

void ThreadPool::Schedule(std::function<void()> fn) {
  CHECK(fn != nullptr);
  eigen_threadpool_->Schedule(fn);
}

int ThreadPool::NumThreads() const { return eigen_threadpool_->NumThreads(); }

}  // namespace research_scann
