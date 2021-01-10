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



#ifndef SCANN_UTILS_WEAK_PTR_CACHE_H_
#define SCANN_UTILS_WEAK_PTR_CACHE_H_

#include <functional>

#include "absl/base/const_init.h"
#include "absl/numeric/int128.h"
#include "absl/synchronization/mutex.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename Output, typename... Inputs>
StatusOr<shared_ptr<const Output>> WeakPtrCache(
    Inputs... inputs,
    std::function<StatusOr<shared_ptr<const Output>>(Inputs...)> base_factory,
    std::function<absl::uint128(Inputs...)> fingerprint_inputs) {
  static auto& cache =
      *(new flat_hash_map<absl::uint128, std::weak_ptr<const Output>>);
  static absl::Mutex mutex(absl::kConstInit);

  absl::MutexLock lock(&mutex);
  const absl::uint128 fp = fingerprint_inputs(inputs...);
  auto it = cache.find(fp);

  auto create_new_output = [&]() -> StatusOr<shared_ptr<const Output>> {
    TF_ASSIGN_OR_RETURN(shared_ptr<const Output> output,
                        base_factory(inputs...));
    cache[fp] = output;
    return output;
  };

  if (it == cache.end()) return create_new_output();

  shared_ptr<const Output> from_cache = it->second.lock();
  if (from_cache) return from_cache;

  return create_new_output();
}

}  // namespace research_scann

#endif
