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

#include "fec_cache.h"

#include <utility>

#include "executor.h"
#include "fec_hashing.h"

namespace brain {
namespace evolution {
namespace amlz {

using ::std::make_pair;
using ::std::vector;

FECCache::FECCache(const FECCacheSpec& spec)
    : spec_(spec), cache_(spec_.cache_size()) {
  CHECK_GT(spec_.num_train_examples(), 0);
  CHECK_GT(spec_.num_valid_examples(), 0);
  CHECK_GT(spec_.cache_size(), 1);
  CHECK(spec_.forget_every() == 0 || spec_.forget_every() > 1);
}

size_t FECCache::Hash(
    const vector<double>& train_errors,
    const vector<double>& valid_errors,
    const IntegerT dataset_index, const IntegerT num_train_examples) {
  return WellMixedHash(train_errors, valid_errors, dataset_index,
                       num_train_examples);
}

std::pair<double, bool> FECCache::Find(const size_t hash) {
  CachedEvaluation* cached = cache_.MutableLookup(hash);
  if (cached == nullptr) {
    return make_pair(kMinFitness, false);
  } else {
    const double fitness = cached->fitness;
    ++cached->count;
    if (spec_.forget_every() != 0 && cached->count >= spec_.forget_every()) {
      cache_.Erase(hash);
    }
    return make_pair(fitness, true);
  }
}

void FECCache::InsertOrDie(
    const size_t hash, const double fitness) {
  CHECK(cache_.Lookup(hash) == nullptr);
  CachedEvaluation* inserted = cache_.Insert(hash, CachedEvaluation(fitness));
  CHECK(inserted != nullptr);
}

// TODO(ereal): test.
void FECCache::Clear() {
  cache_.Clear();
}

IntegerT FECCache::NumTrainExamples() const {
  return spec_.num_train_examples();
}

IntegerT FECCache::NumValidExamples() const {
  return spec_.num_valid_examples();
}

}  // namespace amlz
}  // namespace evolution
}  // namespace brain
