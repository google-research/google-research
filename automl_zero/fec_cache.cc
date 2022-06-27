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

#include "fec_cache.h"

#include <utility>

#include "executor.h"
#include "fec_hashing.h"

namespace automl_zero {

using ::std::make_pair;
using ::std::vector;
using K = LRUCache::K;
using V = LRUCache::V;

LRUCache::LRUCache(IntegerT max_size)
    : max_size_(max_size) {
  CHECK_GT(max_size, 1);
}

V* LRUCache::Insert(const K key, const V& value) {
  // If already inserted, erase it.
  MapIterator found = map_.find(key);
  if (found != map_.end()) EraseImpl(found);
  V* inserted = InsertImpl(key, value);
  MaybeResize();
  return inserted;
}

const V* LRUCache::Lookup(const K key) {
  MapIterator found = map_.find(key);
  if (found == map_.end()) {
    // If not found, return nullptr.
    return nullptr;
  } else {
    // If found, return it.
    return &found->second->second;
  }
}

V* LRUCache::MutableLookup(const K key) {
  MapIterator found = map_.find(key);
  if (found == map_.end()) {
    // If not found, return nullptr.
    return nullptr;
  } else {
    // If found, move it to the front and return it.
    const V value = found->second->second;
    EraseImpl(found);
    return InsertImpl(key, value);
  }
}

void LRUCache::Erase(const K key) {
  MapIterator found = map_.find(key);
  CHECK(found != map_.end());
  EraseImpl(found);
}

void LRUCache::Clear() {
  map_.clear();
  list_.clear();
}

void LRUCache::EraseImpl(MapIterator it) {
  list_.erase(it->second);
  map_.erase(it);
}

V* LRUCache::InsertImpl(const K key, const V& value) {
  list_.push_front(make_pair(key, value));
  ListIterator pushed = list_.begin();
  map_.insert(make_pair(key, pushed));
  return &pushed->second;
}

void LRUCache::MaybeResize() {
  // Keep within size limit.
  while (list_.size() > max_size_) {
    // Erase last element.
    const K erasing = list_.back().first;
    list_.pop_back();
    map_.erase(erasing);
  }
}

FECCache::FECCache(const FECSpec& spec)
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

void FECCache::Clear() {
  cache_.Clear();
}

IntegerT FECCache::NumTrainExamples() const {
  return spec_.num_train_examples();
}

IntegerT FECCache::NumValidExamples() const {
  return spec_.num_valid_examples();
}

}  // namespace automl_zero
