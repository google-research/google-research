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

#ifndef THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_FEC_CACHE_H_
#define THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_FEC_CACHE_H_

#include <functional>
#include <list>
#include <unordered_map>
#include <vector>

#include "definitions.h"
#include "executor.h"
#include "fec_cache.proto.h"

namespace automl_zero {

struct CachedEvaluation {
  double fitness;
  IntegerT count;
  explicit CachedEvaluation(const double fitness) {
    this->fitness = fitness;
    count = 1;
  }
};

class LRUCache {
 public:
  typedef size_t K;
  typedef CachedEvaluation V;

  explicit LRUCache(IntegerT max_size);
  V* Insert(K key, const V& value);
  const V* Lookup(K key);
  V* MutableLookup(K key);
  void Erase(K key);
  void Clear();

 private:
  typedef std::list<std::pair<K, V>> List;
  typedef List::iterator ListIterator;
  typedef std::unordered_map<K, ListIterator> Map;
  typedef Map::iterator MapIterator;

  void EraseImpl(MapIterator it);
  V* InsertImpl(const K key, const V& value);
  void MaybeResize();
  V* MoveToFront(MapIterator it);

  const IntegerT max_size_;
  List list_;  // Least recently used at back.
  Map map_;
};

class FECCache {
 public:
  explicit FECCache(const FECCacheSpec& spec);
  FECCache(const FECCache& other) = delete;
  FECCache& operator=(const FECCache& other) =
      delete;

  size_t Hash(const std::vector<double>& train_errors,
              const std::vector<double>& valid_errors,
              IntegerT dataset_index, IntegerT num_train_examples);

  // Looks up a hash in the cache. Returns a pair: first element is the
  // activation associated with the hash (if found, otherwise kMinActivation)
  // and the second element is whether the hash was found.
  std::pair<double, bool> Find(size_t hash);

  // Inserts a hash into the cache, together with its associated fitness. Call
  // only if the hash was not found.
  void InsertOrDie(size_t hash, double fitness);

  // Notes that a hash in the cache has been seen again. Call only if the hash
  // was found.
  // TODO(ereal): remove from template and caller if/when no longer necessary.
  void UpdateOrDie(size_t hash, double fitness) {}

  // Removes all items in the cache.
  void Clear();

  // Return how many examples should be pretrained, trained, and validated in
  // order to accumulate errors for this cache.
  IntegerT NumTrainExamples() const;
  IntegerT NumValidExamples() const;

 private:
  const FECCacheSpec spec_;

  LRUCache cache_;
};

}  // namespace automl_zero

#endif  // THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_FEC_CACHE_H_
