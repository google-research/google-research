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

#include "frequent.h"

#include <cmath>
#include "utils.h"

namespace sketch {

CuckooHashParams default_params;

IndexCuckooHash::IndexCuckooHash(const std::vector<IntFloatPair>& keys,
                                 int size, const CuckooHashParams& params)
    : keys_(keys), params_(params) {
  hash_tables_.resize(params_.hash_tables);
  hash_a_.resize(params_.hash_tables);
  hash_b_.resize(params_.hash_tables);
  Create(size);
}

void IndexCuckooHash::Reset() {
  for (int i = 0; i < hash_tables_.size(); ++i) {
    int hash_size = hash_tables_[i].size();
    hash_tables_[i].resize(0);
    hash_tables_[i].resize(hash_size, -1);
  }
}

uint IndexCuckooHash::Size() const {
  return hash_tables_.size() * (sizeof(hash_tables_[0]) +
                                hash_tables_[0].size() * sizeof(int)) +
      2 * hash_tables_.size() * sizeof(uint);
}

void IndexCuckooHash::Create(int hash_size) {
  if (hash_size <= 0) {
    hash_size = params_.hash_tables * hash_tables_[0].size();
  }
  hash_max_ = std::ceil(hash_size * params_.resize_factor /
                        params_.hash_tables);
  BitGenerator bit_gen;
  absl::BitGenRef& generator = *bit_gen.BitGen();
  absl::uniform_int_distribution<int> rand_int;

  for (int i = 0; i < params_.hash_tables; ++i) {
    hash_a_[i] = rand_int(generator);
    hash_b_[i] = rand_int(generator);
    hash_tables_[i].resize(0);
    hash_tables_[i].resize(hash_size, -1);
  }
  for (int i = 0; i < keys_.size(); ++i) {
    if (!Update(keys_[i].first, -1, i, false)) {
      // rehashing failed, let us do it again.
      Create(0);
      return;
    }
  }
}

int IndexCuckooHash::Find(uint key) const {
  for (int i = 0; i < hash_tables_.size(); ++i) {
    int hash_index = Hash(hash_a_[i], hash_b_[i], key, hash_max_);
    int key_index = hash_tables_[i][hash_index];
    if (key_index >= 0 && keys_[key_index].first == key) {
      return key_index;
    }
  }
  return -1;
}

bool IndexCuckooHash::Update(uint key, int current, int next, bool rehash) {
  int last_entry = -1;  // the table/column from which we removed this element
  for (int retry = 0; retry < params_.max_retries; ++retry) {
    for (int i = 0; i < hash_tables_.size(); ++i) {
      int hash_index = Hash(hash_a_[i], hash_b_[i], key, hash_max_);
      if (hash_tables_[i][hash_index] == current) {
        hash_tables_[i][hash_index] = next;
        return true;
      }
    }
    if (next == -1) return true;
    // Could not find the entry, kick out someone else
    // However do not kick out the entry just inserted, as that could result
    // in an immediate loop.
    int table_id = (key + retry) % (params_.hash_tables
                                    - (last_entry >= 0 ? 1 : 0));
    if (last_entry >= 0 && table_id >= last_entry) table_id++;
    int hash_index = Hash(hash_a_[table_id], hash_b_[table_id], key, hash_max_);
    int key_index = hash_tables_[table_id][hash_index];
    key = keys_[key_index].first;
    current = -1;  // key will be kicked out, should be put in an empty spot
    hash_tables_[table_id][hash_index] = next;
    next = key_index;
    last_entry = table_id;
  }
  if (!rehash) return false;
  Create(0);
  return true;
}

void IndexCuckooHash::Swap(int loc1, int loc2) {
  std::pair<int, int> loc1_entry, loc2_entry;
  for (int i = 0; i < hash_tables_.size(); ++i) {
    int hash_index = Hash(hash_a_[i], hash_b_[i], keys_[loc1].first, hash_max_);
    if (hash_tables_[i][hash_index] == loc1) {
      loc1_entry = std::make_pair(i, hash_index);
    }
    hash_index = Hash(hash_a_[i], hash_b_[i], keys_[loc2].first, hash_max_);
    if (hash_tables_[i][hash_index] == loc2) {
      loc2_entry = std::make_pair(i, hash_index);
    }
  }
  hash_tables_[loc1_entry.first][loc1_entry.second] = loc2;
  hash_tables_[loc2_entry.first][loc2_entry.second] = loc1;
}

void IndexCuckooHash::Print() const {
  printf("HEAP\n");
  for (int i = 0; i < keys_.size(); ++i) {
    printf("%d %u %f\n", i, keys_[i].first, keys_[i].second);
    for (int j = 0; j < hash_tables_.size(); ++j) {
      int hash_index = Hash(hash_a_[j], hash_b_[j], keys_[i].first,
                            hash_max_);
      printf("%d: hash_index %d key_index %u\n",
               j, hash_index, hash_tables_[j][hash_index]);
    }
  }
  printf("Tables\n");
  for (int i = 0; i < hash_tables_.size(); ++i) {
    printf(" Table %d\n", i);
    for (int j = 0; j < hash_tables_[i].size(); ++j) {
      printf("    %d\n", hash_tables_[i][j]);
    }
  }
}

Frequent::Frequent(uint heap_size)
    : heap_size_(heap_size), delete_threshold_(0),
      counter_heap_(0),
      hash_(counter_heap_, heap_size, default_params) {
  counter_heap_.reserve(heap_size);
}

Frequent::Frequent(uint heap_size, const CuckooHashParams& params)
    : heap_size_(heap_size), delete_threshold_(0), counter_heap_(0),
      hash_(counter_heap_, heap_size, params) {
  counter_heap_.reserve(heap_size);
}

Frequent::Frequent(const Frequent& other)
    : heap_size_(other.heap_size_), delete_threshold_(other.delete_threshold_),
      counter_heap_(other.counter_heap_),
      hash_(counter_heap_, other.heap_size_, other.hash_.GetParams()) {
  counter_heap_.reserve(heap_size_);
}

void Frequent::Reset() {
  counter_heap_.resize(0);
  hash_.Reset();
  ResetMissing();
}

int Frequent::Swap(int loc1, int loc2) {
  hash_.Swap(loc1, loc2);
  IntFloatPair tmp = counter_heap_[loc1];
  counter_heap_[loc1] = counter_heap_[loc2];
  counter_heap_[loc2] = tmp;
  return loc2;
}

bool Frequent::Consistent(const std::string& message) const {
  for (int i = 0; i < counter_heap_.size(); ++i) {
    if (hash_.Find(counter_heap_[i].first) != i) {
      printf("Inconsistency: %s\n", message.c_str());
      printf("Key %u is inconsistent\n", counter_heap_[i].first);
      hash_.Print();
      return false;
    }
  }
  return true;
}

void Frequent::Heapify(int loc) {
  while (loc > 0 &&
         counter_heap_[loc].second < counter_heap_[(loc - 1) / 2].second) {
    loc = Swap(loc, (loc - 1) / 2);
  }
  while (true) {
    int child_loc = 2 * loc + 1;
    if (child_loc >= counter_heap_.size()) break;  // leaf node
    if (child_loc + 1 < counter_heap_.size() &&
        counter_heap_[child_loc + 1].second < counter_heap_[child_loc].second) {
      child_loc++;
    }
    if (counter_heap_[loc].second <= counter_heap_[child_loc].second) break;
    DCHECK(Consistent("Swapping locations " + std::to_string(loc) +
                     " and " + std::to_string(child_loc)));
    loc = Swap(loc, child_loc);
    DCHECK(Consistent("Swapped locations " + std::to_string(loc) +
                     " and " + std::to_string(child_loc)));
  }
}

void Frequent::Add(uint item, float delta) {
  int loc = hash_.Find(item);
  if (loc < 0) {
    IntFloatPair new_pair(item, delta + EstimateMissing(item));
    if (counter_heap_.size() >= heap_size_) {
      if (new_pair.second < counter_heap_[0].second) {
        UpdateMissing(item, new_pair.second);
        return;
      }
      // delete the smallest element from the heap
      UpdateMissing(counter_heap_[0].first, counter_heap_[0].second);
      hash_.Update(counter_heap_[0].first, 0, -1, false);

      loc = 0;
      counter_heap_[0] = new_pair;
      hash_.Update(item, -1, loc, true);
    } else {
      loc = counter_heap_.size();
      counter_heap_.push_back(new_pair);
      hash_.Update(item, -1, loc, true);
    }
  } else {
    counter_heap_[loc].second += delta;
  }
  DCHECK(Consistent("Heapifying location " + std::to_string(loc)));
  Heapify(loc);
  DCHECK(Consistent("Heapifying complete at " + std::to_string(loc)));
}

float Frequent::Estimate(uint item) const {
  int heap_index = hash_.Find(item);
  if (heap_index >= 0) return counter_heap_[heap_index].second;
  return EstimateMissing(item);
}

void Frequent::HeavyHitters(float threshold, std::vector<uint>* items) const {
  items->resize(0);
  items->reserve(counter_heap_.size());
  for (const auto& kv : counter_heap_) {
    if (kv.second > threshold) {
      items->push_back(kv.first);
    }
  }
}

unsigned int Frequent::Size() const {
  return sizeof(Frequent) + hash_.Size() +
      counter_heap_.capacity() * sizeof(IntFloatPair);
}

bool Frequent::Compatible(const Sketch& other_sketch) const {
  const Frequent* other = dynamic_cast<const Frequent*>(&other_sketch);
  if (other == nullptr) return false;
  return CompatibleMissing(*other);
}

void Frequent::Merge(const Sketch& other_sketch) {
  if (!Compatible(other_sketch)) return;
  const Frequent& other = static_cast<const Frequent &>(other_sketch);
  // Merge the counters from other heap
  for (int i = other.counter_heap_.size() - 1; i >= 0; --i) {
    const IntFloatPair& kv = other.counter_heap_[i];
    Add(kv.first, kv.second);
  }
  // Add missing from other
  for (int i = counter_heap_.size() - 1; i >= 0; --i) {
    if (other.hash_.Find(counter_heap_[i].first) < 0) {
      counter_heap_[i].second += other.EstimateMissing(counter_heap_[i].first);
      Heapify(i);  // affects only entries >= i, so not a problem
    }
  }
  MergeMissing(other);
}

}  // namespace sketch
