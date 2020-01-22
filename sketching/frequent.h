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

#ifndef SKETCHING_FREQUENT_H_
#define SKETCHING_FREQUENT_H_

// Implements a slight variant of the Misra-Gries sketch
//  J. Misra and D. Gries. "Finding Repeated Elements". Science of Computer
//  Programming, Vol 2, No 2, 1982, pages 143-152.
// Described in Cormode - "Misra-Gries Summaries". In Ming-Yang Kao (ed).
// Encyclopedia of Algorithms, Springer Verlag, 2014, pages 1-5.

// We also implement a version of Misra-Gries combined with a CountMin sketch,
// which results in a significantly improved performance and accuracy.

#include <vector>

#include "sketch.h"
#include "countmin.h"

namespace sketch {

typedef struct CuckooHashParams {
  int hash_tables = 2;
  float resize_factor = 1.5;
  int max_retries = 20;
} CuckooHashParams;

class IndexCuckooHash {
 public:
  IndexCuckooHash(const std::vector<IntFloatPair>& keys,
                  int size, const CuckooHashParams& params);

  virtual ~IndexCuckooHash() {}

  virtual void Reset();

  virtual uint Size() const;

  // Make each hashtable be of size hash_size, and initialize all entries to -1.
  // If keys_ is non-empty, insert the indices of key elements in the
  // table, indexes by the hashes of the items.
  virtual void Create(int hash_size);

  // Find the index of the item in keys, using the hash_tables.
  // Return -1 if not found
  virtual int Find(uint item) const;

  // Change the index of the item from current to next.
  // To insert an new element, set current = -1.
  // To delete an element, set next = -1.
  // If rehash is true, and in the cuckoo hashing algorithm too many elements
  // have been kicked around, time to rehash.
  // We set rehash to false if the UpdateHash is being called from Create,
  // otherwise it is set to true.
  // The return value is always true, except during Create, where insert can
  // fail if too many elements were evicted.
  virtual bool Update(uint item, int current, int next, bool rehash);

  // We want to swap entries at loc1 and loc2 in keys, so update the hash.
  virtual void Swap(int loc1, int loc2);

  virtual void Print() const;

  const CuckooHashParams& GetParams() const { return params_; }

 protected:
  const std::vector<IntFloatPair>& keys_;
  const CuckooHashParams params_;
  std::vector< std::vector<int> > hash_tables_;
  std::vector<uint> hash_a_;
  std::vector<uint> hash_b_;
  int hash_max_;  // hashes will be computed between 0 .. hash_max_-1
};

class Frequent : public Sketch {
 public:
  Frequent(uint heap_size);

  Frequent(uint heap_size, const CuckooHashParams& params);

  Frequent(const Frequent& other);

  virtual ~Frequent() {}

  virtual void Reset();

  virtual void Add(uint item, float delta);

  virtual float Estimate(uint item) const;

  virtual void HeavyHitters(float threshold, std::vector<uint>* items) const;

  virtual uint Size() const;

  virtual bool Compatible(const Sketch& other) const;

  virtual void Merge(const Sketch& other);

 protected:
  virtual void ResetMissing() {
    delete_threshold_ = 0;
  }

  virtual float EstimateMissing(uint item) const {
    return delete_threshold_;
  }

  virtual void UpdateMissing(uint item, float value) {
    delete_threshold_ = std::max(delete_threshold_, value);
  }

  virtual bool CompatibleMissing(const Frequent& other) const { return true; }

  virtual void MergeMissing(const Frequent& other) {}

 private:
  uint heap_size_;
  float delete_threshold_;
  std::vector<IntFloatPair> counter_heap_;
  IndexCuckooHash hash_;

  // This is called when the item at position loc has been changed (either a
  // new item has been placed at loc, or its weight has changed). Performs
  // a series of Swaps to ensure the heap property - the weight of every
  // parent <= the weights of either child.
  void Heapify(int loc);

  // A utility function to swap the items at loc1 and loc2 in the heap.
  // Also updates the hash tables. Returns loc2, for convenience.
  int Swap(int loc1, int loc2);

  // Check the consistency of the hash with the heap, for debugging.
  bool Consistent(const std::string& message) const;
};

class Frequent_Fallback : public Frequent {
 public:
  Frequent_Fallback(uint heap_size, uint hash_count, uint hash_size) :
      Frequent(heap_size), cm_(CountMinCU(hash_count, hash_size)) {}

  Frequent_Fallback(const Frequent_Fallback& other)
      : Frequent(other), cm_(other.cm_) {}

 protected:
  virtual void ResetMissing() {
    Frequent::ResetMissing();
    cm_.Reset();
  }

  virtual float EstimateMissing(uint item) const {
    return cm_.Estimate(item);
  }

  virtual void UpdateMissing(uint item, float value) {
    cm_.Update(item, value);
  }

  virtual bool CompatibleMissing(const Frequent& other) const {
    if (!Frequent::CompatibleMissing(other)) return false;
    const Frequent_Fallback& other_cast =
        dynamic_cast<const Frequent_Fallback&>(other);
    return cm_.Compatible(other_cast.cm_);
  }

  virtual void MergeMissing(const Frequent& other) {
    Frequent::MergeMissing(other);
    const Frequent_Fallback& other_cast =
        dynamic_cast<const Frequent_Fallback&>(other);
    cm_.Merge(other_cast.cm_);
  }

  virtual unsigned int Size() const {
    return Frequent::Size() + cm_.Size();
  }

 private:
    CountMinCU cm_;
};

}  // namespace sketch

#endif  // SKETCHING_FREQUENT_H_
