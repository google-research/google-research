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

#include <algorithm>
#include <map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "countmin.h"
#include "sketch.h"
#include "utils.h"

namespace sketch {

class Frequent : public Sketch {
 public:
  explicit Frequent(uint heap_size);

  Frequent(const Frequent& other);

  virtual ~Frequent() {}

  virtual void Reset();

  virtual void Add(uint item, float delta);

  virtual float Estimate(uint item) const;

  virtual std::vector<uint> HeavyHitters(float threshold) const;

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
  // map from a weight to the corresponding item. Helps find the smallest item
  // quickly.
  std::multimap<float, uint> weight_to_item_;
  // map from item to an iterator into the weight_to_item_ map.
  absl::flat_hash_map<uint, decltype(weight_to_item_)::const_iterator>
      item_to_weight_;

  uint heap_size_;
  float delete_threshold_ = 0;
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
