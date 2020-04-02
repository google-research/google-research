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

  ~Frequent() override = default;

  void Reset() override;

  void Add(uint item, float delta) override;

  float Estimate(uint item) const override;

  std::vector<uint> HeavyHitters(float threshold) const override;

  uint Size() const override;

  bool Compatible(const Sketch& other) const override;

  void Merge(const Sketch& other) override;

 protected:
  virtual void ResetMissing();

  virtual float EstimateMissing(uint item) const;

  virtual void UpdateMissing(uint item, float value);

  virtual bool CompatibleMissing(const Frequent& other) const;

  virtual void MergeMissing(const Frequent& other);

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

class FrequentFallback : public Frequent {
 public:
  FrequentFallback(uint heap_size, uint hash_count, uint hash_size);

  FrequentFallback(const FrequentFallback& other);

  uint Size() const override;

 protected:
  void ResetMissing() override;

  float EstimateMissing(uint item) const override;

  void UpdateMissing(uint item, float value) override;

  bool CompatibleMissing(const Frequent& other) const override;

  void MergeMissing(const Frequent& other) override;

 private:
    CountMinCU cm_;
};

}  // namespace sketch

#endif  // SKETCHING_FREQUENT_H_
