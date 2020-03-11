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

#ifndef SKETCHING_COUNTMIN_H_
#define SKETCHING_COUNTMIN_H_

// Implements the following algorithms:
// 1. Countmin sketch
//    Cormode and Muthukrishnan. "An Improved Data Stream Summary:
//    The Count-Min Sketch and its Applications". Journal of Algorithms,
//    Volume 55, Number 1, pages 58-75, 2005.
// 2. Hierarchical Countmin (good for finding percentiles and heavy hitters)
//    Cheung-Mon-Chan and Clerot. "Finding Hierarchical Heavy Hitters with
//    the Count Min Sketch". 4th International Workshop on Internet Performance,
//    Simulation, Monitoring and Measurements, 2006.
// 3. CountMinCU, a CountMin with Conservative Update
//    Estan and Varghese. "New Directions in Traffic Measurement and
//    Accounting: Focusing on the Elephants, Ignoring the Mice".
//    ACM Transactions on Computer Systems, Volume 21 Issue 3, August 2003
//    Pages 270-313.
// 4. Hierarchical Countmin with Conservative Update.

#include <vector>

#include "sketch.h"
#include "absl/memory/memory.h"

namespace sketch {

class CountMin : public Sketch {
 public:
  CountMin(uint hash_count, uint hash_size);

  virtual ~CountMin() {}

  virtual void Reset();

  virtual void Add(uint item, float delta);

  virtual float Estimate(uint item) const;

  virtual std::vector<uint> HeavyHitters(float threshold) const;

  virtual uint Size() const;

  virtual bool Compatible(const Sketch& other) const;

  virtual void Merge(const Sketch& other);

  static std::unique_ptr<CountMin> CreateCM(uint hash_count, uint hash_size) {
    return absl::make_unique<CountMin>(CountMin(hash_count, hash_size));
  }

  virtual std::unique_ptr<CountMin> CreateCopy() const {
    return absl::WrapUnique<CountMin>(new CountMin(*this));
  }

 protected:
  const uint hash_size_;
  uint max_item_;
  std::vector<uint> hash_a_;
  std::vector<uint> hash_b_;
  std::vector<std::vector<float> > values_;
};


class CountMinCU : public CountMin {
 public:
  CountMinCU(uint hash_count, uint hash_size) :
      CountMin(hash_count, hash_size) {}

  virtual ~CountMinCU() {}

  void Add(uint item, float delta) override;

  // Add a batch of item_delta pairs. We assume no duplicates, so the items
  // are unique.
  virtual void BatchAdd(const std::vector<IntFloatPair>& item_deltas);

  // Ensure that the value of item will be (at least) value
  void Update(uint item, float value);

  static std::unique_ptr<CountMin> CreateCM_CU(uint hash_count,
                                                uint hash_size) {
    return absl::make_unique<CountMinCU>(CountMinCU(hash_count, hash_size));
  }

  std::unique_ptr<CountMin> CreateCopy() const override {
    return absl::WrapUnique<CountMinCU>(new CountMinCU(*this));
  }
};

class CountMinHierarchical : public Sketch {
 public:
  CountMinHierarchical(uint hash_count, uint hash_size, uint lgN,
                       uint granularity = 1) {
    Initialize(hash_count, hash_size, lgN, granularity, &CountMin::CreateCM);
  }

  CountMinHierarchical(uint hash_count, uint hash_size, uint lgN,
                       uint granularity,
                       std::unique_ptr<CountMin> (*CreateSketch)(uint, uint)) {
    Initialize(hash_count, hash_size, lgN, granularity, CreateSketch);
  }

  CountMinHierarchical(const CountMinHierarchical& other);

  virtual ~CountMinHierarchical() {}

  virtual void Reset();

  virtual void Add(uint item, float delta);

  virtual float Estimate(uint item) const {
    return sketches_[0]->Estimate(item);
  }

  virtual std::vector<uint> HeavyHitters(float threshold) const {
    std::vector<uint> items;
    HeavyHittersRecursive(levels_, 0, threshold, &items);
    return items;
  }

  virtual uint Size() const;

  virtual bool Compatible(const Sketch& other_sketch) const;

  virtual void Merge(const Sketch& other_sketch);

  // Find the sum of the weights of elements from start to end.
  float RangeSum(uint start, uint end) const;

  // Find the element such that "frac" of the total weight is below it.
  uint Quantile(float frac) const;

 protected:
  uint lgN_;
  uint levels_;
  uint granularity_;
  float total_;
  std::vector<std::vector<float> > exact_counts_;
  std::vector<std::unique_ptr<CountMin> > sketches_;

  void Initialize(uint hash_count, uint hash_size, uint lgN, uint granularity,
                  std::unique_ptr<CountMin> (*CreateSketch)(uint, uint));

  // Similar to estimate, but only at the given depth.
  float EstimateAtDepth(int depth, uint item) const;

  // Auxiliary recursive function to compute heavy hitters.
  void HeavyHittersRecursive(uint depth, uint start, float threshold,
                             std::vector<uint>* items) const;

  // Find using binary search, the element such that the total weights of
  // the elements below it are ~= sum (if below is true), otherwise
  // the element such that the total weights of the elements above ~= sum.
  uint FindRange(float sum, bool below) const;
};


class CountMinHierarchicalCU : public CountMinHierarchical {
 public:
  CountMinHierarchicalCU(uint hash_count, uint hash_size, uint lgN,
                          uint granularity = 1) :
      CountMinHierarchical(hash_count, hash_size, lgN, granularity,
                           &CountMinCU::CreateCM_CU) {}

  virtual ~CountMinHierarchicalCU() {}
};

}  // namespace sketch

#endif  // SKETCHING_COUNTMIN_H_
