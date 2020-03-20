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

#ifndef SKETCHING_LOSSY_COUNT_H_
#define SKETCHING_LOSSY_COUNT_H_

// A slightly modified implementation of the Lossy Count implementation
// described in
// "Approximate frequency counts over data streams" by G. S. Manku and
// R. Motwani, in Proceedings of the 28th International Conference on
// Very Large Data Bases, Hong Kong, China, August 2002, section 4.2.
// The paper is available at
//     http://www.vldb.org/conf/2002/S10P03.pdf

// Also implements a version of lossy count with a fallback sketch, to
// improve accuracy dramatically.

#include <vector>

#include "sketch.h"
#include "countmin.h"
#include "utils.h"

namespace sketch {

class LossyCount : public Sketch {
 public:
  explicit LossyCount(uint window_size);

  virtual ~LossyCount() {}

  virtual void Reset();

  // Note that LossyCount only maintains count. However it is ok to
  // update with delta > 1.
  virtual void Add(uint item, float delta);

  virtual void ReadyToEstimate() { MergeCounters(epochs_); }

  virtual float Estimate(uint item) const;

  // LossyCount works best if threshold >= N/window_size, where
  // N is the total number of items added so far.
  virtual std::vector<uint> HeavyHitters(float threshold) const;

  virtual unsigned int Size() const;

  virtual bool Compatible(const Sketch& other_sketch) const;

  virtual void Merge(const Sketch& other_sketch);

 protected:
  uint window_size_;
  uint epochs_ = 0;
  std::vector<IntFloatPair> window_;
  std::vector<IntFloatPair> current_;

  virtual void Forget(const std::vector<IntFloatPair>& forget) {}

  // return estimate for items not found in current
  virtual float EstimateMissing(uint k) const { return epochs_; }

  virtual bool CompatibleMissing(const LossyCount& other) const { return true; }

  // Merge the mechanism for missing values
  virtual void MergeMissing(const LossyCount& other) {
    epochs_ += other.epochs_;
  }

  virtual void ResetMissing() {
    epochs_ = 0;
  }

  // Merge the counts in window_ with current_
  void MergeCounters(float threshold);
};

class LossyCount_Fallback : public LossyCount {
 public:
  LossyCount_Fallback(uint window_size,
                     uint hash_count, uint hash_size) :
      LossyCount(window_size), cm_(CountMinCU(hash_count, hash_size)) {}

  virtual ~LossyCount_Fallback() {}

  unsigned int Size() const override {
    return LossyCount::Size() + cm_.Size() - sizeof(CountMinCU);
  }

  void Forget(const std::vector<IntFloatPair>& forget_pairs) override {
    for (const auto& [item, freq] : forget_pairs) {
      cm_.Update(item, freq);
    }
  }

  float EstimateMissing(uint k) const override {
    return cm_.Estimate(k);
  }

  void ResetMissing() override {
    LossyCount::ResetMissing();
    cm_.Reset();
  }

  bool CompatibleMissing(const LossyCount& other) const override {
    if (!LossyCount::CompatibleMissing(other)) return false;
    const LossyCount_Fallback& other_cast =
        dynamic_cast<const LossyCount_Fallback&>(other);
    return cm_.Compatible(other_cast.cm_);
  }

  void MergeMissing(const LossyCount& other) override {
    LossyCount::MergeMissing(other);
    const LossyCount_Fallback& other_cast =
        dynamic_cast<const LossyCount_Fallback&>(other);
    cm_.Merge(other_cast.cm_);
  }

 private:
  CountMinCU cm_;
};

}  // namespace sketch

#endif  // SKETCHING_LOSSY_COUNT_H_
