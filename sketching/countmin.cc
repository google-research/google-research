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

#include "countmin.h"

#include <cmath>

#include "utils.h"
#include "absl/random/random.h"

namespace sketch {

CountMin::CountMin(uint hash_count, uint hash_size)
    : hash_size_(hash_size), max_item_(0) {
  hash_a_.resize(hash_count);
  hash_b_.resize(hash_count);
  values_.resize(hash_count);

  BitGenerator bit_gen;
  absl::BitGenRef& generator = *bit_gen.BitGen();
  absl::uniform_int_distribution<int> rand_int;
  for (int i = 0; i < hash_count; ++i) {
    values_[i].resize(hash_size);
    hash_a_[i] = rand_int(generator);
    hash_b_[i] = rand_int(generator);
  }
}

void CountMin::Reset() {
  max_item_ = 0;
  for (int i = 0; i < values_.size(); ++i) {
    values_[i].resize(0);
    values_[i].resize(hash_size_, 0);
  }
}

void CountMin::Add(uint item, float delta) {
  max_item_ = std::max(item, max_item_);
  for (int i = 0; i < values_.size(); ++i) {
    values_[i][Hash(hash_a_[i], hash_b_[i], item, hash_size_)] += delta;
  }
}

float CountMin::Estimate(uint item) const {
  float result = values_[0][Hash(hash_a_[0], hash_b_[0], item, hash_size_)];
  for (int i = 1; i < values_.size(); ++i) {
    result = std::min(
        result, values_[i][Hash(hash_a_[i], hash_b_[i], item, hash_size_)]);
  }
  return result;
}

std::vector<uint> CountMin::HeavyHitters(float threshold) const {
  std::vector<uint> items(max_item_);
  for (uint i = 0; i <= max_item_; ++i) {
    if (Estimate(i) > threshold) {
      items.push_back(i);
    }
  }
  return items;
}

uint CountMin::Size() const {
  return sizeof(CountMin) +
      (hash_b_.capacity() + hash_a_.capacity()) * sizeof(uint) +
      values_.capacity() * (sizeof(values_[0]) +
                            values_[0].capacity() * sizeof(float));
}

bool CountMin::Compatible(const Sketch& other_sketch) const {
  const CountMin* other = dynamic_cast<const CountMin*>(&other_sketch);
  if (other == nullptr) return false;
  if (hash_size_ != other->hash_size_) return false;
  if (hash_a_.size() != other->hash_a_.size()) return false;
  for (int i = 0; i < hash_a_.size(); ++i) {
    if (hash_a_[i] != other->hash_a_[i]) return false;
    if (hash_b_[i] != other->hash_b_[i]) return false;
  }
  return true;
}

void CountMin::Merge(const Sketch& other_sketch) {
  if (!Compatible(other_sketch)) return;
  const CountMin& other = static_cast<const CountMin &>(other_sketch);
  max_item_ = std::max(other.max_item_, max_item_);
  for (int i = 0; i < values_.size(); ++i) {
    for (int j = 0; j < values_[i].size(); ++j) {
      values_[i][j] += other.values_[i][j];
    }
  }
}

void CountMinCU::Add(uint item, float delta) {
  Update(item, Estimate(item) + delta);
}

void CountMinCU::BatchAdd(const std::vector<IntFloatPair>& item_deltas) {
  std::vector<IntFloatPair> updates;
  updates.reserve(item_deltas.size());
  for (const auto & item_delta : item_deltas) {
    updates.push_back(std::make_pair(
        item_delta.first, Estimate(item_delta.first) + item_delta.second));
  }
  for (const auto &update : updates) {
    Update(update.first, update.second);
  }
}

void CountMinCU::Update(uint item, float value) {
  max_item_ = std::max(item, max_item_);
  for (int i = 0; i < values_.size(); ++i) {
    uint ind = Hash(hash_a_[i], hash_b_[i], item, hash_size_);
    values_[i][ind] = std::max(values_[i][ind], value);
  }
}

CountMinHierarchical::CountMinHierarchical(const CountMinHierarchical& other) {
  lgN_ = other.lgN_;
  granularity_ = other.granularity_;
  total_ = other.total_;
  levels_ = other.levels_;
  exact_counts_ = other.exact_counts_;

  sketches_.reserve(other.sketches_.size());
  for (int j = 0; j < other.sketches_.size(); ++j) {
    sketches_.push_back(other.sketches_[j]->CreateCopy());
  }
}

void CountMinHierarchical::Initialize(
    uint hash_count, uint hash_size, uint lgN, uint granularity,
    std::unique_ptr<CountMin> (*CreateSketch)(uint, uint)) {
  lgN_ = lgN;
  granularity_ = granularity;
  total_ = 0;
  levels_ = static_cast<int>(ceil(static_cast<float>(lgN) /
                                  static_cast<float>(granularity)));

  uint exact_count_size = floor(log2(hash_count * hash_size));
  exact_counts_.resize(exact_count_size);
  int j = 1;
  for (int i = exact_counts_.size() - 1; i >= 0; --i) {
    exact_counts_[i].resize(1 << (granularity * j));
    j++;
  }
  sketches_.reserve(levels_ - exact_count_size);
  for (int j = exact_count_size; j < levels_; ++j) {
    sketches_.push_back((*CreateSketch)(hash_count, hash_size));
  }
}

void CountMinHierarchical::Reset() {
  total_ = 0;
  for (int i = 0; i < exact_counts_.size(); ++i) {
    for (int j = 0; j < exact_counts_[i].size(); ++j) {
      exact_counts_[i][j] = 0;
    }
  }
  for (int i = 0; i < sketches_.size(); ++i) {
    sketches_[i]->Reset();
  }
}


void CountMinHierarchical::Add(uint item, float delta) {
  total_ += delta;
  for (int i = 0; i < levels_; ++i) {
    if (i >= sketches_.size()) {
      exact_counts_[i - sketches_.size()][item] += delta;
    } else {
      sketches_[i]->Add(item, delta);
    }
    item >>= granularity_;
  }
}

uint CountMinHierarchical::Size() const {
  uint exact_count_space = 0;
  for (int i = 0; i < exact_counts_.size(); ++i) {
    exact_count_space += exact_counts_[i].capacity();
  }
  return sizeof(CountMinHierarchical) +
      exact_counts_.capacity() * sizeof(exact_counts_[0]) +
      exact_count_space * sizeof(float) +
      sketches_.capacity() * (sizeof(sketches_[0]) + sketches_[0]->Size());
}

bool CountMinHierarchical::Compatible(const Sketch& other_sketch) const {
  const CountMinHierarchical* other =
      dynamic_cast<const CountMinHierarchical*>(&other_sketch);
  if (other == nullptr) return false;
  if (lgN_ != other->lgN_) return false;
  if (levels_ != other->levels_) return false;
  if (granularity_ != other->granularity_) return false;
  if (sketches_.size() != other->sketches_.size()) return false;
  for (int i = 0; i < sketches_.size(); ++i) {
    if (!sketches_[i]->Compatible(*(other->sketches_[i]))) return false;
  }
  return true;
}

void CountMinHierarchical::Merge(const Sketch& other_sketch) {
  if (!Compatible(other_sketch)) return;
  const CountMinHierarchical& other =
      static_cast<const CountMinHierarchical &>(other_sketch);
  total_ += other.total_;
  for (int i = 0; i < exact_counts_.size(); ++i) {
    for (int j = 0; j < exact_counts_[i].size(); ++j) {
      exact_counts_[i][j] += other.exact_counts_[i][j];
    }
  }
  for (int i = 0; i < sketches_.size(); ++i) {
    sketches_[i]->Merge(*other.sketches_[i]);
  }
}

float CountMinHierarchical::EstimateAtDepth(int depth, uint item) const {
  if (depth >= levels_) return total_;
  if (depth >= sketches_.size())
    return exact_counts_[depth - sketches_.size()][item];
  return sketches_[depth]->Estimate(item);
}

void CountMinHierarchical::HeavyHittersRecursive(
    uint depth, uint start, float threshold, std::vector<uint>* items) const {
  if (EstimateAtDepth(depth, start) <= threshold) return;
  if (depth == 0) {
    items->push_back(start);
    return;
  }
  uint blocksize = 1 << granularity_;
  uint item_shifted = start << granularity_;
  for (int i = 0; i < blocksize; ++i) {
    HeavyHittersRecursive(depth - 1, item_shifted + i, threshold, items);
  }
}

float CountMinHierarchical::RangeSum(uint start, uint end) const {
  end = std::min(1u << lgN_, end);
  end += 1;  // adjusting for end effects
  float sum = 0.0f;
  for (int depth = 0; depth <= levels_; ++depth) {
    if (start == end) break;
    if (end - start + 1 < (1 << granularity_)) {
      // at the highest level, avoid overcounting
      for (int i = start; i < end; ++i) {
        sum += EstimateAtDepth(depth, i);
      }
      break;
    } else {
      // figure out what needs to be done at each end
      uint leftend = (((start >> granularity_) + 1) << granularity_) - start;
      uint rightend = end - ((end >> granularity_) << granularity_);
      if (leftend > 0 && start < end) {
        for (int i = 0; i < leftend; ++i) {
          sum += EstimateAtDepth(depth, start + i);
        }
      }
      if (rightend > 0 && start < end) {
        for (int i = 0; i < rightend; ++i) {
          sum += EstimateAtDepth(depth, end - i - 1);
        }
      }
      start = (start >> granularity_) + (leftend > 0 ? 1 : 0);
      end >>= granularity_;
    }
  }
  return sum;
}

uint CountMinHierarchical::FindRange(float sum, bool below) const {
  uint top = 1 << lgN_;
  uint low = 0;
  uint high = top;
  while (low < high) {
    uint mid = (low + high) / 2;
    float est = (below ? RangeSum(0, mid) : RangeSum(mid, top));
    if ((below && est < sum) || (!below && est > sum)) {
      high = mid;
    } else {
      low = mid;
    }
  }
  return low;
}

uint CountMinHierarchical::Quantile(float frac) const {
  if (frac <= 0) return 0;
  if (frac >= 1) return 1 << lgN_;
  return (FindRange(total_ * frac, true) +
          FindRange(total_ * (1 - frac), false)) / 2;
}

}  // namespace sketch
