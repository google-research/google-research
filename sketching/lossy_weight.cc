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

#include "lossy_weight.h"

#include <algorithm>

#include "sketch.h"
#include "utils.h"

namespace sketch {

LossyWeight::LossyWeight(uint window_size, uint hash_count, uint hash_size)
    : window_size_(window_size), cm_(CountMinCU(hash_count, hash_size)) {
  counters_.reserve(window_size * 2);
}

void LossyWeight::Reset() {
  accumulated_counters_ = 0;
  counters_.clear();
  cm_.Reset();
}

void LossyWeight::Add(uint item, float delta) {
  counters_.emplace_back(item, delta);
  if (counters_.size() >= window_size_ + accumulated_counters_) {
    MergeCounters();
  }
}

void LossyWeight::ReadyToEstimate() { MergeCounters(); }

float LossyWeight::Estimate(uint item) const {
  const auto& pos = std::lower_bound(counters_.begin(),
                                     counters_.begin() + accumulated_counters_,
                                     std::make_pair(item, 0), cmpByItem);
  return (pos != counters_.end() && pos->first == item) ? pos->second
                                                        : cm_.Estimate(item);
}

std::vector<uint> LossyWeight::HeavyHitters(float threshold) const {
  return FilterOutAboveThreshold(counters_, threshold);
}

unsigned int LossyWeight::Size() const {
  return sizeof(LossyWeight) - sizeof(CountMinCU) + cm_.Size() +
      counters_.capacity() * sizeof(IntFloatPair);
}

bool LossyWeight::Compatible(const Sketch& other_sketch) const {
  const LossyWeight* other = dynamic_cast<const LossyWeight*>(&other_sketch);
  if (other == nullptr) return false;
  if (window_size_ != other->window_size_) return false;
  return cm_.Compatible(other->cm_);
}

void LossyWeight::Merge(const Sketch& other_sketch) {
  if (!Compatible(other_sketch)) return;
  const LossyWeight& other = static_cast<const LossyWeight &>(other_sketch);
  MergeCounters();
  int i = 0;
  int j = 0;
  while (i < accumulated_counters_ || j < other.accumulated_counters_) {
    if (i < accumulated_counters_ && j < other.accumulated_counters_) {
      if (counters_[i].first == other.counters_[j].first) {
        counters_[i].second += other.counters_[j].second;
        i++; j++;
      } else if (counters_[i].first < other.counters_[j].first) {
        counters_[i].second += other.cm_.Estimate(counters_[i].first);
        i++;
      } else {
        counters_.push_back(other.counters_[j++]);
        counters_.back().second += cm_.Estimate(counters_.back().first);
      }
    } else if (i < accumulated_counters_) {
      counters_[i].second += other.cm_.Estimate(counters_[i].first);
      i++;
    } else {
      counters_.push_back(other.counters_[j++]);
      counters_.back().second += cm_.Estimate(counters_.back().first);
    }
  }
  cm_.Merge(other.cm_);
  if (counters_.size() > window_size_) {
    std::sort(counters_.begin(), counters_.end(), cmpByValue);
    for (int i = window_size_; i < counters_.size(); ++i) {
      cm_.Update(counters_[i].first, counters_[i].second);
    }
    counters_.resize(window_size_);
  }
  std::sort(counters_.begin(), counters_.end(), cmpByItem);
  // Also add any unmerged items in other
  for (j = other.accumulated_counters_; j < other.counters_.size(); ++j) {
    Add(other.counters_[j].first, other.counters_[j].second);
  }
}

void LossyWeight::MergeCounters() {
  if (counters_.size() <= accumulated_counters_) return;  // nothing to merge
  std::sort(counters_.begin() + accumulated_counters_, counters_.end(),
            cmpByItem);
  int m = accumulated_counters_;
  for (int i = 0, j = accumulated_counters_;
       i < accumulated_counters_ || j < counters_.size();) {
    int target_idx;
    if (i < accumulated_counters_ &&
        (j >= counters_.size() || counters_[i].first <= counters_[j].first)) {
      target_idx = i++;
    } else {
      counters_[m].first = counters_[j].first;
      counters_[m].second = cm_.Estimate(counters_[m].first) +
                            counters_[j++].second;
      target_idx = m++;
    }
    while (j < counters_.size() &&
           counters_[j].first == counters_[target_idx].first) {
      counters_[target_idx].second += counters_[j++].second;
    }
  }

  if (m > window_size_) {
    static auto comp = [](const IntFloatPair& a, const IntFloatPair& b) {
      if (a.second != b.second) {
        return a.second > b.second;
      }
      return a.first < b.first;
    };

    std::nth_element(
        counters_.begin(),
        counters_.begin() + window_size_ - 1, counters_.begin() + m, comp);
    for (int i = window_size_; i < counters_.size(); ++i) {
      cm_.Update(counters_[i].first, counters_[i].second);
    }
    m = window_size_;
  }

  counters_.resize(m);
  std::sort(counters_.begin(), counters_.end(), cmpByItem);
  accumulated_counters_ = counters_.size();
}

}  // namespace sketch
