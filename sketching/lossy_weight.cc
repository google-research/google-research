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

namespace sketch {

LossyWeight::LossyWeight(uint window_size, uint hash_count, uint hash_size)
    : window_size_(window_size), accumulated_counters_(0),
      cm_(CountMinCU(hash_count, hash_size)) {
  counters_.reserve(window_size * 2);
}

void LossyWeight::Reset() {
  accumulated_counters_ = 0;
  counters_.resize(0);
  cm_.Reset();
}

void LossyWeight::Add(uint item, float delta) {
  counters_.push_back(std::make_pair(item, delta));
  if (counters_.size() >= window_size_ + accumulated_counters_) {
    MergeCounters();
  }
}

float LossyWeight::Estimate(uint item) const {
  const auto& pos = lower_bound(counters_.begin(),
                                counters_.begin() + accumulated_counters_,
                                std::make_pair(item, 0), cmpByItem);
  if (pos != counters_.end() && pos->first == item) return pos->second;
  return cm_.Estimate(item);
}

void LossyWeight::HeavyHitters(float threshold, std::vector<uint>* items) const{
  items->resize(0);
  items->reserve(counters_.size());
  for (const auto& kv : counters_) {
    if (kv.second > threshold) {
      items->push_back(kv.first);
    }
  }
}

unsigned int LossyWeight::Size() const {
  return sizeof(LossyWeight) - sizeof(CountMinCU) + cm_.Size() +
      counters_.capacity() * sizeof(IntFloatPair);
}

bool LossyWeight::Compatible(const Sketch& other_sketch) const {
  const LossyWeight* other = dynamic_cast<const LossyWeight*>(&other_sketch);
  if (other == nullptr) return false;
  if (window_size_ != other->window_size_) return false;
  if (!cm_.Compatible(other->cm_)) return false;
  return true;
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
    sort(counters_.begin(), counters_.end(), cmpByValue);
    for (int i = window_size_; i < counters_.size(); ++i) {
      cm_.Update(counters_[i].first, counters_[i].second);
    }
    counters_.resize(window_size_);
  }
  sort(counters_.begin(), counters_.end(), cmpByItem);
  // Also add any unmerged items in other
  for (j = other.accumulated_counters_; j < other.counters_.size(); ++j) {
    Add(other.counters_[j].first, other.counters_[j].second);
  }
}

void LossyWeight::MergeCounters() {
  if (counters_.size() <= accumulated_counters_) return;  // nothing to merge
  sort(counters_.begin() + accumulated_counters_, counters_.end(), cmpByItem);
  int i = 0;
  int j = accumulated_counters_;
  int m = accumulated_counters_;
  while (i < accumulated_counters_ || j < counters_.size()) {
    if (i < accumulated_counters_ &&
        (j >= counters_.size() || counters_[i].first <= counters_[j].first)) {
      for (; j < counters_.size() && counters_[j].first == counters_[i].first;
           ++j) {
        counters_[i].second += counters_[j].second;
      }
      i++;
    } else {
      counters_[m].first = counters_[j].first;
      counters_[m].second = cm_.Estimate(counters_[m].first) +
                            counters_[j++].second;
      for (; j < counters_.size() && counters_[j].first == counters_[m].first;
           ++j) {
        counters_[m].second += counters_[j].second;
      }
      m++;
    }
  }

  if (m > window_size_) {
    sort(counters_.begin(), counters_.begin() + m, cmpByValue);
    for (i = window_size_; i < m; ++i) {
      cm_.Update(counters_[i].first, counters_[i].second);
    }
    m = window_size_;
  }

  counters_.resize(m);
  sort(counters_.begin(), counters_.end(), cmpByItem);
  accumulated_counters_ = counters_.size();
}

}  // namespace sketch
