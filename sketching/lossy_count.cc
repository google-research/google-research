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

#include "lossy_count.h"
#include <algorithm>

namespace sketch {

LossyCount::LossyCount(uint window_size)
    : window_size_(window_size), epochs_(0) {
  window_.reserve(window_size);
  current_.reserve(window_size * 2);
}

void LossyCount::Reset() {
  window_.resize(0);
  current_.resize(0);
  ResetMissing();
}

void LossyCount::Add(uint item, float delta) {
  window_.push_back(std::make_pair(item, std::max(delta, 1.0f)));
  if (window_.size() >= window_size_) {
    MergeCounters(epochs_ + 1);
    epochs_++;
  }
}

float LossyCount::Estimate(uint item) const {
  const auto& pos = lower_bound(current_.begin(), current_.end(),
                                std::make_pair(item, 0), cmpByItem);
  if (pos != current_.end() && pos->first == item) return pos->second;
  return EstimateMissing(item);
}

void LossyCount::HeavyHitters(float threshold, std::vector<uint>* items) const {
  items->resize(0);
  items->reserve(current_.size());
  for (const auto& kv : current_) {
    if (kv.second > threshold) {
      items->push_back(kv.first);
    }
  }
}

unsigned int LossyCount::Size() const {
  return sizeof(LossyCount) +
      (window_.capacity() + current_.capacity())
      * sizeof(IntFloatPair);
}

bool LossyCount::Compatible(const Sketch& other_sketch) const {
  const LossyCount* other = dynamic_cast<const LossyCount*>(&other_sketch);
  if (other == nullptr) return false;
  if (!CompatibleMissing(*other)) return false;
  if (window_size_ != other->window_size_) return false;
  return true;
}

void LossyCount::Merge(const Sketch& other_sketch) {
  if (!Compatible(other_sketch)) return;
  const LossyCount& other = static_cast<const LossyCount &>(other_sketch);
  std::vector<IntFloatPair> tmp;
  tmp.reserve(current_.capacity());
  int i = 0;
  int j = 0;
  while (i < current_.size() || j < other.current_.size()) {
    IntFloatPair pr;
    if (i < current_.size() && j < other.current_.size()) {
      if (current_[i].first <= other.current_[j].first) {
        pr = current_[i];
        i++;
        if (pr.first == other.current_[j].first) {
          pr.second += other.current_[j].second;
          j++;
        } else {
          pr.second += other.EstimateMissing(pr.first);
        }
      } else {
        pr = other.current_[j];
        pr.second += EstimateMissing(pr.first);
        j++;
      }
    } else if (i < current_.size()) {
      pr = current_[i];
      pr.second += other.EstimateMissing(pr.first);
      i++;
    } else {
      pr = other.current_[j];
      pr.second += EstimateMissing(pr.first);
      j++;
    }
    tmp.push_back(pr);
  }
  current_.swap(tmp);

  MergeMissing(other);
  for (const auto &kv : other.window_) {
    Add(kv.first, kv.second);
  }
}

void LossyCount::MergeCounters(float threshold) {
  if (window_.empty()) return;
  sort(window_.begin(), window_.end(), cmpByItem);
  std::vector<IntFloatPair> tmp;
  tmp.reserve(current_.capacity());
  int i = 0;
  int j = 0;
  std::vector<IntFloatPair> forget;
  while (i < current_.size() || j < window_.size()) {
    IntFloatPair pr;
    if (i < current_.size() &&
        (j >= window_.size() || current_[i].first <= window_[j].first)) {
      pr = current_[i];
      i++;
    } else {
      pr.first = window_[j].first;
      pr.second = EstimateMissing(pr.first);
    }
    for (; j < window_.size() && window_[j].first == pr.first; ++j) {
      pr.second += window_[j].second;
    }
    if (pr.second > threshold) {
      tmp.push_back(pr);
    } else {
      forget.push_back(pr);
    }
  }
  for (const auto& pr : forget) {
    Forget(pr);
  }
  window_.resize(0);
  current_.swap(tmp);
}

}  // namespace sketch
