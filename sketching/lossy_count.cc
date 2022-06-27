// Copyright 2022 The Google Research Authors.
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
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

#include "sketch.h"
#include "utils.h"

namespace sketch {
namespace {

std::vector<IntFloatPair> MergeConsecutiveEqualItems(
    const std::vector<IntFloatPair>& entries) {
  std::vector<IntFloatPair> result;
  for (const auto& entry : entries) {
    if (!result.empty() && entry.first == result.back().first) {
      result.back().second += entry.second;
    } else {
      result.push_back(entry);
    }
  }
  return result;
}

}  // namespace

LossyCount::LossyCount(uint window_size) : window_size_(window_size) {
  window_.reserve(window_size);
  current_.reserve(window_size);
}

LossyCount::LossyCount(const LossyCount& lc) : window_size_(lc.window_size_) {
  window_.reserve(window_size_);
  current_.reserve(window_size_);
  Merge(lc);
}

void LossyCount::Reset() {
  window_.clear();
  current_.clear();
  ResetMissing();
}

void LossyCount::Add(uint item, float delta) {
  window_.emplace_back(item, std::max(delta, 1.0f));
  if (window_.size() >= window_size_) {
    MergeCounters(epochs_ + 1);
    ++epochs_;
  }
}

void LossyCount::ReadyToEstimate() { MergeCounters(epochs_); }

float LossyCount::Estimate(uint item) const {
  const auto& pos = lower_bound(current_.begin(), current_.end(),
                                std::make_pair(item, 0), cmpByItem);
  if (pos != current_.end() && pos->first == item) return pos->second;
  return EstimateMissing(item);
}

std::vector<uint> LossyCount::HeavyHitters(float threshold) const {
  return FilterOutAboveThreshold(current_, threshold);
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

  const LossyCount& other = static_cast<const LossyCount&>(other_sketch);
  std::vector<IntFloatPair> next_current;
  next_current.reserve(current_.capacity());

  int i = 0;
  int j = 0;
  while (i < current_.size() && j < other.current_.size()) {
    const uint this_item = current_[i].first;
    const uint other_item = other.current_[j].first;
    IntFloatPair min_pair;
    if (this_item < other_item) {
      min_pair = std::move(current_[i++]);
      min_pair.second += other.EstimateMissing(min_pair.first);
    } else if (this_item > other_item) {
      min_pair = other.current_[j++];
      min_pair.second += Estimate(min_pair.first);
    } else {  // this_item == other_item
      min_pair = {this_item, current_[i++].second + other.current_[j++].second};
    }
    next_current.push_back(std::move(min_pair));
  }
  while (i < current_.size()) {
    next_current.emplace_back(
        current_[i].first,
        current_[i].second + other.EstimateMissing(current_[i].first));
    ++i;
  }
  while (j < other.current_.size()) {
    next_current.emplace_back(
        other.current_[j].first,
        other.current_[j].second + EstimateMissing(other.current_[j].first));
    ++j;
  }
  current_ = std::move(next_current);

  MergeMissing(other);
  for (const auto &[item, freq] : other.window_) {
    Add(item, freq);
  }
}

void LossyCount::Forget(const std::vector<IntFloatPair>& forget) {}

float LossyCount::EstimateMissing(uint k) const { return epochs_; }

bool LossyCount::CompatibleMissing(const LossyCount& other) const {
  return true;
}

void LossyCount::MergeMissing(const LossyCount& other) {
  epochs_ += other.epochs_;
}

void LossyCount::ResetMissing() { epochs_ = 0; }

void LossyCount::MergeCounters(float threshold) {
  // Early return if window_ is empty as we don't want to clear current_.
  if (window_.empty()) {
    return;
  }
  sort(window_.begin(), window_.end(), cmpByItem);
  std::vector<IntFloatPair> merged_window =
      MergeConsecutiveEqualItems(window_);

  std::vector<IntFloatPair> next_current;
  next_current.reserve(current_.capacity());
  std::vector<IntFloatPair> forget_pairs;
  for (int i = 0, j = 0; i < current_.size() || j < merged_window.size(); ) {
    const uint current_item = i < current_.size()
                                  ? current_[i].first
                                  : std::numeric_limits<uint>::max();
    const uint window_item = j < merged_window.size()
                                 ? merged_window[j].first
                                 : std::numeric_limits<uint>::max();
    IntFloatPair min_pair;
    if (current_item < window_item) {
      min_pair = current_[i++];
    } else if (current_item > window_item) {
      min_pair = {window_item,
                  EstimateMissing(window_item) + merged_window[j].second};
      ++j;
    } else {  // current_item == window_item
      min_pair = {current_item,
                  current_[i++].second + merged_window[j++].second};
    }

    if (min_pair.second > threshold) {
      next_current.push_back(std::move(min_pair));
    } else {
      forget_pairs.push_back(std::move(min_pair));
    }
  }

  Forget(forget_pairs);
  window_.clear();
  current_ = std::move(next_current);
}

LossyCountFallback::LossyCountFallback(uint window_size, uint hash_count,
                                       uint hash_size)
    : LossyCount(window_size), cm_(CountMinCU(hash_count, hash_size)) {}

float LossyCountFallback::EstimateMissing(uint k) const {
  return cm_.Estimate(k);
}

void LossyCountFallback::MergeMissing(const LossyCount& other) {
  LossyCount::MergeMissing(other);
  const LossyCountFallback& other_cast =
      dynamic_cast<const LossyCountFallback&>(other);
  cm_.Merge(other_cast.cm_);
}

bool LossyCountFallback::CompatibleMissing(const LossyCount& other) const {
  if (!LossyCount::CompatibleMissing(other)) return false;
  const LossyCountFallback& other_cast =
      dynamic_cast<const LossyCountFallback&>(other);
  return cm_.Compatible(other_cast.cm_);
}

void LossyCountFallback::ResetMissing() {
  LossyCount::ResetMissing();
  cm_.Reset();
}

void LossyCountFallback::Forget(const std::vector<IntFloatPair>& forget_pairs) {
  for (const auto& [item, freq] : forget_pairs) {
    cm_.Update(item, freq);
  }
}

unsigned int LossyCountFallback::Size() const {
  return LossyCount::Size() + cm_.Size();
}

}  // namespace sketch
