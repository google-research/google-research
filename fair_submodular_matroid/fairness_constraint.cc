// Copyright 2023 The Authors.
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

#include "fairness_constraint.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "matroid.h"
#include "partition_matroid.h"

FairnessConstraint::FairnessConstraint(
    const absl::flat_hash_map<int, int>& colors_map,
    const std::vector<std::pair<int, int>>& bounds) {
  colors_map_ = colors_map;
  bounds_ = bounds;
  ncolors_ = bounds.size();
  current_colorcounts_ = std::vector<int>(ncolors_, 0);
}

void FairnessConstraint::Reset() {
  std::fill(current_colorcounts_.begin(), current_colorcounts_.end(), 0);
  current_set_.clear();
}

bool FairnessConstraint::CanAdd(int element) const {
  assert(!current_set_.count(element));
  int elt_color = colors_map_.at(element);
  return current_colorcounts_[elt_color] + 1 <= bounds_[elt_color].second;
}

void FairnessConstraint::Add(int element) {
  assert(!current_set_.count(element));
  int elt_color = colors_map_.at(element);
  current_colorcounts_[elt_color]++;
  current_set_.insert(element);
}

bool FairnessConstraint::CanRemove(int element) const {
  assert(current_set_.count(element));
  int elt_color = colors_map_.at(element);
  return current_colorcounts_[elt_color] - 1 >= bounds_[elt_color].first;
}

void FairnessConstraint::Remove(int element) {
  assert(current_set_.count(element));
  int elt_color = colors_map_.at(element);
  current_colorcounts_[elt_color]--;
  current_set_.erase(element);
}

int FairnessConstraint::GetColor(int element) const {
  return colors_map_.at(element);
}

int FairnessConstraint::GetColorNum() const { return bounds_.size(); }

std::vector<std::pair<int, int>> FairnessConstraint::GetBounds() const {
  return bounds_;
}

bool FairnessConstraint::IsFeasible(const std::vector<int> elements) {
  std::vector<int> colorcounts = std::vector<int>(ncolors_, 0);
  int elt_color;
  for (int elt : elements) {
    elt_color = colors_map_.at(elt);
    colorcounts[elt_color]++;
    if (colorcounts[elt_color] > bounds_[elt_color].second) return false;
  }
  for (int color = 0; color < ncolors_; color++) {
    if (colorcounts[color] < bounds_[color].first) return false;
  }
  return true;
}

std::unique_ptr<Matroid> FairnessConstraint::LowerBoundsToMatroid() const {
  std::vector<int> ks;
  ks.reserve(bounds_.size());
  for (const std::pair<int, int>& bound : bounds_) {
    ks.push_back(bound.first);
  }
  return std::make_unique<PartitionMatroid>(colors_map_, ks);
}

std::unique_ptr<Matroid> FairnessConstraint::UpperBoundsToMatroid() const {
  std::vector<int> ks;
  ks.reserve(bounds_.size());
  for (const std::pair<int, int>& bound : bounds_) {
    ks.push_back(bound.second);
  }
  return std::make_unique<PartitionMatroid>(colors_map_, ks);
}

std::unique_ptr<FairnessConstraint> FairnessConstraint::Clone() const {
  return std::make_unique<FairnessConstraint>(*this);
}
