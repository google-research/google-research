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

#include "laminar_matroid.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

LaminarMatroid::LaminarMatroid(
    const absl::flat_hash_map<int, std::vector<int>>& groups_map,
    const std::vector<int>& ks)
    : groups_map_(groups_map),
      ks_(ks),
      num_groups_(ks.size()),
      current_grpcards_(num_groups_, 0) {
  // Assume groups are laminar.
}

void LaminarMatroid::Reset() {
  std::fill(current_grpcards_.begin(), current_grpcards_.end(), 0);
  current_set_.clear();
}

bool LaminarMatroid::CanAdd(int element) const {
  assert(!current_set_.count(element));
  const std::vector<int>& elt_groups = groups_map_.at(element);
  for (int elt_group : elt_groups) {
    if (current_grpcards_[elt_group] + 1 > ks_[elt_group]) return false;
  }
  return true;
}

bool LaminarMatroid::CanSwap(int element, int swap) const {
  assert(!current_set_.count(element));
  assert(current_set_.count(swap));
  const std::vector<int>& elt_groups = groups_map_.at(element);
  const std::vector<int>& swap_groups = groups_map_.at(swap);
  for (int elt_group : elt_groups) {
    if (current_grpcards_[elt_group] + 1 > ks_[elt_group] &&
        std::find(swap_groups.begin(), swap_groups.end(), elt_group) ==
            swap_groups.end()) {
      return false;
    }
  }
  return true;
}

void LaminarMatroid::Add(int element) {
  assert(!current_set_.count(element));
  std::vector<int> elt_groups = groups_map_.at(element);
  for (int elt_group : elt_groups) {
    current_grpcards_[elt_group]++;
  }
  current_set_.insert(element);
}

void LaminarMatroid::Remove(int element) {
  assert(current_set_.count(element));
  std::vector<int> elt_groups = groups_map_.at(element);
  for (int elt_group : elt_groups) {
    current_grpcards_[elt_group]--;
  }
  current_set_.erase(element);
}

bool LaminarMatroid::IsFeasible(const std::vector<int>& elements) const {
  std::vector<int> grpcards = std::vector<int>(num_groups_, 0);
  std::vector<int> elt_groups;
  for (int elt : elements) {
    elt_groups = groups_map_.at(elt);
    for (int elt_group : elt_groups) {
      grpcards[elt_group]++;
      if (grpcards[elt_group] > ks_[elt_group]) {
        return false;
      }
    }
  }
  return true;
}

bool LaminarMatroid::CurrentIsFeasible() const {
  for (int i = 0; i < num_groups_; ++i) {
    if (current_grpcards_[i] > ks_[i]) {
      return false;
    }
  }
  return true;
}

std::vector<int> LaminarMatroid::GetCurrent() const {
  return std::vector<int>(current_set_.begin(), current_set_.end());
}

bool LaminarMatroid::InCurrent(int element) const {
  return current_set_.count(element);
}

std::unique_ptr<Matroid> LaminarMatroid::Clone() const {
  return std::make_unique<LaminarMatroid>(*this);
}
