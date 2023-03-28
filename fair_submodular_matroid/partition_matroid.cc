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

#include "partition_matroid.h"

#include <cassert>
#include <memory>
#include <set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "matroid.h"

PartitionMatroid::PartitionMatroid(
    const absl::flat_hash_map<int, int>& groups_map, const std::vector<int>& ks)
    : groups_map_(groups_map),
      ks_(ks),
      num_groups_(ks.size()),
      current_grpcards_(num_groups_, 0) {}

void PartitionMatroid::Reset() {
  for (int i = 0; i < current_grpcards_.size(); ++i) {
    current_grpcards_[i] = 0;
  }
  current_set_.clear();
}

bool PartitionMatroid::CanAdd(int element) const {
  assert(!current_set_.count(element));
  int elt_group = groups_map_.at(element);
  return current_grpcards_[elt_group] + 1 <= ks_[elt_group];
}

bool PartitionMatroid::CanSwap(int element, int swap) const {
  assert(!current_set_.count(element));
  assert(current_set_.count(swap));
  return CanAdd(element) || groups_map_.at(element) == groups_map_.at(swap);
}

std::vector<int> PartitionMatroid::GetAllSwaps(int element) const {
  if (CanAdd(element)) {
    return GetCurrent();
  }
  const int group = groups_map_.at(element);
  std::vector<int> all_swaps;
  for (int swap : current_set_) {
    if (groups_map_.at(swap) == group) {
      all_swaps.push_back(swap);
    }
  }
  return all_swaps;
}

void PartitionMatroid::Add(int element) {
  assert(!current_set_.count(element));
  int elt_group = groups_map_.at(element);
  current_grpcards_[elt_group]++;
  current_set_.insert(element);
}

void PartitionMatroid::Remove(int element) {
  assert(current_set_.count(element));
  int elt_group = groups_map_.at(element);
  current_grpcards_[elt_group]--;
  current_set_.erase(element);
}

bool PartitionMatroid::IsFeasible(const std::vector<int>& elements) const {
  std::vector<int> grpcards = std::vector<int>(num_groups_, 0);
  int elt_group;
  for (int elt : elements) {
    elt_group = groups_map_.at(elt);
    grpcards[elt_group]++;
    if (grpcards[elt_group] > ks_[elt_group]) return false;
  }
  return true;
}

bool PartitionMatroid::CurrentIsFeasible() const {
  for (int i = 0; i < num_groups_; ++i) {
    if (current_grpcards_[i] > ks_[i]) {
      return false;
    }
  }
  return true;
}

std::vector<int> PartitionMatroid::GetCurrent() const {
  return std::vector<int>(current_set_.begin(), current_set_.end());
}

bool PartitionMatroid::InCurrent(int element) const {
  return current_set_.count(element);
}

std::unique_ptr<Matroid> PartitionMatroid::Clone() const {
  return std::make_unique<PartitionMatroid>(*this);
}
