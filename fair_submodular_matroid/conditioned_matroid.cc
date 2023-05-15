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

#include "conditioned_matroid.h"

#include <cassert>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"

ConditionedMatroid::ConditionedMatroid(const Matroid& original,
                                       const std::vector<int>& S)
    : s_(S.begin(), S.end()), original_(original.Clone()) {
  Reset();
}

// Reset to empty set.
void ConditionedMatroid::Reset() {
  original_->Reset();
  for (int el : s_) {
    original_->Add(el);
  }
  current_elements_.clear();
}

// Return whether adding an element would be feasible.
bool ConditionedMatroid::CanAdd(int element) const {
  assert(!current_elements_.count(element));
  if (s_.count(element)) {
    return true;
  }
  return original_->CanAdd(element);
}

// Return whether add element while removing anothe one would be feasible.
bool ConditionedMatroid::CanSwap(int element, int swap) const {
  assert(!current_elements_.count(element));
  assert(current_elements_.count(swap));
  if (s_.count(element)) {
    return true;
  }
  if (s_.count(swap)) {
    return original_->CanAdd(element);
  } else {
    return original_->CanSwap(element, swap);
  }
}

// Add an element. Assumes that the element can be added.
void ConditionedMatroid::Add(int element) {
  assert(!current_elements_.count(element));
  current_elements_.insert(element);
  if (!s_.count(element)) {
    original_->Add(element);
  }
}

// Removes the element.
void ConditionedMatroid::Remove(int element) {
  assert(current_elements_.count(element));
  current_elements_.erase(element);
  if (!s_.count(element)) {
    original_->Remove(element);
  }
}

// Checks if a set is feasible.
bool ConditionedMatroid::IsFeasible(const std::vector<int>& elements) const {
  std::vector<int> elements_plus_S(s_.begin(), s_.end());
  for (int el : elements) {
    if (!s_.count(el)) {
      elements_plus_S.push_back(el);
    }
  }
  return original_->IsFeasible(elements_plus_S);
}

// Return the current set.
std::vector<int> ConditionedMatroid::GetCurrent() const {
  return std::vector<int>(current_elements_.begin(), current_elements_.end());
}

// Returns whether an element is in the current set.
bool ConditionedMatroid::InCurrent(int element) const {
  return current_elements_.count(element);
}

ConditionedMatroid::ConditionedMatroid(
    const absl::flat_hash_set<int>& s,
    const absl::flat_hash_set<int>& current_elements, const Matroid& original)
    : s_(s), current_elements_(current_elements), original_(original.Clone()) {}

// Clone the object.
std::unique_ptr<Matroid> ConditionedMatroid::Clone() const {
  return std::unique_ptr<Matroid>(
      new ConditionedMatroid(s_, current_elements_, *original_));
}
