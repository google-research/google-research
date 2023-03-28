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

#include "uniform_matroid.h"

#include <cassert>
#include <memory>
#include <set>
#include <vector>

UniformMatroid::UniformMatroid(int k) : k_(k) {}

void UniformMatroid::Reset() { current_set_.clear(); }

bool UniformMatroid::CanAdd(int element) const {
  assert(!current_set_.count(element));
  return current_set_.size() + 1 <= k_;
}

bool UniformMatroid::CanSwap(int element, int swap) const {
  assert(!current_set_.count(element));
  assert(current_set_.count(swap));
  return true;
}

std::vector<int> UniformMatroid::GetAllSwaps(int element) const {
  return GetCurrent();
}

void UniformMatroid::Add(int element) {
  assert(!current_set_.count(element));
  current_set_.insert(element);
}

void UniformMatroid::Remove(int element) {
  assert(current_set_.count(element));
  current_set_.erase(element);
}

bool UniformMatroid::IsFeasible(const std::vector<int>& elements) const {
  return elements.size() <= k_;
}

bool UniformMatroid::CurrentIsFeasible() const {
  return current_set_.size() <= k_;
}

std::vector<int> UniformMatroid::GetCurrent() const {
  return std::vector<int>(current_set_.begin(), current_set_.end());
}

bool UniformMatroid::InCurrent(int element) const {
  return current_set_.count(element);
}

std::unique_ptr<Matroid> UniformMatroid::Clone() const {
  return std::make_unique<UniformMatroid>(*this);
}
