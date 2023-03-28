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

#include "matroid.h"

#include <cassert>
#include <vector>

std::vector<int> Matroid::GetAllSwaps(int element) const {
  std::vector<int> all_swaps;
  for (int swap : GetCurrent()) {
    if (CanSwap(element, swap)) {
      all_swaps.push_back(swap);
    }
  }
  return all_swaps;
}

void Matroid::Swap(int element, int swap) {
  assert(InCurrent(swap));
  Remove(swap);
  Add(element);
}

bool Matroid::CurrentIsFeasible() const { return IsFeasible(GetCurrent()); }

bool Matroid::InCurrent(int element) const {
  for (int current_element : GetCurrent()) {
    if (current_element == element) {
      return true;
    }
  }
  return false;
}
