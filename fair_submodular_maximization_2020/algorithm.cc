// Copyright 2020 The Authors.
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

#include "algorithm.h"

bool Algorithm::Feasible(std::vector<std::pair<int, int>> elements) {
  std::vector<int> element_from_colors(bounds_.size(), 0);
  for (const auto& element : elements) {
    element_from_colors[element.second]++;
  }
  int extra_elements_needed = 0;
  for (int i = 0; i < element_from_colors.size(); i++) {
    if (bounds_[i].second < element_from_colors[i]) {
      return false;
    }
    extra_elements_needed +=
        std::max(0, bounds_[i].first - element_from_colors[i]);
  }
  if (elements.size() + extra_elements_needed > cardinality_k_) {
    return false;
  }
  return true;
}
