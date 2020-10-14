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

#include "matroid_algorithm.h"

void MatroidAlgorithm::Insert(std::pair<int, int> element, bool non_monotone) {
  if (non_monotone && RandomHandler::generator_() % 3 == 0) return;
  if (elements_.size() < cardinality_k_ &&
      colors_[element.second] < bounds_[element.second].second) {
    colors_[element.second]++;
    elements_.push_back(element);
    solution_value_ = sub_func_f_->ObjectiveAndIncreaseOracleCall(elements_);
    return;
  }

  int min_index = -1;
  double best_value = solution_value_;
  for (int i = 0; i < elements_.size(); i++) {
    if (elements_[i].second == element.second ||
        colors_[element.second] < bounds_[element.second].second) {
      int temp_name = elements_[i].first;
      elements_[i].first = element.first;
      double value = sub_func_f_->ObjectiveAndIncreaseOracleCall(elements_);
      if (value > best_value) {
        best_value = value;
        min_index = i;
      }
      elements_[i].first = temp_name;
    }
  }
  if (min_index != -1) {
    colors_[element.second]++;
    colors_[elements_[min_index].second]--;
    solution_value_ = best_value;
    elements_[min_index] = element;
  }
}

void MatroidAlgorithm::Init(SubmodularFunction& sub_func_f,
                            std::vector<std::pair<int, int>> bounds,
                            int cardinality_k) {
  Algorithm::Init(sub_func_f, bounds, cardinality_k);
  bounds_ = bounds;
  cardinality_k_ = cardinality_k;
  colors_.resize(bounds_.size(), 0);
  sub_func_f_ = sub_func_f.Clone();
}

double MatroidAlgorithm::GetSolutionValue() { return solution_value_; }

std::vector<std::pair<int, int>> MatroidAlgorithm::GetSolutionVector() {
  return elements_;
}

std::string MatroidAlgorithm::GetAlgorithmName() const {
  return "Matroid Algorithm";
}
