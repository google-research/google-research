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

#include "fair_algorithm_ck.h"

void FairAlgorithmCK::Insert(std::pair<int, int> element, bool non_monotone) {
  elements_.push_back(element);
  universe_.push_back(element);
  double benf =
      sub_func_f_->ObjectiveAndIncreaseOracleCall(elements_) - solution_value_;
  if (Algorithm::Feasible(elements_)) {
    elements_benf_.push_back(benf);
    solution_value_ += benf;
    return;
  }
  elements_.pop_back();
  int min_index = -1;
  double best_value = std::numeric_limits<double>::infinity();
  for (int i = 0; i < elements_.size(); i++) {
    bool feasible = (elements_[i].second == element.second);
    if (!feasible) {
      auto temp_name = elements_[i];
      elements_[i] = element;
      feasible = Algorithm::Feasible(elements_);
      elements_[i] = temp_name;
    }
    if (feasible) {
      if (elements_benf_[i] < best_value) {
        best_value = elements_benf_[i];
        min_index = i;
      }
    }
  }
  if (min_index == -1) return;
  auto copy_element = elements_[min_index];
  elements_[min_index] = element;
  double real_benf =
      sub_func_f_->ObjectiveAndIncreaseOracleCall(elements_) - solution_value_;
  if (min_index != -1 && real_benf > 0) {
    solution_value_ += real_benf;
    elements_benf_[min_index] = benf;
  } else {
    elements_[min_index] = copy_element;
  }
}

void FairAlgorithmCK::Init(SubmodularFunction& sub_func_f,
                           std::vector<std::pair<int, int>> bounds,
                           int cardinality_k) {
  Algorithm::Init(sub_func_f, bounds, cardinality_k);
  bounds_ = bounds;
  sub_func_f_ = sub_func_f.Clone();
  sub_func_f_->Reset();
  solution_value_ = 0;
  elements_.clear();
  universe_.clear();
}

double FairAlgorithmCK::GetSolutionValue() {
  std::vector<int> colors(bounds_.size(), 0);
  for (auto& element : elements_) colors[element.second]++;
  RandomHandler::Shuffle(universe_);
  for (auto& element : universe_)
    if (colors[element.second] < bounds_[element.second].first) {
      elements_.push_back(element);
      colors[element.second]++;
    }
  return sub_func_f_->ObjectiveAndIncreaseOracleCall(elements_);
}

std::vector<std::pair<int, int>> FairAlgorithmCK::GetSolutionVector() {
  return elements_;
}

std::string FairAlgorithmCK::GetAlgorithmName() const {
  return "Fair AlgorithmCK";
}
