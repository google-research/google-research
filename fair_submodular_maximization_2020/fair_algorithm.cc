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

#include "fair_algorithm.h"

void FairAlgorithm::Insert(std::pair<int, int> element, bool non_monotone) {
  elements_.push_back(element);
  universe_.push_back(element);
  if (Algorithm::Feasible(elements_)) {
    solution_value_ = sub_func_f_->ObjectiveAndIncreaseOracleCall(elements_);
    return;
  }
  elements_.pop_back();
  int min_index = -1;
  double best_value = solution_value_;
  for (int i = 0; i < elements_.size(); i++) {
    bool feasible = (elements_[i].second == element.second);
    if (!feasible) {
      auto temp_name = elements_[i];
      elements_[i] = element;
      feasible = Algorithm::Feasible(elements_);
      elements_[i] = temp_name;
    }
    if (feasible) {
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
    solution_value_ = best_value;
    elements_[min_index] = element;
  }
}

void FairAlgorithm::Init(SubmodularFunction& sub_func_f,
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

double FairAlgorithm::GetSolutionValue() {
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

std::vector<std::pair<int, int>> FairAlgorithm::GetSolutionVector() {
  return elements_;
}

std::string FairAlgorithm::GetAlgorithmName() const { return "Fair Algorithm"; }
