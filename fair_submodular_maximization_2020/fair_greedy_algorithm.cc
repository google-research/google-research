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

#include "fair_greedy_algorithm.h"

void FairGreedy::Init(SubmodularFunction& sub_func_f,
                      std::vector<std::pair<int, int>> bounds,
                      int cardinality_k) {
  Algorithm::Init(sub_func_f, bounds, cardinality_k);
  cardinality_k_ = cardinality_k;
  sub_func_f_ = sub_func_f.Clone();
  sub_func_f_->Reset();
  solution_.clear();
  elements_.clear();
}

void FairGreedy::Insert(std::pair<int, int> element, bool non_monotone) {
  elements_.push_back(element);
}

double FairGreedy::GetSolutionValue() {
  while (solution_.size() < cardinality_k_) {
    std::pair<double, std::pair<int, int>> best(-1, std::make_pair(-1, -1));
    solution_.push_back(std::make_pair(-1, -1));
    for (auto& element : elements_) {
      solution_[solution_.size() - 1] = element;
      if (Algorithm::Feasible(solution_)) {
        best = max(
            best,
            make_pair(sub_func_f_->ObjectiveAndIncreaseOracleCall(solution_),
                      element));
      }
    }
    solution_[solution_.size() - 1] = best.second;
  }
  return sub_func_f_->ObjectiveAndIncreaseOracleCall(solution_);
}

std::vector<std::pair<int, int>> FairGreedy::GetSolutionVector() {
  return solution_;
}

std::string FairGreedy::GetAlgorithmName() const { return "Fair greedy"; }
