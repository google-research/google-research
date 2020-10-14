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

#include "fair_random_subset_algorithm.h"

#include "utilities.h"

using std::vector;

void FairRandomSubsetAlgorithm::Init(SubmodularFunction& sub_func_f,
                                     std::vector<std::pair<int, int>> bounds,
                                     int cardinality_k) {
  Algorithm::Init(sub_func_f, bounds, cardinality_k);
  bounds_ = bounds;
  cardinality_k_ = cardinality_k;
  sub_func_f_ = sub_func_f.Clone();
  sub_func_f_->Reset();
  universe_elements_.clear();
  solution_.clear();
}

void FairRandomSubsetAlgorithm::Insert(std::pair<int, int> element,
                                       bool non_monotone) {
  universe_elements_.push_back(element);
}

double FairRandomSubsetAlgorithm::GetSolutionValue() {
  RandomHandler::Shuffle(universe_elements_);
  for (auto& element : universe_elements_) {
    solution_.push_back(element);
    if (!Algorithm::Feasible(solution_)) solution_.pop_back();
  }
  return sub_func_f_->ObjectiveAndIncreaseOracleCall(solution_);
}

std::vector<std::pair<int, int>>
FairRandomSubsetAlgorithm::GetSolutionVector() {
  return solution_;
}

std::string FairRandomSubsetAlgorithm::GetAlgorithmName() const {
  return "Fair random algorithm";
}
