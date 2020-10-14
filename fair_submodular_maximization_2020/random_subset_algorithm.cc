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

#include "random_subset_algorithm.h"

#include "utilities.h"

using std::vector;

void RandomSubsetAlgorithm::Init(SubmodularFunction& sub_func_f,
                                 std::vector<std::pair<int, int>> bounds,
                                 int cardinality_k) {
  cardinality_k_ = cardinality_k;
  sub_func_f_ = sub_func_f.Clone();
  universe_elements_.clear();
  solution_.clear();
}

void RandomSubsetAlgorithm::Insert(std::pair<int, int> element,
                                   bool non_monotone) {
  universe_elements_.push_back(element);
}

double RandomSubsetAlgorithm::GetSolutionValue() {
  sub_func_f_->Reset();
  for (int i = 0; i < cardinality_k_; i++) {
    solution_.push_back(universe_elements_[RandomHandler::generator_() %
                                           universe_elements_.size()]);
  }
  double obj_val = 0.0;
  for (auto& element_in_solution : solution_) {
    obj_val += sub_func_f_->AddAndIncreaseOracleCall(element_in_solution, -1);
  }
  SubmodularFunction::oracle_calls_ -=
      2 * static_cast<int>(solution_.size()) - 1;
  // Should be just one call in our model.
  return obj_val;
}

std::vector<std::pair<int, int>> RandomSubsetAlgorithm::GetSolutionVector() {
  return solution_;
}

std::string RandomSubsetAlgorithm::GetAlgorithmName() const {
  return "Totally random algorithm";
}
