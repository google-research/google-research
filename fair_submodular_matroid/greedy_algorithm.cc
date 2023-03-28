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

#include "greedy_algorithm.h"

#include <memory>
#include <string>
#include <vector>

#include "fairness_constraint.h"
#include "matroid.h"
#include "matroid_intersection.h"
#include "submodular_function.h"

void GreedyAlgorithm::Init(const SubmodularFunction& sub_func_f,
                           const FairnessConstraint& fairness,
                           const Matroid& matroid) {
  Algorithm::Init(sub_func_f, fairness, matroid);
  matroid_by_color_.clear();
  solution_.clear();
  for (int i = 0; i < fairness_->GetColorNum(); ++i) {
    matroid_by_color_.push_back(matroid_->Clone());
  }
  fairness_matroid_ = fairness_->LowerBoundsToMatroid();
}

void GreedyAlgorithm::Insert(int element) {
  const int color = fairness_->GetColor(element);
  Matroid* matroid = matroid_by_color_[color].get();
  if (matroid->CanAdd(element)) {
    matroid->Add(element);
  }
}

double GreedyAlgorithm::GetSolutionValue() {
  std::vector<int> all_elements;
  for (int i = 0; i < matroid_by_color_.size(); ++i) {
    const Matroid* matroid = matroid_by_color_[i].get();
    const std::vector<int> elements = matroid->GetCurrent();
    all_elements.insert(all_elements.end(), elements.begin(), elements.end());
  }
  MaxIntersection(matroid_.get(), fairness_matroid_.get(), all_elements);
  solution_ = matroid_->GetCurrent();
  return sub_func_f_->ObjectiveAndIncreaseOracleCall(solution_);
}

std::vector<int> GreedyAlgorithm::GetSolutionVector() { return solution_; }

std::string GreedyAlgorithm::GetAlgorithmName() const {
  return "Basic greedy algorithm";
}
