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

#include "matroid_intersection_algorithm.h"

#include <memory>
#include <string>
#include <vector>

#include "fairness_constraint.h"
#include "matroid.h"
#include "matroid_intersection.h"
#include "submodular_function.h"

void MatroidIntersectionAlgorithm::Init(const SubmodularFunction& sub_func_f,
                                        const FairnessConstraint& fairness,
                                        const Matroid& matroid) {
  Algorithm::Init(sub_func_f, fairness, matroid);
  universe_elements_.clear();
  solution_vector_.clear();
}

void MatroidIntersectionAlgorithm::Insert(int element) {
  universe_elements_.push_back(element);
}

void MatroidIntersectionAlgorithm::Solve() {
  matroid_->Reset();
  sub_func_f_->Reset();
  auto fairness_matroid = fairness_->UpperBoundsToMatroid();
  SubMaxIntersection(matroid_.get(), fairness_matroid.get(), sub_func_f_.get(),
                     /*const_elements=*/{}, universe_elements_);
  solution_vector_ = matroid_->GetCurrent();
  solution_value_ =
      sub_func_f_->ObjectiveAndIncreaseOracleCall(solution_vector_);
}

double MatroidIntersectionAlgorithm::GetSolutionValue() {
  if (solution_vector_.empty()) {
    Solve();
  }
  return solution_value_;
}

std::vector<int> MatroidIntersectionAlgorithm::GetSolutionVector() {
  if (solution_vector_.empty()) {
    Solve();
  }
  return solution_vector_;
}

std::string MatroidIntersectionAlgorithm::GetAlgorithmName() const {
  return "Matroid intersection algorithm";
}
