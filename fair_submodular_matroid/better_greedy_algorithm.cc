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

#include "better_greedy_algorithm.h"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "fairness_constraint.h"
#include "matroid.h"
#include "matroid_intersection.h"
#include "submodular_function.h"

BetterGreedyAlgorithm::BetterGreedyAlgorithm(bool minimal)
    : minimal_(minimal) {}

void BetterGreedyAlgorithm::Init(const SubmodularFunction& sub_func_f,
                                 const FairnessConstraint& fairness,
                                 const Matroid& matroid) {
  GreedyAlgorithm::Init(sub_func_f, fairness, matroid);
  function_by_color_.clear();
  for (int i = 0; i < fairness_->GetColorNum(); ++i) {
    function_by_color_.push_back(sub_func_f_->Clone());
  }
}

void BetterGreedyAlgorithm::Insert(int element) {
  const int color = fairness_->GetColor(element);
  Matroid* matroid = matroid_by_color_[color].get();
  SubmodularFunction* sub_func = function_by_color_[color].get();
  if (matroid->CanAdd(element)) {
    matroid->Add(element);
    sub_func->Add(element);
  } else {
    const std::vector<int> all_swaps = matroid->GetAllSwaps(element);
    if (all_swaps.empty()) {
      return;
    }
    const int best_swap = *std::min_element(
        all_swaps.begin(), all_swaps.end(), [&sub_func](int lhs, int rhs) {
          return sub_func->RemovalDeltaAndIncreaseOracleCall(lhs) <
                 sub_func->RemovalDeltaAndIncreaseOracleCall(rhs);
        });
    if (sub_func->RemovalDeltaAndIncreaseOracleCall(best_swap) <
        sub_func->DeltaAndIncreaseOracleCall(element)) {
      matroid->Swap(element, best_swap);
      sub_func->Swap(element, best_swap);
    }
  }
}

double BetterGreedyAlgorithm::GetSolutionValue() {
  // Get feasible solution.
  std::vector<int> all_elements;
  for (int i = 0; i < matroid_by_color_.size(); ++i) {
    const Matroid* matroid = matroid_by_color_[i].get();
    const std::vector<int> elements = matroid->GetCurrent();
    all_elements.insert(all_elements.end(), elements.begin(), elements.end());
  }
  MaxIntersection(matroid_.get(), fairness_matroid_.get(), all_elements);
  std::vector<int> solution = matroid_->GetCurrent();
  assert(fairness_->IsFeasible(solution));

  if (!minimal_) {
    // Populate fairness_ and sub_func_f_.
    for (int element : solution) {
      fairness_->Add(element);
      sub_func_f_->Add(element);
    }

    // Add more elements greedily.
    std::vector<int> elements_left;
    std::sort(all_elements.begin(), all_elements.end());
    std::sort(solution.begin(), solution.end());
    std::set_difference(all_elements.begin(), all_elements.end(),
                        solution.begin(), solution.end(),
                        std::inserter(elements_left, elements_left.begin()));
    std::sort(elements_left.begin(), elements_left.end(),
              [this](int lhs, int rhs) {
                return sub_func_f_->DeltaAndIncreaseOracleCall(lhs) >
                       sub_func_f_->DeltaAndIncreaseOracleCall(rhs);
              });
    for (int element : elements_left) {
      if (fairness_->CanAdd(element) && matroid_->CanAdd(element)) {
        fairness_->Add(element);
        matroid_->Add(element);
        solution.push_back(element);
      }
    }
  }

  solution_ = solution;
  return sub_func_f_->ObjectiveAndIncreaseOracleCall(solution_);
}

std::string BetterGreedyAlgorithm::GetAlgorithmName() const {
  return "Slightly better greedy algorithm";
}
