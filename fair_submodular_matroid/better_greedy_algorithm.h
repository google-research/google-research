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

#ifndef FAIR_SUBMODULAR_MATROID_BETTER_GREEDY_ALGORITHM_H_
#define FAIR_SUBMODULAR_MATROID_BETTER_GREEDY_ALGORITHM_H_

#include <memory>
#include <string>
#include <vector>

#include "fairness_constraint.h"
#include "greedy_algorithm.h"
#include "matroid.h"
#include "submodular_function.h"

class BetterGreedyAlgorithm : public GreedyAlgorithm {
 public:
  explicit BetterGreedyAlgorithm(bool minimal = false);

  // Initialize the algorithm state.
  void Init(const SubmodularFunction& sub_func_f,
            const FairnessConstraint& fairness,
            const Matroid& matroid) override;

  // Handles insertion of an element.
  void Insert(int element) override;

  // Gets current solution value.
  double GetSolutionValue() override;

  // Gets the name of the algorithm.
  std::string GetAlgorithmName() const override;

 private:
  // A copy of the original submodular function for each color class.
  std::vector<std::unique_ptr<SubmodularFunction>> function_by_color_;
  // if true find feasible solution with only ell_c elts in each color
  bool minimal_;
};

#endif  // FAIR_SUBMODULAR_MATROID_BETTER_GREEDY_ALGORITHM_H_
