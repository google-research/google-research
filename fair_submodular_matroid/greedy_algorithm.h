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

#ifndef FAIR_SUBMODULAR_MATROID_GREEDY_ALGORITHM_H_
#define FAIR_SUBMODULAR_MATROID_GREEDY_ALGORITHM_H_

#include <memory>
#include <string>
#include <vector>

#include "algorithm.h"
#include "fairness_constraint.h"
#include "matroid.h"
#include "submodular_function.h"

class GreedyAlgorithm : public Algorithm {
 public:
  // Initialize the algorithm state.
  void Init(const SubmodularFunction& sub_func_f,
            const FairnessConstraint& fairness,
            const Matroid& matroid) override;

  // Handles insertion of an element.
  void Insert(int element) override;

  // Gets current solution value.
  double GetSolutionValue() override;

  // Gets current solution. Only call this after calling GetSolutionValue().
  std::vector<int> GetSolutionVector() override;

  // Gets the name of the algorithm.
  std::string GetAlgorithmName() const override;

 protected:
  // A copy of the original matroid for each color class.
  std::vector<std::unique_ptr<Matroid>> matroid_by_color_;

  // The partition matroid derived from the lower bounds of the fairness
  // constraint.
  std::unique_ptr<Matroid> fairness_matroid_;

  // The final solution set.
  std::vector<int> solution_;
};

#endif  // FAIR_SUBMODULAR_MATROID_GREEDY_ALGORITHM_H_
