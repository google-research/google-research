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

#ifndef FAIR_SUBMODULAR_MATROID_TWO_PASS_ALGORITHM_WITH_CONDITIONED_MATROID_H_
#define FAIR_SUBMODULAR_MATROID_TWO_PASS_ALGORITHM_WITH_CONDITIONED_MATROID_H_

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "algorithm.h"

class TwoPassAlgorithmWithConditionedMatroid : public Algorithm {
 public:
  // Initialize the algorithm state.
  void Init(const SubmodularFunction& sub_func_f,
            const FairnessConstraint& fairness,
            const Matroid& matroid) override;

  // Handles insertion of an element.
  void Insert(int element) override;

  // Gets current solution value.
  double GetSolutionValue() override;

  // Gets current solution.
  std::vector<int> GetSolutionVector() override;

  // Gets the name of the algorithm.
  std::string GetAlgorithmName() const override;

  // Returns the number of passes.
  int GetNumberOfPasses() const override { return 2; }

 private:
  void FirstPass();
  void GreedyFirstPass();
  void FindFeasibleSolution();
  void DivideSolution();
  std::vector<int> SecondPass(std::vector<int> start_solution);

  // Upper and Lower bounds.
  std::vector<std::pair<int, int>> bounds_;

  // All the elements in the universe.
  std::vector<int> universe_elements_;

  // Solutions per color.
  std::vector<std::vector<int>> per_color_solutions_;

  std::vector<int> first_round_solution_;

  std::vector<std::vector<int>> lower_bound_solutions_;

  // The weight of the elements when added to the solution.
  std::map<int, double> weights_;

  // The final solution.
  std::vector<int> final_solution_;
};

#endif  // FAIR_SUBMODULAR_MATROID_TWO_PASS_ALGORITHM_WITH_CONDITIONED_MATROID_H_
