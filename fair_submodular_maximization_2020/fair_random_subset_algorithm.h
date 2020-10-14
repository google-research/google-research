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

//
//  Fair Random Subset Algorithm
//
// This algorithm returns a random subetset of the universe as the solution that
// satisfies the lower and upper bound constraints.
//

#ifndef FAIR_SUBMODULAR_MAXIMIZATION_2020_FAIR_RANDOM_SUBSET_ALGORITHM_H_
#define FAIR_SUBMODULAR_MAXIMIZATION_2020_FAIR_RANDOM_SUBSET_ALGORITHM_H_

#include "algorithm.h"
#include "utilities.h"

class FairRandomSubsetAlgorithm : public Algorithm {
 public:
  // Initializes the algorithm state (also saves bounds and k).
  void Init(SubmodularFunction& sub_func_f,
            std::vector<std::pair<int, int>> bounds, int cardinality_k);

  // Handles insertion of an element.
  void Insert(std::pair<int, int> element, bool non_monotone = false);

  // Gets current solution value.
  double GetSolutionValue();

  // Gets current solution.
  std::vector<std::pair<int, int>> GetSolutionVector();

  // Gets the name of the algorithm.
  std::string GetAlgorithmName() const;

 private:
  // Cardinality constraint.
  int cardinality_k_;

  // The solution (i.e. sampled elements), all the elements in the universe.
  std::vector<std::pair<int, int>> solution_, universe_elements_;

  // Submodulaer function.
  std::unique_ptr<SubmodularFunction> sub_func_f_;

  // The lower and upper bounds constraints.
  std::vector<std::pair<int, int>> bounds_;
};

#endif  // FAIR_SUBMODULAR_MAXIMIZATION_2020_FAIR_RANDOM_SUBSET_ALGORITHM_H_
