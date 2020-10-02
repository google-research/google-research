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
//  Random Subset Algorithm
//
// This algorithm returns a random subetset of the universe as the solution.

#ifndef FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_RANDOM_SUBSET_ALGORITHM_H_
#define FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_RANDOM_SUBSET_ALGORITHM_H_

#include "algorithm.h"
#include "utilities.h"

class RandomSubsetAlgorithm : public Algorithm {
 public:
  void Initialization(const SubmodularFunction& sub_func_f, int cardinality_k);
  void Insert(int element);
  void Erase(int element);

  double GetSolutionValue();
  std::vector<int> GetSolutionVector();
  std::string GetAlgorithmName() const;

 private:
  // Cardinality constraint.
  int cardinality_k_;
  // The solution (i.e. sampled elements), all the elements in the universe.
  std::vector<int> solution_, universe_elements_;
  // Submodulaer function.
  std::unique_ptr<SubmodularFunction> sub_func_f_;
};

#endif  // FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_RANDOM_SUBSET_ALGORITHM_H_
