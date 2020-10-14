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
// A fancier version of Greedy:
// Keeps a solution always ready, recomputes it when elements arrive/leave
// faster, but keeps k "prefix" copies of function f, so uses more memory
// should be equivalent to SimpleGreedy in what it outputs.
//

#ifndef ANONYMOUS_FAIR_SUBMODULAR_MAXIMIZATION_2019_GREEDY_ALGORITHM_H_
#define ANONYMOUS_FAIR_SUBMODULAR_MAXIMIZATION_2019_GREEDY_ALGORITHM_H_

#include <set>

#include "algorithm.h"
#include "utilities.h"

class Greedy : public Algorithm {
 public:
  // Initializes the algorithm and its dependencies.
  void Init(SubmodularFunction& sub_func_f,
            std::vector<std::pair<int, int>> bounds, int cardinality_k);

  // The algorithm that runs in case an elements in inserted.
  void Insert(std::pair<int, int> element, bool non_monotone = false);

  // Returns the value of the current solution.
  double GetSolutionValue();

  // Returns the elements in the solution.
  std::vector<std::pair<int, int>> GetSolutionVector();

  // Return the name of the algrithm.
  std::string GetAlgorithmName() const;

 private:
  // Cardinality Constraint.
  int cardinality_k_;

  // Set of the elements.
  std::set<std::pair<int, int>> elements_;

  // Set of the selected elements.
  std::vector<std::pair<int, int>> solution_;

  // Submodulaer function.
  std::unique_ptr<SubmodularFunction> sub_func_f_;
};

#endif  // ANONYMOUS_FAIR_SUBMODULAR_MAXIMIZATION_2019_GREEDY_ALGORITHM_H_
