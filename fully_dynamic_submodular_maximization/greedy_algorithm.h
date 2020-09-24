// Copyright 2020 The Google Research Authors.
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

// A fancier version of Greedy:
// Keeps a solution always ready, recomputes it when elements arrive/leave
// faster, but keeps k "prefix" copies of function f, so uses more memory
// should be equivalent to SimpleGreedy in what it outputs.

#ifndef FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_GREEDY_ALGORITHM_H_
#define FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_GREEDY_ALGORITHM_H_

#include <unordered_set>

#include "algorithm.h"
#include "utilities.h"

class Greedy : public Algorithm {
 public:
  // Initializes the algorithm and its dependencies.
  void Init(const SubmodularFunction& sub_func_f, int cardinality_k);

  // The algorithm that runs in case an elements in inserted.
  void Insert(int element);

  // The algorithm that runs in case an elements in erased.
  void Erase(int element);

  // Returns the value of the current solution.
  double GetSolutionValue();

  // Returns the elements in the solution.
  std::vector<int> GetSolutionVector();

  // Return the name of the algrithm.
  std::string GetAlgorithmName() const;

 private:
  int cardinality_k_;
  std::unordered_set<int> available_elements_;

  // partial_F_[i] = function after inserting i elements.
  std::vector<std::unique_ptr<SubmodularFunction>> partial_F_;
  std::vector<int> solution_;

  // obj_val[i] = f(solution[0..i-1]).
  std::vector<double> obj_vals_;

  // Insert an element into the solution (at the end).
  void InsertIntoSolution(int element);

  // Remove the last element from the solution.
  void RemoveLastElement();

  // Complete the solution to k elements.
  void Complete();
};

#endif  // FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_GREEDY_ALGORITHM_H_
