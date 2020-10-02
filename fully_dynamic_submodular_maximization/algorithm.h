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
//      Interface for an algorithm
//

// To maximize a submodular function under a cardinality constraint of k
//  for a stream of insertions and deletions.

#ifndef FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_ALGORITHM_H_
#define FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_ALGORITHM_H_

#include "submodular_function.h"

class Algorithm {
 public:
  // Initializes the algorithm state (also save f and k).
  virtual void Init(const SubmodularFunction& sub_func_f,
                    int cardinality_k) = 0;

  // Handles insertion of an element.
  virtual void Insert(int element) = 0;

  // Handles deletion of an element.
  virtual void Erase(int element) = 0;

  // Gets current solution value.
  virtual double GetSolutionValue() = 0;

  // Gets current solution.
  virtual std::vector<int> GetSolutionVector() = 0;

  // Get name of algorithm.
  virtual std::string GetAlgorithmName() const = 0;

  virtual ~Algorithm() {}
};

#endif  // FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_ALGORITHM_H_
