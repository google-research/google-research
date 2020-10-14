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
// The general class for the algorithms. Our algorithms and the baselines are
// inherited from this class.
//

#ifndef FAIR_SUBMODULAR_MAXIMIZATION_2020_ALGORITHM_H_
#define FAIR_SUBMODULAR_MAXIMIZATION_2020_ALGORITHM_H_

#include "submodular_function.h"

class Algorithm {
 public:
  // Initializes the algorithm state (also saves bounds and k).
  virtual void Init(SubmodularFunction& sub_func_f,
                    std::vector<std::pair<int, int>> bounds,
                    int cardinality_k) {
    bounds_ = bounds;
    cardinality_k_ = cardinality_k;
  }

  // Handles insertion of an element.
  virtual void Insert(std::pair<int, int> element,
                      bool non_monotone = false) = 0;

  // Gets current solution value.
  virtual double GetSolutionValue() = 0;

  // Gets current solution.
  virtual std::vector<std::pair<int, int>> GetSolutionVector() = 0;

  // Gets the name of the algorithm.
  virtual std::string GetAlgorithmName() const = 0;

  // Determines if a set is feasible, i.e., satisfies the upper bounds and can
  // be extended to satisfy the lower bounds.
  bool Feasible(std::vector<std::pair<int, int>> elements);

  virtual ~Algorithm() {}

 private:
  // Pairs of lower and upper bounds.
  std::vector<std::pair<int, int>> bounds_;

  // The cardinality constraint.
  int cardinality_k_;
};

#endif  // FAIR_SUBMODULAR_MAXIMIZATION_2020_ALGORITHM_H_
