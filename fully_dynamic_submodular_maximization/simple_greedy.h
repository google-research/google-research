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
//  Greedy Algorithm
//
// First version, simple: stores basically nothing (just the available
// universe), and lazily runs greedy whenever getSolutionValue() is called.

#ifndef FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_SIMPLE_GREEDY_H_
#define FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_SIMPLE_GREEDY_H_

#include "memory"
#include "algorithm.h"
#include "unordered_set"

class SimpleGreedy : public Algorithm {
 public:
  void Init(const SubmodularFunction& sub_func_f, int cardinality_k);
  void Insert(int element);
  void Erase(int element);
  double GetSolutionValue();
  std::vector<int> GetSolutionVector();
  std::string GetAlgorithmName() const;

 private:
  int cardinality_k_;
  std::unordered_set<int> available_elements_;
  std::unique_ptr<SubmodularFunction> sub_func_f_;
};

#endif  // FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_SIMPLE_GREEDY_H_
