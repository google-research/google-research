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

#ifndef FAIR_SUBMODULAR_MATROID_ALGORITHM_H_
#define FAIR_SUBMODULAR_MATROID_ALGORITHM_H_

#include <memory>
#include <string>
#include <vector>

#include "fairness_constraint.h"
#include "matroid.h"
#include "submodular_function.h"

// Any algorithm should be used as follows:
// * Init()
// * set SubmodularFunction::oracle_calls_ = 0
// * n times Insert()
// * if two-pass: BeginNextPass(), then again n times Insert()
// * GetSolutionValue() (obligatory! the algorithm might only compute the final
//   solution here)
// * GetSolutionVector() (optional)
// * read SubmodularFunction::oracle_calls_

class Algorithm {
 public:
  // Initialize the algorithm state.
  // Default implementation only saves the three parameters' clones into the
  // object.
  virtual void Init(const SubmodularFunction& sub_func_f,
                    const FairnessConstraint& fairness, const Matroid& matroid);

  // Handles insertion of an element.
  virtual void Insert(int element) = 0;

  // Gets current solution value.
  virtual double GetSolutionValue() = 0;

  // Gets current solution. Only call this after calling GetSolutionValue().
  virtual std::vector<int> GetSolutionVector() = 0;

  // Gets the name of the algorithm.
  virtual std::string GetAlgorithmName() const = 0;

  // Returns the number of passes the algorithm makes (1 or 2 for us). Default
  // is 1.
  virtual int GetNumberOfPasses() const;

  // Signal to the algorithm that the next pass is beginning.
  // (Do not call before the first pass, or after the last pass.
  //  Do not call at all for single-pass algorithms.)
  virtual void BeginNextPass();

  virtual ~Algorithm() = default;

 protected:
  // Color lower and upper bounds.
  std::unique_ptr<FairnessConstraint> fairness_;

  // Submodular function.
  std::unique_ptr<SubmodularFunction> sub_func_f_;

  // Matroid.
  std::unique_ptr<Matroid> matroid_;
};

#endif  // FAIR_SUBMODULAR_MATROID_ALGORITHM_H_
