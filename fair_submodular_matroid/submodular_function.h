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

#ifndef FAIR_SUBMODULAR_MATROID_SUBMODULAR_FUNCTION_H_
#define FAIR_SUBMODULAR_MATROID_SUBMODULAR_FUNCTION_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// A submodular function object maintains a current solution set S,
// but does *not* maintain its value (that should be maintained by the user,
// i.e., the algorithm, or alternatively one can e.g. call
// `ObjectiveAndIncreaseOracleCall()` at the end for the final solution).

class SubmodularFunction {
 public:
  static int64_t oracle_calls_;

  virtual ~SubmodularFunction() = default;

  // Sets S = empty set.
  virtual void Reset() = 0;

  // there used to be a function Init() here too, but removed it

  // Return the objective value of a set 'S' and also increases oracle_calls.
  // Does not depend on the current state of the object.
  double ObjectiveAndIncreaseOracleCall(const std::vector<int>& elements) const;

  // Adds a new element to set S. Does not return any value and does not cost an
  // oracle call.
  virtual void Add(int element) = 0;

  // Removes an element from S. Does not return any value and does not cost an
  // oracle call. Assumes (without checking) that e is in S.
  virtual void Remove(int element) = 0;

  // Swap an element for one already in the solution.
  virtual void Swap(int element, int swap);

  // Returns the delta if this element were added (but does not add it),
  // and also increases oracle_calls.
  double DeltaAndIncreaseOracleCall(int element);

  // Adds element if and only if its contribution is >= thre and also increases
  // oracle_calls. Returns the contribution increase (if added, otherwise 0).
  virtual double AddAndIncreaseOracleCall(int element, double thre);

  // Computes f(S) - f(S - e). Does not remove e. Costs one oracle call.
  // Assumes (without checking) that e is in S.
  double RemovalDeltaAndIncreaseOracleCall(int element);

  // Remove an element e from S.
  // Return f(S) - f(S - e).
  // Costs one oracle call.
  // Assumes (without checking) that e is in S.
  virtual double RemoveAndIncreaseOracleCall(int element);

  // Returns the universe of the utility function, as pairs.
  // The first element of the pair is the name of the element and the second is
  // its color.
  virtual const std::vector<int>& GetUniverse() const = 0;

  // Get name of utility function.
  virtual std::string GetName() const = 0;

  // Clone the object (see e.g. GraphUtility for an example).
  virtual std::unique_ptr<SubmodularFunction> Clone() const = 0;

  // Gets geometrically increasing sequence of estimates for OPT.
  // Should be always run on an empty function.
  std::vector<double> GetOptEstimates(
      int upper_bound_on_size_of_any_feasible_set);

 protected:
  // Computes f(S u {e}) - f(S).
  virtual double Delta(int element) = 0;

  // Computes f(S) - f(S - e).
  // Assumes (without checking) that e is in S.
  virtual double RemovalDelta(int element) = 0;

  // Computes f(S).
  // Does not depend on the current state of the object.
  virtual double Objective(const std::vector<int>& elements) const = 0;
};

#endif  // FAIR_SUBMODULAR_MATROID_SUBMODULAR_FUNCTION_H_
