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

#ifndef FAIR_SUBMODULAR_MATROID_MOVIES_FACILITY_LOCATION_FUNCTION_H_
#define FAIR_SUBMODULAR_MATROID_MOVIES_FACILITY_LOCATION_FUNCTION_H_

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "submodular_function.h"

class MoviesFacilityLocationFunction : public SubmodularFunction {
 public:
  MoviesFacilityLocationFunction();

  // Sets S = empty set.
  void Reset() override;

  // Adds a new element to set S.
  void Add(int movie) override;

  // Removes an element from S.
  void Remove(int movie) override;

  // Removes an element from S and increases the oracle calls.
  // Not necessary, but overloaded for efficiency
  double RemoveAndIncreaseOracleCall(int movie) override;

  // Returns the universe of the utility function, as pairs.
  const std::vector<int>& GetUniverse() const override;

  // Get name of utility function.
  std::string GetName() const override;

  // Clone the object.
  std::unique_ptr<SubmodularFunction> Clone() const override;

 protected:
  // Computes f(S u {e}) - f(S).
  double Delta(int e) override;

  // Computes f(S) - f(S - e).
  double RemovalDelta(int e) override;

  // Computes f(S).
  double Objective(const std::vector<int>& elements) const override;

  // Needed for accessing Delta and RemovalDelta.
  friend class MoviesMixedUtilityFunction;

 private:
  // max_sim[i] = {0.0} u { sim(i,j) : j in S }
  std::vector<std::multiset<double, std::greater<double>>> max_sim_;
};

#endif  // FAIR_SUBMODULAR_MATROID_MOVIES_FACILITY_LOCATION_FUNCTION_H_
