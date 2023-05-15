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

#ifndef FAIR_SUBMODULAR_MATROID_MOVIES_MIXED_UTILITY_FUNCTION_H_
#define FAIR_SUBMODULAR_MATROID_MOVIES_MIXED_UTILITY_FUNCTION_H_

#include <memory>
#include <string>
#include <vector>

#include "movies_facility_location_function.h"
#include "movies_user_utility_function.h"

class MoviesMixedUtilityFunction : public SubmodularFunction {
 public:
  MoviesMixedUtilityFunction(int user, double _alpha);

  // Sets S = empty set.
  void Reset() override;

  // Adds a new element to set S.
  void Add(int movie) override;

  // Removes an element from S.
  void Remove(int movie) override;

  // Computes f(S) - f(S - e).
  double RemovalDelta(int movie) override;

  // Returns the universe of the utility function, as pairs.
  const std::vector<int>& GetUniverse() const override;

  // Get name of utility function.
  std::string GetName() const override;

  // Clone the object.
  std::unique_ptr<SubmodularFunction> Clone() const override;

  // Removes an element from S and increases the oracle calls.
  // Not necessary, but overloaded for efficiency
  double RemoveAndIncreaseOracleCall(int movie) override;

 protected:
  // Computes f(S u {e}) - f(S).
  double Delta(int movie) override;

  // Computes f(S).
  double Objective(const std::vector<int>& movies) const override;

 private:
  // The mixed utility function is a weighted mixture of the facility location
  // and user utility functions, with the formula:
  //   alpha_ * mf_ + (1 - alpha_) * mu_
  // where mf_ is the facility location utility function and mu_ is the user
  // utility function.
  MoviesFacilityLocationFunction mf_;
  MoviesUserUtilityFunction mu_;
  double alpha_;
};

#endif  // FAIR_SUBMODULAR_MATROID_MOVIES_MIXED_UTILITY_FUNCTION_H_
