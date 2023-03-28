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

#ifndef FAIR_SUBMODULAR_MATROID_MOVIES_USER_UTILITY_FUNCTION_H_
#define FAIR_SUBMODULAR_MATROID_MOVIES_USER_UTILITY_FUNCTION_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "submodular_function.h"

class MoviesUserUtilityFunction : public SubmodularFunction {
 public:
  explicit MoviesUserUtilityFunction(int user);

  // Sets S = empty set.
  void Reset() override;

  // Adds a new element to set S.
  void Add(int e) override;

  // Removes an element from S.
  void Remove(int e) override;

  // Returns the universe of the utility function, as pairs.
  const std::vector<int>& GetUniverse() const override;

  // Get name of utility function.
  std::string GetName() const override;

  // Clone the object.
  std::unique_ptr<SubmodularFunction> Clone() const override;

 protected:
  // Computes f(S u {e}) - f(S).
  double Delta(int movie) override;

  // Computes f(S) - f(S - e).
  double RemovalDelta(int movie) override;

  // Needed for accessing Delta and RemovalDelta.
  double Objective(const std::vector<int>& elements) const override;

  // Needed to access Delta and RemovalDelta.
  friend class MoviesMixedUtilityFunction;

 private:
  const int user_;  // ID of user for whom the movie quality is computed.
  // Elements currently present in the solution.
  absl::flat_hash_set<int> present_elements_;
};

#endif  // FAIR_SUBMODULAR_MATROID_MOVIES_USER_UTILITY_FUNCTION_H_
