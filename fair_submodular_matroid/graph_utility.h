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

#ifndef FAIR_SUBMODULAR_MATROID_GRAPH_UTILITY_H_
#define FAIR_SUBMODULAR_MATROID_GRAPH_UTILITY_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "graph.h"
#include "submodular_function.h"

class GraphUtility : public SubmodularFunction {
 public:
  explicit GraphUtility(Graph& graph);

  // Removes all the data that it has.
  void Reset() override;

  // Returns the elements of the universe.
  const std::vector<int>& GetUniverse() const override;

  // Returns the name.
  std::string GetName() const override;

  // Returns a deep copy of the object.
  std::unique_ptr<SubmodularFunction> Clone() const override;

  // Adds a new element to set S.
  void Add(int element) override;

  // Removes a new element to set S.
  void Remove(int element) override;

  // Not necessary, but overloaded for efficiency
  double RemoveAndIncreaseOracleCall(int element) override;

 protected:
  // Computes f(S u {e}) - f(S).
  double Delta(int element) override;

  // Returns the value of the given elements.
  double Objective(const std::vector<int>& elements) const override;

  // Computes f(S) - f(S - e).
  // Assumes (without checking) that e is in S.
  double RemovalDelta(int element) override;

 private:
  // The underlying graph.
  const Graph& graph_;

  // Can also implement with unordered_map, might be faster sometimes.
  std::vector<int> present_elements_;  // elements in current solution

  // Counts how many element in current solution cover each element
  absl::flat_hash_set<int> existing_elements_;
};

#endif  // FAIR_SUBMODULAR_MATROID_GRAPH_UTILITY_H_
