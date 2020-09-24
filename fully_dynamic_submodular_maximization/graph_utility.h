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

// The submodular (coverage) function for graph utility.

#ifndef FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_GRAPH_UTILITY_H_
#define FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_GRAPH_UTILITY_H_

#include <algorithm>
#include <memory>

#include "graph.h"
#include "submodular_function.h"
#include "utilities.h"

class GraphUtility : public SubmodularFunction {
 public:
  GraphUtility(const std::string& graph_name);

  // Removes all the data that it has.
  void Reset();

  // The delta added by inserting this element.
  double Delta(int element) const;

  // Adds element e.
  void Add(int element);

  // Returns the elements of the universe.
  const vector<int>& GetUniverse() const;

  // Returns the name.
  std::string GetName() const;
  std::unique_ptr<SubmodularFunction> Clone() const;

 private:
  Graph& graph_;
  // Can also implement with unordered_set, might be faster sometimes.
  vector<bool> present_elements_;
};

#endif  // FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_GRAPH_UTILITY_H_
