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

#include "graph_utility.h"

#include <stddef.h>

#include <algorithm>
#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "submodular_function.h"
#include "utilities.h"

constexpr int max_num_elements = 500000000;

GraphUtility::GraphUtility(Graph& graph) : graph_(graph) {
  int max_el = *std::max_element(graph_.GetCoverableVertices().begin(),
                                 graph_.GetCoverableVertices().end());
  // Sanity-check if the data read and stored correctly.
  // Serves as an upper bound on number of vertices as well.
  if (max_el > max_num_elements) {
    Fail("looks like vertices were not renumbered?");
  }
  present_elements_.assign(max_el + 1, 0);
  existing_elements_.clear();
}

void GraphUtility::Reset() {
  present_elements_.assign(present_elements_.size(), 0);
  existing_elements_.clear();
}

double GraphUtility::Delta(int element) {
  int val = 0;
  for (int x : graph_.GetNeighbors(element)) {
    if (!present_elements_[x]) {
      ++val;
    }
  }
  return val;
}

void GraphUtility::Add(int element) {
  existing_elements_.insert(element);
  for (int x : graph_.GetNeighbors(element)) {
    present_elements_[x]++;
  }
}

void GraphUtility::Remove(int element) {
  assert(existing_elements_.erase(element) == 1);
  for (int x : graph_.GetNeighbors(element)) {
    present_elements_[x]--;
  }
}

double GraphUtility::RemovalDelta(int element) {
  assert(existing_elements_.count(element));

  int val = 0;
  for (int x : graph_.GetNeighbors(element)) {
    val += present_elements_[x] == 1;
  }
  return val;
}

// Not necessary, but overloaded for efficiency
double GraphUtility::RemoveAndIncreaseOracleCall(int element) {
  ++oracle_calls_;

  assert(existing_elements_.erase(element) == 1);

  int val = 0;
  for (int x : graph_.GetNeighbors(element)) {
    val += present_elements_[x] == 1;
    present_elements_[x]--;
  }
  return val;
}

double GraphUtility::Objective(const std::vector<int>& elements) const {
  absl::flat_hash_set<int> covered_vertices;
  for (const int it : elements) {
    for (int x : graph_.GetNeighbors(it)) {
      covered_vertices.insert(x);
    }
  }

  return covered_vertices.size();
}

const std::vector<int>& GraphUtility::GetUniverse() const {
  return graph_.GetUniverseVertices();
}

std::string GraphUtility::GetName() const {
  return std::string("graph (") + graph_.GetName() + ")";
}

std::unique_ptr<SubmodularFunction> GraphUtility::Clone() const {
  return std::make_unique<GraphUtility>(*this);
}
