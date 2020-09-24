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

#include "graph_utility.h"

GraphUtility::GraphUtility(const std::string& graph_name)
    : graph_(Graph::GetGraph(graph_name)) {
  int max_el = *max_element(graph_.GetCoverableVertices().begin(),
                            graph_.GetCoverableVertices().end());
  // Used to check if the data read and stored correctly.
  // Serves as an upper bound on number of vertices as well.
  static const int max_num_elements = 500000000;
  if (max_el > max_num_elements) {
    Fail("looks like vertices were not renumbered?");
  }
  present_elements_.assign(max_el + 1, false);
}

void GraphUtility::Reset() {
  present_elements_.assign(present_elements_.size(), false);
}

double GraphUtility::Delta(int element) const {
  int val = 0;
  for (int x : graph_.GetNeighbors(element)) {
    if (!present_elements_[x]) {
      ++val;
    }
  }
  return val;
}

void GraphUtility::Add(int element) {
  for (int x : graph_.GetNeighbors(element)) {
    present_elements_[x] = true;
  }
}

const vector<int>& GraphUtility::GetUniverse() const {
  return graph_.GetUniverseVertices();
}

std::string GraphUtility::GetName() const {
  return std::string("graph (") + graph_.GetName() + ")";
}

std::unique_ptr<SubmodularFunction> GraphUtility::Clone() const {
  return std::make_unique<GraphUtility>(*this);
}
