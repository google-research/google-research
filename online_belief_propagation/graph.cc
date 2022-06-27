// Copyright 2022 The Google Research Authors.
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

#include "graph.h"

#include <assert.h>

#include <algorithm>

Graph::Graph() : sorted_(true) {}

Graph::~Graph() {}

void Graph::InsertVertex() { adjacency_lists_.emplace_back(); }

void Graph::InsertEdge(int u, int v) {
  // Both endpoints of the input edge should be within range, and they should be
  // distinct from each other.
  assert(u >= 0 && v >= 0);
  assert(u < adjacency_lists_.size() && v < adjacency_lists_.size());
  assert(u != v);

  adjacency_lists_[u].push_back(v);
  adjacency_lists_[v].push_back(u);
  // The graph is assumed to not be sorted unless Sort() has been called since
  // the last insertion.
  sorted_ = false;
}

int Graph::GetVertexNumber() const { return adjacency_lists_.size(); }

const std::vector<int>& Graph::GetNeighborhood(int vertex) const {
  // The input vertex should be within range.
  assert(vertex >= 0);
  assert(vertex < adjacency_lists_.size());

  return adjacency_lists_[vertex];
}

bool Graph::IsSorted() const { return sorted_; }

void Graph::Sort() {
  if (sorted_) {
    return;
  }
  for (std::vector<int>& adjacency_list : adjacency_lists_) {
    std::sort(adjacency_list.begin(), adjacency_list.end());
  }
  sorted_ = true;
}
