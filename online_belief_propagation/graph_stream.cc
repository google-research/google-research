// Copyright 2021 The Google Research Authors.
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

#include "graph_stream.h"

#include <assert.h>

#include <algorithm>

GraphStream::GraphStream(const Graph* graph) : graph_(graph), time_(0) {
  assert(graph_->IsSorted());
}

GraphStream::~GraphStream() {}

int GraphStream::GetCurrentVertexNumber() const { return time_; }

int GraphStream::GetTotalVertexNumber() const {
  return graph_->GetVertexNumber();
}

std::vector<int> GraphStream::GetCurrentNeighborhood(int vertex) const {
  // Input vertex must be within range.
  assert(vertex >= 0);
  assert(vertex < time_);

  std::vector<int> current_neighborhood;
  for (int i : graph_->GetNeighborhood(vertex)) {
    if (i >= time_) return current_neighborhood;
    current_neighborhood.push_back(i);
  }
  return current_neighborhood;
}

bool GraphStream::CheckEdge(int u, int v) const {
  // Input vertices must be in range.
  assert(u >= 0 && v >= 0);
  assert(u < time_ && v < time_);

  // Perform binary search on u_neighborhood, the adjacency list of u.
  const std::vector<int>& u_neighborhood = graph_->GetNeighborhood(u);
  return binary_search(u_neighborhood.begin(), u_neighborhood.end(), v);
}

void GraphStream::Step() {
  // time_ cannot be increased beyond the total number of vertices.
  assert(time_ < graph_->GetVertexNumber());
  time_++;
}

void GraphStream::Restart() { time_ = 0; }
