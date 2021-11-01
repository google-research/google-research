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

#ifndef ONLINE_BELIEF_PROPAGATION_GRAPH_STREAM_H_
#define ONLINE_BELIEF_PROPAGATION_GRAPH_STREAM_H_

#include <vector>

#include "graph.h"

class GraphStream {
 public:
  GraphStream(const Graph* graph);
  ~GraphStream();

  // Return the current number of vertices.
  int GetCurrentVertexNumber() const;

  // Return the number of vertices at the end of stream.
  int GetTotalVertexNumber() const;

  // Prune the adjacency list of vertex to include only vertices below time_.
  std::vector<int> GetCurrentNeighborhood(int vertex) const;

  // Return whether the edge (u,v) is in the graph, by performing binary
  // search on the adjacency list of u. If either of u or v is not below time_
  // automatically return false.
  bool CheckEdge(int u, int v) const;

  // Increase time_ by 1.
  void Step();

  // Reset time_ to 0.
  void Restart();

 private:
  // The underlying graph. Should be sorted at time of construction.
  const Graph* graph_;

  // The number of vertices of the underlying graph that have been processed.
  // Since the vertices of graph_ are assumed to be in temporal order, these are
  // assumed to be exactly the vertices labeled by 0..time_-1.
  // time_ is set to 0 at construction.
  int time_;
};

#endif  // ONLINE_BELIEF_PROPAGATION_GRAPH_STREAM_H_
