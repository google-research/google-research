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

#ifndef ONLINE_BELIEF_PROPAGATION_GRAPH_H_
#define ONLINE_BELIEF_PROPAGATION_GRAPH_H_

#include <vector>

// Simple adjacency list implementation of an undirected graph. Vertices are
// labeled by nonnegative integers.
class Graph {
 public:
  // The graph always starts out empty after construction; vertices and
  // edges can be added incrementally.
  Graph();
  ~Graph();

  // Increases the size of adjacency_lists_ by one without adding any edges.
  void InsertVertex();

  // Adds the undirected edge (u,v) to both adjacency lists. Both endpoints are
  // assumed to be a legal vertex id, that is an integer between 0 and
  // adjacency_lists_.size() - 1. We do not verify that the edge doesn't
  // already exist as this would be too costly in an unordered graph.
  void InsertEdge(int u, int v);

  // Returns the number of vertices.
  int GetVertexNumber() const;

  // Returns the adjacency list of a specific vertex. The input should be a
  // legal vertex id.
  const std::vector<int>& GetNeighborhood(int vertex) const;

  // Sorts the adjacency lists of the graph. Enables optimal processing in graph
  // streams.
  void Sort();

  // Returns whether or not the adjacenct lists are currently sorted.
  bool IsSorted() const;

 private:
  // The adjacency lists where the graph data is stored. adjacency_lists_[i] is
  // the adjacency list of vertex "i".
  std::vector<std::vector<int>> adjacency_lists_;

  // Stores whether or not the adjacency lists are currently sorted to avoid
  // sorting redundantly.
  bool sorted_;
};

#endif  // ONLINE_BELIEF_PROPAGATION_GRAPH_H_
