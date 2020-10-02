// Copyright 2020 The Authors.
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

//
//   Graphs (influence maximization, coverage function)
//
// Class that stores a graph (e.g. a social network)
// with special support for DBLP.

// Input format:
// For normal graphs we expect on edge per line. Each edge is expected to be two
// space separated integers.
// For DBLP, each line is expected to be an edge followed by the year.

#ifndef FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_GRAPH_H_
#define FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_GRAPH_H_

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "utilities.h"

using std::unordered_map;
using std::unordered_set;
using std::vector;

class Graph {
 public:
  // Forbids copying.
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;

  Graph(const std::string& name);

  // Returns the graph.
  static Graph& GetGraph(const std::string& name) {
    static unordered_map<std::string, Graph> name_to_graph;
    if (!name_to_graph.count(name)) {
      // Next line: "nameToGraph[name] = Graph(name)".
      name_to_graph.emplace(std::piecewise_construct, forward_as_tuple(name),
                            forward_as_tuple(name));
    }
    return name_to_graph.at(name);
  }

  // Returns the coverable vertices, i.e., the vertices with an ingoing edge.
  const vector<int>& GetCoverableVertices() const;

  // Returns the all the vertices that have an outgoing edge.
  const vector<int>& GetUniverseVertices() const;

  // Returns the list of neighbors of a vertex.
  const vector<int>& GetNeighbors(int vertex_i) const;

  // Returns the name of the graph.
  const std::string& GetName() const;

  // Name of dataset.
  const std::string name_;
  int64_t numEdges_;
  int numVertices_;

  // Neighbors[i] = list of i's neighbors.
  vector<vector<int>> neighbors_;

  // Those that have an edge out of them (the universe of f).
  vector<int> leftVertices_;

  // Those that have an edge to them (what we're covering).
  // If graph is bipartite (e.g. DBLP), these are disjoint.
  // If graph is non-bipartite, these are (usually/ideally) equal.
  vector<int> rightVertices_;

  // Publication dates for DBLP.
  unordered_map<int, unordered_set<int>> publicationDates_;
};

#endif  // FULLY_DYNAMIC_SUBMODULAR_MAXIMIZATION_GRAPH_H_
