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

#include "graph.h"

#include "absl/container/node_hash_map.h"

using std::unordered_set;
using std::vector;

Graph::Graph(const std::string& name) : name_(name) {
  // Technical comment: would prefer to make this constructor private,
  // but this breaks a crucial line in getGraph.

  // Update this part with the file name.
  static const  unordered_map<std::string, std::string> name_to_filename =
      {{"amazon", "../datasets/amazon/graph-relabeled-big-amazon.txt"},
       {"enron", "../datasets/enron/Email-Enron.txt"},
       {"twitter", "../datasets/twitter/graph-relabel-twitter.txt"},
       {"dblp", "../datasets/dblp/dblp-graph.txt"},
       {"pokec", "../datasets/pokec/soc-pokec-relationships.txt"},
       {"friendster", "../datasets/friendster/friendster.txt"}};

  if (!name_to_filename.count(name_)) {
    Fail("unknown graph name");
  }
  const std::string& file_name = name_to_filename.at(name_);

  std::cerr << "reading graph from " << file_name << "..." << std::endl;
  std::ifstream input(file_name);
  if (!input) {
    Fail("graph file does not exist");
  }
  // renumber[x] = new number of x
  unordered_map<int64_t, int> renumber;
  numVertices_ = 0;
  numEdges_ = 0;
  const bool is_dblp = (name_ == "dblp");
  int64_t first_endpoint, second_endpoint;
  unordered_set<int> leftVertices, rightVertices;
  while (input >> first_endpoint >> second_endpoint) {
    if (!renumber.count(first_endpoint)) {
      renumber[first_endpoint] = numVertices_;
      ++numVertices_;
      neighbors_.push_back({});
    }
    leftVertices.insert(renumber[first_endpoint]);
    if (!renumber.count(second_endpoint)) {
      renumber[second_endpoint] = numVertices_;
      ++numVertices_;
      neighbors_.push_back({});
    }
    rightVertices.insert(renumber[second_endpoint]);
    numEdges_++;
    neighbors_[renumber[first_endpoint]].push_back(renumber[second_endpoint]);
    // Note: our graphs are directed.
    // But in some cases you may want to also add the reverse edge.

    if (is_dblp) {
      int year;
      input >> year;
      publicationDates_[renumber[first_endpoint]].insert(year);
    }
  }
  leftVertices_.assign(leftVertices.begin(), leftVertices.end());
  rightVertices_.assign(rightVertices.begin(), rightVertices.end());
  std::cerr << "read graph with " << numVertices_ << " vertices "
       << "and " << numEdges_ << " edges" << std::endl;
}

const vector<int>& Graph::GetCoverableVertices() const {
  return rightVertices_;
}

const vector<int>& Graph::GetUniverseVertices() const { return leftVertices_; }

const vector<int>& Graph::GetNeighbors(int vertex_i) const {
  return neighbors_[vertex_i];
}

const std::string& Graph::GetName() const { return name_; }
