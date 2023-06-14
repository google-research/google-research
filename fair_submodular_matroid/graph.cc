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

#include "graph.h"

#include <stddef.h>
#include <stdint.h>

#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "utilities.h"

struct hash_pair {
  size_t operator()(const std::pair<int, int>& p) const {
    return std::hash<int>{}(p.first) ^ std::hash<int>{}(p.second);
  }
};

Graph::Graph(const std::string& name) : name_(name) {
  // Technical comment: would prefer to make this constructor private,
  // but this breaks a crucial line in getGraph.

  // Update this part with the file name if adding more datasets.
  // The first file in the vector contains edges, the second colors of vertices,
  // and the third groups of vertices.
  static const auto* const name_to_filename =
      new std::map<std::string, std::vector<std::string>>{
          // Edit to add datasources in the format:
          //
          // {"graph name", {
          //   "file path to graph", "file path to colors", "file path to
          //   groups"}}
          //
          // See the README for more details.
      };

  if (!name_to_filename->count(name_)) {
    Fail("unknown graph name");
  }
  const std::string& file_name = name_to_filename->at(name_)[0];

  std::cerr << "reading graph from " << file_name << " ..." << std::endl;
  std::ifstream input(file_name);
  if (!input) {
    Fail(
        "Graph file does not exist. Please refer to README on where to"
        "download the datasets from/to.");
  }
  // renumber[x] = new number of x
  absl::btree_map<int64_t, int> renumber;
  num_vertices_ = 0;
  num_edges_ = 0;
  int64_t first_endpoint, second_endpoint;
  std::set<int> left_vertices, right_vertices;
  while (input >> first_endpoint >> second_endpoint) {
    if (!renumber.count(first_endpoint)) {
      renumber[first_endpoint] = num_vertices_;
      ++num_vertices_;
      neighbors_.push_back({});
    }
    left_vertices.insert(renumber[first_endpoint]);
    if (!renumber.count(second_endpoint)) {
      renumber[second_endpoint] = num_vertices_;
      ++num_vertices_;
      neighbors_.push_back({});
    }
    right_vertices.insert(renumber[second_endpoint]);
    num_edges_++;
    neighbors_[renumber[first_endpoint]].push_back(renumber[second_endpoint]);
    // Note: our graphs are directed.
    // But in some cases you may want to also add the reverse edge.
  }
  left_vertices_.assign(left_vertices.begin(), left_vertices.end());
  right_vertices_.assign(right_vertices.begin(), right_vertices.end());

  const std::string& file_name_colors = name_to_filename->at(name_)[1];
  std::cerr << "reading colors from " << file_name_colors << " ..."
            << std::endl;
  std::ifstream input_colors(file_name_colors);
  if (!input_colors) Fail("Color file does not exist.");

  int vertex, color;
  std::map<int, int> renumber_color;
  num_colors_ = 0;
  colors_map_.clear();
  while (input_colors >> vertex >> color) {
    if (!renumber_color.count(color)) {
      renumber_color[color] = num_colors_;
      num_colors_++;
      colors_cards_.push_back(0);
    }

    if (!renumber.count(vertex)) {
      std::cerr << "It seems that vertex " << vertex
                << " does not exist in the graph." << std::endl;
    } else {
      colors_map_[renumber[vertex]] = renumber_color[color];
      if (left_vertices.count(renumber[vertex]))
        // only count elements in V
        colors_cards_[renumber_color[color]]++;
    }
  }

  const std::string& file_name_groups = name_to_filename->at(name_)[2];
  std::cerr << "reading groups from " << file_name_groups << " ..."
            << std::endl;
  std::ifstream input_groups(file_name_groups);
  if (!input_groups) Fail("Group file does not exist.");

  int group;
  std::map<int, int> renumber_group;
  num_groups_ = 0;
  groups_map_.clear();

  while (input_groups >> vertex >> group) {
    if (!renumber_group.count(group)) {
      renumber_group[group] = num_groups_;
      num_groups_++;
      groups_cards_.push_back(0);
    }

    if (!renumber.count(vertex)) {
      std::cerr << "It seems that vertex " << vertex
                << " does not exist in the graph." << std::endl;
    } else {
      groups_map_[renumber[vertex]] = renumber_group[group];
      if (left_vertices.count(renumber[vertex]))
        // only count elements in V
        groups_cards_[renumber_group[group]]++;
    }
  }

  std::cerr << "read graph with " << num_vertices_ << " vertices (of which "
            << left_vertices.size() << " are in V) and " << num_edges_
            << " edges" << std::endl;
  std::cerr << "# of vertices with colors " << colors_map_.size() << std::endl;
  std::cerr << "# of vertices with groups " << groups_map_.size() << std::endl;

  std::cout << "colors cardinalities: ";
  for (int i = 0; i < colors_cards_.size(); i++) {
    std::cout << colors_cards_[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "groups cardinalities: ";
  for (int i = 0; i < groups_cards_.size(); i++) {
    std::cout << groups_cards_[i] << " ";
  }
  std::cout << std::endl;
}

const std::vector<int>& Graph::GetCoverableVertices() const {
  return right_vertices_;
}

const std::vector<int>& Graph::GetUniverseVertices() const {
  return left_vertices_;
}

const std::vector<int>& Graph::GetNeighbors(int vertex_i) const {
  return neighbors_[vertex_i];
}

const std::string& Graph::GetName() const { return name_; }

const std::vector<int>& Graph::GetColorsCards() const { return colors_cards_; }

const std::vector<int>& Graph::GetGroupsCards() const { return groups_cards_; }

const std::map<int, int>& Graph::GetColorsMap() const { return colors_map_; }

const std::map<int, int>& Graph::GetGroupsMap() const { return groups_map_; }
