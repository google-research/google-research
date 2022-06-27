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

#include "tree_embedding.h"

#include <iostream>
namespace fast_k_means {

void TreeEmbedding::BuildTree(const vector<vector<int> >& input_points) {
  id_space.push_back(map<int, vector<int> >());
  space_id.push_back(map<vector<int>, int>());
  // Constructing the first layer of the tree.
  for (int i = 0; i < input_points.size(); i++) {
    if (space_id[height].find(input_points[i]) == space_id[height].end()) {
      id_space[height][first_unused_id] = input_points[i];
      space_id[height][input_points[i]] = first_unused_id++;
      number_points.push_back(0);
      children.push_back(vector<int>(0));
      points_in_node.push_back(vector<int>(0));
    }
    number_points[space_id[height][input_points[i]]]++;
    points_in_node[space_id[height][input_points[i]]].push_back(i);
  }
  // If the size is one, then we have reached the root and construction is done.
  while (space_id[height].size() > 1) {
    id_space.push_back(map<int, vector<int> >());
    space_id.push_back(map<vector<int>, int>());
    for (const auto& e : space_id[height]) {
      vector<int> e_space = e.first;
      int e_int = e.second;
      for (int i = 0; i < e_space.size(); i++) e_space[i] /= 2;
      if (space_id[height + 1].find(e_space) == space_id[height + 1].end()) {
        id_space[height + 1][first_unused_id] = e_space;
        space_id[height + 1][e_space] = first_unused_id++;
        number_points.push_back(0);
        children.push_back(vector<int>(0));
        points_in_node.push_back(vector<int>(0));
      }
      int current_id = space_id[height + 1][e_space];
      number_points[current_id] += number_points[e_int];
      children[current_id].push_back(e_int);
      for (auto points : points_in_node[e_int])
        points_in_node[current_id].push_back(points);
    }
    height++;
  }
  root = space_id[height++].begin()->second;
}

}  // namespace fast_k_means
