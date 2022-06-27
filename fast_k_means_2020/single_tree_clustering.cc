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

#include "single_tree_clustering.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <set>

#include "preprocess_input_points.h"
#include "tree_embedding.h"

namespace fast_k_means {

using std::set;
void SingleTreeClustering::InitializeTree(const vector<vector<double>>& input,
                                          double scaling_factor) {
  // Preprocessing the input.
  input_ = PreProcessInputPoints::ScaleToIntSpace(input, scaling_factor);
  PreProcessInputPoints::ShiftToDimensionsZero(&input_);
  PreProcessInputPoints::RandomShiftSpace(&input_);

  // Embedding the input_ to a tree.
  tree_.BuildTree(input_);
  // Setting all the initial distances to infinity.
  closets_open_center = vector<int>(input_.size(), -1);
}

vector<pair<int, uint64_t>> SingleTreeClustering::ComputeCostAndOpen(
    int center, bool open_center) {
  // The new distances if this center gets opened.
  vector<pair<int, uint64_t>> updated_distances;
  // The nodes that their distance is updated
  set<int> updated_nodes;
  vector<int> center_coordinate = input_[center];
  for (int i = 0; i < tree_.height; i++) {
    // The node of the tree in this height.
    int node = tree_.space_id[i].find(center_coordinate)->second;
    // The cost of nodes above this height will not change.
    if (has_open_center_[node]) break;
    if (open_center) has_open_center_[node] = true;
    for (auto point : tree_.points_in_node[node]) {
      if (updated_nodes.find(point) == updated_nodes.end()) {
        updated_distances.push_back(
            pair<int, uint64_t>(point, static_cast<uint64_t>(1) << (2 * i)));
        if (open_center) closets_open_center[point] = center;
        updated_nodes.insert(point);
      }
    }
    for (int j = 0; j < center_coordinate.size(); j++)
      center_coordinate[j] /= 2;
  }
  return updated_distances;
}

}  // namespace fast_k_means
