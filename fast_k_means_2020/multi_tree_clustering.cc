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

#include "multi_tree_clustering.h"

#include <cstdint>
#include <iostream>
#include <limits>

#include "random_handler.h"
#include "single_tree_clustering.h"

namespace fast_k_means {

void MultiTreeClustering::InitializeTree(const vector<vector<double>>& input,
                                         int number_of_trees,
                                         double scaling_factor) {
  single_trees_ = vector<SingleTreeClustering>(number_of_trees);
  for (int i = 0; i < single_trees_.size(); i++)
    single_trees_[i].InitializeTree(input, scaling_factor);
  closets_open_center = vector<int>(input.size());
  distance_to_center =
      vector<unsigned long long_t>(input.size(), std::numeric_limits<unsigned long long_t>::max());
  number_of_points_ = input.size();
  // Finding the boundries of the tree.
  binary_tree_boundery_ = 1;
  while (binary_tree_boundery_ < input.size()) binary_tree_boundery_ *= 2;
  // To ensure that it fits the binary tree
  binary_tree_value_ = vector<unsigned long long_t>(2 * binary_tree_boundery_,
                                        std::numeric_limits<unsigned long long_t>::max());
}

unsigned long long_t MultiTreeClustering::ComputeCostAndOpen(int center, bool open_center) {
  // To keep the previous costs in case open_center is false.
  map<int, unsigned long long_t> old_costs;
  // The amount of the improment that opening this center gives.
  unsigned long long_t improvement = 0;
  // This the first time so there is in previous cost so improvement is not
  // defined.
  if (!open_center &&
      distance_to_center[0] == std::numeric_limits<unsigned long long_t>::max())
    return improvement;
  for (int i = 0; i < single_trees_.size(); i++)
    for (pair<int, unsigned long long_t>& update :
         single_trees_[i].ComputeCostAndOpen(center, open_center))
      if (update.second < distance_to_center[update.first]) {
        if (old_costs.find(update.first) == old_costs.end())
          old_costs[update.first] = distance_to_center[update.first];
        improvement += distance_to_center[update.first] - update.second;
        distance_to_center[update.first] = update.second;
        if (open_center) {
          closets_open_center[update.first] = center;
          UpdateDistance(update, 0, binary_tree_boundery_, 1);
        }
      }
  if (!open_center)
    for (auto& update : old_costs)
      distance_to_center[update.first] = update.second;
  return improvement;
}

void MultiTreeClustering::UpdateDistance(pair<int, unsigned long long_t> update, int left,
                                         int right, int binary_tree_id) {
  // Finishing Condition
  if (left + 1 >= right) {
    binary_tree_value_[binary_tree_id] = update.second;
    return;
  }
  int middle = (left + right) / 2;
  if (update.first < middle)
    UpdateDistance(update, left, middle, binary_tree_id * 2);
  else
    UpdateDistance(update, middle, right, binary_tree_id * 2 + 1);
  if (binary_tree_value_[binary_tree_id * 2] !=
          std::numeric_limits<unsigned long long_t>::max() &&
      binary_tree_value_[binary_tree_id * 2 + 1] !=
          std::numeric_limits<unsigned long long_t>::max())
    binary_tree_value_[binary_tree_id] =
        binary_tree_value_[binary_tree_id * 2] +
        binary_tree_value_[binary_tree_id * 2 + 1];
  else
    binary_tree_value_[binary_tree_id] =
        std::min(binary_tree_value_[binary_tree_id * 2],
                 binary_tree_value_[binary_tree_id * 2 + 1]);
}

int MultiTreeClustering::SampleAPoint() {
  // The first point sampled is base on uniform distribution
  if (binary_tree_value_[1] == std::numeric_limits<unsigned long long_t>::max()) {
    return RandomHandler::eng() % number_of_points_;
  }
  unsigned long long_t chosen_prob = RandomHandler::eng() % binary_tree_value_[1];
  return SampleAPointRecurse(chosen_prob, 0, binary_tree_boundery_, 1);
}

int MultiTreeClustering::SampleAPointRecurse(unsigned long long_t chosen_prob, int left,
                                             int right, int binary_tree_id) {
  if (left + 1 >= right) return left;
  int middle = (left + right) / 2;
  if (chosen_prob < binary_tree_value_[binary_tree_id * 2])
    return SampleAPointRecurse(chosen_prob, left, middle, binary_tree_id * 2);
  return SampleAPointRecurse(
      chosen_prob - binary_tree_value_[binary_tree_id * 2], middle, right,
      binary_tree_id * 2 + 1);
}
}  //  namespace fast_k_means
