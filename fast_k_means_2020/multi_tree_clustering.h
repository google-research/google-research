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

// Tree embedding algorithm with multiple tree.

#ifndef FAST_K_MEANS_2020_MULTI_TREE_CLUSTERING_H_
#define FAST_K_MEANS_2020_MULTI_TREE_CLUSTERING_H_

#include <cstdint>
#include <iostream>
#include <vector>

#include "single_tree_clustering.h"

namespace fast_k_means {

using std::pair;
using std::vector;

class MultiTreeClustering {
 public:
  // Initializes the tree and datasets. Nothing technical.
  void InitializeTree(const vector<vector<double>>& input, int number_of_trees,
                      double scaling_factor);

  // Returns the benefit of opening center with optional flag to open.
  uint64_t ComputeCostAndOpen(int center, bool open_center);

  // Samples a point according to distances in the tree.
  // The correct distance are D^2.
  int SampleAPoint();

  // Keeps the id of the closest center of each point. The centers are computed
  // based on tree distances but replaced with actual distances.
  vector<int> closets_open_center;

  // The distance of each point to the closest open center.
  vector<uint64_t> distance_to_center;

 private:
  void UpdateDistance(pair<int, uint64_t> update, int left, int right,
                      int binary_tree_id);

  // Recursive function to sample points based on binary trees.
  // Improves the runtime to log n.
  int SampleAPointRecurse(uint64_t chosen_prob, int left, int right,
                          int binary_tree_id);

  // Single trees that we use for computing the distances.
  vector<SingleTreeClustering> single_trees_;

  // The distances kept in binary tree so could be easily sampled.
  vector<uint64_t> binary_tree_value_;

  // The size of the binary tree, e.g., the number points that we sample from.
  // Some point might be empty size the input size might not be a power of two.
  int binary_tree_boundery_;

  // Total number of points, i.e., input.size().
  int number_of_points_;
};
}  //  namespace fast_k_means

#endif  // FAST_K_MEANS_2020_MULTI_TREE_CLUSTERING_H_
