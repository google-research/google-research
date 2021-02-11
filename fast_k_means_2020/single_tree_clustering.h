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

// Tree embedding algorithm with only a single tree.

#ifndef FAST_K_MEANS_2020_SINGLE_TREE_CLUSTERING_H_
#define FAST_K_MEANS_2020_SINGLE_TREE_CLUSTERING_H_

#include <cstdint>
#include <vector>

#include "preprocess_input_points.h"
#include "tree_embedding.h"

namespace fast_k_means {

using std::pair;
using std::vector;

class SingleTreeClustering {
 public:
  // Updates the input by preprocessing.
  // Construct the tree embedding.
  void InitializeTree(const vector<vector<double>>& input,
                      double scaling_factor);

  // Returns the benefit of opening center with optional flag to open.
  // It returns the points and their new cost.
  vector<pair<int, uint64_t>> ComputeCostAndOpen(int center, bool open_center);

  // Keeps the id of the closest center of each point.
  vector<int> closets_open_center;

 private:
  // The tree embedding.
  TreeEmbedding tree_;

  // Preproccessed input points.
  vector<vector<int>> input_;

  // For each node of the tree, keeps if it has an open center in its sub-tree.
  map<int, bool> has_open_center_;
};
}  //  namespace fast_k_means

#endif  // FAST_K_MEANS_2020_SINGLE_TREE_CLUSTERING_H_
