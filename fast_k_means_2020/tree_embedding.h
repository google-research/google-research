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

#ifndef FAST_K_MEANS_2020_TREE_EMBEDDING_H_
#define FAST_K_MEANS_2020_TREE_EMBEDDING_H_

#include <map>
#include <vector>

namespace fast_k_means {

using std::vector;
using std::map;

class TreeEmbedding {
 public:
  // Builds the tree, basically upadte the below data structures.
  void BuildTree(const vector<vector<int> >& input_points );

  // Mapping of the space to ids of the nodes of the tree, used for embedding
  // the input to tree. More details are provided in the paper.
  vector<map<vector<int>, int>> space_id;

  // Mapping of the ids of the nodes of the tree to the space they cover, used
  // for embedding the input to tree. More details are provided in the paper.
  vector<map<int, vector<int>>> id_space;

  // The children of a node of in the tree.
  vector<vector<int>> children;

  // Height of the tree.
  int height = 0;

  // Root of the tree.
  int root;

  // The first unused id in building the tree embedding. Each node of the tree
  // is assigned an id. We use this variable to find the next unused id.
  int first_unused_id = 0;

  // Number of the points of the input in a node of the tree.
  vector<int> number_points;

  // List of points in the sub-tree of a node of the tree.
  vector<vector<int>> points_in_node;
};

}  // namespace fast_k_means

#endif  // FAST_K_MEANS_2020_TREE_EMBEDDING_H_
