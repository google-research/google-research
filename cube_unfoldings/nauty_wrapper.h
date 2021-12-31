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

#ifndef CUBE_UNFOLDINGS_NAUTY_WRAPPER_H_
#define CUBE_UNFOLDINGS_NAUTY_WRAPPER_H_
#include <vector>

#include "perfect_matchings.h"

namespace cube_unfoldings {
using Generator = std::vector<int>;

struct TreeData {
  // All the edges in the complement of the tree.
  std::vector<Edge> complement_edges;
  // Generators of the automorphism group.
  std::vector<Generator> generators;
  // Number of nodes.
  size_t n;

  static TreeData FromString(const char* tree);
};
}  // namespace cube_unfoldings

#endif  // CUBE_UNFOLDINGS_NAUTY_WRAPPER_H_
