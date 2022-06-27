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

#include "perfect_matchings.h"

#include <stdio.h>

namespace cube_unfoldings {
namespace {
// Tries to extend the matching in `matching` with edges in `edges` (with index
// >= `first_edge`). Keeps track in `used_node` of what nodes are already part
// of the matching, and skips edges that have either end already in the
// matching.
void AllPerfectMatchings(Matching& matching, const std::vector<Edge>& edges,
                         int first_edge, std::vector<char>& used_nodes,
                         absl::FunctionRef<void(const Matching&)> cb) {
  if (matching.size() * 2 == used_nodes.size()) {
    cb(matching);
    return;
  }
  for (int i = first_edge; i < edges.size(); i++) {
    std::array<int, 2> edge = edges[i];
    if (used_nodes[edge[0]] || used_nodes[edge[1]]) continue;
    // Adds an edge to the matching
    matching.push_back(edge);
    used_nodes[edge[0]] = true;
    used_nodes[edge[1]] = true;
    AllPerfectMatchings(matching, edges, i + 1, used_nodes, cb);
    // Removes the edge again, since we collected already all corresponding
    // matchings.
    matching.pop_back();
    used_nodes[edge[0]] = false;
    used_nodes[edge[1]] = false;
  }
}
}  // namespace

void AllPerfectMatchings(const std::vector<Edge>& edges, size_t n,
                         absl::FunctionRef<void(const Matching&)> cb) {
  Matching matching;
  std::vector<char> used_nodes(n, 0);
  AllPerfectMatchings(matching, edges, 0, used_nodes, cb);
}
}  // namespace cube_unfoldings
