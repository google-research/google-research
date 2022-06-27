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

#include "base/dominance.h"

#include <algorithm>
#include <limits>
#include <list>
#include <vector>

#include "glog/logging.h"

namespace zebraix {
namespace base {

void ScanRankRanges(DominanceGraph* d_graph) {
  d_graph->prime_max = std::numeric_limits<int>::min();
  d_graph->prime_min = std::numeric_limits<int>::max();
  d_graph->obverse_max = std::numeric_limits<int>::min();
  d_graph->obverse_min = std::numeric_limits<int>::max();
  bool has_source = false;
  bool has_sink = false;
  for (int i = 0; i < d_graph->d_nodes.size(); ++i) {
    bool new_prime_max = d_graph->d_nodes[i].prime_rank > d_graph->prime_max;
    bool new_prime_min = d_graph->d_nodes[i].prime_rank < d_graph->prime_min;
    bool new_obverse_max =
        d_graph->d_nodes[i].obverse_rank > d_graph->obverse_max;
    bool new_obverse_min =
        d_graph->d_nodes[i].obverse_rank < d_graph->obverse_min;

    if (new_prime_max) {
      d_graph->prime_max = d_graph->d_nodes[i].prime_rank;
    }
    if (new_prime_min) {
      d_graph->prime_min = d_graph->d_nodes[i].prime_rank;
    }
    if (new_obverse_max) {
      d_graph->obverse_max = d_graph->d_nodes[i].obverse_rank;
    }
    if (new_obverse_min) {
      d_graph->obverse_min = d_graph->d_nodes[i].obverse_rank;
    }

    if (new_prime_max || new_obverse_max) {
      has_sink = new_prime_max && new_obverse_max;
    }
    if (new_prime_min || new_obverse_min) {
      has_source = new_prime_min && new_obverse_min;
    }
  }

  d_graph->source_imputed = !has_source;
  d_graph->sink_imputed = !has_sink;

  if (d_graph->source_imputed) {
    --d_graph->prime_min;
    --d_graph->obverse_min;
  }
  if (d_graph->sink_imputed) {
    ++d_graph->prime_max;
    ++d_graph->obverse_max;
  }
  const int node_count = d_graph->prime_max - d_graph->prime_min + 1;
  CHECK_EQ(node_count, d_graph->obverse_max - d_graph->obverse_min + 1);
}

void FleshOutGraphNodes(DominanceGraph* d_graph) {
  if (d_graph->source_imputed) {
    DominanceNode dnode;
    dnode.import_index = kSourceImportPseudoIndex;
    dnode.prime_rank = d_graph->prime_min;
    dnode.obverse_rank = d_graph->obverse_min;
    d_graph->d_nodes.push_back(dnode);
  }
  if (d_graph->sink_imputed) {
    DominanceNode dnode;
    dnode.import_index = kSinkImportPseudoIndex;
    dnode.prime_rank = d_graph->prime_max;
    dnode.obverse_rank = d_graph->obverse_max;
    d_graph->d_nodes.push_back(dnode);
  }

  std::sort(d_graph->d_nodes.begin(), d_graph->d_nodes.end(), &RankCmp);
}

void ConnectGraph(DominanceGraph* d_graph) {
  const int node_count = d_graph->prime_max - d_graph->prime_min + 1;
  std::vector<std::list<int>> scratch_parents(node_count);
  std::vector<std::list<int>> scratch_children(node_count);

  for (int c = node_count - 1; c >= 0; --c) {
    const int child_obverse = d_graph->d_nodes[c].obverse_rank;
    int max_parent_obverse = -1;  // For each node, keep max parent rank.
    for (int p = c - 1; p >= 0; --p) {
      const int parent_obverse = d_graph->d_nodes[p].obverse_rank;
      if ((parent_obverse < child_obverse) &&
          (parent_obverse > max_parent_obverse)) {
        max_parent_obverse = parent_obverse;
        scratch_parents[c].push_back(p);
        scratch_children[p].push_back(c);
      }
    }
  }

  // Create efficient shrink-wrapped vector edge structures.
  for (int i = 0; i < node_count; ++i) {
    d_graph->d_nodes[i].parents = std::vector<int>(scratch_parents[i].cbegin(),
                                                   scratch_parents[i].cend());
    d_graph->d_nodes[i].parents.shrink_to_fit();
    d_graph->d_nodes[i].children = std::vector<int>(
        scratch_children[i].cbegin(), scratch_children[i].cend());
    d_graph->d_nodes[i].children.shrink_to_fit();
  }
}

}  // namespace base
}  // namespace zebraix
