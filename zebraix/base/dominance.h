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

#ifndef ZEBRAIX_BASE_DOMINANCE_H_
#define ZEBRAIX_BASE_DOMINANCE_H_

#include <vector>

namespace zebraix {
namespace base {

// These are indices that are impossible in protobuf input repeated messages.
static constexpr int kSinkImportPseudoIndex = -1;
static constexpr int kSourceImportPseudoIndex = -2;

// DominanceNode indices index the vector of nodes in a DominanceGraph.
struct DominanceNode {
  int prime_rank = -1;
  int obverse_rank = -1;
  // The import_index is used contextually. If the data comes from a proto, it
  // indexes the original proto source so that, for example, one can pull
  // rendering information.
  int import_index = -1;
  std::vector<int> parents;
  std::vector<int> children;
};

struct DominanceGraph {
  // Ranges of ranks that include any imputed source or sink.
  int prime_min;
  int prime_max;
  int obverse_min;
  int obverse_max;

  // Sources and sinks are only imputed if there is not already one.
  int source_index;
  bool source_imputed;
  int sink_index;
  bool sink_imputed;

  std::vector<DominanceNode> d_nodes;
};

inline bool RankCmp(const DominanceNode& i, const DominanceNode& j) {
  return i.prime_rank < j.prime_rank;
}

// Scan nodes for ranges of ranks. Ascertain if source and/or sink nodes need to
// be imputed.
void ScanRankRanges(DominanceGraph* d_graph);
// Create source and sink nodes as required. Sort nodes in increasing order of
// prime rank.
void FleshOutGraphNodes(DominanceGraph* d_graph);

// Apply order-dimension 2 properties to construct graph connections (edges).
void ConnectGraph(DominanceGraph* d_graph);

}  // namespace base
}  // namespace zebraix

#endif  // ZEBRAIX_BASE_DOMINANCE_H_
