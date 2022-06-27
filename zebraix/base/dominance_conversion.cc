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

#include "base/dominance_conversion.h"

#include <vector>

#include "base/dominance.h"
#include "base/zebraix_graph.proto.h"

namespace zebraix {
namespace base {

void SelfMergeLayoutIntoBaseValues(zebraix_proto::ZebraixGraph* layout) {
  for (int i = 0; i < layout->nodes_size(); ++i) {
    zebraix_proto::Node p_node_copy;
    p_node_copy.CopyFrom(layout->nodes(i));
    zebraix_proto::Node* fixedup_node = layout->mutable_nodes(i);
    // Copy default node values, then merge the original node back in.
    fixedup_node->CopyFrom(layout->base_node());
    fixedup_node->MergeFrom(p_node_copy);
  }
}

void FillGraphFromProto(const zebraix_proto::ZebraixGraph& p_graph,
                        DominanceGraph* d_graph) {
  const int p_node_count = p_graph.nodes_size();
  d_graph->d_nodes.resize(p_node_count);

  for (int i = 0; i < p_node_count; ++i) {
    d_graph->d_nodes[i].import_index = i;
  }

  for (int i = 0; i < p_node_count; ++i) {
    int import_index = d_graph->d_nodes[i].import_index;
    const zebraix_proto::Node& p_node = p_graph.nodes(import_index);
    d_graph->d_nodes[i].prime_rank = p_node.prime_rank();
    d_graph->d_nodes[i].obverse_rank = p_node.obverse_rank();
  }
}

}  // namespace base
}  // namespace zebraix
