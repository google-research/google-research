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

#include "misc/misc_render.h"

#include <vector>

#include "base/dominance.h"
#include "base/zebraix_graph.proto.h"
#include "misc/misc_proto.h"

using zebraix::base::DominanceNode;
using zebraix::misc::AnchorToOctant;
using zebraix::misc::CompassToOctant;

namespace zebraix {
namespace misc {

void ApplyNodeLogic(const std::vector<DominanceNode>& d_nodes,
                    const zebraix_proto::Layout& proto_layout,
                    std::vector<zebraix_proto::Node>* target_nodes) {
  const int node_count = target_nodes->size();
  for (int p = 0; p < node_count; ++p) {
    zebraix_proto::Node* p_node = &(*target_nodes)[p];

    if (p_node->display() == zebraix_proto::SHOW_HIDE_AUTO) {
      p_node->set_display(zebraix_proto::SHOW);
    }
  }

  for (int p = 0; p < node_count; ++p) {
    zebraix_proto::Node* p_node = &(*target_nodes)[p];

    if (p_node->display() == zebraix_proto::SHOW_HIDE_AUTO) {
      p_node->set_display(zebraix_proto::SHOW);
    }

    if (p_node->anchor() == zebraix_proto::ANCHOR_AUTO) {
      int child_count = 0;
      int parent_count = 0;
      for (const auto child : d_nodes[p].children) {
        if ((*target_nodes)[child].display() != zebraix_proto::HIDE) {
          ++child_count;
        }
      }
      for (const auto parent : d_nodes[p].parents) {
        if ((*target_nodes)[parent].display() != zebraix_proto::HIDE) {
          ++parent_count;
        }
      }

      zebraix_proto::LabelAnchor anchor =
          child_count == parent_count
              ? zebraix_proto::BR
              : (child_count > parent_count
                     ? (parent_count == 0 ? zebraix_proto::R
                                          : zebraix_proto::BR)
                     : (child_count == 0 ? zebraix_proto::BL
                                         : zebraix_proto::BL));
      p_node->set_anchor(OctantToAnchor(
          AnchorToOctant(anchor) + CompassToOctant(proto_layout.direction())));
    }

    if (p_node->compass() == zebraix_proto::DIRECTION_AUTO) {
      p_node->set_compass(
          OctantToCompass(AnchorToOctant(p_node->anchor()) + 4));
    }
  }
}

}  // namespace misc
}  // namespace zebraix
