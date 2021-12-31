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

#ifndef ZEBRAIX_MISC_MISC_RENDER_H_
#define ZEBRAIX_MISC_MISC_RENDER_H_

// Functions that apply rendering rules to .

#include <cmath>
#include <limits>
#include <vector>

#include "base/dominance.h"
#include "base/zebraix_graph.proto.h"

namespace zebraix {
namespace misc {

static constexpr char kLayoutDefaultPbTxt[] =
    "base_node {"
    "   label_text: \"Missing label\","
    "   prime_rank: -1,"
    "   node_radius: 10.5,"
    "   label_radius: 18.0,"
    "}"
    "layout {"
    "  direction: E,"
    "  sep_points: 72.0,"
    "}";
static constexpr char kImputedSource[] =
    "label_text: \"Extrapolated source\",";  // Ranks edited in.
static constexpr char kImputedSink[] =
    "label_text: \"Extrapolated sink\",";  // Ranks edited in.

inline void NormalizeVector(double* x, double* y) {
  double norm = 1.0 / std::sqrt(*x * *x + *y * *y +
                                std::numeric_limits<double>::epsilon());
  *x = *x * norm;
  *y = *y * norm;
}

inline zebraix_proto::ShowHide LeastShowy(zebraix_proto::ShowHide v1,
                                          zebraix_proto::ShowHide v2) {
  if (v1 == v2) {
    return v1;
  } else if (v1 == zebraix_proto::HIDE || v2 == zebraix_proto::HIDE) {
    return zebraix_proto::HIDE;
  } else if (v1 == zebraix_proto::GHOST || v2 == zebraix_proto::GHOST) {
    return zebraix_proto::GHOST;
  } else if (v1 == zebraix_proto::WAYPOINT || v2 == zebraix_proto::WAYPOINT) {
    return zebraix_proto::WAYPOINT;
  } else {
    return v1;
  }
}

void ApplyNodeLogic(const std::vector<zebraix::base::DominanceNode>& d_nodes,
                    const zebraix_proto::Layout& proto_layout,
                    std::vector<zebraix_proto::Node>* target_nodes);

}  // namespace misc
}  // namespace zebraix

#endif  // ZEBRAIX_MISC_MISC_RENDER_H_
