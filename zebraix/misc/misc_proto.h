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

#ifndef ZEBRAIX_MISC_MISC_PROTO_H_
#define ZEBRAIX_MISC_MISC_PROTO_H_

// Functions that interpret the content of proto graph layout.

#include "base/zebraix_graph.proto.h"

namespace zebraix {
namespace misc {

inline int AnchorToOctant(zebraix_proto::LabelAnchor anchor) {
  switch (anchor) {
    case zebraix_proto::R:
      return 0;
    case zebraix_proto::TR:
      return 1;
    case zebraix_proto::T:
      return 2;
    case zebraix_proto::TL:
      return 3;
    case zebraix_proto::L:
      return 4;
    case zebraix_proto::BL:
      return 5;
    case zebraix_proto::B:
      return 6;
    case zebraix_proto::BR:
      return 7;
    case zebraix_proto::ANCHOR_AUTO:
    default:
      return 0;
  }
}

inline zebraix_proto::LabelAnchor OctantToAnchor(int octant) {
  switch (octant & 7) {
    case 0:
      return zebraix_proto::R;
    case 1:
      return zebraix_proto::TR;
    case 2:
      return zebraix_proto::T;
    case 3:
      return zebraix_proto::TL;
    case 4:
      return zebraix_proto::L;
    case 5:
      return zebraix_proto::BL;
    case 6:
      return zebraix_proto::B;
    case 7:
    default:
      return zebraix_proto::BR;
  }
}

inline int CompassToOctant(zebraix_proto::LayoutDirection compass) {
  switch (compass) {
    case zebraix_proto::E:
      return 0;
    case zebraix_proto::NE:
      return 1;
    case zebraix_proto::N:
      return 2;
    case zebraix_proto::NW:
      return 3;
    case zebraix_proto::W:
      return 4;
    case zebraix_proto::SW:
      return 5;
    case zebraix_proto::S:
      return 6;
    case zebraix_proto::SE:
      return 7;
    case zebraix_proto::DIRECTION_AUTO:
    default:
      return 0;
  }
}

inline zebraix_proto::LayoutDirection OctantToCompass(int octant) {
  switch (octant & 7) {
    case 0:
      return zebraix_proto::E;
    case 1:
      return zebraix_proto::NE;
    case 2:
      return zebraix_proto::N;
    case 3:
      return zebraix_proto::NW;
    case 4:
      return zebraix_proto::W;
    case 5:
      return zebraix_proto::SW;
    case 6:
      return zebraix_proto::S;
    case 7:
    default:
      return zebraix_proto::SE;
  }
}

}  // namespace misc
}  // namespace zebraix

#endif  // ZEBRAIX_MISC_MISC_PROTO_H_
