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

#ifndef ZEBRAIX_RENDER_RENDER_STRUCTS_H_
#define ZEBRAIX_RENDER_RENDER_STRUCTS_H_

#include <iostream>
#include <string>

#include "third_party/cairo/cairo.h"  // IWYU pragma: keep
#include "base/zebraix_graph.proto.h"  // IWYU pragma: keep
// IWYU pragma: no_include "render/zebraix_rendersvg.h"
// IWYU pragma: no_include "render/zebraix_layout.h"

namespace zebraix {
namespace render {

enum class InboundDumpChoices {
  kNoDump = 0,
  kDefaults = 1,
  kImport = 2,
  // kMerged = 3,  // Withdrawn.
  // kImputed = 4, // Withdrawn.
  kEdges = 5,
  kDeAuto = 6,
};

struct ZebraixLabel {
  std::string label_text;
  zebraix_proto::LabelAnchor anchor;
};

struct ZebraixNode {
  double orig_centre_x;
  double orig_centre_y;
  double centre_x;
  double centre_y;
  double radius;

  double label_anchor_x;
  double label_anchor_y;
  ZebraixLabel label;

  zebraix_proto::ShowHide display;
};

struct ZebraixEdge {
  int start_node_index;
  int finish_node_index;

  bool forward_arrow;
  bool reverse_arrow;

  zebraix_proto::ShowHide display;
  zebraix_proto::ArrowDirection arrow_direction;
};

struct ZebraixCanvas {
  double canvas_width;
  double canvas_height;
  double canvas_x_offset;
  double canvas_y_offset;

  cairo_matrix_t global_tran;
};

struct ZebraixConfig {
  InboundDumpChoices dump_inbound_graph = InboundDumpChoices::kNoDump;
  bool draw_label_ticks = false;
  bool label_with_ranks = false;
  bool vanish_waypoints = false;
};

template <class M>
inline void DebugDumpGraph(const M& message, const ZebraixConfig& layout_config,
                           InboundDumpChoices choice) {
  if (layout_config.dump_inbound_graph == choice) {
    std::cerr << message.DebugString() << std::endl;
  }
}

}  // namespace render
}  // namespace zebraix

#endif  // ZEBRAIX_RENDER_RENDER_STRUCTS_H_
