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

#include <iostream>
#include <vector>


#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/string_view.h"
#include "render/render_structs.h"
#include "render/zebraix_layout.h"
#include "render/zebraix_rendersvg.h"

using zebraix::render::BuildSampleLayout;
using zebraix::render::InboundDumpChoices;
using zebraix::render::ZebraixCairoSvg;
using zebraix::render::ZebraixLayout;

ABSL_FLAG(int, dump_inbound_graph, 0,
          "Dump debug string of graph, often in proto form");
ABSL_FLAG(
    bool, generate_sample_graph, false,
    "Generate a simple sample of drawing elements, and do not read input");
ABSL_FLAG(bool, draw_label_ticks, false,
          "Draw short lines from node centres to label anchor points");
ABSL_FLAG(bool, label_with_ranks, false,
          "Replace labels with (prime-rank,obverse-rank)");
ABSL_FLAG(bool, vanish_waypoints, false,
          "Make waypoints shrink to nothing, and no default arrows to them");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::vector<char*> positional_args = absl::ParseCommandLine(argc, argv);

  if (positional_args.size() != 3) {
    std::cerr << "Usage:  " << argv[0] << " <input-file> <output-file>"
              << std::endl;
    return -1;
  }

  ZebraixLayout graph_layout;

  graph_layout.GetConfig().dump_inbound_graph =
      static_cast<InboundDumpChoices>(absl::GetFlag(FLAGS_dump_inbound_graph));
  graph_layout.GetConfig().draw_label_ticks =
      absl::GetFlag(FLAGS_draw_label_ticks);
  graph_layout.GetConfig().label_with_ranks =
      absl::GetFlag(FLAGS_label_with_ranks);
  graph_layout.GetConfig().vanish_waypoints =
      absl::GetFlag(FLAGS_vanish_waypoints);

  if (absl::GetFlag(FLAGS_generate_sample_graph)) {
    BuildSampleLayout(&graph_layout);
  } else {
    graph_layout.BuildLayoutFromFile(positional_args[1]);
  }

  const char* out_file = positional_args[2];

  {
    // Scoped to control destruction.
    ZebraixCairoSvg zcs(graph_layout.GetCanvas(), out_file);
    zcs.RenderToSvg(graph_layout);
    // ZebraixCairoSvg destructor performs important tear-down.
  }

  // Tear down required to avoid sanitizer errors over global caches.
  ZebraixCairoSvg::GlobalTearDown();
  return 0;
}
