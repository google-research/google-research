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

#include "render/zebraix_layout.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "base/integral_types.h"
#include "glog/logging.h"
#include "file/base/helpers.h"
#include "file/base/options.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "third_party/cairo/cairo.h"  // IWYU pragma: keep
#include "base/dominance.h"
#include "base/dominance_conversion.h"
#include "base/zebraix_graph.proto.h"
#include "misc/misc_proto.h"
#include "misc/misc_render.h"
#include "render/render_structs.h"

using zebraix::base::DominanceNode;
using zebraix::misc::ApplyNodeLogic;
using zebraix::misc::CompassToOctant;
using zebraix::misc::LeastShowy;
using zebraix::misc::NormalizeVector;

namespace zebraix {
namespace render {

namespace {

void InitRNodes(const std::vector<zebraix_proto::Node>& proto_nodes_,
                std::vector<ZebraixNode>* r_nodes) {
  const int node_count = proto_nodes_.size();
  r_nodes->resize(node_count);
}

// Also sets starting values for zebraix_nodes_.
//
// Uses Cairo transformation math.
void LayoutLayout(const DominanceGraph& d_graph,
                  const std::vector<zebraix_proto::Node>& proto_nodes_,
                  const zebraix_proto::Layout& proto_layout,
                  const ZebraixLayout& layout,  // Only for config_.
                  ZebraixCanvas* overall, std::vector<ZebraixNode>* r_nodes) {
  const int node_count = proto_nodes_.size();
  constexpr double kMaxCanvasDim =
      100 * 72.0;  // Use a realistic size limit rather than numeric limits.

  double min_centre_x = kMaxCanvasDim;
  double min_centre_y = kMaxCanvasDim;
  double max_centre_x = -kMaxCanvasDim;
  double max_centre_y = -kMaxCanvasDim;

  double node_sep = proto_layout.sep_points();

  // Initial graph is in NE direction, so calculate relative rotation.
  cairo_matrix_t graph_rotation;
  cairo_matrix_init_rotate(&graph_rotation,
                           0.25 *
                               (proto_layout.octant_rotation() +
                                CompassToOctant(proto_layout.direction())) *
                               M_PI);
  cairo_matrix_rotate(&graph_rotation,
                      -0.25 * CompassToOctant(zebraix_proto::NE) * M_PI);

  // Form stretch matrix.
  double alpha = proto_layout.stretch_alpha();
  double beta = proto_layout.stretch_beta();
  if (proto_layout.base_transform() == zebraix_proto::SQUARE_UP) {
    if (beta == 0.0) {  // Only modify if default.
      beta = -1.0 / std::sqrt(node_count);
    }
  }
  cairo_matrix_t graph_stretch;
  cairo_matrix_init(&graph_stretch, alpha, beta, beta, alpha, 0.0, 0.0);

  // Global matrix transformation for use when drawing, say, grid lines.
  // The origin is shifted with a canvas offset, so only the matrix is
  // required.
  cairo_matrix_t* global_tran = &overall->global_tran;
  cairo_matrix_init_scale(global_tran, node_sep, node_sep);
  cairo_matrix_multiply(global_tran, global_tran, &graph_stretch);
  cairo_matrix_multiply(global_tran, global_tran, &graph_rotation);

  for (int i = 0; i < node_count; ++i) {
    const zebraix_proto::Node& proto_node = proto_nodes_[i];
    ZebraixNode* new_node = &(*r_nodes)[i];

    double label_radius = proto_node.label_radius();

    new_node->orig_centre_x =
        (d_graph.d_nodes[i].prime_rank - d_graph.prime_min) * node_sep;
    new_node->orig_centre_y =
        (d_graph.d_nodes[i].obverse_rank - d_graph.obverse_min) * node_sep;

    new_node->radius = proto_node.node_radius();
    if (layout.GetConfig().vanish_waypoints &&
        (proto_node.display() == zebraix_proto::WAYPOINT)) {
      new_node->radius = 0.0;
    }

    cairo_matrix_t label_rotation;
    cairo_matrix_init_rotate(
        &label_rotation, 0.25 * CompassToOctant(proto_node.compass()) * M_PI);

    new_node->label_anchor_x = label_radius;
    new_node->label_anchor_y = 0.0;
    cairo_matrix_transform_distance(&label_rotation, &new_node->label_anchor_x,
                                    &new_node->label_anchor_y);

    if (layout.GetConfig().label_with_ranks) {
      new_node->label.label_text = absl::StrFormat(
          "(%d,%d)", proto_node.prime_rank(), proto_node.obverse_rank());
    } else {
      new_node->label.label_text = proto_node.label_text();
    }
    new_node->label.anchor = proto_node.anchor();

    new_node->display = proto_node.display();

    new_node->centre_x = new_node->orig_centre_x;
    new_node->centre_y = new_node->orig_centre_y;

    // Transform centre and label pos.
    cairo_matrix_transform_distance(&graph_stretch, &new_node->centre_x,
                                    &new_node->centre_y);
    cairo_matrix_transform_distance(&graph_rotation, &new_node->centre_x,
                                    &new_node->centre_y);
    cairo_matrix_transform_distance(&graph_rotation, &new_node->label_anchor_x,
                                    &new_node->label_anchor_y);

    double preferred_distance = proto_node.distance();
    if (preferred_distance != 0.0) {
      // Since we traverse nodes in topological sort, we know that parent node
      // locations are already established.
      double parent_average_x = 0.0;
      double parent_average_y = 0.0;
      int parent_count = 0;
      for (auto parent : d_graph.d_nodes[i].parents) {
        const ZebraixNode& r_node = (*r_nodes)[parent];
        if (r_node.display != zebraix_proto::HIDE) {
          parent_average_x += r_node.centre_x;
          parent_average_y += r_node.centre_y;
          ++parent_count;
        }
      }
      if (parent_count > 0) {
        parent_average_x = parent_average_x / parent_count;
        parent_average_y = parent_average_y / parent_count;
        double displacement_x = new_node->centre_x - parent_average_x;
        double displacement_y = new_node->centre_y - parent_average_y;
        NormalizeVector(&displacement_x, &displacement_y);
        new_node->centre_x =
            parent_average_x + displacement_x * preferred_distance;
        new_node->centre_y =
            parent_average_y + displacement_y * preferred_distance;
      }
    }

    if (new_node->display != zebraix_proto::HIDE) {
      min_centre_x = std::min(min_centre_x, new_node->centre_x);
      min_centre_y = std::min(min_centre_y, new_node->centre_y);
      max_centre_x = std::max(max_centre_x, new_node->centre_x);
      max_centre_y = std::max(max_centre_y, new_node->centre_y);
    }
  }

  double base_margin = proto_layout.base_margin();
  if (base_margin == 0.0) {
    base_margin = node_sep;
  }
  min_centre_x -= 1.5 * base_margin;
  min_centre_y -= 0.75 * base_margin;
  max_centre_x += 1.5 * base_margin;
  max_centre_y += 0.75 * base_margin;

  overall->canvas_width = max_centre_x - min_centre_x;
  overall->canvas_height = max_centre_y - min_centre_y;
  overall->canvas_x_offset = -min_centre_x;
  overall->canvas_y_offset = max_centre_y;
}

void BuildRendererNodes(const DominanceGraph& d_graph,
                        const zebraix_proto::ZebraixGraph& proto_layout_,
                        std::vector<zebraix_proto::Node>* proto_nodes_) {
  proto_nodes_->resize(d_graph.d_nodes.size());
  for (int i = 0; i < d_graph.d_nodes.size(); ++i) {
    const DominanceNode& d_node = d_graph.d_nodes[i];
    if (d_node.import_index == base::kSourceImportPseudoIndex) {
      zebraix_proto::Node& source = (*proto_nodes_)[i];
      source.CopyFrom(proto_layout_.base_node());
      zebraix_proto::Node default_source;
      CHECK(proto2::TextFormat::ParseFromString(misc::kImputedSource,
                                                &default_source));
      source.MergeFrom(default_source);

      source.set_prime_rank(d_node.prime_rank);
      source.set_obverse_rank(d_node.obverse_rank);

      source.set_display(proto_layout_.layout().source_display());
      if (source.display() == zebraix_proto::SHOW_HIDE_AUTO) {
        source.set_display(zebraix_proto::SHOW);
      }
    } else if (d_node.import_index == base::kSinkImportPseudoIndex) {
      zebraix_proto::Node* sink = &(*proto_nodes_)[i];
      sink->CopyFrom(proto_layout_.base_node());
      zebraix_proto::Node default_sink;
      CHECK(proto2::TextFormat::ParseFromString(misc::kImputedSink,
                                                &default_sink));
      sink->MergeFrom(default_sink);

      sink->set_prime_rank(d_node.prime_rank);
      sink->set_obverse_rank(d_node.obverse_rank);

      sink->set_display(proto_layout_.layout().sink_display());
      if (sink->display() == zebraix_proto::SHOW_HIDE_AUTO) {
        sink->set_display(zebraix_proto::HIDE);
      }
    } else {
      const zebraix_proto::Node& p_node =
          proto_layout_.nodes(d_node.import_index);
      (*proto_nodes_)[i].CopyFrom(p_node);
    }
  }
}

void BuildRendererEdges(
    const zebraix_proto::ZebraixGraph& layout,
    const std::vector<DominanceNode>& d_nodes,
    const std::vector<ZebraixNode>& r_nodes,
    absl::flat_hash_map<std::pair<int, int>, ZebraixEdge>* r_edges) {
  zebraix_proto::Edge base_edge = layout.base_edge();
  if (base_edge.arrow() == zebraix_proto::ARROW_AUTO) {
    base_edge.set_arrow(zebraix_proto::ARROW_FORWARD);
  }
  // const zebraix_proto::Edge& base_edge = layout.base_edge();

  const int node_count = d_nodes.size();

  for (int i = 0; i < node_count; ++i) {
    for (auto child : d_nodes[i].children) {
      ZebraixEdge& r_edge = (*r_edges)[std::make_pair(i, child)];
      r_edge.start_node_index = i;
      r_edge.finish_node_index = child;
      r_edge.arrow_direction = base_edge.arrow();

      if (base_edge.display() == zebraix_proto::SHOW_HIDE_AUTO) {
        r_edge.display = LeastShowy(r_nodes[i].display, r_nodes[child].display);
      } else {
        r_edge.display = base_edge.display();
      }
    }
  }

  absl::flat_hash_map<int, int> prime_to_canonical;
  for (int i = 0; i < node_count; ++i) {
    int prime_rank = d_nodes[i].prime_rank;
    prime_to_canonical[prime_rank] = i;
  }

  for (const auto& proto_edge : layout.edges()) {
    ZebraixEdge& r_edge =
        (*r_edges)[std::make_pair(prime_to_canonical[proto_edge.parent()],
                                  prime_to_canonical[proto_edge.child()])];
    if (proto_edge.display() != zebraix_proto::SHOW_HIDE_AUTO) {
      r_edge.display = proto_edge.display();
    }

    r_edge.arrow_direction = proto_edge.arrow();
  }

  for (int i = 0; i < node_count; ++i) {
    for (auto child : d_nodes[i].children) {
      ZebraixEdge& r_edge = (*r_edges)[std::make_pair(i, child)];
      const auto arrow_choice = r_edge.arrow_direction;
      switch (arrow_choice) {
        case zebraix_proto::ARROW_AUTO:
        case zebraix_proto::ARROW_FORWARD:
          r_edge.forward_arrow = true;
          break;
        case zebraix_proto::ARROW_BIDRECTIONAL:
          r_edge.forward_arrow = true;
          r_edge.reverse_arrow = true;
          break;
        case zebraix_proto::ARROW_REVERSE:
          r_edge.reverse_arrow = true;
          break;
        case zebraix_proto::ARROW_NONE:
        default:
          break;
      }
    }
  }
}

}  // namespace

void ZebraixLayout::BuildLayoutFromLayoutProto(
    const zebraix_proto::ZebraixGraph& input_proto) {
  using zebraix::base::ScanRankRanges;
  using zebraix::base::SelfMergeLayoutIntoBaseValues;

  proto_layout_.MergeFrom(input_proto);
  SelfMergeLayoutIntoBaseValues(&proto_layout_);

  FillGraphFromProto(proto_layout_, &dominance_graph_);
  ScanRankRanges(&dominance_graph_);
  FleshOutGraphNodes(&dominance_graph_);

  CHECK_GT(proto_layout_.nodes_size(), 0);

  ConnectGraph(&dominance_graph_);

  BuildRendererNodes(dominance_graph_, proto_layout_, &this->proto_nodes_);

  if (config_.dump_inbound_graph == InboundDumpChoices::kEdges) {
    DumpEdges();
  }

  ApplyNodeLogic(dominance_graph_.d_nodes, proto_layout_.layout(),
                 &proto_nodes_);

  DebugDumpGraph(proto_layout_, this->config_, InboundDumpChoices::kDeAuto);
  InitRNodes(proto_nodes_, &zebraix_nodes_);

  // Progress marker.
  LayoutLayout(dominance_graph_, proto_nodes_, proto_layout_.layout(), *this,
               &canvas_, &zebraix_nodes_);

  BuildRendererEdges(proto_layout_, dominance_graph_.d_nodes, zebraix_nodes_,
                     &zebraix_edges_);
}

void ZebraixLayout::BuildLayoutFromFile(const std::string& input_filename) {
  CHECK(proto2::TextFormat::ParseFromString(misc::kLayoutDefaultPbTxt,
                                            &proto_layout_));
  DebugDumpGraph(proto_layout_, this->config_, InboundDumpChoices::kDefaults);

  std::string initial_layout_pb_txt;
  // In future consider "preferred" ::util::Status GetTextProto.
  CHECK_OK(file::GetContents(input_filename, &initial_layout_pb_txt,
                             file::Defaults()));
  BuildLayoutFromProtoString(initial_layout_pb_txt);
}

void ZebraixLayout::DumpEdges() {
  const int node_count = proto_nodes_.size();

  for (int i = 0; i < node_count; ++i) {
    std::cerr << proto_nodes_[i].label_text() << ":" << std::endl;
    for (auto parent : dominance_graph_.d_nodes[i].parents) {
      std::cerr << "   " << proto_nodes_[parent].label_text() << " -->"
                << std::endl;
    }
    for (auto child : dominance_graph_.d_nodes[i].children) {
      std::cerr << "   --> " << proto_nodes_[child].label_text() << std::endl;
    }
  }
}

void ZebraixLayout::BuildLayoutFromProtoString(
    const std::string& proto_string) {
  zebraix_proto::ZebraixGraph loaded_layout;

  CHECK(proto2::TextFormat::ParseFromString(proto_string, &loaded_layout));
  DebugDumpGraph(loaded_layout, this->config_, InboundDumpChoices::kImport);

  BuildLayoutFromLayoutProto(loaded_layout);
}

}  // namespace render
}  // namespace zebraix
