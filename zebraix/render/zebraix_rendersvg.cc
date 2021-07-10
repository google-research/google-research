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

#include "render/zebraix_rendersvg.h"

#include <math.h>

#include <iostream>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "absl/container/flat_hash_map.h"
#include "third_party/cairo/cairo-svg.h"
#include "third_party/cairo/cairo.h"  // IWYU pragma: keep
// IWYU pragma: no_include "third_party/cairo/v1_12_16/src/cairo.h"
#include "third_party/fontconfig/fontconfig/fontconfig.h"
// IWYU pragma: no_include "third_party/glib/src/gobject/gobject.h"
#include "base/dominance.h"
#include "misc/misc_proto.h"
#include "misc/misc_render.h"
#include "render/render_structs.h"
#include "render/zebraix_layout.h"
#include "third_party/pango/pango/pangocairo.h"
#include "third_party/pango/pango/pangoft2.h"
// IWYU pragma: no_include "third_party/pango/src/pango/pango-font.h"
// IWYU pragma: no_include "third_party/pango/src/pango/pango-fontmap.h"
// IWYU pragma: no_include "third_party/pango/src/pango/pango-layout.h"
// IWYU pragma: no_include "third_party/pango/src/pango/pango-types.h"

using zebraix::misc::NormalizeVector;
using zebraix::misc::OctantToAnchor;
using zebraix::render::ZebraixCanvas;
using zebraix::render::ZebraixLayout;

namespace zebraix {
namespace render {

namespace {
// TODO(b/193166687): Move constants into base "default" config.
constexpr double kLineWidth = 1.5;
constexpr double kArrowSize = 12.0;
constexpr double kLabelFontSize = 12.0;
constexpr double kDashLength = 6.0;
constexpr double kArrowAngle = 25.0;
constexpr double kArrowOffsetFraction = 1.25;
constexpr double kDotLength = kLineWidth;
constexpr double kGridGray = 0.60;
static double kGridDashLength[2] = {4, 4};

void CairoRenderLabel(double x, double y, const ZebraixLabel& label,
                      cairo_t* cr) {
  CHECK(!label.label_text.empty());
  PangoLayout* layout;
  PangoContext* pango_context;

  PangoFontDescription* font_description;
  font_description = pango_font_description_new();
  pango_font_description_set_family(font_description, "sans");
  pango_font_description_set_absolute_size(font_description,
                                           kLabelFontSize * PANGO_SCALE);

  pango_context = pango_cairo_create_context(cr);
  layout = pango_layout_new(pango_context);
  pango_layout_set_font_description(layout, font_description);
  pango_layout_set_text(layout, label.label_text.c_str(), -1);

  cairo_set_source_rgb(cr, 0.0, 0.0, 1.0);

  double adjust_x = 0.0;
  double adjust_y = 0.0;

  PangoRectangle logical_extents;
  pango_layout_get_extents(layout, nullptr /*=ink_extents*/, &logical_extents);
  switch (label.anchor) {
    case zebraix_proto::TL:
      break;
    case zebraix_proto::T:
      adjust_x = 0.5 * logical_extents.width;
      break;
    case zebraix_proto::TR:
      adjust_x = 1.0 * logical_extents.width;
      break;
    case zebraix_proto::R:
      adjust_x = 1.0 * logical_extents.width;
      adjust_y = -0.5 * logical_extents.height;
      break;
    case zebraix_proto::BR:
      adjust_x = 1.0 * logical_extents.width;
      adjust_y = -1.0 * logical_extents.height;
      break;
    case zebraix_proto::B:
      adjust_x = 0.5 * logical_extents.width;
      adjust_y = -1.0 * logical_extents.height;
      break;
    case zebraix_proto::BL:
      adjust_y = -1.0 * logical_extents.height;
      break;
    case zebraix_proto::L:
      adjust_y = -0.5 * logical_extents.height;
      break;
    default:
      CHECK(false);
      break;
  }

  cairo_move_to(cr, x - adjust_x / PANGO_SCALE, -(y - adjust_y / PANGO_SCALE));
  pango_cairo_show_layout(cr, layout);

  pango_cairo_font_map_set_default(nullptr);
  g_object_unref(layout);
  g_object_unref(pango_context);
  pango_font_description_free(font_description);
}

void CairoRenderEdge(const ZebraixEdge& r_edge,
                     const std::vector<ZebraixNode>& nodes, cairo_t* cr) {
  const ZebraixNode& start_node = nodes[r_edge.start_node_index];
  const ZebraixNode& finish_node = nodes[r_edge.finish_node_index];
  const zebraix_proto::ShowHide display = r_edge.display;
  if (display == zebraix_proto::HIDE) {
    return;
  }
  if (display == zebraix_proto::GHOST) {
    cairo_set_dash(cr, &kDashLength, 1, kArrowOffsetFraction * kDashLength);
  }

  double delta_x = finish_node.centre_x - start_node.centre_x;
  double delta_y = finish_node.centre_y - start_node.centre_y;
  NormalizeVector(&delta_x, &delta_y);
  double forward_tip_x = finish_node.centre_x - delta_x * finish_node.radius;
  double forward_tip_y = finish_node.centre_y - delta_y * finish_node.radius;
  double reverse_tip_x = start_node.centre_x + delta_x * start_node.radius;
  double reverse_tip_y = start_node.centre_y + delta_y * start_node.radius;

  double arrow_1_x = -delta_x * kArrowSize;
  double arrow_1_y = -delta_y * kArrowSize;
  double arrow_2_x = arrow_1_x;
  double arrow_2_y = arrow_1_y;
  cairo_matrix_t arrow_rotate;

  cairo_matrix_init_rotate(&arrow_rotate, kArrowAngle / 180.0 * M_PI);
  cairo_matrix_transform_distance(&arrow_rotate, &arrow_1_x, &arrow_1_y);
  cairo_matrix_init_rotate(&arrow_rotate, -kArrowAngle / 180.0 * M_PI);
  cairo_matrix_transform_distance(&arrow_rotate, &arrow_2_x, &arrow_2_y);

  // Draw main line (of arrow) from destination to source, so that dashed lines
  // are drawn at arrow.
  cairo_set_source_rgb(cr, 0, 0, 0);
  cairo_set_line_width(cr, kLineWidth);

  cairo_move_to(cr, forward_tip_x, -forward_tip_y);
  cairo_line_to(cr, reverse_tip_x, -reverse_tip_y);
  cairo_stroke(cr);

  cairo_set_dash(cr, &kDashLength, 0, 0.0);

  if (r_edge.forward_arrow) {
    cairo_move_to(cr, forward_tip_x, -forward_tip_y);
    cairo_line_to(cr, forward_tip_x + arrow_1_x, -(forward_tip_y + arrow_1_y));
    cairo_stroke(cr);
    cairo_move_to(cr, forward_tip_x, -forward_tip_y);
    cairo_line_to(cr, forward_tip_x + arrow_2_x, -(forward_tip_y + arrow_2_y));
    cairo_stroke(cr);
  }
  if (r_edge.reverse_arrow) {
    cairo_move_to(cr, reverse_tip_x, -reverse_tip_y);
    cairo_line_to(cr, reverse_tip_x - arrow_1_x, -(reverse_tip_y - arrow_1_y));
    cairo_stroke(cr);
    cairo_move_to(cr, reverse_tip_x, -reverse_tip_y);
    cairo_line_to(cr, reverse_tip_x - arrow_2_x, -(reverse_tip_y - arrow_2_y));
    cairo_stroke(cr);
  }
}

void CairoRenderGrid(const ZebraixLayout& graph_layout, cairo_t* cr) {
  const double grid_thickness =
      graph_layout.GetMainLayoutProto().layout().grid_thickness();
  if (grid_thickness != 0.0) {
    cairo_set_source_rgb(cr, 1.0 - kGridGray, 1.0 - kGridGray, 1.0 - kGridGray);
    cairo_set_line_width(cr, grid_thickness);

    for (int x = graph_layout.GetDominance().prime_min;
         x <= graph_layout.GetDominance().prime_max; ++x) {
      if (x == 0) {
        cairo_set_dash(cr, kGridDashLength, 0, 0.0);
      } else {
        cairo_set_dash(cr, kGridDashLength, 2, 0.0);
      }

      double y_begin = graph_layout.GetDominance().obverse_min - 0.5;
      double y_end = graph_layout.GetDominance().obverse_max + 0.5;
      double x_begin = 1.0 * x;
      double x_end = x_begin;

      cairo_matrix_transform_distance(&graph_layout.GetCanvas().global_tran,
                                      &x_begin, &y_begin);
      cairo_matrix_transform_distance(&graph_layout.GetCanvas().global_tran,
                                      &x_end, &y_end);
      cairo_move_to(cr, x_begin, -y_begin);
      cairo_line_to(cr, x_end, -y_end);
      cairo_stroke(cr);
    }
    for (int y = graph_layout.GetDominance().obverse_min;
         y <= graph_layout.GetDominance().obverse_max; ++y) {
      if (y == 0) {
        cairo_set_dash(cr, kGridDashLength, 0, 0.0);
      } else {
        cairo_set_dash(cr, kGridDashLength, 2, 0.0);
      }

      double x_begin = graph_layout.GetDominance().prime_min - 0.5;
      double x_end = graph_layout.GetDominance().prime_max + 0.5;
      double y_begin = 1.0 * y;
      double y_end = y_begin;

      cairo_matrix_transform_distance(&graph_layout.GetCanvas().global_tran,
                                      &x_begin, &y_begin);
      cairo_matrix_transform_distance(&graph_layout.GetCanvas().global_tran,
                                      &x_end, &y_end);
      cairo_move_to(cr, x_begin, -y_begin);
      cairo_line_to(cr, x_end, -y_end);
      cairo_stroke(cr);
    }
    cairo_set_dash(cr, kGridDashLength, 0, 0.0);
  }

  cairo_set_line_width(cr, kLineWidth);
}

void CairoRenderNode(const ZebraixNode& r_node,
                     const ZebraixLayout& graph_layout, cairo_t* cr) {
  if (r_node.display == zebraix_proto::HIDE) {
    return;
  }
  if (r_node.display == zebraix_proto::GHOST) {
    cairo_set_dash(cr, &kDashLength, 1, 0.0);
  } else if (r_node.display == zebraix_proto::WAYPOINT) {
    cairo_set_dash(cr, &kDotLength, 1, 0.0);
  }

  cairo_set_source_rgb(cr, 0, 0, 0);

  if (r_node.radius != 0) {
    cairo_move_to(cr, r_node.centre_x + r_node.radius, -r_node.centre_y);
    cairo_arc(cr, r_node.centre_x, -r_node.centre_y, r_node.radius, 0.0 * M_PI,
              2.0 * M_PI);
    cairo_stroke(cr);
  }

  cairo_set_dash(cr, &kDashLength, 0, 0.0);

  if (!r_node.label.label_text.empty() &&
      (r_node.display != zebraix_proto::HIDE)) {
    CairoRenderLabel(r_node.centre_x + r_node.label_anchor_x,
                     r_node.centre_y + r_node.label_anchor_y, r_node.label, cr);
  }

  if (graph_layout.GetConfig().draw_label_ticks &&
      (r_node.display != zebraix_proto::HIDE)) {
    cairo_move_to(cr, r_node.centre_x, -r_node.centre_y);
    cairo_line_to(cr, r_node.centre_x + r_node.label_anchor_x,
                  -(r_node.centre_y + r_node.label_anchor_y));
    cairo_stroke(cr);
  }
}

void CairoRenderLayout(const ZebraixLayout& graph_layout, cairo_t* cr) {
  CairoRenderGrid(graph_layout, cr);
  cairo_set_line_width(cr, kLineWidth);

  for (const auto& r_node : graph_layout.GetNodes()) {
    CairoRenderNode(r_node, graph_layout, cr);
  }

  for (int p = 0; p < graph_layout.GetNodes().size(); ++p) {
    for (const auto c : graph_layout.GetDominance().d_nodes[p].children) {
      CairoRenderEdge(
          graph_layout.GetEdges().find(std::make_pair(p, c))->second,
          graph_layout.GetNodes(), cr);
    }
  }
}

}  // namespace

// TODO(b/193165551): Remove and create better exemplary samples and tests.
void BuildSampleLayout(ZebraixLayout* graph_layout) {
  constexpr double kNodeRadius = 10.5;
  constexpr double kLabelRadius = 16.5;
  constexpr double kNodeSep = 72.0;

  graph_layout->GetNodes().resize(9);
  graph_layout->GetDominance().d_nodes.resize(9);
  for (int i = 0; i < 8; ++i) {
    ZebraixNode& r_node = graph_layout->GetNodes()[i];
    ZebraixEdge& r_edge = graph_layout->GetEdges()[std::make_pair(8, i)];
    r_node.radius = kNodeRadius;

    cairo_matrix_t rotation_matrix;
    cairo_matrix_init_rotate(&rotation_matrix, 0.25 * i * M_PI);

    r_node.centre_x = kNodeSep;
    r_node.centre_y = 0.0;
    cairo_matrix_transform_distance(&rotation_matrix, &r_node.centre_x,
                                    &r_node.centre_y);

    r_node.label_anchor_x = kLabelRadius;
    r_node.label_anchor_y = 0.0;
    cairo_matrix_transform_distance(&rotation_matrix, &r_node.label_anchor_x,
                                    &r_node.label_anchor_y);

    r_node.label.label_text = "Node label";
    r_node.label.anchor = OctantToAnchor(i + 4);

    r_node.display = zebraix_proto::SHOW;

    r_edge.start_node_index = 8;
    r_edge.finish_node_index = i;
    r_edge.forward_arrow = true;
    r_edge.display = zebraix_proto::SHOW;
    graph_layout->GetDominance().d_nodes[i].parents.resize(1);
    graph_layout->GetDominance().d_nodes[i].parents = {8};
  }
  graph_layout->GetDominance().d_nodes[8].children.resize(8);
  graph_layout->GetDominance().d_nodes[8].children = {0, 1, 2, 3, 4, 5, 6, 7};
  ZebraixNode& r_node = graph_layout->GetNodes()[8];
  r_node.radius = kNodeRadius;
  r_node.centre_x = 0.0;
  r_node.centre_y = 0.0;

  graph_layout->GetNodes()[2].label.label_text = "Hello, world";
  graph_layout->GetNodes()[6].label.label_text = "Eat a fluffy soufflé";

  graph_layout->GetNodes()[1].display = zebraix_proto::GHOST;
  graph_layout->GetNodes()[7].display = zebraix_proto::GHOST;
  graph_layout->GetEdges()[std::make_pair(8, 1)].display = zebraix_proto::GHOST;
  graph_layout->GetEdges()[std::make_pair(8, 7)].display = zebraix_proto::GHOST;

  CHECK(graph_layout->GetNodes()[1].label.anchor == zebraix_proto::BL);
  CHECK(graph_layout->GetNodes()[5].label.anchor == zebraix_proto::TR);

  graph_layout->GetCanvas().canvas_width = 72.0 * 5;
  graph_layout->GetCanvas().canvas_height = 72.0 * 4;
  graph_layout->GetCanvas().canvas_x_offset =
      0.5 * graph_layout->GetCanvas().canvas_width;
  graph_layout->GetCanvas().canvas_y_offset =
      0.5 * graph_layout->GetCanvas().canvas_height;
}

void ZebraixCairoSvg::GlobalTearDown() {
  // Global caches, etc. Explicit destruction required to avoid terminal
  // memory leak errors.
  cairo_debug_reset_static_data();
  FcFini();  // FontConfig.
}

ZebraixCairoSvg::ZebraixCairoSvg(const ZebraixCanvas& overall,
                                 const char* out_file) {
  surface = cairo_svg_surface_create(out_file, overall.canvas_width,
                                     overall.canvas_height);
  CHECK_EQ(cairo_surface_status(surface), CAIRO_STATUS_SUCCESS);

  cr = cairo_create(surface);
  CHECK_EQ(cairo_status(cr), CAIRO_STATUS_SUCCESS);
  cairo_translate(cr, overall.canvas_x_offset, 1.0 * overall.canvas_y_offset);
}

ZebraixCairoSvg::~ZebraixCairoSvg() {
  // Explicit destruction required to avoid terminal memory leak errors.
  cairo_surface_destroy(surface);
  cairo_destroy(cr);
}

void ZebraixCairoSvg::RenderToSvg(const ZebraixLayout& layout) {
  CairoRenderLayout(layout, cr);
  cairo_surface_finish(surface);
}

}  // namespace render
}  // namespace zebraix
