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

#ifndef ZEBRAIX_RENDER_ZEBRAIX_RENDERSVG_H_
#define ZEBRAIX_RENDER_ZEBRAIX_RENDERSVG_H_

#include "third_party/cairo/cairo.h"  // IWYU pragma: keep
#include "base/dominance.h"
#include "render/render_structs.h"
#include "render/zebraix_layout.h"

namespace zebraix {
namespace render {

class ZebraixCairoSvg {
 public:
  ZebraixCairoSvg(const ZebraixCanvas& overall, const char* out_file);
  ~ZebraixCairoSvg();

  void RenderToSvg(const ZebraixLayout& layout);

  static void GlobalTearDown();

 private:
  cairo_surface_t* surface;
  cairo_t* cr;
};

void BuildSampleLayout(ZebraixLayout* graph_layout);

}  // namespace render
}  // namespace zebraix

#endif  // ZEBRAIX_RENDER_ZEBRAIX_RENDERSVG_H_
