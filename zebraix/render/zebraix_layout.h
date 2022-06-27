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

#ifndef ZEBRAIX_RENDER_ZEBRAIX_LAYOUT_H_
#define ZEBRAIX_RENDER_ZEBRAIX_LAYOUT_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "base/dominance.h"
#include "base/zebraix_graph.pb.h"
#include "base/zebraix_graph.proto.h"
#include "render/render_structs.h"
// IWYU pragma: no_include "render/zebraix_rendersvg.h"

using zebraix::base::DominanceGraph;

namespace zebraix {
namespace render {

struct ZebraixLayout {
 public:
  ZebraixLayout() = default;

  void BuildLayoutFromFile(const std::string& input_filename);
  void BuildLayoutFromProtoString(const std::string& proto_string);
  void BuildLayoutFromLayoutProto(
      const zebraix_proto::ZebraixGraph& input_proto);

  const zebraix_proto::ZebraixGraph& GetMainLayoutProto() const {
    return proto_layout_;
  }
  const DominanceGraph& GetDominance() const { return dominance_graph_; }
  DominanceGraph& GetDominance() { return dominance_graph_; }
  const ZebraixCanvas& GetCanvas() const { return canvas_; }
  ZebraixCanvas& GetCanvas() { return canvas_; }
  const std::vector<ZebraixNode>& GetNodes() const { return zebraix_nodes_; }
  std::vector<ZebraixNode>& GetNodes() { return zebraix_nodes_; }
  const absl::flat_hash_map<std::pair<int, int>, ZebraixEdge>& GetEdges()
      const {
    return zebraix_edges_;
  }
  absl::flat_hash_map<std::pair<int, int>, ZebraixEdge>& GetEdges() {
    return zebraix_edges_;
  }
  const ZebraixConfig& GetConfig() const { return config_; }
  ZebraixConfig& GetConfig() { return config_; }

  void DumpEdges();

 private:
  zebraix_proto::ZebraixGraph proto_layout_;
  DominanceGraph dominance_graph_;
  std::vector<zebraix_proto::Node> proto_nodes_;
  ZebraixCanvas canvas_;
  std::vector<ZebraixNode> zebraix_nodes_;
  // Map edges by pair of <parent, child>.
  absl::flat_hash_map<std::pair<int, int>, ZebraixEdge> zebraix_edges_;
  ZebraixConfig config_;
};

}  // namespace render
}  // namespace zebraix

#endif  // ZEBRAIX_RENDER_ZEBRAIX_LAYOUT_H_
