// Copyright 2025 The Google Research Authors.
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

#ifndef GRAPH_H_
#define GRAPH_H_

#include <algorithm>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace geo_algorithms {

struct ArcIndex {
  int node;
  int index;

  friend bool operator==(const ArcIndex& a, const ArcIndex& b) {
    return a.node == b.node && a.index == b.index;
  }

  friend bool operator!=(const ArcIndex& a, const ArcIndex& b) {
    return !(a == b);
  }

  template <typename H>
  friend H AbslHashValue(H h, const ArcIndex& a) {
    return H::combine(std::move(h), a.node, a.index);
  }

  friend bool operator<(const ArcIndex& a, const ArcIndex& b) {
    return (a.node < b.node) || ((a.node == b.node) && (a.index < b.index));
  }
};

struct ResidualIndex {
  ArcIndex orig;

  friend bool operator==(const ResidualIndex& a, const ResidualIndex& b) {
    return a.orig == b.orig;
  }

  friend bool operator!=(const ResidualIndex& a, const ResidualIndex& b) {
    return !(a == b);
  }

  template <typename H>
  friend H AbslHashValue(H h, const ResidualIndex& a) {
    return H::combine(std::move(h), a.orig);
  }
};

struct AOrRIndex {
  ArcIndex forward;
  ResidualIndex residual;
  bool is_forward;
};

class Graph {
 public:
  struct Node {
    int id = -1;
    double lat = -720.0;
    double lng = -720.0;
  };

  struct Arc {
    int dst = -1;
    int num = -1;
    int64_t cost = -1;
  };

  Graph(std::vector<std::vector<Arc>>&& arcs, std::vector<Node>&& nodes);

  static std::unique_ptr<Graph> LoadFromFiles(const std::string& arcs_file,
                                              const std::string& lat_lngs_file,
                                              const std::string& cost_name);

  static std::unique_ptr<Graph> LoadFromDirectory(
      const std::string& graph_dir, const std::string& cost_name) {
    return Graph::LoadFromFiles(absl::StrCat(graph_dir, "/arcs.tsv"),
                                absl::StrCat(graph_dir, "/nodes.tsv"),
                                cost_name);
  }

  static std::unique_ptr<Graph> Reverse(const Graph& graph);

  bool HasArc(ArcIndex idx) const {
    if (idx.node >= NumNodes()) {
      return false;
    }
    if (idx.index >= arcs_[idx.node].size()) {
      return false;
    }
    Arc candidate_arc = arcs_[idx.node][idx.index];
    CHECK_EQ(candidate_arc.dst, ArcIndexDst(idx));
    return true;
  }

  int64_t CostForArc(ArcIndex idx) const {
    return arcs_[idx.node][idx.index].cost;
  }

  const ArcIndex ArcIndexFor(int src, int dst) const {
    const std::vector<Arc> src_arcs = ArcsForNode(src);
    return {src, (int)std::distance(
                     src_arcs.begin(),
                     std::find_if(src_arcs.begin(), src_arcs.end(),
                                  [dst](Arc a) { return a.dst == dst; }))};
  }

  const absl::StatusOr<ArcIndex> ReversedArcIfExists(ArcIndex idx) const {
    int dst = ArcIndexDst(idx);
    ArcIndex reversal = ArcIndexFor(dst, idx.node);
    if (reversal.index >= ArcsForNode(dst).size()) {
      return absl::NotFoundError("No reversal exists.");
    }
    return reversal;
  }

  const int ArcIndexDst(ArcIndex idx) const {
    CHECK_GE(idx.index, 0);
    return arcs_[idx.node][idx.index].dst;
  }

  const int ArcIndexNum(ArcIndex idx) const {
    CHECK_GE(idx.index, 0);
    return arcs_[idx.node][idx.index].num;
  }

  int AOrRIndexSrc(AOrRIndex idx) const {
    if (idx.is_forward) {
      return idx.forward.node;
    } else {
      return ArcIndexDst(idx.residual.orig);
    }
  }

  int AOrRIndexDst(AOrRIndex idx) const {
    if (idx.is_forward) {
      return ArcIndexDst(idx.forward);
    } else {
      return idx.residual.orig.node;
    }
  }

  const std::vector<Arc>& ArcsForNode(int node_id) const {
    return arcs_.at(node_id);
  }
  const Node& GetNode(int node_id) const { return nodes_.at(node_id); }
  int NumNodes() const { return nodes_.size(); }
  int NumArcs() const { return num_arcs_; }

  const std::vector<Node>& nodes() const { return nodes_; }

 private:
  int CalculateNumArcs() {
    int total_size = 0;
    for (int i = 0; i < arcs_.size(); ++i) {
      total_size += arcs_[i].size();
    }
    return total_size;
  }

  std::vector<std::vector<Arc>> arcs_;
  std::vector<Node> nodes_;
  const int num_arcs_;
};

}  // namespace geo_algorithms

#endif  // GRAPH_H_
