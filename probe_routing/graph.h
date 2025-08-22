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
};

int GetIntFeature(const absl::flat_hash_map<std::string, std::string>& row,
                  const std::string& feature_name);

double GetDoubleFeature(
    const absl::flat_hash_map<std::string, std::string>& row,
    const std::string& feature_name);

class MultiCostGraph {
 public:
  struct Node {
    int id = -1;
    double lat = -720.0;
    double lng = -720.0;
  };

  struct Arc {
    int dst = -1;
    int num = -1;
    std::vector<double> cost_vector;
  };

  MultiCostGraph(const std::vector<std::string>& base_cost_names,
                 std::vector<std::vector<Arc>>&& arcs,
                 std::vector<Node>&& nodes);

  static std::unique_ptr<MultiCostGraph> LoadFromFiles(
      const std::string& arcs_file, const std::string& lat_lngs_file);

  static std::unique_ptr<MultiCostGraph> LoadFromDirectory(
      const std::string& graph_dir) {
    return MultiCostGraph::LoadFromFiles(absl::StrCat(graph_dir, "/arcs.tsv"),
                                         absl::StrCat(graph_dir, "/nodes.tsv"));
  }

  static std::unique_ptr<MultiCostGraph> Reverse(const MultiCostGraph& graph);

  const std::vector<double> CostForArc(ArcIndex idx) const {
    return arcs_[idx.node][idx.index].cost_vector;
  }

  const ArcIndex ArcIndexFor(int src, int dst) const {
    const std::vector<Arc> src_arcs = ArcsForNode(src);
    return {src, (int)std::distance(
                     src_arcs.begin(),
                     std::find_if(src_arcs.begin(), src_arcs.end(),
                                  [dst](Arc a) { return a.dst == dst; }))};
  }

  const int ArcIndexDst(ArcIndex idx) const {
    return arcs_[idx.node][idx.index].dst;
  }

  const int ArcIndexNum(ArcIndex idx) const {
    return arcs_[idx.node][idx.index].num;
  }

  const std::vector<Arc>& ArcsForNode(int node_id) const {
    return arcs_.at(node_id);
  }
  const Node& GetNode(int node_id) const { return nodes_.at(node_id); }
  int NumNodes() const { return nodes_.size(); }
  int NumArcs() const { return num_arcs_; }

  std::vector<double> ZeroCostVector() const {
    return std::vector<double>(base_cost_names_.size(), 0.0);
  }

  // Returns the default cost weights [1, 0, 0, ...], which considers only the
  // first cost objective and ignores the others.
  std::vector<double> DefaultCostWeights() const {
    std::vector<double> weights(base_cost_names_.size(), 0.0);
    weights[0] = 1.0;
    return weights;
  }

  static std::vector<double> SixDefaultCostWeights() {
    std::vector<double> weights(6, 0.0);
    weights[0] = 1.0;
    return weights;
  }

  // Returns cost weights as a vector for weights specified by name.
  std::vector<double> CostWeightsFromMap(
      const absl::flat_hash_map<std::string, double>& weights_map) const;

  const std::vector<std::string>& base_cost_names() const {
    return base_cost_names_;
  }
  const std::vector<Node>& nodes() const { return nodes_; }

 private:
  int CalculateNumArcs() {
    int total_size = 0;
    for (int i = 0; i < arcs_.size(); ++i) {
      total_size += arcs_[i].size();
    }
    return total_size;
  }

  std::vector<std::string> base_cost_names_;
  std::vector<std::vector<Arc>> arcs_;
  std::vector<Node> nodes_;
  const int num_arcs_;
};

}  // namespace geo_algorithms

#endif  // GRAPH_H_
