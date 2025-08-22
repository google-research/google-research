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

#include "graph.h"

#include <cmath>

#include "absl/strings/numbers.h"
#include "tsv_utils.h"

namespace geo_algorithms {

using utils_tsv::TsvReader;

std::unique_ptr<Graph> Graph::Reverse(const Graph& graph) {
  std::vector<Graph::Node> nodes;
  for (const Graph::Node node : graph.nodes()) {
    nodes.push_back(node);
  }

  std::vector<std::vector<Graph::Arc>> arcs(nodes.size());
  for (const Graph::Node node : graph.nodes()) {
    for (const Graph::Arc arc : graph.ArcsForNode(node.id)) {
      Graph::Arc reversed_arc;
      reversed_arc.dst = node.id;
      reversed_arc.num = arc.num;
      reversed_arc.cost = arc.cost;
      arcs[arc.dst].push_back(reversed_arc);
    }
  }
  return std::unique_ptr<Graph>(new Graph(std::move(arcs), std::move(nodes)));
}

std::unique_ptr<Graph> Graph::LoadFromFiles(const std::string& arcs_file,
                                            const std::string& lat_lngs_file,
                                            const std::string& cost_name) {
  // Load nodes
  std::vector<Graph::Node> nodes;
  {
    TsvReader reader(lat_lngs_file);
    for (int node_id = 0; !reader.AtEnd(); node_id++) {
      const absl::flat_hash_map<std::string, std::string> row =
          reader.ReadRow();
      double lat, lng;
      CHECK(absl::SimpleAtod(row.at("lat"), &lat));
      CHECK(absl::SimpleAtod(row.at("lng"), &lng));
      nodes.push_back({.id = node_id, .lat = lat, .lng = lng});
    }
  }

  // Load arcs
  std::vector<std::vector<Graph::Arc>> arcs(nodes.size());
  int num_arcs = 0;
  for (TsvReader reader(arcs_file); !reader.AtEnd();) {
    const absl::flat_hash_map<std::string, std::string> row = reader.ReadRow();

    int src, dst;
    CHECK(absl::SimpleAtoi(row.at("src"), &src));
    CHECK_GE(src, 0);
    CHECK_LT(src, nodes.size());
    CHECK(absl::SimpleAtoi(row.at("dst"), &dst));
    CHECK_GE(dst, 0);
    CHECK_LT(dst, nodes.size());

    double precost;
    CHECK(absl::SimpleAtod(row.at(cost_name), &precost));
    const int64_t cost = (int64_t)(precost * 10000.0);

    arcs[src].push_back({
        .dst = dst,
        .num = num_arcs,
        .cost = cost,
    });
    num_arcs++;
  }
  return std::unique_ptr<Graph>(new Graph(std::move(arcs), std::move(nodes)));
}

Graph::Graph(std::vector<std::vector<Arc>>&& arcs, std::vector<Node>&& nodes)
    : arcs_(std::move(arcs)),
      nodes_(std::move(nodes)),
      num_arcs_(CalculateNumArcs()) {
  for (const std::vector<Arc>& arc_list : arcs_) {
    for (const Arc& arc : arc_list) {
      CHECK(arc.num != -1);
      CHECK(arc.cost >= 0);
    }
  }
}

}  // namespace geo_algorithms
