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

#include "absl/strings/numbers.h"
#include "tsv_utils.h"

namespace geo_algorithms {

using utils_tsv::TsvReader;

int GetIntFeature(const absl::flat_hash_map<std::string, std::string>& row,
                  const std::string& feature_name) {
  int value;
  CHECK(absl::SimpleAtoi(row.at(feature_name), &value));
  return value;
}

double GetDoubleFeature(
    const absl::flat_hash_map<std::string, std::string>& row,
    const std::string& feature_name) {
  double value;
  CHECK(absl::SimpleAtod(row.at(feature_name), &value));
  return value;
}

// Returns a value between 0.0 and 1.0, with 0.0 meaning definitely not a turn,
// and 1.0 meaning definitely a turn.
double GetTurnness(double turn_degrees) {
  if (turn_degrees < 20) return 0.0;
  if (turn_degrees < 40) return (turn_degrees - 20) / 20;
  return 1.0;
}

std::unique_ptr<MultiCostGraph> MultiCostGraph::Reverse(
    const MultiCostGraph& graph) {
  std::vector<std::string> base_cost_names = graph.base_cost_names();
  std::vector<MultiCostGraph::Node> nodes;
  for (const MultiCostGraph::Node node : graph.nodes()) {
    nodes.push_back(node);
  }

  std::vector<std::vector<MultiCostGraph::Arc>> arcs(nodes.size());
  for (const MultiCostGraph::Node node : graph.nodes()) {
    for (const MultiCostGraph::Arc arc : graph.ArcsForNode(node.id)) {
      MultiCostGraph::Arc reversed_arc;
      reversed_arc.dst = node.id;
      reversed_arc.num = arc.num;
      reversed_arc.cost_vector = arc.cost_vector;
      arcs[arc.dst].push_back(reversed_arc);
    }
  }
  return std::unique_ptr<MultiCostGraph>(
      new MultiCostGraph(base_cost_names, std::move(arcs), std::move(nodes)));
}

std::unique_ptr<MultiCostGraph> MultiCostGraph::LoadFromFiles(
    const std::string& arcs_file, const std::string& lat_lngs_file) {
  // Load nodes
  std::vector<MultiCostGraph::Node> nodes;
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
  std::vector<std::vector<MultiCostGraph::Arc>> arcs(nodes.size());
  int num_arcs = 0;
  for (TsvReader reader(arcs_file); !reader.AtEnd();) {
    const absl::flat_hash_map<std::string, std::string> row = reader.ReadRow();

    const int src = GetIntFeature(row, "src");
    CHECK_GE(src, 0);
    CHECK_LT(src, nodes.size());
    const int dst = GetIntFeature(row, "dst");
    CHECK_GE(dst, 0);
    CHECK_LT(dst, nodes.size());

    const double dist_meters = GetDoubleFeature(row, "dist_meters");
    const double duration_secs = GetDoubleFeature(row, "duration_secs");
    const double turn_degrees = std::abs(GetDoubleFeature(row, "turn_degrees"));

    const int has_stop = GetIntFeature(row, "has_stop");
    const int has_traffic_light = GetIntFeature(row, "has_traffic_light");
    // Road types are 0-12, higher values are more local it seems.
    const int road_type = GetIntFeature(row, "road_type");

    arcs[src].push_back({
        .dst = dst,
        .num = num_arcs,
        .cost_vector =
            {
                duration_secs,
                dist_meters,
                GetTurnness(turn_degrees),
                static_cast<double>(has_stop),
                static_cast<double>(has_traffic_light),
                (road_type >= 10 ? duration_secs : 0.0),
            },
    });
    num_arcs++;
  }

  const std::vector<std::string> cost_names = {
      "duration_secs", "dist_meters",       "turnness",
      "has_stop",      "has_traffic_light", "local_road_duration",
  };
  return std::unique_ptr<MultiCostGraph>(
      new MultiCostGraph(cost_names, std::move(arcs), std::move(nodes)));
}

MultiCostGraph::MultiCostGraph(const std::vector<std::string>& base_cost_names,
                               std::vector<std::vector<Arc>>&& arcs,
                               std::vector<Node>&& nodes)
    : base_cost_names_(base_cost_names),
      arcs_(std::move(arcs)),
      nodes_(std::move(nodes)),
      num_arcs_(CalculateNumArcs()) {
  for (const std::vector<Arc>& arc_list : arcs_) {
    for (const Arc& arc : arc_list) {
      CHECK_EQ(arc.cost_vector.size(), base_cost_names_.size());
      CHECK(arc.num != -1);
    }
  }
}

std::vector<double> MultiCostGraph::CostWeightsFromMap(
    const absl::flat_hash_map<std::string, double>& weights_map) const {
  std::vector<double> weights_vec;
  for (const std::string& cost_name : base_cost_names_) {
    const auto it = weights_map.find(cost_name);
    weights_vec.push_back(it == weights_map.end() ? 0.0 : it->second);
  }
  return weights_vec;
}

}  // namespace geo_algorithms
