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

#include "penalty_runner.h"

#include "neg_weight_graph_search.h"
#include "path_store.h"

namespace geo_algorithms {

std::string PenaltyRunner::GetShortParamSuffix() const {
  return absl::StrCat(
      "__",
      graph_name_.has_value() ? absl::StrCat("sg_", graph_name_.value()) : "",
      graph_name_.has_value() ? "__" : "", penalty_mode_, "__", "penalty_",
      penalty_multiplier_);
}

std::string PenaltyRunner::GetPathStoreDirectoryName() const {
  return absl::StrCat(base_graph_directory_, "/", "scenario_path_store",
                      GetShortParamSuffix());
}

std::string GetPathStoreFilename(int src, int dst, int path_num) {
  return absl::StrCat("path__src_", src, "__dst_", dst, "__num_", path_num,
                      ".tsv");
}

// arcs penalized in order to obtain path number path_num
std::string GetPathStorePenalizedArcsFilename(int src, int dst, int path_num) {
  return absl::StrCat("penalized_arcs__src_", src, "__dst_", dst, "__num_",
                      path_num, ".tsv");
}

std::string GetPathStoreEiFilename(int src, int dst, int path_num) {
  return absl::StrCat("ei__src_", src, "__dst_", dst, "__num_", path_num,
                      ".tsv");
}

const double PI = 3.14159265358979323846;
const double EARTH_RADIUS = 6371;  // in kilometers

double deg2rad(double deg) { return deg * PI / 180.0; }

double haversine(double lat1, double lon1, double lat2, double lon2) {
  double dLat = deg2rad(lat2 - lat1);
  double dLon = deg2rad(lon2 - lon1);

  double a = sin(dLat / 2) * sin(dLat / 2) + cos(deg2rad(lat1)) *
                                                 cos(deg2rad(lat2)) *
                                                 sin(dLon / 2) * sin(dLon / 2);

  double c = 2 * atan2(sqrt(a), sqrt(1 - a));
  double distance = EARTH_RADIUS * c;

  return distance;
}

double SetHaversine(const std::pair<double, double>& lat_lng,
                    const std::vector<std::pair<double, double>>& lat_lng_set) {
  double min_dist = -1.0;
  for (const auto& [lat, lng] : lat_lng_set) {
    double dist = haversine(lat_lng.first, lat_lng.second, lat, lng);
    if (min_dist == -1.0 || dist < min_dist) {
      min_dist = dist;
    }
  }
  CHECK_GE(min_dist, 0.0);
  return min_dist;
}

std::pair<std::vector<ArcIndex>, ArcIndex> GetPenalizedArcsVersionA(
    const std::vector<AOrRIndex>& path, const std::vector<ArcIndex>& eis_so_far,
    const absl::flat_hash_map<std::string, std::unique_ptr<Graph>>&
        auxiliary_costs) {
  std::vector<ArcIndex> forward_path;
  for (const AOrRIndex& e : path) {
    CHECK(e.is_forward);
    if (!IsNull(e)) {
      forward_path.push_back(e.forward);
    }
  }
  CHECK(IsNull(path[path.size() - 1]));

  // define X

  int64_t min_road_type = -1;
  absl::flat_hash_set<ArcIndex> X;
  for (const ArcIndex& e : forward_path) {
    int64_t new_cost = auxiliary_costs.at("road_type")->CostForArc(e);
    if (min_road_type == -1 || new_cost < min_road_type) {
      min_road_type = new_cost;
      X.clear();
    }
    if (new_cost == min_road_type) {
      X.insert(e);
    }
  }

  // define Y

  absl::flat_hash_set<ArcIndex> Y;
  if (!eis_so_far.empty()) {
    ArcIndex ei = eis_so_far[eis_so_far.size() - 1];
    int64_t ei_length = auxiliary_costs.at("dist_meters")->CostForArc(ei);

    for (const ArcIndex& e : X) {
      int64_t e_length = auxiliary_costs.at("dist_meters")->CostForArc(e);
      if (e_length >= 0.8 * ei_length) {
        Y.insert(e);
      }
    }
  }

  // define Z, the final list of candidates to filter by number of lanes
  absl::flat_hash_set<ArcIndex> Z;
  if (Y.empty()) {
    // filter by max length
    int64_t max_length = -1;
    for (const ArcIndex& e : X) {
      int64_t e_length = auxiliary_costs.at("dist_meters")->CostForArc(e);
      if (e_length > max_length) {
        max_length = e_length;
        Z.clear();
      }
      if (e_length == max_length) {
        Z.insert(e);
      }
    }
  } else {
    // filter by set distance
    std::vector<std::pair<double, double>> ei_lat_lngs;
    for (const ArcIndex& ei : eis_so_far) {
      ei_lat_lngs.push_back(std::make_pair(
          auxiliary_costs.at("dist_meters")->GetNode(ei.node).lat,
          auxiliary_costs.at("dist_meters")->GetNode(ei.node).lng));
    }

    double max_crow_dist = -1.0;
    for (const ArcIndex& e : Y) {
      std::pair<double, double> e_lat_lng = std::make_pair(
          auxiliary_costs.at("dist_meters")->GetNode(e.node).lat,
          auxiliary_costs.at("dist_meters")->GetNode(e.node).lng);

      double set_crow_dist = SetHaversine(e_lat_lng, ei_lat_lngs);
      if (set_crow_dist > max_crow_dist) {
        max_crow_dist = set_crow_dist;
        Z.clear();
        Z.insert(e);  // done here to guard against numerical stability issues
      }
      if (set_crow_dist == max_crow_dist) {
        Z.insert(e);
      }
    }
  }

  // filter by number of lanes

  int64_t max_num_lanes = -1;
  std::vector<ArcIndex> final_eip1_candidates;
  for (const ArcIndex& e : Z) {
    int64_t num_lanes = auxiliary_costs.at("num_lanes")->CostForArc(e);
    if (num_lanes > max_num_lanes) {
      max_num_lanes = num_lanes;
      final_eip1_candidates.clear();
    }
    if (num_lanes == max_num_lanes) {
      final_eip1_candidates.push_back(e);
    }
  }

  ArcIndex eip1 = final_eip1_candidates[0];

  // get nearby segments

  int eip1_index = -1;
  for (int i = 0; i < forward_path.size(); i++) {
    if (forward_path[i] == eip1) {
      CHECK_EQ(eip1_index, -1);
      eip1_index = i;
    }
  }
  CHECK_GE(eip1_index, 0);

  std::vector<ArcIndex> nearby_segments;
  int start_index = std::max(0, eip1_index - 5);
  int end_index = std::min((int)forward_path.size(), eip1_index + 6);
  for (int i = start_index; i < end_index; i++) {
    nearby_segments.push_back(forward_path[i]);
  }

  return std::make_pair(nearby_segments, eip1);
}

int SimpleVersionACutoff(int length) {
  if (length < 10) {
    return 0;
  }
  if (length < 40) {
    return 2;
  }
  if (length < 70) {
    return 4;
  }
  if (length < 100) {
    return 7;
  }
  if (length < 150) {
    return 10;
  }
  if (length < 300) {
    return 20;
  }
  if (length < 1000) {
    return 30;
  }
  return 100;
}

std::pair<std::vector<ArcIndex>, ArcIndex> GetPenalizedArcsSimpleVersionA(
    const std::vector<AOrRIndex>& path, const std::vector<ArcIndex>& eis_so_far,
    const absl::flat_hash_map<std::string, std::unique_ptr<Graph>>&
        auxiliary_costs,
    int num_nearby_segments) {
  CHECK_GE(num_nearby_segments, 0);
  std::vector<ArcIndex> forward_path;
  for (const AOrRIndex& e : path) {
    CHECK(e.is_forward);
    if (!IsNull(e)) {
      forward_path.push_back(e.forward);
    }
  }
  CHECK(IsNull(path[path.size() - 1]));

  // define W: the set of all segments that are sufficiently
  // far from the origin and destination of the path to prevent blockage

  absl::flat_hash_set<ArcIndex> W;
  int offset = SimpleVersionACutoff(forward_path.size());
  for (int i = offset; i < (forward_path.size() - offset); i++) {
    W.insert(forward_path[i]);
  }

  // define X

  int64_t min_road_type = -1;
  absl::flat_hash_set<ArcIndex> X;
  for (const ArcIndex& e : W) {
    int64_t new_cost = auxiliary_costs.at("road_type")->CostForArc(e);
    if (min_road_type == -1 || new_cost < min_road_type) {
      min_road_type = new_cost;
      X.clear();
    }
    if (new_cost == min_road_type) {
      X.insert(e);
    }
  }

  // define Z, the final list of candidates to filter by number of lanes
  absl::flat_hash_set<ArcIndex> Z;
  // filter by max length
  int64_t max_length = -1;
  for (const ArcIndex& e : X) {
    int64_t e_length = auxiliary_costs.at("dist_meters")->CostForArc(e);
    if (e_length > max_length) {
      max_length = e_length;
      Z.clear();
    }
    if (e_length == max_length) {
      Z.insert(e);
    }
  }

  // filter by number of lanes

  int64_t max_num_lanes = -1;
  std::vector<ArcIndex> final_eip1_candidates;
  for (const ArcIndex& e : Z) {
    int64_t num_lanes = auxiliary_costs.at("num_lanes")->CostForArc(e);
    if (num_lanes > max_num_lanes) {
      max_num_lanes = num_lanes;
      final_eip1_candidates.clear();
    }
    if (num_lanes == max_num_lanes) {
      final_eip1_candidates.push_back(e);
    }
  }

  ArcIndex eip1 = final_eip1_candidates[0];

  // get nearby segments

  int eip1_index = -1;
  for (int i = 0; i < forward_path.size(); i++) {
    if (forward_path[i] == eip1) {
      CHECK_EQ(eip1_index, -1);
      eip1_index = i;
    }
  }
  CHECK_GE(eip1_index, 0);

  std::vector<ArcIndex> nearby_segments;
  int start_index = std::max(offset, eip1_index - num_nearby_segments);
  int end_index = std::min((int)forward_path.size() - offset,
                           eip1_index + num_nearby_segments + 1);
  for (int i = start_index; i < end_index; i++) {
    nearby_segments.push_back(forward_path[i]);
  }

  return std::make_pair(nearby_segments, eip1);
}

std::pair<std::vector<ArcIndex>, ArcIndex> PenaltyRunner::GetPenalizedArcs(
    const std::vector<AOrRIndex>& path, const std::vector<ArcIndex>& eis_so_far,
    const absl::flat_hash_map<std::string, std::unique_ptr<Graph>>&
        auxiliary_costs) {
  if (penalty_mode_ == "important_arcs_only_simple_version_A") {
    return GetPenalizedArcsSimpleVersionA(path, eis_so_far, auxiliary_costs,
                                          /*num_nearby_segments = */ 5);
  } else if (penalty_mode_ == "most_important_arc_only_simple_version_A") {
    return GetPenalizedArcsSimpleVersionA(path, eis_so_far, auxiliary_costs,
                                          /*num_nearby_segments = */ 0);
  } else if (penalty_mode_ == "important_arcs_only_version_A") {
    return GetPenalizedArcsVersionA(path, eis_so_far, auxiliary_costs);
  } else {
    LOG(FATAL) << "Invalid penalty mode: " << penalty_mode_;
    CHECK(false);
  }
}

void CheckIsValidType(const std::string& type) {
  if (type != "whole_path" && type != "important_arcs_only_version_A" &&
      type != "important_arcs_only_simple_version_A" &&
      type != "most_important_arc_only_simple_version_A") {
    LOG(FATAL) << "Invalid penalty mode: " << type;
    CHECK(false);
  }
}

PenaltyRunner::PenaltyRunner(
    const absl::flat_hash_map<ArcIndex, int64_t>& default_costs,
    const Graph& graph, const absl::optional<std::string> graph_name,
    const std::string& base_graph_directory,
    const absl::flat_hash_map<std::string, std::unique_ptr<Graph>>&
        auxiliary_costs,
    const int old_src, const int old_dst, const int new_src, const int new_dst,
    const std::string& penalty_mode, const double penalty_multiplier,
    const double num_penalized_paths, const bool break_if_segment_repenalized)
    : graph_name_(graph_name),
      base_graph_directory_(base_graph_directory),
      penalty_mode_(penalty_mode),
      penalty_multiplier_(penalty_multiplier),
      break_if_segment_repenalized_(break_if_segment_repenalized) {
  absl::flat_hash_set<ArcIndex> P;
  PathStore path_store(GetPathStoreDirectoryName());
  std::vector<ArcIndex> all_eis;

  for (int i = 0; i < num_penalized_paths; i++) {
    Search::CostModel costs = Search::CostModel{
        .default_costs = default_costs,
        .updated_forward_costs = updated_forward_costs_,
        .nonneg_weights = true,
    };
    // generate or load path
    std::string path_filename = GetPathStoreFilename(old_src, old_dst, i);
    absl::StatusOr<std::vector<AOrRIndex>> path_stat =
        path_store.ReadPathOrFailIfAbsent(path_filename);
    std::vector<AOrRIndex> path;
    if (path_stat.ok()) {
      LOG(INFO) << "Loading path " << i << " from " << path_filename << "...";
      path = *path_stat;
    } else {
      LOG(INFO) << "Generating path " << i << "...";
      const auto path_result = Search(graph, costs, new_src).FindPath(new_dst);
      if (!path_result.ok()) {
        LOG(INFO) << "No path exists! Continuing to next OD pair...";
        stats_ = ScenarioStats{.P_plotly = "NULL"};
        return;
      }
      path = path_result->path;
      path_store.SavePathOrFailIfPresent(path_filename, path);
    }

    // draw path
    std::string path_color;
    if (i == 0) {
      path_color = "#FF00FF|stroke_weight=4";
    } else if (i == (num_penalized_paths - 1)) {
      path_color = "#008800|stroke_weight=4";
    } else {
      path_color = "#FF0000";
    }
    std::string plotly_path_string = GetPlotlyPathString(
        graph, path, absl::StrCat("Src", old_src, "Dst", old_dst, "SP", i),
        path_color);
    if (i == 0) {
      stats_.shortest_path_plotly = plotly_path_string;
    } else if (i == (num_penalized_paths - 1)) {
      stats_.P_plotly = plotly_path_string;
    }
    path_plotly_strings_.push_back(plotly_path_string);
    if (i == (num_penalized_paths - 1)) {
      path_smallplotly_strings_.push_back(plotly_path_string);
    }

    P.clear();
    for (int j = 0; j < path.size() - 1; j++) {
      AOrRIndex e = path[j];
      P.insert(e.forward);

      CHECK(e.is_forward);
      all_path_arcs_.insert(e.forward);
    }
    paths_.push_back(P);

    // get penalized arcs on path

    absl::flat_hash_set<ArcIndex> new_penalized_arcs;
    if (penalty_mode_ == "whole_path") {
      new_penalized_arcs = P;
    } else {
      // caching penalized arcs to allow arc selection to be non-deterministic
      std::string penalized_arc_fname =
          GetPathStorePenalizedArcsFilename(old_src, old_dst, i + 1);
      std::string eip1_fname = GetPathStoreEiFilename(old_src, old_dst, i + 1);
      absl::StatusOr<std::vector<AOrRIndex>> penalized_arc_stat =
          path_store.ReadPathOrFailIfAbsent(penalized_arc_fname);
      absl::StatusOr<std::vector<AOrRIndex>> eip1_stat =
          path_store.ReadPathOrFailIfAbsent(eip1_fname);
      CHECK_EQ(penalized_arc_stat.ok(), eip1_stat.ok());
      std::vector<ArcIndex> penalized_arc_list;
      ArcIndex eip1;

      if (penalized_arc_stat.ok()) {
        for (const AOrRIndex& e : *penalized_arc_stat) {
          CHECK(e.is_forward);
          penalized_arc_list.push_back(e.forward);
        }
        CHECK_EQ(eip1_stat->size(), 1);
        CHECK(eip1_stat->at(0).is_forward);
        eip1 = (*eip1_stat)[0].forward;
      } else {
        auto p = GetPenalizedArcs(path, all_eis, auxiliary_costs);
        penalized_arc_list = p.first;
        eip1 = p.second;
        if (penalty_mode_ == "most_important_arc_only_simple_version_A") {
          CHECK_EQ(penalized_arc_list.size(), 1);
        }

        std::vector<AOrRIndex> penalized_arc_list_vector;
        for (const ArcIndex& e : penalized_arc_list) {
          penalized_arc_list_vector.push_back(
              AOrRIndex{.forward = e, .is_forward = true});
        }
        path_store.SavePathOrFailIfPresent(penalized_arc_fname,
                                           penalized_arc_list_vector);
        path_store.SavePathOrFailIfPresent(
            eip1_fname, {AOrRIndex{.forward = eip1, .is_forward = true}});
      }

      all_eis.push_back(eip1);
      for (const ArcIndex& e : penalized_arc_list) {
        new_penalized_arcs.insert(e);
      }
      std::string penalty_color = "#FF0000|stroke_weight=9";
      std::vector<AOrRIndex> penalized_arc_list_vector;
      for (const ArcIndex& e : penalized_arc_list) {
        penalized_arc_list_vector.push_back(
            AOrRIndex{.forward = e, .is_forward = true});
      }
      if (i != (num_penalized_paths - 1)) {
        std::vector<std::string> penalty_stretch_plotlys =
            GetPlotlyPathStrings(graph, penalized_arc_list_vector,
                                 absl::StrCat("Penalty", i + 1), penalty_color);
        for (const std::string& plotly_string : penalty_stretch_plotlys) {
          penalty_plotly_strings_.push_back(plotly_string);
        }
      }
    }

    // change arc penalties

    for (const ArcIndex& e : new_penalized_arcs) {
      all_penalized_arcs_.insert(e);
      if (updated_forward_costs_.contains(e)) {
        if (break_if_segment_repenalized) {
          LOG(INFO) << "Segment repenalized! Breaking...";
          stats_ = ScenarioStats{.P_plotly = "CUT"};
          return;
        }
        updated_forward_costs_[e] =
            (int64_t)(penalty_multiplier_ * updated_forward_costs_.at(e));
      } else {
        updated_forward_costs_[e] =
            (int64_t)(penalty_multiplier_ * default_costs.at(e));
      }
    }
  }
}

}  // namespace geo_algorithms
