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

#include "plotly_path_string.h"

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace geo_algorithms {

bool IsNull(const AOrRIndex& e) {
  return e.is_forward && (e.forward.index == -1);
}

std::vector<std::string> GetPlotlyPathStrings(
    const Graph& graph,
    const std::vector<AOrRIndex>& path,  // may or may not have null edge at end
    const std::string& name, const std::string& color) {
  std::vector<std::vector<std::string>> coordinates;
  std::vector<std::string> current_coordinates;
  int prev_end_id = -1;
  for (const AOrRIndex& e : path) {
    int start_id = graph.AOrRIndexSrc(e);
    if (IsNull(e)) continue;
    if (start_id != prev_end_id) {
      if (prev_end_id != -1) {
        coordinates.push_back(current_coordinates);
        current_coordinates.clear();
      }
      Graph::Node start = graph.nodes()[start_id];
      current_coordinates.push_back(absl::StrCat(start.lat, ":", start.lng));
    }
    Graph::Node end = graph.nodes()[graph.AOrRIndexDst(e)];
    current_coordinates.push_back(absl::StrCat(end.lat, ":", end.lng));
    prev_end_id = graph.AOrRIndexDst(e);
  }
  coordinates.push_back(current_coordinates);

  CHECK_GE(coordinates.size(), 1);

  if (coordinates.size() == 1) {
    return {absl::StrCat(name, "Line|undirected|", color, ": ",
                         absl::StrJoin(coordinates[0], ","))};
  } else {
    std::vector<std::string> plotlys;
    for (int i = 0; i < coordinates.size(); i++) {
      std::vector<std::string> coords = coordinates[i];
      plotlys.push_back(absl::StrCat(name, "Part", i, "Line|undirected|", color,
                                     ": ", absl::StrJoin(coords, ",")));
    }
    return plotlys;
  }
}

std::string GetPlotlyPathString(
    const Graph& graph,
    const std::vector<AOrRIndex>& path,  // may or may not have null edge at end
    const std::string& name, const std::string& color) {
  std::vector<std::string> result =
      GetPlotlyPathStrings(graph, path, name, color);
  CHECK_EQ(result.size(), 1);
  return result[0];
}

}  // namespace geo_algorithms
