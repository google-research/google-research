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

#ifndef PENALTY_RUNNER_H_
#define PENALTY_RUNNER_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "graph.h"
#include "plotly_path_string.h"

namespace geo_algorithms {

void CheckIsValidType(const std::string& type);

struct ScenarioStats {
  std::string shortest_path_plotly;
  std::string P_plotly;
  double frac_explanation_in_paths;
  double frac_explanation_in_penalized_set;
  int num_arcs_in_explanation;
  int num_components_in_explanation;
  double scenario_runtime_seconds;
  double solver_runtime_seconds;
};

class PenaltyRunner {
 public:
  PenaltyRunner(const absl::flat_hash_map<ArcIndex, int64_t>& default_costs,
                const Graph& graph,
                const absl::optional<std::string> graph_name,
                const std::string& base_graph_directory,
                const absl::flat_hash_map<std::string, std::unique_ptr<Graph>>&
                    auxiliary_costs,
                const int old_src, const int old_dst, const int new_src,
                const int new_dst, const std::string& penalty_mode,
                const double penalty_multiplier,
                const double num_penalized_paths,
                const bool break_if_segment_repenalized);

  ScenarioStats Stats() { return stats_; }

  absl::flat_hash_set<ArcIndex> AllPenalizedArcs() {
    return all_penalized_arcs_;
  }

  absl::flat_hash_set<ArcIndex> AllPathArcs() { return all_path_arcs_; }

  absl::flat_hash_map<ArcIndex, int64_t> UpdatedForwardCosts() {
    return updated_forward_costs_;
  }

  std::vector<std::string> PathPlotlyStrings() { return path_plotly_strings_; }

  std::vector<std::string> PathSmallplotlyStrings() {
    return path_smallplotly_strings_;
  }

  std::vector<std::string> PenaltyPlotlyStrings() {
    return penalty_plotly_strings_;
  }

  absl::flat_hash_set<ArcIndex> GetPath(int i) { return paths_[i]; }

  absl::flat_hash_set<ArcIndex> GetLastPath() { return paths_.back(); }

  int GetNumPaths() { return paths_.size(); }

 private:
  std::string GetShortParamSuffix() const;
  std::string GetPathStoreDirectoryName() const;
  std::pair<std::vector<ArcIndex>, ArcIndex> GetPenalizedArcs(
      const std::vector<AOrRIndex>& path, const std::vector<ArcIndex>& all_eis,
      const absl::flat_hash_map<std::string, std::unique_ptr<Graph>>&
          auxiliary_costs);

  const absl::optional<std::string> graph_name_;
  const std::string base_graph_directory_;
  const std::string penalty_mode_;
  const double penalty_multiplier_;
  const bool break_if_segment_repenalized_;
  ScenarioStats stats_;
  std::vector<std::string> path_plotly_strings_;
  std::vector<std::string> path_smallplotly_strings_;
  std::vector<std::string> penalty_plotly_strings_;
  absl::flat_hash_set<ArcIndex> all_penalized_arcs_;
  absl::flat_hash_set<ArcIndex> all_path_arcs_;
  absl::flat_hash_map<ArcIndex, int64_t> updated_forward_costs_;
  std::vector<absl::flat_hash_set<ArcIndex>> paths_;
};

void CheckIsValidType(const std::string& type);

}  // namespace geo_algorithms

#endif  // PENALTY_RUNNER_H_
