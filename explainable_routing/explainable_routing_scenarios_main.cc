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

#include <cmath>
#include <fstream>
#include <iostream>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"
#include "file_util.h"
#include "formulation.h"
#include "graph.h"
#include "path_store.h"
#include "penalty_runner.h"
#include "plotly_path_string.h"
#include "residual_problem.h"

ABSL_FLAG(std::string, subgraph_name,
          // "NONE",
          // "wp_A",
          // "iaosv_A",
          "wp_A_plus_iaosv_A",
          "Name of subgraph type to use, with NONE if running on full graph.");
ABSL_FLAG(std::string, penalty_mode,
          // "whole_path",
          // "important_arcs_only_version_A",
          "important_arcs_only_simple_version_A",
          // "most_important_arc_only_simple_version_A",
          "Penalize the whole path or only important arcs"
          " to find the next path.");
ABSL_FLAG(double, penalty_multiplier,
          // 1.1,
          100000.0, "Multiplier applied in generating secondary paths.");
ABSL_FLAG(int, num_penalized_paths, 5,
          "Number of times to penalize before stopping.");
ABSL_FLAG(double, other_multiplier, 2.0,
          "Amount to multiply non-penalized non-path arcs by.");
ABSL_FLAG(double, path_only_other_multiplier, 1.0,
          "Amount to multiply non-penalized path arcs by.");
ABSL_FLAG(bool, use_normalized_objective, true,
          "If true, normalize objective by upper minus lower bound per arc."
          " Otherwise, use the objective value directly.");
ABSL_FLAG(bool, use_fast_solver, false,
          "If true, use the fast residual problem implementation."
          "Might give different results from the slow version, "
          "but they will still be correct.");
ABSL_FLAG(bool, break_if_segment_repenalized, true,
          "If true, break if a segment is about to be penalized twice."
          " Should be set to true if segments are effectively deleted."
          " Not part of the suffix as the code will never find"
          " an explanation at all.");
ABSL_FLAG(int64_t, t_upper_bound, 1000,
          "T values between 1 and this value inclusive");
ABSL_FLAG(
    std::string, graph_directory,
    // "BADEN-PROCESSED DIRECTORY HERE",
    "WASHINGTON-PROCESSED DIRECTORY HERE",
    "Location where graph input is stored.");
ABSL_FLAG(std::string, plotly_output_subdirectory_prefix,
          "explainable_output/plotly",
          "Location to print penalized arcs"
          " to within graph_directory. Params will be appended.");
ABSL_FLAG(std::string, queries_filename,
          // "short_queries_random_100.tsv",
          // "medium_queries_random_100.tsv",
          // "long_queries_random_100.tsv",
          "seattle_medium_queries_random_300.tsv", "Location of query pairs.");

namespace geo_algorithms {
namespace {

std::vector<std::pair<int, int>> GetQueries() {
  // no need to use traffic directory here
  std::string full_queries_filename =
      absl::StrCat(absl::GetFlag(FLAGS_graph_directory), "/",
                   absl::GetFlag(FLAGS_queries_filename));
  return GetQueriesFromFile(full_queries_filename);
}

std::string GetFullParamSuffix() {
  std::string subgraph_term = "";
  if (absl::GetFlag(FLAGS_subgraph_name) != "NONE") {
    subgraph_term = absl::StrCat("__sg_", absl::GetFlag(FLAGS_subgraph_name));
  }
  std::string fast_solver_term =
      absl::GetFlag(FLAGS_use_fast_solver) ? "_ufs" : "";
  std::string normalize_objective_term = "";
  std::string tub_term = "";
  if (absl::GetFlag(FLAGS_use_normalized_objective)) {
    normalize_objective_term = "_uno";
    tub_term = absl::StrCat("_tub_", absl::GetFlag(FLAGS_t_upper_bound));
  } else {
    normalize_objective_term = "";
    tub_term = "";
  }
  return absl::StrCat(subgraph_term, "__", absl::GetFlag(FLAGS_penalty_mode),
                      "__", "p_", absl::GetFlag(FLAGS_penalty_multiplier),
                      normalize_objective_term, tub_term, fast_solver_term,
                      "_om_", absl::GetFlag(FLAGS_other_multiplier), "_poom_",
                      absl::GetFlag(FLAGS_path_only_other_multiplier), "_npp_",
                      absl::GetFlag(FLAGS_num_penalized_paths));
}

std::string GetCycleStoreDirectoryName() {
  return absl::StrCat(absl::GetFlag(FLAGS_graph_directory), "/",
                      "scenario_cycle_store", GetFullParamSuffix());
}

std::string GetCycleStoreFnamePrefix(int src, int dst) {
  return absl::StrCat("cycle__src_", src, "__dst_", dst);
}

std::string GetCutStoreDirectoryName() {
  return absl::StrCat(absl::GetFlag(FLAGS_graph_directory), "/",
                      "scenario_cut_store", GetFullParamSuffix());
}

std::string GetPlotlyDirectoryName() {
  return absl::StrCat(absl::GetFlag(FLAGS_graph_directory), "/",
                      absl::GetFlag(FLAGS_plotly_output_subdirectory_prefix),
                      GetFullParamSuffix());
}

std::string GetPlotlyFilePath(int src, int dst) {
  return absl::StrCat(GetPlotlyDirectoryName(), "/", "plotly__src_", src,
                      "__dst_", dst, ".txt");
}

std::string GetSmallplotlyFilePath(int src, int dst) {
  return absl::StrCat(GetPlotlyDirectoryName(), "/", "smallplotly__src_", src,
                      "__dst_", dst, ".txt");
}

std::string RemoveFtype(const std::string& filename) {
  std::vector<std::string> parts = absl::StrSplit(filename, '.');
  if (parts.size() < 2) {
    return filename;
  }
  return absl::StrJoin(parts.begin(), parts.end() - 1, ".");
}

std::string GetSummaryPlotlyFilePath() {
  return absl::StrCat(GetPlotlyDirectoryName(), "/", "summary_plotly__",
                      RemoveFtype(absl::GetFlag(FLAGS_queries_filename)),
                      ".txt");
}

std::string GetSummaryStatsFilePath() {
  return absl::StrCat(GetPlotlyDirectoryName(), "/", "summary_stats__",
                      RemoveFtype(absl::GetFlag(FLAGS_queries_filename)),
                      ".txt");
}

int GetNumComponents(const std::vector<std::pair<int, int>>& edges) {
  absl::flat_hash_map<int, absl::flat_hash_set<int>> adj_dict;
  for (const auto& [u, v] : edges) {
    adj_dict[u].insert(v);
    adj_dict[v].insert(u);
  }

  absl::flat_hash_set<int> visited;
  int num_components = 0;
  while (!adj_dict.empty()) {
    absl::flat_hash_set<int> component;
    std::queue<int> q;
    q.push(adj_dict.begin()->first);
    while (!q.empty()) {
      int u = q.front();
      q.pop();
      if (visited.contains(u)) {
        continue;
      }
      visited.insert(u);
      component.insert(u);
      for (int v : adj_dict[u]) {
        q.push(v);
      }
    }
    for (int u : component) {
      adj_dict.erase(u);
    }
    num_components++;
  }

  return num_components;
}

absl::flat_hash_map<ArcIndex, int64_t> GetNormalizedT(
    const absl::flat_hash_map<ArcIndex, int64_t>& L,
    const absl::flat_hash_map<ArcIndex, int64_t>& U) {
  CHECK_EQ(L.size(), U.size());

  absl::flat_hash_map<ArcIndex, int64_t> T;
  for (const auto& [e, _] : U) {
    if (U.at(e) == L.at(e)) {
      T[e] = 0;
    } else {
      T[e] = 1 + (int64_t)((absl::GetFlag(FLAGS_t_upper_bound) * L.at(e)) /
                           U.at(e));  // add 1 to ensure that it is positive
    }
  }
  return T;
}

ScenarioStats GenerateAndRunScenario(
    const absl::flat_hash_map<ArcIndex, int64_t>& default_costs,
    const Graph& graph, const absl::optional<std::string> graph_name,
    const absl::flat_hash_map<std::string, std::unique_ptr<Graph>>&
        auxiliary_costs,
    const int old_src, const int old_dst, const int new_src,
    const int new_dst) {
  // generate scenario
  PenaltyRunner penalty_runner(
      default_costs, graph, graph_name, absl::GetFlag(FLAGS_graph_directory),
      auxiliary_costs, old_src, old_dst, new_src, new_dst,
      absl::GetFlag(FLAGS_penalty_mode),
      absl::GetFlag(FLAGS_penalty_multiplier),
      absl::GetFlag(FLAGS_num_penalized_paths),
      absl::GetFlag(FLAGS_break_if_segment_repenalized));

  ScenarioStats stats = penalty_runner.Stats();
  if (stats.P_plotly == "CUT" || stats.P_plotly == "NULL") {
    return stats;
  }

  std::string plotly_fpath = GetPlotlyFilePath(old_src, old_dst);
  std::ofstream plotly_file;
  plotly_file.open(plotly_fpath);

  std::string smallplotly_fpath = GetSmallplotlyFilePath(old_src, old_dst);
  std::ofstream smallplotly_file;
  smallplotly_file.open(smallplotly_fpath);

  absl::flat_hash_map<ArcIndex, int64_t> updated_forward_costs =
      penalty_runner.UpdatedForwardCosts();
  absl::flat_hash_set<ArcIndex> P = penalty_runner.GetLastPath();

  absl::flat_hash_set<ArcIndex> all_path_arcs = penalty_runner.AllPathArcs();
  absl::flat_hash_set<ArcIndex> all_penalized_arcs =
      penalty_runner.AllPenalizedArcs();

  std::vector<std::string> path_plotly_strings =
      penalty_runner.PathPlotlyStrings();
  std::vector<std::string> path_smallplotly_strings =
      penalty_runner.PathSmallplotlyStrings();
  std::vector<std::string> penalty_plotly_strings =
      penalty_runner.PenaltyPlotlyStrings();

  for (const ArcIndex& e : P) {
    if (all_path_arcs.contains(e)) {
      all_path_arcs.erase(e);
    }
    if (all_penalized_arcs.contains(e)) {
      all_penalized_arcs.erase(e);
    }
  }

  // actually print paths and penalty arcs to plotly file
  // with shortest path being printed last for better visibility
  // followed by path to explain
  CHECK_GE(path_plotly_strings.size(), 2);
  for (int i = 1; i < (path_plotly_strings.size() - 1); i++) {
    plotly_file << path_plotly_strings[i] << "\n";
  }
  plotly_file << path_plotly_strings[0] << "\n";
  plotly_file << path_plotly_strings[path_plotly_strings.size() - 1] << "\n";
  plotly_file << "\n";
  for (const std::string& plotly_string : penalty_plotly_strings) {
    plotly_file << plotly_string << "\n";
  }
  plotly_file << "\n";

  // smallplotly file only contains the given path
  CHECK_EQ(path_smallplotly_strings.size(), 1);
  smallplotly_file << path_smallplotly_strings[0] << "\n";
  smallplotly_file << "\n";

  absl::flat_hash_map<ArcIndex, int64_t> U;
  for (const Graph::Node& node : graph.nodes()) {
    for (int i = 0; i < graph.ArcsForNode(node.id).size(); i++) {
      ArcIndex e = ArcIndex{.node = node.id, .index = i};
      if (updated_forward_costs.contains(e)) {
        U[e] = updated_forward_costs.at(e);
      } else if (all_path_arcs.contains(e)) {
        U[e] = (int64_t)(default_costs.at(e) *
                         absl::GetFlag(FLAGS_path_only_other_multiplier));
      } else {
        U[e] = (int64_t)(default_costs.at(e) *
                         absl::GetFlag(FLAGS_other_multiplier));
      }
    }
  }

  absl::flat_hash_map<ArcIndex, int64_t> T;
  if (absl::GetFlag(FLAGS_use_normalized_objective)) {
    T = GetNormalizedT(default_costs, U);
  } else {
    for (const auto& [e, _] : U) {
      T[e] = 1;
    }
  }

  Formulation problem{.G = graph,
                      .L = default_costs,
                      .U = U,
                      .T = T,
                      .P = P,
                      .path_src = new_src,
                      .path_dst = new_dst};
  absl::Time start_time = absl::Now();
  absl::StatusOr<std::pair<FlowSolution, CutSolution>> solution =
      absl::GetFlag(FLAGS_use_fast_solver)
          ? FastSolve(problem, GetCycleStoreDirectoryName(),
                      GetCutStoreDirectoryName(),
                      GetCycleStoreFnamePrefix(old_src, old_dst))
          : Solve(problem, GetCycleStoreDirectoryName(),
                  GetCutStoreDirectoryName(),
                  GetCycleStoreFnamePrefix(old_src, old_dst));
  stats.solver_runtime_seconds =
      absl::ToDoubleSeconds(absl::Now() - start_time);
  CHECK_OK(solution);
  CutSolution cut = solution->second;
  int explanation_arc_count = 0;
  int num_exp_arcs_in_paths = 0;
  int num_exp_arcs_in_penalized_set = 0;
  std::vector<std::pair<int, int>> explanation_arcs;
  for (const auto& [e, we] : cut.w) {
    if (we != problem.L.at(e)) {
      std::string plotly_arc_string = GetPlotlyPathString(
          graph, {AOrRIndex{.forward = e, .is_forward = true}},
          absl::StrCat("E", explanation_arc_count), "#0000FF|stroke_weight=6");
      plotly_file << plotly_arc_string << "\n";

      std::string smallplotly_arc_string = GetPlotlyPathString(
          graph, {AOrRIndex{.forward = e, .is_forward = true}},
          absl::StrCat("E", explanation_arc_count), "#FF0000|stroke_weight=9");
      smallplotly_file << smallplotly_arc_string << "\n";

      explanation_arc_count++;
      if (all_path_arcs.contains(e)) {
        num_exp_arcs_in_paths++;
      }
      if (all_penalized_arcs.contains(e)) {
        num_exp_arcs_in_penalized_set++;
      }
      explanation_arcs.push_back(std::make_pair(e.node, graph.ArcIndexDst(e)));
      // theorem check: no explanation arc is in P
      CHECK(!problem.P.contains(e));
      // theorem check: when there are only two paths and only ei is penalized
      // then the explanation is on the first path
      if ((absl::GetFlag(FLAGS_penalty_mode) ==
           "most_important_arc_only_simple_version_A") &&
          (absl::GetFlag(FLAGS_num_penalized_paths) == 2)) {
        CHECK(all_path_arcs.contains(e));
      }
    }
  }
  stats.frac_explanation_in_paths =
      (double)num_exp_arcs_in_paths / (double)explanation_arc_count;
  stats.frac_explanation_in_penalized_set =
      (double)num_exp_arcs_in_penalized_set / (double)explanation_arc_count;
  stats.num_arcs_in_explanation = explanation_arc_count;
  stats.num_components_in_explanation = GetNumComponents(explanation_arcs);
  plotly_file.close();
  smallplotly_file.close();
  return stats;
}

std::string GetStatSummaryString(
    const std::vector<std::pair<double, std::string>>& values_and_files) {
  std::vector<std::pair<double, std::string>> sorted_values_and_files =
      values_and_files;
  std::sort(sorted_values_and_files.begin(), sorted_values_and_files.end());

  std::vector<double> TILES = {0.0,  0.02, 0.05, 0.1,  0.25, 0.5,
                               0.75, 0.9,  0.95, 0.98, 1.0};

  std::vector<std::string> stats;

  for (double q : TILES) {
    int rank = std::max(0, std::min((int)(q * values_and_files.size()),
                                    (int)values_and_files.size() - 1));
    stats.push_back(absl::StrCat(q, ":", sorted_values_and_files[rank].first,
                                 "::", sorted_values_and_files[rank].second));
  }
  return absl::StrJoin(stats, "\n");
}

struct SubgraphParams {
  std::string penalty_mode;
  double penalty_multiplier;
  double num_penalized_paths;
  bool break_if_segment_repenalized;
};

const absl::flat_hash_map<std::string, SubgraphParams> GetSubgraphParamsDict() {
  absl::flat_hash_map<std::string, SubgraphParams> result;
  result["wp_A"] = {.penalty_mode = "whole_path",
                    .penalty_multiplier = 1.1,
                    .num_penalized_paths = 30,
                    .break_if_segment_repenalized = false};
  result["iaosv_A"] = {.penalty_mode = "important_arcs_only_simple_version_A",
                       .penalty_multiplier = 100000.0,
                       .num_penalized_paths = 30,
                       .break_if_segment_repenalized = true};
  return result;
}

std::tuple<Graph, absl::flat_hash_map<std::string, std::unique_ptr<Graph>>, int,
           int>
GetArcInducedSubgraph(
    const Graph& graph,
    const absl::flat_hash_map<std::string, std::unique_ptr<Graph>>&
        auxiliary_costs,
    const absl::flat_hash_set<ArcIndex>& subgraph_arcs, const int old_src,
    const int old_dst) {
  // sort subgraph arcs to ensure consistent node and arc mapping
  std::vector<ArcIndex> sorted_subgraph_arcs(subgraph_arcs.begin(),
                                             subgraph_arcs.end());
  std::sort(sorted_subgraph_arcs.begin(), sorted_subgraph_arcs.end());

  // node mapping
  absl::flat_hash_map<int, int> old_to_new_id;
  std::vector<int> new_to_old_id;
  CHECK_GE(old_src, 0);
  CHECK_GE(old_dst, 0);
  CHECK_LT(old_src, graph.NumNodes());
  CHECK_LT(old_dst, graph.NumNodes());
  old_to_new_id[old_src] = 0;
  new_to_old_id.push_back(old_src);
  old_to_new_id[old_dst] = 1;
  new_to_old_id.push_back(old_dst);
  std::vector<int> all_old_ids;
  for (const ArcIndex& e : sorted_subgraph_arcs) {
    int arc_src = e.node;
    int arc_dst = graph.ArcIndexDst(e);
    all_old_ids.push_back(arc_src);
    all_old_ids.push_back(arc_dst);
  }
  // ensures that subgraph node mapping is always the same
  std::sort(all_old_ids.begin(), all_old_ids.end());
  for (int i = 0; i < all_old_ids.size(); i++) {
    if (((i == 0) || (all_old_ids[i] != all_old_ids[i - 1])) &&
        (all_old_ids[i] != old_src) && (all_old_ids[i] != old_dst)) {
      old_to_new_id[all_old_ids[i]] = new_to_old_id.size();
      new_to_old_id.push_back(all_old_ids[i]);
    }
  }

  CHECK_EQ(old_to_new_id.size(), new_to_old_id.size());

  // setup nodes and arcs
  std::vector<Graph::Node> new_nodes;
  std::vector<std::vector<Graph::Arc>> new_arcs(new_to_old_id.size());
  absl::flat_hash_map<std::string, std::vector<std::vector<Graph::Arc>>>
      new_auxiliary_arcs;
  for (const auto& [name, _] : auxiliary_costs) {
    new_auxiliary_arcs[name] =
        std::vector<std::vector<Graph::Arc>>(new_to_old_id.size());
  }
  for (int new_id = 0; new_id < new_to_old_id.size(); new_id++) {
    int old_id = new_to_old_id[new_id];
    new_nodes.push_back(Graph::Node{.id = new_id,
                                    .lat = graph.GetNode(old_id).lat,
                                    .lng = graph.GetNode(old_id).lng});
  }

  int arcs_so_far = 0;
  for (const ArcIndex& e : sorted_subgraph_arcs) {
    int old_src = e.node;
    int old_dst = graph.ArcIndexDst(e);
    int new_src = old_to_new_id.at(old_src);
    int new_dst = old_to_new_id.at(old_dst);
    new_arcs[new_src].push_back(Graph::Arc{
        .dst = new_dst, .num = arcs_so_far, .cost = graph.CostForArc(e)});
    for (const auto& [name, aux_graph] : auxiliary_costs) {
      int64_t cost = aux_graph->CostForArc(e);
      new_auxiliary_arcs[name][new_src].push_back(
          Graph::Arc{.dst = new_dst, .num = arcs_so_far, .cost = cost});
    }
    arcs_so_far++;
  }

  // make graphs
  std::vector<Graph::Node> main_new_nodes = new_nodes;
  absl::flat_hash_map<std::string, std::vector<Graph::Node>>
      auxiliary_new_nodes;
  for (const auto& [name, _] : new_auxiliary_arcs) {
    auxiliary_new_nodes[name] = new_nodes;
  }
  Graph new_graph(std::move(new_arcs), std::move(main_new_nodes));
  absl::flat_hash_map<std::string, std::unique_ptr<Graph>> new_auxiliary_costs;
  for (const auto& [name, arcs] : new_auxiliary_arcs) {
    new_auxiliary_costs[name] =
        std::make_unique<Graph>(std::move(new_auxiliary_arcs.at(name)),
                                std::move(auxiliary_new_nodes.at(name)));
  }

  return {new_graph, std::move(new_auxiliary_costs), old_to_new_id.at(old_src),
          old_to_new_id.at(old_dst)};
}

absl::flat_hash_map<ArcIndex, int64_t> GetDefaultCosts(const Graph& graph) {
  absl::flat_hash_map<ArcIndex, int64_t> default_costs;
  for (const Graph::Node& node : graph.nodes()) {
    for (int i = 0; i < graph.ArcsForNode(node.id).size(); i++) {
      ArcIndex e = ArcIndex{.node = node.id, .index = i};
      default_costs[e] = graph.CostForArc(e);
    }
  }
  return default_costs;
}

std::tuple<Graph, absl::flat_hash_map<std::string, std::unique_ptr<Graph>>, int,
           int>
GetSubgraph(const absl::flat_hash_map<ArcIndex, int64_t>& default_costs,
            const Graph& graph,
            const absl::flat_hash_map<std::string, std::unique_ptr<Graph>>&
                auxiliary_costs,
            const int old_src, const int old_dst) {
  CHECK_NE(absl::GetFlag(FLAGS_subgraph_name), "NONE");
  const auto subgraph_params = GetSubgraphParamsDict();
  if (subgraph_params.contains(absl::GetFlag(FLAGS_subgraph_name))) {
    LOG(INFO) << "Constructing single subgraph "
              << absl::GetFlag(FLAGS_subgraph_name) << "...";
    SubgraphParams params =
        subgraph_params.at(absl::GetFlag(FLAGS_subgraph_name));
    PenaltyRunner penalty_runner(
        default_costs, graph, absl::nullopt,
        absl::GetFlag(FLAGS_graph_directory), auxiliary_costs, old_src, old_dst,
        old_src, old_dst, params.penalty_mode, params.penalty_multiplier,
        params.num_penalized_paths, params.break_if_segment_repenalized);
    absl::flat_hash_set<ArcIndex> subgraph_arcs = penalty_runner.AllPathArcs();
    return GetArcInducedSubgraph(graph, auxiliary_costs, subgraph_arcs, old_src,
                                 old_dst);
  } else if (absl::GetFlag(FLAGS_subgraph_name).find("_plus_") !=
             std::string::npos) {
    std::vector<std::string> parts =
        absl::StrSplit(absl::GetFlag(FLAGS_subgraph_name), "_plus_");
    CHECK_GE(parts.size(), 2);
    absl::flat_hash_set<ArcIndex> subgraph_arcs;
    for (const std::string& part : parts) {
      LOG(INFO) << "Constructing subgraph part " << part << "...";
      CHECK(subgraph_params.contains(part));
      SubgraphParams params = subgraph_params.at(part);
      PenaltyRunner penalty_runner(
          default_costs, graph, absl::nullopt,
          absl::GetFlag(FLAGS_graph_directory), auxiliary_costs, old_src,
          old_dst, old_src, old_dst, params.penalty_mode,
          params.penalty_multiplier, params.num_penalized_paths,
          params.break_if_segment_repenalized);
      subgraph_arcs.insert(penalty_runner.AllPathArcs().begin(),
                           penalty_runner.AllPathArcs().end());
    }
    return GetArcInducedSubgraph(graph, auxiliary_costs, subgraph_arcs, old_src,
                                 old_dst);
  } else {
    LOG(FATAL) << "Invalid subgraph name: "
               << absl::GetFlag(FLAGS_subgraph_name);
    CHECK(false);
  }
}

void Main() {
  std::string penalty_mode = absl::GetFlag(FLAGS_penalty_mode);
  CheckIsValidType(absl::GetFlag(FLAGS_penalty_mode));
  LOG(INFO) << "Loading graph...";
  std::unique_ptr<Graph> graph_ptr = Graph::LoadFromDirectory(
      absl::GetFlag(FLAGS_graph_directory), "duration_secs");
  Graph graph = *graph_ptr;
  LOG(INFO) << "Generating default costs for penalty runner...";
  absl::flat_hash_map<ArcIndex, int64_t> default_costs = GetDefaultCosts(graph);
  absl::flat_hash_map<std::string, std::unique_ptr<Graph>> auxiliary_costs;
  LOG(INFO) << "Loading dist_meters graph...";
  auxiliary_costs["dist_meters"] = Graph::LoadFromDirectory(
      absl::GetFlag(FLAGS_graph_directory), "dist_meters");
  LOG(INFO) << "Loading road_type graph...";
  auxiliary_costs["road_type"] = Graph::LoadFromDirectory(
      absl::GetFlag(FLAGS_graph_directory), "road_type");
  LOG(INFO) << "Loading num_lanes graph...";
  auxiliary_costs["num_lanes"] = Graph::LoadFromDirectory(
      absl::GetFlag(FLAGS_graph_directory), "num_lanes");

  LOG(INFO) << "Number of nodes: " << graph.NumNodes();
  LOG(INFO) << "Number of arcs: " << graph.NumArcs();
  LOG(INFO) << "Loading queries...";
  std::vector<std::pair<int, int>> queries = GetQueries();
  fs::create_directories(GetPlotlyDirectoryName());
  std::vector<std::pair<ScenarioStats, std::string>> all_stat_filename_pairs;
  int num_null = 0;
  int num_cut = 0;
  for (int i = 0; i < queries.size(); i++) {
    LOG(INFO) << "Running query " << i << "/" << queries.size() << " from "
              << queries[i].first << " to " << queries[i].second << "...";
    const auto& [src, dst] = queries[i];
    ScenarioStats new_stats;
    if (absl::GetFlag(FLAGS_subgraph_name) == "NONE") {
      new_stats = GenerateAndRunScenario(default_costs, graph, absl::nullopt,
                                         auxiliary_costs, src, dst, src, dst);
    } else {
      auto [subgraph, subgraph_auxiliary_costs, new_src, new_dst] =
          GetSubgraph(default_costs, graph, auxiliary_costs, src, dst);
      absl::flat_hash_map<ArcIndex, int64_t> subgraph_default_costs =
          GetDefaultCosts(subgraph);
      absl::optional<std::string> subgraph_name =
          absl::GetFlag(FLAGS_subgraph_name);
      absl::Time start = absl::Now();
      new_stats = GenerateAndRunScenario(
          subgraph_default_costs, subgraph, subgraph_name,
          subgraph_auxiliary_costs, src, dst, new_src, new_dst);
      absl::Time end = absl::Now();
      new_stats.scenario_runtime_seconds = absl::ToDoubleSeconds(end - start);
      LOG(INFO) << "Solver time taken: " << new_stats.solver_runtime_seconds;
      LOG(INFO) << "Total time taken: " << new_stats.scenario_runtime_seconds;
    }
    if (new_stats.P_plotly == "NULL") {
      LOG(INFO) << "No path exists! Skipping...";
      num_null++;
      continue;
    }
    if (new_stats.P_plotly == "CUT") {
      LOG(INFO) << "Small cut with faraway edges exists. Skipping...";
      num_cut++;
      continue;
    }
    all_stat_filename_pairs.push_back(
        std::make_pair(new_stats, GetPlotlyFilePath(src, dst)));
  }

  LOG(INFO) << "Stats and plotly file writing...";
  std::ofstream summary_stats_file;
  summary_stats_file.open(GetSummaryStatsFilePath());
  std::ofstream summary_plotly_file;
  summary_plotly_file.open(GetSummaryPlotlyFilePath());
  absl::flat_hash_map<std::string, std::vector<std::pair<double, std::string>>>
      stats_by_name;
  for (int i = 0; i < all_stat_filename_pairs.size(); i++) {
    const auto [stats, filename] = all_stat_filename_pairs[i];
    summary_plotly_file << "Scenario" << i << stats.shortest_path_plotly
                        << "\n";
    summary_plotly_file << "Scenario" << i << stats.P_plotly << "\n";
    stats_by_name["frac_explanation_in_paths"].push_back(
        std::make_pair(stats.frac_explanation_in_paths, filename));
    stats_by_name["frac_explanation_in_penalized_set"].push_back(
        std::make_pair(stats.frac_explanation_in_penalized_set, filename));
    stats_by_name["num_arcs_in_explanation"].push_back(
        std::make_pair(stats.num_arcs_in_explanation, filename));
    stats_by_name["num_components_in_explanation"].push_back(
        std::make_pair(stats.num_components_in_explanation, filename));
    stats_by_name["solver_runtime_seconds"].push_back(
        std::make_pair(stats.solver_runtime_seconds, filename));
    stats_by_name["scenario_runtime_seconds"].push_back(
        std::make_pair(stats.scenario_runtime_seconds, filename));
  }

  summary_stats_file << "num_null: " << num_null << " / " << queries.size()
                     << "\n";
  summary_stats_file << "num_cut: " << num_cut << " / " << queries.size()
                     << "\n";
  LOG(INFO) << "num_null: " << num_null << " / " << queries.size() << "\n";
  LOG(INFO) << "num_cut: " << num_cut << " / " << queries.size() << "\n";
  for (const auto& [name, values] : stats_by_name) {
    summary_stats_file << name << " " << GetStatSummaryString(values) << "\n";
    LOG(INFO) << name << " " << GetStatSummaryString(values);
  }
  summary_stats_file.close();
  summary_plotly_file.close();
}

}  // namespace
}  // namespace geo_algorithms

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  geo_algorithms::Main();
  return 0;
}
