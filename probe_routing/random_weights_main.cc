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

#include <algorithm>
#include <complex>
#include <cstddef>
#include <fstream>
#include <string>

#include "absl/base/log_severity.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "file_util.h"
#include "graph.h"
#include "graph_search.h"
#include "graph_utils.h"
#include "random_weights_utils.h"
#include "tsv_utils.h"

ABSL_FLAG(std::string, tsv_directory, "*-PROCESSED DIRECTORY NAME",
          "Place where input TSV files are stored.");
ABSL_FLAG(std::string, real_traffic_subdir, "highway_traffic_2.0",
          "Subdir where input TSV files for traffic graph are stored. Also set "
          "use_clusters flag if highway.");
ABSL_FLAG(std::string, resampled_traffic_subdir,
          "resampled_highway_traffic_2.0",
          "Resampled traffic. Also set use_clusters flag if highway.");
ABSL_FLAG(double, epsilon, 0.1, "Approximation error.");
ABSL_FLAG(std::string, queries_filename, "medium_queries_random_100.tsv",
          "List of source-target queries to assess.");
ABSL_FLAG(std::string, compare_paths_subdir, "compare_paths",
          "Subdirectory of tsv directory where viewable paths are stored.");
ABSL_FLAG(
    bool, use_clusters, true,
    absl::StrCat("Whether or not the traffic and resampled traffic files ",
                 "use clustered randomness. The answer must be the same ",
                 "for both files and, if true, the randomness clusters ",
                 "must be specified by clusters.tsv."));

namespace geo_algorithms {
namespace {

namespace fs = std::filesystem;

enum TrafficType { NONE, REAL, RESAMPLED, MERGED };

using utils_tsv::TsvReader;
using utils_tsv::TsvRow;
using utils_tsv::TsvSpec;
using utils_tsv::TsvWriter;

std::string TrafficRunDirectoryName() {
  absl::Time now = absl::Now();
  absl::TimeZone lax;

  // Because LoadTimeZone() may fail, it is always safer to load time zones×
  // checking its return value:
  if (!absl::LoadTimeZone("America/Los_Angeles", &lax)) {
    // Handle failure
  }

  return absl::StrCat(absl::GetFlag(FLAGS_tsv_directory), "/traffic_run_",
                      absl::FormatTime("%Y-%m-%d-%H:%M:%S", now, lax));
}

std::string GetGraphName(TrafficType type, double threshold_scale) {
  switch (type) {
    case NONE:
      return "no_traffic__graph";

    case REAL:
      return absl::StrCat(absl::GetFlag(FLAGS_real_traffic_subdir), "__graph");

    case RESAMPLED:
      return absl::StrCat(absl::GetFlag(FLAGS_resampled_traffic_subdir),
                          "__graph");

    // threshold graph, followed by lesser graph, followed by greater graph
    case MERGED:
      return absl::StrCat("merged__", "no_traffic__",
                          absl::GetFlag(FLAGS_resampled_traffic_subdir), "__",
                          absl::GetFlag(FLAGS_real_traffic_subdir), "__",
                          threshold_scale, "__graph");
  }
}

std::string GraphDirectoryName(TrafficType traffic_type) {
  switch (traffic_type) {
    case NONE:
      return absl::GetFlag(FLAGS_tsv_directory);
    case REAL:
      return absl::StrCat(absl::GetFlag(FLAGS_tsv_directory), "/",
                          absl::GetFlag(FLAGS_real_traffic_subdir));
    case RESAMPLED:
      return absl::StrCat(absl::GetFlag(FLAGS_tsv_directory), "/",
                          absl::GetFlag(FLAGS_resampled_traffic_subdir));
    case MERGED:
      return "ERROR";
  }
}

std::unique_ptr<MultiCostGraph> GetGraph(TrafficType type) {
  LOG(INFO) << "Loading graph from " << GraphDirectoryName(type) << "...";
  std::unique_ptr<MultiCostGraph> graph =
      MultiCostGraph::LoadFromDirectory(GraphDirectoryName(type));
  MultiCostGraph* graph_ptr = graph.get();
  LOG(INFO) << "Number of nodes: " << graph_ptr->nodes().size() << "";
  int edges = 0;
  for (const auto& node : graph_ptr->nodes()) {
    edges += graph_ptr->ArcsForNode(node.id).size();
  }
  LOG(INFO) << "Number of edges: " << edges << "";
  return graph;
}

void SetupOutput(std::string run_directory_name) {
  fs::create_directory(run_directory_name);
  fs::create_directory(absl::StrCat(run_directory_name, "/",
                                    absl::GetFlag(FLAGS_compare_paths_subdir)));
}

std::vector<std::pair<int, int>> GetQueries() {
  // no need to use traffic directory here
  std::string full_queries_filename =
      absl::StrCat(absl::GetFlag(FLAGS_tsv_directory), "/",
                   absl::GetFlag(FLAGS_queries_filename));
  return GetQueriesFromFile(full_queries_filename);
}

double ReplacementThreshold(int n_vertices, double cost_lb,
                            double threshold_scale) {
  return cost_lb * absl::GetFlag(FLAGS_epsilon) * absl::GetFlag(FLAGS_epsilon) /
         (threshold_scale * std::log(n_vertices));
}

std::string GetCostDistortionTsvPath(std::string run_directory_name) {
  return absl::StrCat(run_directory_name, "/distortion.tsv");
}

void WriteSubsettedPathToBoundingBox(const MultiCostGraph& graph,
                                     WeightedCostSearch::Result path,
                                     std::ofstream& file, int n_points) {
  CHECK_GE(n_points, 2);
  int size = path.path.size();
  int spacing = ((size - 1) - (size - 1) % (n_points - 1)) / (n_points - 1);
  int next_value = 0;
  int n_so_far = 0;

  for (int i = 0; i < size; i++) {
    if (size < n_points || i == next_value) {
      ArcIndex arc = path.path[i];
      file << graph.nodes()[arc.node].lat << ":" << graph.nodes()[arc.node].lng;
      if (i != size - 1) {
        file << ",\n";
      }
      next_value += spacing + (n_so_far < ((size - 1) % (n_points - 1)));
      n_so_far++;
      if (n_so_far == n_points) {
        CHECK_EQ(i, path.path.size() - 1);
      }
    }
  }
  if (path.path.size() >= n_points) {
    CHECK_EQ(n_so_far, n_points);
  } else {
    CHECK_LE(n_so_far, n_points);
  }
}

void WriteSubsettedPathsToBoundingBox(const MultiCostGraph& graph,
                                      WeightedCostSearch::Result merged_path,
                                      WeightedCostSearch::Result real_path,
                                      std::string filename, int n_points) {
  std::ofstream file;
  file.open(filename);
  file << "MergedLine|#FF0000: ";
  WriteSubsettedPathToBoundingBox(graph, merged_path, file, n_points);
  file << "\n";
  file << "\n";
  file << "RealLine|#0000FF: ";
  WriteSubsettedPathToBoundingBox(graph, real_path, file, n_points);
  file << "\n";
  file.close();
}

std::string ComparePathsFilename(std::string run_directory_name, int origin,
                                 int destination) {
  return absl::StrCat(run_directory_name, "/",
                      absl::GetFlag(FLAGS_compare_paths_subdir), "/", "path_",
                      origin, "_", destination, ".tsv");
}

bool ClusterFilesEqual() {
  std::string filename1 =
      absl::StrCat(absl::GetFlag(FLAGS_tsv_directory), "/",
                   absl::GetFlag(FLAGS_real_traffic_subdir), "/clusters.tsv");
  std::string filename2 = absl::StrCat(
      absl::GetFlag(FLAGS_tsv_directory), "/",
      absl::GetFlag(FLAGS_resampled_traffic_subdir), "/clusters.tsv");
  std::ifstream file1(filename1, std::ios::binary);
  std::ifstream file2(filename2, std::ios::binary);

  if (!file1.is_open() || !file2.is_open()) {
    return false;  // Error opening files
  }

  while (file1 && file2) {
    if (file1.get() != file2.get()) {
      return false;
    }
  }

  return file1.eof() && file2.eof();
}

std::string ClusterFilename() {
  CHECK(absl::GetFlag(FLAGS_use_clusters));
  return absl::StrCat(absl::GetFlag(FLAGS_tsv_directory), "/",
                      absl::GetFlag(FLAGS_real_traffic_subdir),
                      "/clusters.tsv");
}

std::vector<int> GetClusterAssignment(int num_arcs) {
  std::vector<int> cluster_assignment;
  if (absl::GetFlag(FLAGS_use_clusters)) {
    CHECK(ClusterFilesEqual());
    TsvReader cluster_reader(ClusterFilename());
    for (; !cluster_reader.AtEnd();) {
      absl::flat_hash_map<std::string, std::string> cluster_row =
          cluster_reader.ReadRow();
      cluster_assignment.push_back(GetIntFeature(cluster_row, "cluster_id"));
    }
    CHECK_EQ(cluster_assignment.size(), num_arcs);
  } else {
    for (int i = 0; i < num_arcs; i++) {
      cluster_assignment.push_back(i);
    }
  }

  return cluster_assignment;
}

absl::flat_hash_map<std::string, double> CostDistortionExperiment(
    std::string run_directory_name, double threshold_scale) {
  absl::flat_hash_map<std::string, double> important_logged_values;
  std::unique_ptr<MultiCostGraph> no_traffic_graph = GetGraph(NONE);
  std::vector<std::pair<int, int>> queries = GetQueries();

  std::vector<std::pair<std::pair<int, int>, double>> no_traffic_results;
  PathCache no_traffic_cache(no_traffic_graph.get(),
                             GetGraphName(NONE, threshold_scale),
                             absl::GetFlag(FLAGS_tsv_directory));
  int n_failed = 0;
  std::vector<double> costs;
  LOG(INFO) << "Computing no-traffic paths...";
  for (int i = 0; i < queries.size(); i++) {
    LOG(INFO) << "Query " << i << "...";
    auto [source, target] = queries[i];
    absl::StatusOr<WeightedCostSearch::Result> result =
        no_traffic_cache.GetPath(source, target);
    if (result.ok()) {
      no_traffic_results.push_back(
          std::make_pair(std::make_pair(source, target),
                         WeightedCost(no_traffic_graph->DefaultCostWeights(),
                                      result->cost_vector)));
    } else {
      LOG(INFO) << source << " to " << target
                << " search failed due to connectivity";
      n_failed++;
    }
  }

  LOG(INFO) << "No-traffic paths found. Sorting queries by value...";
  std::sort(no_traffic_results.begin(), no_traffic_results.end(),
            [](const auto& a, const auto& b) {
              if (a.second != b.second) return a.second < b.second;
              if (a.first.first != b.first.first) {
                return a.first.first < b.first.first;
              }
              return a.first.second < b.first.second;
            });

  LOG(INFO) << "Reading cluster assignment...";
  std::vector<int> cluster_assignment =
      GetClusterAssignment(no_traffic_graph->NumArcs());

  LOG(INFO) << "Preparing traffic experiment...";
  double min_cost = no_traffic_results[0].second;
  double current_cost_lb = min_cost;
  double COST_RANGE_RATIO = 2.0;
  int n_nodes = no_traffic_graph->NumNodes();
  LOG(INFO) << "Making real traffic graph...";
  std::unique_ptr<MultiCostGraph> real_graph = GetGraph(REAL);
  LOG(INFO) << "Making fake traffic graph...";
  std::unique_ptr<MultiCostGraph> fake_graph = GetGraph(RESAMPLED);
  LOG(INFO) << "Making " << current_cost_lb << " to "
            << COST_RANGE_RATIO * current_cost_lb << " merged graph...";
  double replacement_threshold =
      ReplacementThreshold(n_nodes, current_cost_lb, threshold_scale);
  auto [merged_graph_ptr, real_clusters_fraction] =
      GetMergedGraphAndRealClustersFraction(*no_traffic_graph, *fake_graph,
                                            *real_graph, replacement_threshold,
                                            cluster_assignment);
  std::unique_ptr<MultiCostGraph> merged_graph =
      std::unique_ptr<MultiCostGraph>(merged_graph_ptr);
  important_logged_values[absl::StrCat("real_clusters_fraction_",
                                       replacement_threshold)] =
      real_clusters_fraction;
  LOG(INFO) << "Making path caches...";
  PathCache real_cache(real_graph.get(), GetGraphName(REAL, threshold_scale),
                       absl::GetFlag(FLAGS_tsv_directory));
  PathCache merged_cache(merged_graph.get(),
                         GetGraphName(MERGED, threshold_scale),
                         absl::GetFlag(FLAGS_tsv_directory));

  LOG(INFO) << "Starting main experiment...";
  std::vector<std::string> column_names{
      "source", "target",   "no_traffic", "real",  "merged", "merged_in_real",
      "m/r",    "merged/r", "n_real",     "n_all", "r/a"};
  TsvSpec columns(column_names);
  TsvWriter writer(GetCostDistortionTsvPath(run_directory_name), &columns);
  std::vector<double> merged_graph_ratios;
  std::vector<double> real_graph_ratios;
  for (int i = 0; i < no_traffic_results.size(); i++) {
    auto [query, cost] = no_traffic_results[i];
    if (cost > COST_RANGE_RATIO * current_cost_lb) {
      current_cost_lb *= COST_RANGE_RATIO;
      LOG(INFO) << "Remaking " << current_cost_lb << " to "
                << COST_RANGE_RATIO * current_cost_lb << " merged graph...";
      replacement_threshold =
          ReplacementThreshold(n_nodes, current_cost_lb, threshold_scale);
      auto [merged_graph_ptr, real_clusters_fraction] =
          GetMergedGraphAndRealClustersFraction(
              *no_traffic_graph, *fake_graph, *real_graph,
              replacement_threshold, cluster_assignment);
      merged_graph.reset(merged_graph_ptr);
      important_logged_values[absl::StrCat("real_clusters_fraction_",
                                           replacement_threshold)] =
          real_clusters_fraction;
      merged_cache.ReplaceMergedGraph(merged_graph.get());
    }

    LOG(INFO) << "Query " << i << "/" << no_traffic_results.size() << "...";

    absl::StatusOr<WeightedCostSearch::Result> real_result =
        real_cache.GetPath(query.first, query.second);
    absl::StatusOr<WeightedCostSearch::Result> merged_result =
        merged_cache.GetPath(query.first, query.second);
    int n_real_clusters_in_ball =
        no_traffic_cache.GetClusterCountWithinDistanceAboveThreshold(
            query.first, current_cost_lb * COST_RANGE_RATIO,
            ReplacementThreshold(n_nodes, current_cost_lb, threshold_scale),
            cluster_assignment);
    int n_all_clusters_in_ball =
        no_traffic_cache.GetClusterCountWithinDistanceAboveThreshold(
            query.first, current_cost_lb * COST_RANGE_RATIO, 0.0,
            cluster_assignment);
    double real_cost_for_merged_path = WeightedCost(
        no_traffic_graph->DefaultCostWeights(),
        EvaluatePathInNewGraph(*merged_graph, *real_graph, *merged_result));

    LOG(INFO) << "Printing paths to viewable format...";
    std::string comparison_filename =
        ComparePathsFilename(run_directory_name, query.first, query.second);
    WriteSubsettedPathsToBoundingBox(*real_graph, *merged_result, *real_result,
                                     comparison_filename, 500);

    TsvRow row(&columns);
    row.Add("source", query.first);
    row.Add("target", query.second);
    row.Add("no_traffic", cost);
    double real_cost = WeightedCost(no_traffic_graph->DefaultCostWeights(),
                                    real_result->cost_vector);
    row.Add("real", real_cost);
    double merged_cost = WeightedCost(no_traffic_graph->DefaultCostWeights(),
                                      merged_result->cost_vector);
    row.Add("merged", merged_cost);
    row.Add("merged_in_real", real_cost_for_merged_path);
    row.Add("m/r", std::abs(merged_cost / real_cost - 1.0));
    row.Add("merged/r", real_cost_for_merged_path / real_cost);
    row.Add("n_real", n_real_clusters_in_ball);
    row.Add("n_all", n_all_clusters_in_ball);
    row.Add("r/a", (n_real_clusters_in_ball + 0.0) / n_all_clusters_in_ball);
    merged_graph_ratios.push_back(std::abs(merged_cost / real_cost - 1.0));
    real_graph_ratios.push_back(real_cost_for_merged_path / real_cost);
    writer.WriteRow(row);
  }
  int min_index = std::distance(
      real_graph_ratios.begin(),
      std::min_element(real_graph_ratios.begin(), real_graph_ratios.end()));
  auto [query0, cost0] = no_traffic_results[min_index];
  LOG(INFO) << "Min pair is "
            << ComparePathsFilename(run_directory_name, query0.first,
                                    query0.second);
  fs::copy(
      ComparePathsFilename(run_directory_name, query0.first, query0.second),
      absl::StrCat(run_directory_name, "/min_path.txt"),
      fs::copy_options::overwrite_existing);
  int max_index = std::distance(
      real_graph_ratios.begin(),
      std::max_element(real_graph_ratios.begin(), real_graph_ratios.end()));
  auto [query1, cost1] = no_traffic_results[max_index];
  LOG(INFO) << "Max pair is "
            << ComparePathsFilename(run_directory_name, query1.first,
                                    query1.second);

  fs::copy(
      ComparePathsFilename(run_directory_name, query1.first, query1.second),
      absl::StrCat(run_directory_name, "/max_path.txt"),
      fs::copy_options::overwrite_existing);
  std::sort(merged_graph_ratios.begin(), merged_graph_ratios.end());
  std::sort(real_graph_ratios.begin(), real_graph_ratios.end());
  double IMPORTANT_PERCENTILE = 90;
  for (int i = 0; i < merged_graph_ratios.size(); i++) {
    LOG(INFO) << "Merged graph ratio " << i << ": " << merged_graph_ratios[i];
    if (i == IMPORTANT_PERCENTILE) {
      important_logged_values[absl::StrCat("merged_graph_ratio_", i)] =
          merged_graph_ratios[i];
    }
  }
  for (int i = 0; i < real_graph_ratios.size(); i++) {
    LOG(INFO) << "Real graph ratio " << i << ": " << real_graph_ratios[i];
    if (i == IMPORTANT_PERCENTILE) {
      important_logged_values[absl::StrCat("real_graph_ratio_", i)] =
          real_graph_ratios[i];
    }
  }
  return important_logged_values;
}

}  // namespace
}  // namespace geo_algorithms

int main(int argc, char* argv[]) {
  std::string run_directory_name = geo_algorithms::TrafficRunDirectoryName();
  geo_algorithms::SetupOutput(run_directory_name);
  LOG(WARNING) << "Real traffic directory: "
               << absl::GetFlag(FLAGS_real_traffic_subdir);
  LOG(WARNING) << "Fake traffic directory: "
               << absl::GetFlag(FLAGS_resampled_traffic_subdir);
  LOG(WARNING) << "Queries filename: " << absl::GetFlag(FLAGS_queries_filename);
  LOG(WARNING) << "Epsilon: " << absl::GetFlag(FLAGS_epsilon);
  LOG(WARNING) << "Clusters: " << absl::GetFlag(FLAGS_use_clusters);

  std::vector<double> all_threshold_scales = {
      0.00001, 0.000012, 0.000014, 0.000016, 0.000018, 0.00002, 0.00004,
      0.00005, 0.00006,  0.00007,  0.00008,  0.00009,  0.0001,  0.001,
      0.003,   0.01,     0.03,     0.1,      0.3,      1.0};

  std::vector<absl::flat_hash_map<std::string, double>> results;
  for (double t : all_threshold_scales) {
    LOG(INFO) << "COLLECTING INFO FOR SCALE " << t << "...";
    results.push_back(
        geo_algorithms::CostDistortionExperiment(run_directory_name, t));
  }

  LOG(INFO) << "DONE";

  for (int i = 0; i < all_threshold_scales.size(); i++) {
    LOG(INFO) << "Results for threshold " << all_threshold_scales[i] << ": ";
    for (const auto& [key, value] : results[i]) {
      LOG(INFO) << key << ": " << value;
    }
  }

  return 0;
}
