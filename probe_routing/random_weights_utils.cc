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

#include "random_weights_utils.h"

#include <math.h>

#include <iostream>
#include <limits>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "graph.h"
#include "graph_utils.h"
#include "tsv_utils.h"

ABSL_FLAG(std::string, file_not_found_msg, "FILE NOT FOUND", "File Message");
ABSL_FLAG(std::string, not_connected_msg, "NOT CONNECTED",
          "Vertices not connected message");
ABSL_FLAG(std::string, path_cache_prefix, "path_cache",
          "Prefix to graph name for directory that contains saved paths.");

namespace geo_algorithms {

using utils_tsv::TsvReader;
using utils_tsv::TsvRow;
using utils_tsv::TsvSpec;
using utils_tsv::TsvWriter;
namespace fs = std::filesystem;

void CheckNodes(const MultiCostGraph& no_traffic_graph,
                const MultiCostGraph& resampled_traffic_graph,
                const MultiCostGraph& real_traffic_graph) {
  CHECK_EQ(no_traffic_graph.NumNodes(), resampled_traffic_graph.NumNodes());
  CHECK_EQ(no_traffic_graph.NumNodes(), real_traffic_graph.NumNodes());
  for (int i = 0; i < no_traffic_graph.NumNodes(); i++) {
    CHECK_EQ(no_traffic_graph.nodes()[i].id,
             resampled_traffic_graph.nodes()[i].id);
    CHECK_EQ(no_traffic_graph.nodes()[i].id, real_traffic_graph.nodes()[i].id);
  }
}

std::vector<double> EvaluatePathInNewGraph(
    const MultiCostGraph& old_graph, const MultiCostGraph& new_graph,
    const WeightedCostSearch::Result& path) {
  // LOG(INFO) << "A";
  CHECK_EQ(old_graph.base_cost_names().size(),
           new_graph.base_cost_names().size());
  // LOG(INFO) << "B";
  std::vector<double> cost = new_graph.ZeroCostVector();
  // LOG(INFO) << "C";
  int i = 0;
  for (ArcIndex arc : path.path) {
    if (arc.index != -1) {
      CHECK_EQ(old_graph.ArcIndexDst(arc), new_graph.ArcIndexDst(arc));
      // LOG(INFO) << "E" << i;
      cost = AddCostVectors(cost, new_graph.CostForArc(arc));
      // LOG(INFO) << "F" << i;
      i++;
    }
  }
  // LOG(INFO) << "D";
  return cost;
}

absl::flat_hash_map<int, double> GetClusterCosts(
    const MultiCostGraph& graph, const std::vector<int>& cluster_assignment) {
  absl::flat_hash_map<int, double> cluster_costs;
  std::vector<double> w = graph.DefaultCostWeights();
  for (int node_id = 0; node_id < graph.NumNodes(); node_id++) {
    for (const auto& arc : graph.ArcsForNode(node_id)) {
      int cluster = cluster_assignment[arc.num];
      if (!cluster_costs.contains(cluster)) {
        cluster_costs[cluster] = 0.0;
      }
      cluster_costs[cluster] += WeightedCost(w, arc.cost_vector);
    }
  }
  return cluster_costs;
}

std::pair<MultiCostGraph*, double> GetMergedGraphAndRealClustersFraction(
    const MultiCostGraph& no_traffic_graph,
    const MultiCostGraph& resampled_traffic_graph,
    const MultiCostGraph& real_traffic_graph, double replacement_threshold,
    const std::vector<int>& cluster_assignment) {
  CheckNodes(no_traffic_graph, resampled_traffic_graph, real_traffic_graph);
  std::vector<MultiCostGraph::Node> merged_nodes = no_traffic_graph.nodes();
  std::vector<std::vector<MultiCostGraph::Arc>> merged_arcs(
      no_traffic_graph.NumNodes());
  absl::flat_hash_map<int, double> no_traffic_cluster_costs =
      GetClusterCosts(no_traffic_graph, cluster_assignment);
  int resampled_edges = 0;
  int real_edges = 0;
  for (int node_id = 0; node_id < no_traffic_graph.NumNodes(); node_id++) {
    for (int arc_id = 0; arc_id < no_traffic_graph.ArcsForNode(node_id).size();
         arc_id++) {
      ArcIndex arc_idx = ArcIndex{.node = node_id, .index = arc_id};
      int dst = no_traffic_graph.ArcIndexDst(arc_idx);
      CHECK_EQ(dst, resampled_traffic_graph.ArcIndexDst(arc_idx));
      CHECK_EQ(dst, real_traffic_graph.ArcIndexDst(arc_idx));
      int num = no_traffic_graph.ArcIndexNum(arc_idx);
      CHECK_EQ(num, resampled_traffic_graph.ArcIndexNum(arc_idx));
      CHECK_EQ(num, real_traffic_graph.ArcIndexNum(arc_idx));
      std::vector<double> real_traffic_cost =
          real_traffic_graph.CostForArc(arc_idx);
      std::vector<double> resampled_traffic_cost =
          resampled_traffic_graph.CostForArc(arc_idx);
      if (no_traffic_cluster_costs[cluster_assignment[num]] <
          replacement_threshold) {
        merged_arcs[node_id].push_back(MultiCostGraph::Arc{
            .dst = dst,
            .num = no_traffic_graph.ArcIndexNum(arc_idx),
            .cost_vector = resampled_traffic_cost,
        });
        resampled_edges++;
      } else {
        merged_arcs[node_id].push_back(MultiCostGraph::Arc{
            .dst = dst,
            .num = no_traffic_graph.ArcIndexNum(arc_idx),
            .cost_vector = real_traffic_cost,
        });
        real_edges++;
      }
    }
    CHECK_EQ(no_traffic_graph.ArcsForNode(node_id).size(),
             merged_arcs[node_id].size());
  }
  LOG(INFO) << "Threshold: " << replacement_threshold;
  LOG(INFO) << "Resampled/real edges: " << resampled_edges << "/" << real_edges;
  LOG(INFO) << "Real edges fraction:"
            << (real_edges + 0.0) / (real_edges + resampled_edges);
  int resampled_clusters = 0;
  int real_clusters = 0;
  for (const auto& cluster_cost : no_traffic_cluster_costs) {
    if (cluster_cost.second < replacement_threshold) {
      resampled_clusters++;
    } else {
      real_clusters++;
    }
  }
  LOG(INFO) << "Resampled/real clusters: " << resampled_clusters << "/"
            << real_clusters;
  double real_clusters_fraction =
      (real_clusters + 0.0) / (real_clusters + resampled_clusters);
  LOG(INFO) << "Real clusters fraction:" << real_clusters_fraction;
  return std::make_pair(
      new MultiCostGraph(no_traffic_graph.base_cost_names(),
                         std::move(merged_arcs), std::move(merged_nodes)),
      real_clusters_fraction);
}

PathCache::PathCache(MultiCostGraph* graph, const std::string& graph_name,
                     const std::string& output_dir)
    : output_dir_(output_dir), graph_(graph), graph_name_(graph_name) {
  fs::create_directories(cache_dir());
}

absl::StatusOr<WeightedCostSearch::Result> PathCache::GetPath(int source,
                                                              int target) {
  absl::StatusOr<WeightedCostSearch::Result> result =
      ReadPathFromFiles(source, target);
  if (result.status().message() == absl::GetFlag(FLAGS_file_not_found_msg)) {
    LOG(INFO) << "Path from " << source << " to " << target
              << " not found; generating...";
    WeightedCostSearch search(*graph_, graph_->DefaultCostWeights(), source);
    absl::StatusOr<WeightedCostSearch::Result> path = search.Search(target);
    SavePathToFiles(source, target, path);
    return path;
  } else {
    LOG(INFO) << "Loaded path from " << source << " to " << target
              << " from cache.";
    return result;
  }
}

void PathCache::SavePathToFiles(
    int source, int target, absl::StatusOr<WeightedCostSearch::Result> path) {
  std::string path_cache =
      absl::StrCat(cache_dir(), "/", path_cache_filename(source, target));
  std::string cost_cache =
      absl::StrCat(cache_dir(), "/", cost_cache_filename(source, target));
  LOG(INFO) << "Path cache: " << path_cache;
  LOG(INFO) << "Cost cache: " << cost_cache;
  CHECK(!fs::exists(path_cache));
  CHECK(!fs::exists(cost_cache));

  std::vector<std::string> path_column_names{"node", "index"};
  TsvSpec path_columns(path_column_names);
  TsvWriter path_writer(path_cache, &path_columns);

  std::vector<std::string> cost_column_names(graph_->base_cost_names());
  TsvSpec cost_columns(cost_column_names);
  TsvWriter cost_writer(cost_cache, &cost_columns);

  if (path.ok()) {
    TsvRow cost_row(&cost_columns);
    for (int i = 0; i < graph_->base_cost_names().size(); i++) {
      cost_row.Add(graph_->base_cost_names()[i], path->cost_vector[i]);
    }
    cost_writer.WriteRow(cost_row);

    for (ArcIndex arc : path->path) {
      TsvRow path_row(&path_columns);
      path_row.Add("node", arc.node);
      path_row.Add("index", arc.index);
      path_writer.WriteRow(path_row);
    }
  }
  // empty files expected if path not found
}

absl::StatusOr<WeightedCostSearch::Result> PathCache::ReadPathFromFiles(
    int source, int target) {
  std::string path_cache =
      absl::StrCat(cache_dir(), "/", path_cache_filename(source, target));
  std::string cost_cache =
      absl::StrCat(cache_dir(), "/", cost_cache_filename(source, target));
  CHECK_EQ(fs::exists(path_cache), fs::exists(cost_cache));

  if (fs::exists(path_cache)) {
    WeightedCostSearch::Result result;
    TsvReader cost_reader(cost_cache);

    if (cost_reader.AtEnd()) {
      return absl::NotFoundError(absl::GetFlag(FLAGS_not_connected_msg));
    } else {
      absl::flat_hash_map<std::string, std::string> cost_row =
          cost_reader.ReadRow();
      for (int i = 0; i < graph_->base_cost_names().size(); i++) {
        result.cost_vector.push_back(
            GetDoubleFeature(cost_row, graph_->base_cost_names()[i]));
      }
      CHECK(cost_reader.AtEnd());
      for (TsvReader path_reader(path_cache); !path_reader.AtEnd();) {
        absl::flat_hash_map<std::string, std::string> path_row =
            path_reader.ReadRow();
        result.path.push_back(
            ArcIndex{.node = GetIntFeature(path_row, "node"),
                     .index = GetIntFeature(path_row, "index")});
      }
    }
    return result;
  } else {
    return absl::NotFoundError(absl::GetFlag(FLAGS_file_not_found_msg));
  }
}

std::string PathCache::path_cache_filename(int source, int target) {
  return absl::StrCat("path_", source, "_", target, ".tsv");
}

std::string PathCache::cost_cache_filename(int source, int target) {
  return absl::StrCat("cost_", source, "_", target, ".tsv");
}

std::string PathCache::cache_dir() {
  return absl::StrCat(output_dir_, "/", absl::GetFlag(FLAGS_path_cache_prefix),
                      "__", graph_name_);
}

std::string PathCache::counts_filename(int source) {
  return absl::StrCat("counts_", source, ".tsv");
}

int PathCache::GetClusterCountWithinDistanceAboveThreshold(
    int source, double distance_threshold, double weight_threshold,
    const std::vector<int>& cluster_assignment) {
  CountData count_data = GetCounts(cluster_assignment, source);
  CHECK_EQ(count_data.count.size(), count_data.distance.size());
  CHECK_EQ(count_data.count.size(), count_data.weight.size());
  int count = 0;
  for (int i = 0; i < count_data.count.size(); i++) {
    if (count_data.distance[i] <= distance_threshold &&
        count_data.weight[i] >= weight_threshold) {
      count = std::max(count, count_data.count[i]);
    }
  }
  return count;
}

PathCache::CountData PathCache::GetCounts(
    const std::vector<int>& cluster_assignment, int source) {
  absl::StatusOr<CountData> result = ReadCountsFromFile(source);
  if (result.status().message() == absl::GetFlag(FLAGS_file_not_found_msg)) {
    LOG(INFO) << "Counts for " << source << " not found; generating...";
    CountData counts;
    double MULTIPLE = 1.1;
    double min_weight = std::numeric_limits<double>::max();
    double max_weight = 0;
    LOG(INFO) << "Computing min/max weights...";
    for (MultiCostGraph::Node node : graph_->nodes()) {
      for (MultiCostGraph::Arc arc : graph_->ArcsForNode(node.id)) {
        double weight =
            WeightedCost(graph_->DefaultCostWeights(), arc.cost_vector);
        if (weight > 0.0) {
          min_weight = std::min(min_weight, weight);
          max_weight = std::max(max_weight, weight);
        }
      }
    }
    double min_distance = min_weight;
    double max_distance = 3.0 * graph_->NumNodes() * max_weight;
    max_weight *= graph_->NumArcs();  // needed for cluster case, as clusters
                                      // can have many edges

    LOG(INFO) << "Computing all distances from " << source << "...";

    WeightedCostSearch search(*graph_, graph_->DefaultCostWeights(), source);

    LOG(INFO) << "Setting up count object...";

    int n_weight_idxs = 0;
    int n_distance_idxs = 0;
    for (double w = min_weight; w <= max_weight; w *= MULTIPLE) {
      n_weight_idxs++;
    }
    std::vector<double> precounts;
    for (double d = min_distance; d <= max_distance; d *= MULTIPLE) {
      n_distance_idxs++;
      for (double w = min_weight; w <= max_weight; w *= MULTIPLE) {
        counts.distance.push_back(d);
        counts.weight.push_back(w);
        counts.count.push_back(0);
        precounts.push_back(0);
      }
    }

    LOG(INFO) << "Inverting clustering...";

    absl::flat_hash_map<int, std::vector<ArcIndex>> clusters;
    for (int node = 0; node < graph_->NumNodes(); node++) {
      for (int arc_id = 0; arc_id < graph_->ArcsForNode(node).size();
           arc_id++) {
        ArcIndex arc_idx = ArcIndex{.node = node, .index = arc_id};
        clusters[cluster_assignment[graph_->ArcIndexNum(arc_idx)]].push_back(
            arc_idx);
      }
    }
    absl::flat_hash_map<int, double> cluster_costs =
        GetClusterCosts(*graph_, cluster_assignment);

    LOG(INFO) << "Collecting counts...";

    for (const auto& cluster : clusters) {
      std::vector<bool> cluster_has_edge;
      for (int i = 0; i < precounts.size(); i++) {
        cluster_has_edge.push_back(false);
      }

      for (ArcIndex arc_idx : cluster.second) {
        absl::StatusOr<std::vector<double>> cost = search.Cost(arc_idx.node);
        if (cost.ok()) {
          // std::cerr << "A";
          double distance = WeightedCost(graph_->DefaultCostWeights(), *cost);
          if (distance == 0.0) continue;
          // std::cerr << "B";
          // std::cerr << "distance = " << distance << "\n";
          int distance_idx = (int)std::ceil(std::log(distance / min_distance) /
                                            std::log(MULTIPLE));
          // std::cerr << "C";
          CHECK_GE(distance_idx, 0);
          // std::cerr << "D";
          CHECK_LE(distance_idx, n_distance_idxs - 1);
          // std::cerr << "E";
          double weight = cluster_costs.at(cluster.first);
          // std::cerr << "F";
          // hack for graphs that have zero edge weights
          if (weight == 0.0) weight = min_weight;
          int weight_idx = (int)std::floor(std::log(weight / min_weight) /
                                           std::log(MULTIPLE));
          // std::cerr << "G";
          CHECK_GE(weight_idx, 0);
          // std::cerr << "J";
          CHECK_LE(weight_idx, n_weight_idxs - 1);
          // std::cerr << "K";
          int real_idx = distance_idx * n_weight_idxs + weight_idx;
          // std::cerr << "H";
          // std::cerr << distance_idx << " * " << n_weight_idxs <<
          //     " + " << weight_idx << " = " << real_idx << "\n";
          CHECK_LE(distance, counts.distance[real_idx]);
          CHECK_GE(distance, counts.distance[real_idx] / MULTIPLE);
          CHECK_GE(weight, counts.weight[real_idx]);
          CHECK_LE(weight, counts.weight[real_idx] * MULTIPLE);
          // std::cerr << "I";
          cluster_has_edge[real_idx] = true;
        }
      }
      for (int i = 0; i < precounts.size(); i++) {
        precounts[i] += cluster_has_edge[i];
      }
    }

    LOG(INFO) << "Summing...";
    for (int dist0 = 0; dist0 < n_distance_idxs; dist0++) {
      for (int dist1 = 0; dist1 < n_distance_idxs; dist1++) {
        for (int weight0 = 0; weight0 < n_weight_idxs; weight0++) {
          for (int weight1 = 0; weight1 < n_weight_idxs; weight1++) {
            int idx0 = dist0 * n_weight_idxs + weight0;
            int idx1 = dist1 * n_weight_idxs + weight1;
            if (dist0 >= dist1 && weight0 <= weight1) {
              counts.count[idx0] += precounts[idx1];
            }
          }
        }
      }
    }

    for (int dist = 0; dist < n_distance_idxs - 1; dist++) {
      for (int weight = 1; weight < n_weight_idxs; weight++) {
        int idx = dist * n_weight_idxs + weight;
        int other_idx = (dist + 1) * n_weight_idxs + weight - 1;
        CHECK_LE(counts.count[idx], counts.count[other_idx]);
      }
    }

    LOG(INFO) << "Saving counts...";
    SaveCountsToFile(source, counts);
    return counts;
  } else {
    LOG(INFO) << "Loaded counts for " << source << " from cache.";
    return *result;
  }
}

void PathCache::SaveCountsToFile(int source, CountData counts) {
  std::string counts_cache =
      absl::StrCat(cache_dir(), "/", counts_filename(source));
  CHECK(!fs::exists(counts_cache));
  CHECK_EQ(counts.count.size(), counts.distance.size());
  CHECK_EQ(counts.count.size(), counts.weight.size());
  std::vector<std::string> counts_column_names{"distance", "weight", "count"};
  TsvSpec counts_columns(counts_column_names);
  TsvWriter counts_writer(counts_cache, &counts_columns);

  for (int i = 0; i < counts.count.size(); i++) {
    TsvRow counts_row(&counts_columns);

    counts_row.Add("distance", counts.distance[i]);
    counts_row.Add("weight", counts.weight[i]);
    counts_row.Add("count", counts.count[i]);
    counts_writer.WriteRow(counts_row);
  }
}

absl::StatusOr<PathCache::CountData> PathCache::ReadCountsFromFile(int source) {
  std::string counts_cache =
      absl::StrCat(cache_dir(), "/", counts_filename(source));
  if (fs::exists(counts_cache)) {
    CountData counts;
    for (TsvReader counts_reader(counts_cache); !counts_reader.AtEnd();) {
      absl::flat_hash_map<std::string, std::string> counts_row =
          counts_reader.ReadRow();
      counts.count.push_back(GetIntFeature(counts_row, "count"));
      counts.distance.push_back(GetDoubleFeature(counts_row, "distance"));
      counts.weight.push_back(GetDoubleFeature(counts_row, "weight"));
    }
    return counts;
  } else {
    return absl::NotFoundError(absl::GetFlag(FLAGS_file_not_found_msg));
  }
}

}  // namespace geo_algorithms
