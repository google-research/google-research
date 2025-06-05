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

#include "path_store.h"

#include "neg_weight_graph_search.h"

namespace geo_algorithms {

using utils_tsv::TsvReader;
using utils_tsv::TsvRow;
using utils_tsv::TsvSpec;
using utils_tsv::TsvWriter;

void PathStore::SavePathOrFailIfPresent(const std::string& filename,
                                        const std::vector<AOrRIndex>& path) {
  std::string full_path = absl::StrCat(directory_, "/", filename);
  CHECK(!fs::exists(full_path));

  TsvSpec path_columns({"is_forward", "node", "index"});
  TsvWriter path_writer(full_path, &path_columns);

  for (const AOrRIndex& e : path) {
    TsvRow path_row(&path_columns);
    path_row.Add("is_forward", e.is_forward);
    if (e.is_forward) {
      path_row.Add("node", e.forward.node);
      path_row.Add("index", e.forward.index);
    } else {
      path_row.Add("node", e.residual.orig.node);
      path_row.Add("index", e.residual.orig.index);
    }
    path_writer.WriteRow(path_row);
  }
}

absl::StatusOr<std::vector<AOrRIndex>> PathStore::ReadPathOrFailIfAbsent(
    const std::string& filename) {
  std::string full_path = absl::StrCat(directory_, "/", filename);
  if (!fs::exists(full_path)) {
    return absl::NotFoundError(absl::StrCat("Path not found at ", full_path));
  }

  std::vector<AOrRIndex> result;
  for (TsvReader path_reader(full_path); !path_reader.AtEnd();) {
    absl::flat_hash_map<std::string, std::string> path_row =
        path_reader.ReadRow();
    int node, index;
    bool is_forward;
    CHECK(absl::SimpleAtoi(path_row.at("node"), &node));
    CHECK(absl::SimpleAtoi(path_row.at("index"), &index));
    CHECK(absl::SimpleAtob(path_row.at("is_forward"), &is_forward));
    if (is_forward) {
      result.push_back(
          AOrRIndex{.forward = ArcIndex{.node = node, .index = index},
                    .is_forward = true});
    } else {
      result.push_back(AOrRIndex{
          .residual =
              ResidualIndex{.orig = ArcIndex{.node = node, .index = index}},
          .is_forward = false});
    }
  }
  return result;
}

void CutStore::SaveCutOrFailIfPresent(const std::string& filename,
                                      const CutSolution& cut) {
  std::string full_path = absl::StrCat(directory_, "/", filename);
  CHECK(!fs::exists(full_path));
  int64_t max_distance = cut.d.at(problem_.path_dst);

  // only save w that are greater than the lower bound to save space
  TsvSpec cut_columns({"node", "d"});
  TsvWriter cut_writer(full_path, &cut_columns);

  absl::flat_hash_set<int> written_nodes;
  for (const auto& [v, dv] : cut.d) {
    if (dv <= max_distance) {
      TsvRow cut_row(&cut_columns);
      cut_row.Add("node", v);
      cut_row.Add("d", dv);
      cut_writer.WriteRow(cut_row);
      written_nodes.insert(v);
    }
  }

  CHECK(written_nodes.contains(problem_.path_src));
  CHECK(written_nodes.contains(problem_.path_dst));
}

absl::StatusOr<CutSolution> CutStore::ReadCutOrFailIfAbsent(
    const std::string& filename) {
  std::string full_path = absl::StrCat(directory_, "/", filename);
  if (!fs::exists(full_path)) {
    return absl::NotFoundError(absl::StrCat("Cut not found at ", full_path));
  }

  CutSolution result;
  for (TsvReader cut_reader(full_path); !cut_reader.AtEnd();) {
    absl::flat_hash_map<std::string, std::string> cut_row =
        cut_reader.ReadRow();
    int node;
    int64_t d;
    CHECK(absl::SimpleAtoi(cut_row.at("node"), &node));
    CHECK(absl::SimpleAtoi(cut_row.at("d"), &d));
    result.d[node] = d;
  }
  CHECK_EQ(result.d.at(problem_.path_src), 0);
  CHECK(result.d.contains(problem_.path_dst));
  for (const auto& node : problem_.G.nodes()) {
    if (!result.d.contains(node.id)) {
      result.d[node.id] = result.d.at(problem_.path_dst);
    }
  }
  for (const auto& node : problem_.G.nodes()) {
    for (int i = 0; i < problem_.G.ArcsForNode(node.id).size(); i++) {
      ArcIndex e = ArcIndex{.node = node.id, .index = i};
      CHECK(result.d.contains(e.node));
      CHECK(result.d.contains(problem_.G.ArcIndexDst(e)));
      int64_t we = result.d.at(problem_.G.ArcIndexDst(e)) - result.d.at(e.node);
      result.w[e] = std::max(we, problem_.L.at(e));
    }
  }
  return result;
}

void SlowCutStore::SaveCutOrFailIfPresent(const std::string& filename,
                                          const CutSolution& cut) {
  std::string full_path = absl::StrCat(directory_, "/", filename);
  CHECK(!fs::exists(full_path));

  // only save w that are strictly greater than the lower bound to save space
  TsvSpec cut_columns({"node", "index", "w"});
  TsvWriter cut_writer(full_path, &cut_columns);

  for (const auto& [e, w] : cut.w) {
    if (w > problem_.L.at(e)) {
      TsvRow cut_row(&cut_columns);
      cut_row.Add("node", e.node);
      cut_row.Add("index", e.index);
      cut_row.Add("w", w);
      cut_writer.WriteRow(cut_row);
    }
  }
}

absl::StatusOr<CutSolution> SlowCutStore::ReadCutOrFailIfAbsent(
    const std::string& filename) {
  std::string full_path = absl::StrCat(directory_, "/", filename);
  if (!fs::exists(full_path)) {
    return absl::NotFoundError(absl::StrCat("Cut not found at ", full_path));
  }

  CutSolution result;
  for (const auto& [e, le] : problem_.L) {
    result.w[e] = le;
  }
  absl::flat_hash_map<ArcIndex, int64_t> updated_arcs;
  for (TsvReader cut_reader(full_path); !cut_reader.AtEnd();) {
    absl::flat_hash_map<std::string, std::string> cut_row =
        cut_reader.ReadRow();
    int node, index;
    int64_t w;
    CHECK(absl::SimpleAtoi(cut_row.at("node"), &node));
    CHECK(absl::SimpleAtoi(cut_row.at("index"), &index));
    CHECK(absl::SimpleAtoi(cut_row.at("w"), &w));
    ArcIndex e = ArcIndex{.node = node, .index = index};
    CHECK_GE(w, problem_.L.at(e));
    result.w[e] = w;
    updated_arcs[e] = w;
  }

  Search::CostModel costs = Search::CostModel{
      .default_costs = problem_.L,
      .updated_forward_costs = updated_arcs,
      .nonneg_weights = true,
  };

  Search search(problem_.G, costs, problem_.path_src);
  for (const auto& node : problem_.G.nodes()) {
    absl::StatusOr<int64_t> cost = search.Cost(node.id);
    if (cost.ok()) {
      result.d[node.id] = *cost;
    } else {
      result.d[node.id] = std::numeric_limits<int64_t>::max();
    }
  }
  return result;
}

}  // namespace geo_algorithms
