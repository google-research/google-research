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

#include "residual_problem.h"

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_join.h"
#include "path_store.h"
#include "plotly_path_string.h"

namespace geo_algorithms {

// Residual Problem

void ResidualProblem::ClearVars() {
  flow_.f.clear();
  flow_.a.clear();
  flow_.b.clear();
  wh_W_ = 0;
  wh_kappa_.clear();
  wh_c_.clear();
  wh_kappa_residual_.clear();
  wh_c_residual_.clear();
  residual_.df.clear();
  residual_.df_residual.clear();
  residual_is_nonzero_ = false;
  cut_.d.clear();
  cut_.w.clear();
}

void ResidualProblem::ResetFlow(const FlowSolution& flow,
                                const std::string& cycle_fname) {
  ClearVars();
  flow_ = flow;

  // braintex step 1: check exclusivity of a and b
  for (const auto& [e, ae] : flow.a) {
    if (ae > 0) {
      if (flow.b.contains(e)) {
        CHECK_EQ(flow.b.at(e), 0);
      }
    }
  }
  for (const auto& [e, be] : flow.b) {
    if (be > 0) {
      if (flow.a.contains(e)) {
        CHECK_EQ(flow.a.at(e), 0);
      }
    }
  }

  // step 4: for loop for edges that also generates non-P residual edges
  for (const auto& [e, fe] : flow.f) {
    if (fe < problem_.T.at(e)) {
      wh_W_ -= problem_.L.at(e) * fe;
      UpdateKappaAndC(e, -problem_.L.at(e), problem_.T.at(e) - fe);
    } else {
      wh_W_ -= problem_.U.at(e) * (problem_.T.at(e) - fe) - problem_.L.at(e);
      UpdateKappaAndC(e, -problem_.U.at(e), 0);
    }

    ResidualIndex res_e = ResidualIndex{.orig = e};
    if (fe > problem_.T.at(e)) {
      UpdateKappaAndC(res_e, problem_.U.at(e), fe - problem_.T.at(e));
    } else if (fe > 0) {
      UpdateKappaAndC(res_e, problem_.L.at(e), fe);
    }
  }

  // step 4: for loop for P residual edges
  for (const ArcIndex& e : problem_.P) {
    ResidualIndex res_e = ResidualIndex{.orig = e};
    if (!flow.f.contains(e) || flow.f.at(e) <= 0) {
      UpdateKappaAndC(res_e, problem_.L.at(e), 0);
    }
  }

  Optimize(cycle_fname);
}

void ResidualProblem::UpdateKappaAndC(ArcIndex e, int64_t kappa, int64_t c) {
  absl::StatusOr<ArcIndex> rev_e = problem_.G.ReversedArcIfExists(e);
  if (rev_e.ok()) {
    ResidualIndex res_rev_e = ResidualIndex{.orig = *rev_e};
    CHECK(!wh_kappa_residual_.contains(res_rev_e) || !wh_kappa_.contains(e));
    CHECK(!wh_c_residual_.contains(res_rev_e) || !wh_c_.contains(e));

    if (wh_kappa_residual_.contains(res_rev_e)) {
      if (wh_kappa_residual_.at(res_rev_e) < kappa) {
        wh_kappa_residual_.erase(res_rev_e);
        wh_c_residual_.erase(res_rev_e);

        wh_kappa_[e] = kappa;
        wh_c_[e] = c;
      }
      return;
    }
  }

  // know that overlapping residual edge does not exist
  if (wh_kappa_.contains(e)) {
    if (wh_kappa_.at(e) < kappa) {
      wh_kappa_.erase(e);
      wh_c_.erase(e);

      wh_kappa_[e] = kappa;
      wh_c_[e] = c;
    }
  } else {
    wh_kappa_[e] = kappa;
    wh_c_[e] = c;
  }
}

void ResidualProblem::UpdateKappaAndC(ResidualIndex res_e, int64_t kappa,
                                      int64_t c) {
  ArcIndex e = res_e.orig;
  absl::StatusOr<ArcIndex> stat_rev_e = problem_.G.ReversedArcIfExists(e);
  if (stat_rev_e.ok()) {
    ArcIndex rev_e = *stat_rev_e;
    CHECK(!wh_kappa_residual_.contains(res_e) || !wh_kappa_.contains(rev_e));
    CHECK(!wh_c_residual_.contains(res_e) || !wh_c_.contains(rev_e));

    if (wh_kappa_.contains(rev_e)) {
      if (wh_kappa_.at(rev_e) < kappa) {
        wh_kappa_.erase(rev_e);
        wh_c_.erase(rev_e);

        wh_kappa_residual_[res_e] = kappa;
        wh_c_residual_[res_e] = c;
      }
      return;
    }
  }

  // know that overlapping non-residual edge does not exist
  if (wh_kappa_residual_.contains(res_e)) {
    if (wh_kappa_residual_.at(res_e) < kappa) {
      wh_kappa_residual_.erase(res_e);
      wh_c_residual_.erase(res_e);

      wh_kappa_residual_[res_e] = kappa;
      wh_c_residual_[res_e] = c;
    }
  } else {
    wh_kappa_residual_[res_e] = kappa;
    wh_c_residual_[res_e] = c;
  }
}

bool ResidualProblem::IsImprovable() { return residual_is_nonzero_; }

absl::StatusOr<FlowSolution> ResidualProblem::GetUpdatedSolution() {
  if (!residual_is_nonzero_) {
    return absl::NotFoundError("At optimality.");
  }
  FlowSolution new_flow = flow_;
  for (const auto& [e, dfe] : residual_.df) {
    if (dfe == std::numeric_limits<int64_t>::max()) {
      return absl::NotFoundError("Unbounded.");
    }
    new_flow.f[e] += dfe;
  }
  for (const auto& [res_e, dfres_e] : residual_.df_residual) {
    if (dfres_e == std::numeric_limits<int64_t>::max()) {
      return absl::NotFoundError("Unbounded.");
    }
    new_flow.f[res_e.orig] -= dfres_e;
  }

  for (const auto& [e, fe] : new_flow.f) {
    if (fe <= problem_.T.at(e)) {
      new_flow.a[e] = problem_.T.at(e) - fe;
      new_flow.b[e] = 0;
    } else {
      new_flow.a[e] = 0;
      new_flow.b[e] = fe - problem_.T.at(e);
    }
  }
  return new_flow;
}

absl::StatusOr<CutSolution> ResidualProblem::CutCertificate() {
  if (residual_is_nonzero_) {
    return absl::NotFoundError("Not at optimality yet.");
  }
  return cut_;
}

void ResidualProblem::Optimize(const std::string& cycle_fname) {
  absl::flat_hash_map<ArcIndex, int64_t> updated_forward_costs;
  absl::flat_hash_map<ResidualIndex, int64_t> residual_costs;
  for (const auto& [e, kappa] : wh_kappa_) {
    updated_forward_costs[e] = -kappa;
  }
  for (const auto& [res_e, kappa] : wh_kappa_residual_) {
    residual_costs[res_e] = -kappa;
  }
  Search::CostModel costs = Search::CostModel{
      .default_costs = problem_.L,
      .updated_forward_costs = updated_forward_costs,
      .residual_costs = residual_costs,
      .nonneg_weights = false,
  };

  std::vector<AOrRIndex> cycle;

  PathStore cycle_store(cycle_store_directory_);
  absl::StatusOr<std::vector<AOrRIndex>> cycle_stat =
      cycle_store.ReadPathOrFailIfAbsent(cycle_fname);

  SlowCutStore cut_store(problem_, cut_store_directory_);
  // reusing cycle fname to reduce boilerplate
  absl::StatusOr<CutSolution> cut_stat =
      cut_store.ReadCutOrFailIfAbsent(cycle_fname);

  CHECK(!cycle_stat.ok() || !cut_stat.ok());

  if (cycle_stat.ok()) {
    LOG(INFO) << "Loading cycle from " << cycle_fname << "...";
    cycle = *cycle_stat;
    residual_is_nonzero_ = true;
  } else if (cut_stat.ok()) {
    LOG(INFO) << "No cycle found at " << cycle_fname << ", loading cut instead"
              << "...";
    cut_ = *cut_stat;
    residual_is_nonzero_ = false;
  } else {
    LOG(INFO) << "No cycle found at " << cycle_fname << ", computing...";
    Search search(problem_.G, costs, problem_.path_src);
    residual_is_nonzero_ = search.NegativeCycleExists();
    if (residual_is_nonzero_) {
      cycle = *search.GetNegativeCycleIfExists();
      cycle_store.SavePathOrFailIfPresent(cycle_fname, cycle);
    } else {
      CHECK(cycle.empty());
      LOG(INFO) << "No cut found at " << cycle_fname << ", computing...";
      for (const auto& node : problem_.G.nodes()) {
        absl::StatusOr<int64_t> result = search.Cost(node.id);
        if (result.ok()) {
          cut_.d[node.id] = *result;
        } else {
          cut_.d[node.id] = std::numeric_limits<int64_t>::max();
        }
        cut_.d[node.id] =
            std::min(cut_.d.at(node.id), *search.Cost(problem_.path_dst));
      }
      for (const auto& node : problem_.G.nodes()) {
        const auto& arcs = problem_.G.ArcsForNode(node.id);
        for (int i = 0; i < arcs.size(); i++) {
          ArcIndex idx = ArcIndex{.node = node.id, .index = i};
          cut_.w[idx] = std::max(
              cut_.d.at(problem_.G.ArcIndexDst(idx)) - cut_.d.at(node.id),
              problem_.L.at(idx));
        }
      }
      cut_store.SaveCutOrFailIfPresent(cycle_fname, cut_);
    }
  }

  if (residual_is_nonzero_) {
    LOG(INFO) << "Length of cycle being augmented: " << cycle.size();
    int64_t min_cycle_capacity = std::numeric_limits<int64_t>::max();
    for (const AOrRIndex& e : cycle) {
      if (e.is_forward) {
        int64_t capacity = wh_c_.contains(e.forward) ? wh_c_.at(e.forward)
                                                     : problem_.T.at(e.forward);
        if (capacity > 0) {
          min_cycle_capacity = std::min(min_cycle_capacity, capacity);
        }
      } else {
        int64_t capacity = wh_c_residual_.at(e.residual);
        if (capacity > 0) {
          min_cycle_capacity = std::min(min_cycle_capacity, capacity);
        }
      }
    }
    for (const AOrRIndex& e : cycle) {
      if (e.is_forward) {
        residual_.df[e.forward] = min_cycle_capacity;
      } else {
        residual_.df_residual[e.residual] = min_cycle_capacity;
      }
    }
  }
}

// ----------------------------------------------

// Fast Residual Problem

void FastResidualProblem::ClearVars() {
  flow_.f.clear();
  flow_.a.clear();
  flow_.b.clear();
  soft_capacities_.clear();
  soft_capacities_residual_.clear();
  // no hard forward capacities
  hard_capacities_residual_.clear();
  weight_below_soft_.clear();
  weight_below_soft_residual_.clear();
  weight_above_soft_.clear();
  weight_above_soft_residual_.clear();
  residual_.df.clear();
  residual_.df_residual.clear();
  residual_is_nonzero_ = false;
  cut_.d.clear();
  cut_.w.clear();
}

void FastResidualProblem::ResetFlow(const FlowSolution& flow,
                                    const std::string& cycle_fname) {
  ClearVars();
  flow_ = flow;

  absl::flat_hash_set<ArcIndex> flow_and_P;
  for (const auto& [e, fe] : flow.f) {
    flow_and_P.insert(e);
  }
  for (const auto& e : problem_.P) {
    flow_and_P.insert(e);
  }

  // braintex step 1: check exclusivity of a and b
  for (const auto& [e, ae] : flow.a) {
    if (ae > 0) {
      if (flow.b.contains(e)) {
        CHECK_EQ(flow.b.at(e), 0);
      }
    }
  }
  for (const auto& [e, be] : flow.b) {
    if (be > 0) {
      if (flow.a.contains(e)) {
        CHECK_EQ(flow.a.at(e), 0);
      }
    }
  }

  // generate all capacities and weights from the flow, ignoring a and b
  // for simplicity
  for (const auto& e : flow_and_P) {
    int64_t fe = flow.f.contains(e) ? flow.f.at(e) : 0;
    // forward arc creation
    if (fe < problem_.T.at(e)) {
      soft_capacities_[e] = problem_.T.at(e) - fe;
      // infinite hard capacity
      weight_below_soft_[e] = problem_.L.at(e);
      weight_above_soft_[e] = problem_.U.at(e);
    } else {  // fe >= problem_.T.at(e)
      soft_capacities_[e] = std::numeric_limits<int64_t>::max();
      // infinite hard capacity
      weight_below_soft_[e] = problem_.U.at(e);
      // nothing above soft
    }

    ResidualIndex res_e = ResidualIndex{.orig = e};
    // residual arc creation
    if (fe > problem_.T.at(e)) {
      soft_capacities_residual_[res_e] = fe - problem_.T.at(e);
      if (problem_.P.contains(e)) {
        hard_capacities_residual_[res_e] = std::numeric_limits<int64_t>::max();
      } else {
        hard_capacities_residual_[res_e] = fe;
      }
      weight_below_soft_residual_[res_e] = -problem_.U.at(e);
      weight_above_soft_residual_[res_e] = -problem_.L.at(e);
    } else if ((fe > 0) || (problem_.P.contains(e))) {
      if (problem_.P.contains(e)) {
        soft_capacities_residual_[res_e] = std::numeric_limits<int64_t>::max();
        hard_capacities_residual_[res_e] = std::numeric_limits<int64_t>::max();
      } else {
        soft_capacities_residual_[res_e] = fe;
        hard_capacities_residual_[res_e] = fe;
      }
      weight_below_soft_residual_[res_e] = -problem_.L.at(e);
      // nothing above soft
    }  // no residual arc when fe = 0 and e not in P
  }

  Optimize(cycle_fname);
}

bool FastResidualProblem::IsImprovable() { return residual_is_nonzero_; }

absl::StatusOr<FlowSolution> FastResidualProblem::GetUpdatedSolution() {
  if (!residual_is_nonzero_) {
    return absl::NotFoundError("At optimality.");
  }
  FlowSolution new_flow = flow_;
  for (const auto& [e, dfe] : residual_.df) {
    if (dfe == std::numeric_limits<int64_t>::max()) {
      return absl::NotFoundError("Unbounded.");
    }
    new_flow.f[e] += dfe;
  }
  for (const auto& [res_e, dfres_e] : residual_.df_residual) {
    if (dfres_e == std::numeric_limits<int64_t>::max()) {
      return absl::NotFoundError("Unbounded.");
    }
    new_flow.f[res_e.orig] -= dfres_e;
  }

  for (const auto& [e, fe] : new_flow.f) {
    if (fe <= problem_.T.at(e)) {
      new_flow.a[e] = problem_.T.at(e) - fe;
      new_flow.b[e] = 0;
    } else {
      new_flow.a[e] = 0;
      new_flow.b[e] = fe - problem_.T.at(e);
    }
  }
  return new_flow;
}

absl::StatusOr<CutSolution> FastResidualProblem::CutCertificate() {
  if (residual_is_nonzero_) {
    return absl::NotFoundError("Not at optimality yet.");
  }
  return cut_;
}

absl::StatusOr<int64_t> FastResidualProblem::GetWeight(AOrRIndex e,
                                                       int64_t dfe) {
  int64_t soft_c = GetSoftCapacity(e);
  int64_t hard_c = GetHardCapacity(e);
  if (e.is_forward) {
    bool use_default = !flow_.f.contains(e.forward);
    CHECK_EQ(hard_c, std::numeric_limits<int64_t>::max());
    if (dfe < soft_c) {
      return use_default ? problem_.L.at(e.forward)
                         : weight_below_soft_.at(e.forward);
    } else {
      return use_default ? problem_.U.at(e.forward)
                         : weight_above_soft_.at(e.forward);
    }
  } else {
    if (dfe < soft_c) {
      return weight_below_soft_residual_.at(e.residual);
    } else if (dfe < hard_c) {
      return weight_above_soft_residual_.at(e.residual);
    } else {
      return absl::NotFoundError("Capacity exceeded.");
    }
  }
}

int64_t FastResidualProblem::GetSoftCapacity(AOrRIndex e) {
  if (e.is_forward) {
    bool use_default = !flow_.f.contains(e.forward);
    return use_default ? problem_.T.at(e.forward)
                       : soft_capacities_.at(e.forward);
  } else {
    return soft_capacities_residual_.at(e.residual);
  }
}

int64_t FastResidualProblem::GetHardCapacity(AOrRIndex e) {
  if (e.is_forward) {
    return std::numeric_limits<int64_t>::max();
  } else {
    return hard_capacities_residual_.at(e.residual);
  }
}

void FastResidualProblem::Optimize(const std::string& cycle_fname) {
  Search::CostModel costs = Search::CostModel{
      .default_costs = default_path_search_weights_,
      .updated_forward_costs = weight_below_soft_,
      .residual_costs = weight_below_soft_residual_,
      .nonneg_weights = false,
  };

  std::vector<AOrRIndex> cycle;

  PathStore cycle_store(cycle_store_directory_);
  absl::StatusOr<std::vector<AOrRIndex>> cycle_stat =
      cycle_store.ReadPathOrFailIfAbsent(cycle_fname);

  SlowCutStore cut_store(problem_, cut_store_directory_);
  // reusing cycle fname to reduce boilerplate
  absl::StatusOr<CutSolution> cut_stat =
      cut_store.ReadCutOrFailIfAbsent(cycle_fname);

  CHECK(!cycle_stat.ok() || !cut_stat.ok());

  if (cycle_stat.ok()) {
    LOG(INFO) << "Loading cycle from " << cycle_fname << "...";
    cycle = *cycle_stat;
    residual_is_nonzero_ = true;
  } else if (cut_stat.ok()) {
    LOG(INFO) << "No cycle found at " << cycle_fname << ", loading cut instead"
              << "...";
    cut_ = *cut_stat;
    residual_is_nonzero_ = false;
  } else {
    LOG(INFO) << "No cycle found at " << cycle_fname << ", computing...";
    Search search(problem_.G, costs, problem_.path_src);
    residual_is_nonzero_ = search.NegativeCycleExists();
    if (residual_is_nonzero_) {
      cycle = *search.GetNegativeCycleIfExists();
      std::vector<std::string> cycle_strings =
          GetPlotlyPathStrings(problem_.G, cycle, "Cycle", "#FF0000");
      for (const std::string& cycle_string : cycle_strings) {
        std::cout << cycle_string << "\n";
      }
      cycle_store.SavePathOrFailIfPresent(cycle_fname, cycle);
    } else {
      CHECK(cycle.empty());
      LOG(INFO) << "No cut found at " << cycle_fname << ", computing...";
      for (const auto& node : problem_.G.nodes()) {
        absl::StatusOr<int64_t> result = search.Cost(node.id);
        if (result.ok()) {
          cut_.d[node.id] = *result;
        } else {
          cut_.d[node.id] = std::numeric_limits<int64_t>::max();
        }
        cut_.d[node.id] =
            std::min(cut_.d.at(node.id), *search.Cost(problem_.path_dst));
      }
      for (const auto& node : problem_.G.nodes()) {
        const auto& arcs = problem_.G.ArcsForNode(node.id);
        for (int i = 0; i < arcs.size(); i++) {
          ArcIndex idx = ArcIndex{.node = node.id, .index = i};
          cut_.w[idx] = std::max(
              cut_.d.at(problem_.G.ArcIndexDst(idx)) - cut_.d.at(node.id),
              problem_.L.at(idx));
        }
      }
      cut_store.SaveCutOrFailIfPresent(cycle_fname, cut_);
    }
  }

  // key difference with slow version: augmenting cycle until positive
  // even when residual capacities are exceeded
  if (residual_is_nonzero_) {
    LOG(INFO) << "Length of cycle being augmented: " << cycle.size();

    int64_t min_hard_capacity = std::numeric_limits<int64_t>::max();
    std::string plotly_hard_arc;
    for (const AOrRIndex& e : cycle) {
      int64_t hard_c = GetHardCapacity(e);
      if (hard_c < min_hard_capacity) {
        min_hard_capacity = hard_c;
        plotly_hard_arc =
            GetPlotlyPathStrings(problem_.G, {e}, "Hard", "#008800")[0];
      }
    }
    std::cout << plotly_hard_arc << "\n";

    std::vector<std::pair<int64_t, int>> sorted_soft_c_index_pairs;
    for (int i = 0; i < cycle.size(); i++) {
      sorted_soft_c_index_pairs.push_back(
          std::make_pair(GetSoftCapacity(cycle[i]), i));
    }
    sorted_soft_c_index_pairs.push_back(std::make_pair(min_hard_capacity, -1));
    std::sort(sorted_soft_c_index_pairs.begin(),
              sorted_soft_c_index_pairs.end());

    int64_t df_so_far = 0;
    int64_t prev_df_so_far = 0;
    int64_t cycle_weight_so_far = 0;
    for (const AOrRIndex& e : cycle) {
      cycle_weight_so_far += *GetWeight(e, 0);
    }
    CHECK_LT(cycle_weight_so_far, 0);

    int num_times_optimized = 0;
    for (const auto& [soft_c, index] : sorted_soft_c_index_pairs) {
      if (soft_c > df_so_far) {
        num_times_optimized++;
        // LOG(INFO) << "Cycle weight so far: " << cycle_weight_so_far << " at
        // soft c: " << soft_c; about to start building a new cycle weight. take
        // stock of previous
        CHECK_LT(df_so_far, min_hard_capacity);
        if (cycle_weight_so_far >= 0) {
          // hit first positive cycle, so go up to that
          break;
        }
        if (soft_c >= min_hard_capacity) {
          df_so_far = min_hard_capacity;
          break;
        }
        prev_df_so_far = df_so_far;
        df_so_far = soft_c;
      }
      CHECK_GE(index, 0);
      AOrRIndex e = cycle[index];
      int64_t old_weight = *GetWeight(e, prev_df_so_far);
      int64_t new_weight = *GetWeight(e, df_so_far);
      cycle_weight_so_far += (new_weight - old_weight);
    }

    CHECK_LE(df_so_far, min_hard_capacity);
    CHECK_GT(df_so_far, prev_df_so_far);
    cycle_weight_so_far = 0;
    for (const AOrRIndex& e : cycle) {
      CHECK_EQ(*GetWeight(e, prev_df_so_far), *GetWeight(e, df_so_far - 1));
      cycle_weight_so_far += *GetWeight(e, prev_df_so_far);
    }
    CHECK_LT(cycle_weight_so_far, 0);

    LOG(INFO) << "N increases: " << num_times_optimized;
    LOG(INFO) << "DF: " << df_so_far;
    LOG(INFO) << "Hard capacity: " << min_hard_capacity;

    for (const AOrRIndex& e : cycle) {
      if (e.is_forward) {
        residual_.df[e.forward] = df_so_far;
      } else {
        residual_.df_residual[e.residual] = df_so_far;
      }
    }
  }
}

// ----------------------------------------------

// Other Methods

bool CorrectArcSet(const Formulation& problem,
                   const absl::flat_hash_map<ArcIndex, int64_t> v,
                   bool has_default_value) {
  for (const auto& [e, _] : v) {
    if (!problem.G.HasArc(e)) {
      return false;
    }
  }
  return has_default_value || (v.size() == problem.G.NumArcs());
}

bool CorrectNodeSet(const Formulation& problem,
                    const absl::flat_hash_map<int, int64_t> v) {
  for (const auto& [n, _] : v) {
    if ((n >= problem.G.NumNodes()) || (n < 0)) {
      return false;
    }
  }
  return (v.size() == problem.G.NumNodes());
}

void CheckFormulationWellDefined(const Formulation& problem) {
  // arc checks
  for (const ArcIndex& e : problem.P) {
    CHECK(problem.G.HasArc(e));
  }

  for (const auto& [e, _] : problem.L) {
    CHECK(problem.G.HasArc(e));
  }

  for (const auto& [e, _] : problem.U) {
    CHECK(problem.G.HasArc(e));
  }

  CHECK_EQ(problem.L.size(), problem.G.NumArcs());
  CHECK_EQ(problem.U.size(), problem.G.NumArcs());

  // L and U key sets are now equal to the edge set of G

  // L vs U checks
  for (const auto& [e, Le] : problem.L) {
    CHECK_GE(problem.U.at(e), Le);
  }
}

void CheckFeasible(const Formulation& problem, const CutSolution& cut) {
  CHECK(CorrectArcSet(problem, cut.w, false));
  CHECK(CorrectNodeSet(problem, cut.d));

  for (const auto& node : problem.G.nodes()) {
    for (int i = 0; i < problem.G.ArcsForNode(node.id).size(); i++) {
      int src = node.id;
      ArcIndex e = ArcIndex{.node = src, .index = i};
      int dst = problem.G.ArcIndexDst(e);

      CHECK_LE(cut.w.at(e), problem.U.at(e));
      CHECK_GE(cut.w.at(e), problem.L.at(e));
      int64_t diff = cut.d.at(dst) - cut.d.at(src);
      CHECK_LE(diff, cut.w.at(e));
      if (problem.P.contains(e)) {
        CHECK_EQ(diff, cut.w.at(e));
      }
    }
  }
}

void CheckFeasible(const Formulation& problem, const FlowSolution& flow) {
  CHECK(CorrectArcSet(problem, flow.f, true));
  CHECK(CorrectArcSet(problem, flow.a, true));
  CHECK(CorrectArcSet(problem, flow.b, true));

  absl::flat_hash_map<ArcIndex, int64_t> edge_var_sums;
  absl::flat_hash_map<int, int64_t> incoming_sums;
  absl::flat_hash_map<int, int64_t> outgoing_sums;
  for (const auto& [e, fe] : flow.f) {
    int src = e.node;
    int dst = problem.G.ArcIndexDst(e);
    edge_var_sums[e] += fe;
    incoming_sums[dst] += fe;
    outgoing_sums[src] += fe;
    if (!problem.P.contains(e)) {
      CHECK_GE(fe, 0);
    }
  }

  for (const auto& [e, ae] : flow.a) {
    edge_var_sums[e] += ae;
    CHECK_GE(ae, 0);
  }

  for (const auto& [e, be] : flow.b) {
    edge_var_sums[e] -= be;
    CHECK_GE(be, 0);
  }

  for (const auto& [e, sum_e] : edge_var_sums) {
    CHECK_EQ(sum_e, problem.T.at(e));
  }

  for (const auto& [v, sum] : incoming_sums) {
    if (outgoing_sums.contains(v)) {
      CHECK_EQ(sum, outgoing_sums.at(v));
    } else {
      CHECK_EQ(sum, 0);
    }
  }
  for (const auto& [v, sum] : outgoing_sums) {
    if (!incoming_sums.contains(v)) {
      CHECK_EQ(sum, 0);
    }
  }
}

int64_t FlowObjective(const Formulation& problem, const FlowSolution& flow) {
  int64_t objective = 0;
  for (const auto& [e, ae] : flow.a) {
    objective += problem.L.at(e) * (ae - problem.T.at(e));
  }
  for (const auto& [e, be] : flow.b) {
    objective -= problem.U.at(e) * be;
  }
  return objective;
}

int64_t CutObjective(const Formulation& problem, const CutSolution& cut) {
  int64_t objective = 0;
  for (const auto& [e, we] : cut.w) {
    objective += problem.T.at(e) * (we - problem.L.at(e));
  }
  return objective;
}

FlowSolution Initialize(const Formulation& problem) {
  return FlowSolution();  // all maps empty
  // as a coordinates all are T(e) intially
  // and f,b coordinates are all 0 initially
}

absl::StatusOr<std::pair<FlowSolution, CutSolution>> Solve(
    const Formulation& problem, const std::string& cycle_store_directory,
    const std::string& cut_store_directory,
    const std::string& cycle_fname_prefix) {
  CheckFormulationWellDefined(problem);

  FlowSolution flow = Initialize(problem);
  LOG(INFO) << "Iteration 0...";
  ResidualProblem residual(problem, cycle_store_directory, cut_store_directory);
  residual.ResetFlow(flow, absl::StrCat(cycle_fname_prefix, "__cycle_0.tsv"));

  int iter = 1;
  while (residual.IsImprovable()) {
    LOG(INFO) << "Iteration " << iter << "...";
    // can return unbounded status
    absl::StatusOr<FlowSolution> stat_flow = residual.GetUpdatedSolution();
    if (stat_flow.ok()) {
      flow = *stat_flow;
    } else {
      return stat_flow.status();
    }
    residual.ResetFlow(
        flow, absl::StrCat(cycle_fname_prefix, "__cycle_", iter, ".tsv"));
    iter++;
  }

  LOG(INFO) << "Computing cut solution...";
  CutSolution cut = *residual.CutCertificate();
  LOG(INFO) << "Checking feasibility and optimality...";
  CheckFeasible(problem, flow);
  CheckFeasible(problem, cut);
  CHECK_EQ(CutObjective(problem, cut), FlowObjective(problem, flow));
  LOG(INFO) << "Solution found.";

  return std::make_pair(flow, cut);
}

absl::StatusOr<std::pair<FlowSolution, CutSolution>> FastSolve(
    const Formulation& problem, const std::string& cycle_store_directory,
    const std::string& cut_store_directory,
    const std::string& cycle_fname_prefix) {
  CheckFormulationWellDefined(problem);

  LOG(INFO) << "Running fast solve.";

  FlowSolution flow = Initialize(problem);
  LOG(INFO) << "Iteration 0...";
  FastResidualProblem residual(problem, cycle_store_directory,
                               cut_store_directory);
  residual.ResetFlow(flow, absl::StrCat(cycle_fname_prefix, "__cycle_0.tsv"));

  int iter = 1;
  while (residual.IsImprovable()) {
    LOG(INFO) << "Objective: " << FlowObjective(problem, flow);
    LOG(INFO) << "Iteration " << iter << "...";
    // can return unbounded status
    absl::StatusOr<FlowSolution> stat_flow = residual.GetUpdatedSolution();
    if (stat_flow.ok()) {
      flow = *stat_flow;
    } else {
      return stat_flow.status();
    }
    residual.ResetFlow(
        flow, absl::StrCat(cycle_fname_prefix, "__cycle_", iter, ".tsv"));
    iter++;
  }

  LOG(INFO) << "Computing cut solution...";
  CutSolution cut = *residual.CutCertificate();
  LOG(INFO) << "Checking feasibility and optimality...";
  CheckFeasible(problem, flow);
  CheckFeasible(problem, cut);
  CHECK_EQ(CutObjective(problem, cut), FlowObjective(problem, flow));
  LOG(INFO) << "Solution found.";

  return std::make_pair(flow, cut);
}

}  // namespace geo_algorithms
