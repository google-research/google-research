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

#ifndef RESIDUAL_PROBLEM_H_
#define RESIDUAL_PROBLEM_H_

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "formulation.h"
#include "graph.h"
#include "neg_weight_graph_search.h"

namespace geo_algorithms {

class ResidualProblem {
 public:
  ResidualProblem(const Formulation& problem,
                  const std::string& cycle_store_directory,
                  const std::string& cut_store_directory)
      : problem_(problem),
        cycle_store_directory_(cycle_store_directory),
        cut_store_directory_(cut_store_directory) {}

  void ResetFlow(const FlowSolution& flow, const std::string& cycle_fname);

  bool IsImprovable();

  absl::StatusOr<FlowSolution> GetUpdatedSolution();

  absl::StatusOr<CutSolution> CutCertificate();

 private:
  const Formulation& problem_;
  const std::string cycle_store_directory_;
  const std::string cut_store_directory_;
  FlowSolution flow_;

  int64_t wh_W_;
  // key set contained in F. for f_e = 0, default = T(e)
  absl::flat_hash_map<ArcIndex, int64_t> wh_c_;
  // key set equal to wh_F_res_ + wh_F_P_. no default
  absl::flat_hash_map<ResidualIndex, int64_t> wh_c_residual_;
  // for f_e = 0, default = -L(e)
  absl::flat_hash_map<ArcIndex, int64_t> wh_kappa_;
  absl::flat_hash_map<ResidualIndex, int64_t> wh_kappa_residual_;

  ResidualSolution residual_;
  bool residual_is_nonzero_;
  CutSolution cut_;

  void ClearVars();

  void UpdateKappaAndC(ArcIndex e, int64_t kappa, int64_t c);

  void UpdateKappaAndC(ResidualIndex res_e, int64_t kappa, int64_t c);

  void Optimize(const std::string& cycle_fname);
};

class FastResidualProblem {
 public:
  FastResidualProblem(const Formulation& problem,
                      const std::string& cycle_store_directory,
                      const std::string& cut_store_directory)
      : problem_(problem),
        cycle_store_directory_(cycle_store_directory),
        cut_store_directory_(cut_store_directory) {
    for (const auto& [e, _] : problem_.L) {
      if (problem_.P.contains(e)) {
        default_path_search_weights_[e] = problem_.L.at(e);
      } else if (problem_.T.at(e) == 0) {
        default_path_search_weights_[e] = problem_.U.at(e);
      } else {
        default_path_search_weights_[e] = problem_.L.at(e);
      }
    }
  }

  void ResetFlow(const FlowSolution& flow, const std::string& cycle_fname);

  bool IsImprovable();

  absl::StatusOr<FlowSolution> GetUpdatedSolution();

  absl::StatusOr<CutSolution> CutCertificate();

 private:
  const Formulation& problem_;
  const std::string cycle_store_directory_;
  const std::string cut_store_directory_;
  absl::flat_hash_map<ArcIndex, int64_t> default_path_search_weights_;
  FlowSolution flow_;

  // if forward key not present, default = T(e). if backward key not present,
  // no default.
  absl::flat_hash_map<ArcIndex, int64_t> soft_capacities_;
  absl::flat_hash_map<ResidualIndex, int64_t> soft_capacities_residual_;
  // guaranteed to be greater than soft capacities.
  // any key not present defaults to infinity
  // forward arcs have infinite hard capacity
  absl::flat_hash_map<ResidualIndex, int64_t> hard_capacities_residual_;
  // if forward key not present, default = L(e). if backward key not present,
  // no default
  absl::flat_hash_map<ArcIndex, int64_t> weight_below_soft_;
  absl::flat_hash_map<ResidualIndex, int64_t> weight_below_soft_residual_;
  // if forward key not present, default = U(e). if backward key not present,
  // no default
  absl::flat_hash_map<ArcIndex, int64_t> weight_above_soft_;
  absl::flat_hash_map<ResidualIndex, int64_t> weight_above_soft_residual_;

  ResidualSolution residual_;
  bool residual_is_nonzero_;
  CutSolution cut_;

  void ClearVars();

  // returns error if flow is not feasible
  absl::StatusOr<int64_t> GetWeight(AOrRIndex e, int64_t dfe);

  int64_t GetSoftCapacity(AOrRIndex e);

  int64_t GetHardCapacity(AOrRIndex e);

  void Optimize(const std::string& cycle_fname);
};

int64_t FlowObjective(const Formulation& problem, const FlowSolution& flow);

int64_t CutObjective(const Formulation& problem, const CutSolution& cut);

void CheckFormulationWellDefined(const Formulation& problem);

bool CorrectArcSet(const Formulation& problem,
                   const absl::flat_hash_map<ArcIndex, int64_t> v,
                   bool has_default_value);

bool CorrectNodeSet(const Formulation& problem,
                    const absl::flat_hash_map<int, int64_t> v);

void CheckFeasibility(const Formulation& problem, const CutSolution& cut);

void CheckFeasibility(const Formulation& problem, const FlowSolution& flow);

FlowSolution Initialize(const Formulation& problem);

// can only return unbounded status
absl::StatusOr<std::pair<FlowSolution, CutSolution>> Solve(
    const Formulation& problem, const std::string& cycle_store_directory,
    const std::string& cut_store_directory,
    const std::string& cycle_fname_prefix);

// can only return unbounded status
absl::StatusOr<std::pair<FlowSolution, CutSolution>> FastSolve(
    const Formulation& problem, const std::string& cycle_store_directory,
    const std::string& cut_store_directory,
    const std::string& cycle_fname_prefix);

}  // namespace geo_algorithms

#endif  // RESIDUAL_PROBLEM_H_
