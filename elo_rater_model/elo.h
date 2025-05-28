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

// Copyright 2023 The Google Research Authors.
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

#ifndef ELO_H_
#define ELO_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/strings/string_view.h"
#include "elo_fast_math.h"

struct ELOSettings {
  float alpha = 1e-5;
  float rater_prior_strength = 100.0;
  static constexpr float kInitialElo = 2000;

  // Probability that a wins against b.
  static float WinProbability(float elo_a, float elo_b) {
    return ::WinProbability(elo_a, elo_b);
  }

  // log_e of the probability that a wins against b.
  static float LogWinProbability(float elo_a, float elo_b) {
    return ::LogWinProbability(elo_a, elo_b);
  }
};

// Exposed for testing.
double ComputeEloLogLikelihood(const ELOSettings& settings,
                               const Eigen::VectorXd& elos,
                               const Eigen::VectorXd& rater_reliability,
                               const std::vector<Eigen::MatrixXd>& win_count,
                               const Eigen::VectorXd& rater_correct_goldens,
                               const Eigen::VectorXd& rater_incorrect_goldens);

Eigen::VectorXd ComputeEloGradientWrtElo(
    const ELOSettings& settings, const Eigen::VectorXd& elos,
    const Eigen::VectorXd& rater_reliability,
    const std::vector<Eigen::MatrixXd>& win_count);

Eigen::VectorXd ComputeEloGradientWrtRater(
    const ELOSettings& settings, const Eigen::VectorXd& elos,
    const Eigen::VectorXd& rater_reliability,
    const std::vector<Eigen::MatrixXd>& win_count,
    const Eigen::VectorXd& rater_correct_goldens,
    const Eigen::VectorXd& rater_incorrect_goldens);

Eigen::VectorXd ComputeEloHessianDiagonalWrtRater(
    const ELOSettings& settings, const Eigen::VectorXd& elos,
    const Eigen::VectorXd& rater_reliability,
    const std::vector<Eigen::MatrixXd>& win_count,
    const Eigen::VectorXd& rater_correct_goldens,
    const Eigen::VectorXd& rater_incorrect_goldens);

Eigen::MatrixXd ComputeEloHessianWrtElo(
    const ELOSettings& settings, const Eigen::VectorXd& elos,
    const Eigen::VectorXd& rater_reliability,
    const std::vector<Eigen::MatrixXd>& win_count);

struct ELOScore {
  float elo;
  float p99_low;
  float p99_hi;
};

struct Suggestion {
  float weight;
  std::string method_a;
  std::string method_b;
};

class ELOData {
 public:
  ELOData(ELOSettings settings, size_t num_question_kinds,
          const std::vector<std::string>& methods);

  void AddQuestion(const std::optional<std::vector<bool>>&
                       golden_question_correct_answers_are_a,
                   absl::string_view method_a, absl::string_view method_b,
                   const std::vector<std::string>& choices, size_t rater_index);

  size_t GetNumQuestions() const;
  const std::vector<std::string>& GetMethods() const;

  // Propose questions to be asked to raters.
  const std::vector<Suggestion>& SuggestQuestions();

  // Computes the ELO scores that best explain the given set of answers (i.e.
  // give the highest probability for the answers under the ELO/BTL model), for
  // each question in the given set of answers.
  // Also computes (approximate) 99-percentile credible intervals.
  const std::vector<std::vector<ELOScore>>& ComputeELOScores();

  // Computes rater reliability for a given question kind.
  Eigen::VectorXd RaterRandomProbability(size_t question_kind);

  // Computes the "divergence" of the current distribution over ELO scores
  // versus the given vector of assumed-correct ELO scores. In particular,
  // averages the KL divergence of true vs assumed beliefs of match outcomes.
  // Useful for simulations.
  float ComputeEloDivergence(const std::vector<float>& ground_truth_elos,
                             size_t question_index);

  ~ELOData();

  ELOData(ELOData&&);
  ELOData& operator=(ELOData&&);

 private:
  struct Data;
  std::unique_ptr<Data> data_;
  ELOSettings settings_;
};

#endif  // ELO_H_
