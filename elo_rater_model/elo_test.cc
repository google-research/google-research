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

#include "elo.h"

#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "elo_fast_math.h"

namespace {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Field;
using ::testing::FloatNear;

void AddAns(ELOData& data, absl::string_view a, absl::string_view b) {
  data.AddQuestion(std::nullopt, a, b, {std::string(a), std::string(b)}, 0);
}

void AddGoldenAns(ELOData& data, bool correct) {
  std::string chosen = correct ? "x" : "y";
  data.AddQuestion({{true, true}}, "x", "y", {chosen, chosen}, 0);
}

std::vector<std::string> Methods() { return {"a", "b", "c", "d"}; }

TEST(ELO, BasicELOTie) {
  ELOData data(ELOSettings(), 1, Methods());
  AddAns(data, "a", "b");
  AddAns(data, "b", "c");
  AddAns(data, "c", "a");
  auto elos = data.ComputeELOScores();

  EXPECT_THAT(elos, ElementsAre(ElementsAre(
                        Field(&ELOScore::elo, FloatNear(2000, 1)),
                        Field(&ELOScore::elo, FloatNear(2000, 1)),
                        Field(&ELOScore::elo, FloatNear(2000, 1)),
                        Field(&ELOScore::elo, FloatNear(2000, 1)))));
}

TEST(ELO, BasicELOWinner) {
  ELOData data(ELOSettings(), 2, Methods());
  AddAns(data, "a", "b");
  AddAns(data, "b", "c");
  AddAns(data, "a", "c");

  // Ensure that we consider the rater to be fully reliable.
  for (size_t i = 0; i < 10000; i++) {
    AddGoldenAns(data, true);
  }

  auto elos = data.ComputeELOScores();

  EXPECT_THAT(
      elos,
      ElementsAre(ElementsAre(Field(&ELOScore::elo, FloatNear(2334, 1)),
                              Field(&ELOScore::elo, FloatNear(2000, 1)),
                              Field(&ELOScore::elo, FloatNear(1665, 1)),
                              Field(&ELOScore::elo, FloatNear(2000, 1))),
                  ElementsAre(Field(&ELOScore::elo, FloatNear(1665, 1)),
                              Field(&ELOScore::elo, FloatNear(2000, 1)),
                              Field(&ELOScore::elo, FloatNear(2334, 1)),
                              Field(&ELOScore::elo, FloatNear(2000, 1)))));
}

TEST(ELO, ELOErrorBar) {
  ELOData data(ELOSettings(), 2, Methods());
  for (size_t i = 0; i < 4; i++) {
    AddAns(data, "a", "b");
    AddAns(data, "b", "c");
    AddAns(data, "a", "c");
  }
  AddAns(data, "b", "a");
  AddAns(data, "c", "b");
  AddAns(data, "c", "a");

  // Ensure that we consider the rater to be fully reliable.
  for (size_t i = 0; i < 10000; i++) {
    AddGoldenAns(data, true);
  }

  auto elos = data.ComputeELOScores();

  EXPECT_THAT(
      elos,
      ElementsAre(
          ElementsAre(AllOf(Field(&ELOScore::elo, FloatNear(2162, 1)),
                            Field(&ELOScore::p99_hi, FloatNear(2689, 1)),
                            Field(&ELOScore::p99_low, FloatNear(1637, 1))),
                      AllOf(Field(&ELOScore::elo, FloatNear(2000, 1)),
                            Field(&ELOScore::p99_hi, FloatNear(2513, 1)),
                            Field(&ELOScore::p99_low, FloatNear(1486, 1))),
                      AllOf(Field(&ELOScore::elo, FloatNear(1837, 1)),
                            Field(&ELOScore::p99_hi, FloatNear(2363, 1)),
                            Field(&ELOScore::p99_low, FloatNear(1311, 1))),
                      AllOf(Field(&ELOScore::elo, FloatNear(2000, 1)),
                            Field(&ELOScore::p99_hi, FloatNear(3410, 1)),
                            Field(&ELOScore::p99_low, FloatNear(590, 1)))),
          ElementsAre(AllOf(Field(&ELOScore::elo, FloatNear(1837, 1)),
                            Field(&ELOScore::p99_hi, FloatNear(2363, 1)),
                            Field(&ELOScore::p99_low, FloatNear(1311, 1))),
                      AllOf(Field(&ELOScore::elo, FloatNear(2000, 1)),
                            Field(&ELOScore::p99_hi, FloatNear(2513, 1)),
                            Field(&ELOScore::p99_low, FloatNear(1486, 1))),
                      AllOf(Field(&ELOScore::elo, FloatNear(2162, 1)),
                            Field(&ELOScore::p99_hi, FloatNear(2689, 1)),
                            Field(&ELOScore::p99_low, FloatNear(1637, 1))),
                      AllOf(Field(&ELOScore::elo, FloatNear(2000, 1)),
                            Field(&ELOScore::p99_hi, FloatNear(3410, 1)),
                            Field(&ELOScore::p99_low, FloatNear(590, 1)))))

  );
}

TEST(ELO, ELOTable) {
  ELOData data(ELOSettings(), 2, Methods());
  // According to ELO probability tables, a rating difference of 250 corresponds
  // to about 80.83% win probability. Test that we can recover this ELO
  // difference.
  for (size_t i = 0; i < 8083; i++) {
    AddAns(data, "a", "b");
  }
  for (size_t i = 0; i < 1917; i++) {
    AddAns(data, "b", "a");
  }
  // Ensure that we consider the rater to be fully reliable.
  for (size_t i = 0; i < 10000; i++) {
    AddGoldenAns(data, true);
  }
  auto elos = data.ComputeELOScores();
  EXPECT_THAT(elos[0][0].elo - elos[0][1].elo, FloatNear(250, 1));
}

TEST(ELO, ELOSuggestions) {
  ELOData data(ELOSettings(), 2, Methods());
  for (size_t i = 0; i < 1000; i++) {
    AddAns(data, "a", "b");
    AddAns(data, "b", "a");
  }
  // c and d are identical, but d has 10x less data. Thus, all questions that
  // involve d should have higher weight.
  for (size_t i = 0; i < 1000; i++) {
    AddAns(data, "c", "b");
  }
  for (size_t i = 0; i < 100; i++) {
    AddAns(data, "d", "b");
    AddAns(data, "b", "c");
  }
  for (size_t i = 0; i < 10; i++) {
    AddAns(data, "b", "d");
  }
  auto suggestions = data.SuggestQuestions();
  std::sort(suggestions.begin(), suggestions.end(),
            [](Suggestion& a, Suggestion& b) { return a.weight > b.weight; });

  for (size_t i = 0; i < 3; i++) {
    EXPECT_THAT(std::max(suggestions[i].method_a, suggestions[i].method_b),
                Eq("d"));
  }
}

TEST(ELO, ELOSuggestionsNoAnswersForMethod) {
  ELOData data(ELOSettings(), 2, Methods());
  for (size_t i = 0; i < 1000; i++) {
    AddAns(data, "a", "b");
    AddAns(data, "b", "a");
  }
  // There are no answers for d, but a lot for other pairs. Thus, all questions
  // that involve d should have higher weight.
  for (size_t i = 0; i < 1000; i++) {
    AddAns(data, "c", "b");
  }
  for (size_t i = 0; i < 100; i++) {
    AddAns(data, "b", "c");
  }
  auto suggestions = data.SuggestQuestions();

  std::sort(suggestions.begin(), suggestions.end(),
            [](Suggestion& a, Suggestion& b) { return a.weight > b.weight; });

  for (size_t i = 0; i < 3; i++) {
    EXPECT_THAT(std::max(suggestions[i].method_a, suggestions[i].method_b),
                Eq("d"));
  }
}

TEST(ELO, ELOSuggestionsNoAnswers) {
  ELOData data(ELOSettings(), 2, Methods());
  auto suggestions = data.SuggestQuestions();
  EXPECT_EQ(suggestions.size(), 6);
}

TEST(ELO, CheckGradients) {
  ELOSettings settings;
  settings.alpha = 1e-3;
  settings.rater_prior_strength = 1e-2;
  Eigen::VectorXd elos(3);
  elos << 100.0, 2.0, -100.0;
  std::vector<Eigen::MatrixXd> win_count = {
      Eigen::MatrixXd{{0.0, 10.0, 20.0}, {20.0, 0.0, 10.0}, {30.0, 0.0, 0.0}},
      Eigen::MatrixXd{{0.0, 20.0, 10.0}, {10.0, 0.0, 20.0}, {0.0, 30.0, 0.0}},
      Eigen::MatrixXd{{0.0, 15.0, 15.0}, {15.0, 0.0, 15.0}, {30.0, 30.0, 0.0}},
  };
  Eigen::VectorXd rater_correct_goldens(3);
  rater_correct_goldens << 1.0, 2.0, 3.0;
  Eigen::VectorXd rater_incorrect_goldens(3);
  rater_incorrect_goldens << 1.0, 4.0, 0.0;
  Eigen::VectorXd rater_reliability(3);
  rater_reliability << -1.0, 0.0, 1.0;

  {
    Eigen::VectorXd elos_shifted = elos.array() + ELOSettings::kInitialElo;
    LOG(INFO) << "log likelihood "
              << ComputeEloLogLikelihood(
                     settings, elos_shifted, rater_reliability, win_count,
                     rater_correct_goldens, rater_incorrect_goldens);
  }

  {
    constexpr double kEps = 1e-4;
    constexpr double kErrorLimit = 1e-8;
    double max = 0.0;
    auto gradient =
        ComputeEloGradientWrtElo(settings, elos, rater_reliability, win_count);
    LOG(INFO) << "gradient elo " << gradient;
    for (size_t i = 0; i < elos.size(); i++) {
      Eigen::VectorXd eloseps = elos.array() + ELOSettings::kInitialElo;
      eloseps(i) += kEps;
      double up = ComputeEloLogLikelihood(settings, eloseps, rater_reliability,
                                          win_count, rater_correct_goldens,
                                          rater_incorrect_goldens);
      eloseps = elos.array() + ELOSettings::kInitialElo;
      eloseps(i) -= kEps;
      double down = ComputeEloLogLikelihood(
          settings, eloseps, rater_reliability, win_count,
          rater_correct_goldens, rater_incorrect_goldens);
      // ComputeEloGradientWrtElo includes a - sign.
      double estgrad = (down - up) / (2 * kEps);
      double diff = std::abs(estgrad - gradient(i));
      double relerr = diff / std::max(std::abs(estgrad), std::abs(gradient(i)));
      max = std::max(max, diff);
      EXPECT_LT(diff, kErrorLimit)
          << "ELO gradient " << i << " rel error " << relerr;
    }

    fprintf(stderr, "max error elo1: %e\n", max);
  }

  {
    constexpr double kEps = 1e-4;
    constexpr double kErrorLimit = 1e-9;
    double max = 0.0;
    auto hessian =
        ComputeEloHessianWrtElo(settings, elos, rater_reliability, win_count);
    LOG(INFO) << "hessian ELO " << hessian;
    for (size_t a = 0; a < elos.size(); a++) {
      for (size_t b = 0; b < elos.size(); b++) {
        Eigen::VectorXd eloseps = elos;
        eloseps(a) += kEps;
        double up = ComputeEloGradientWrtElo(settings, eloseps,
                                             rater_reliability, win_count)(b);
        eloseps = elos;
        eloseps(a) -= kEps;
        double down = ComputeEloGradientWrtElo(settings, eloseps,
                                               rater_reliability, win_count)(b);
        double esthess = (up - down) / (2 * kEps);
        double diff = std::abs(esthess - hessian(a, b));
        max = std::max(max, diff);
        EXPECT_LT(diff, kErrorLimit) << "hessian " << a << " " << b;
      }
    }
    fprintf(stderr, "max error elo2: %e\n", max);
  }

  {
    constexpr double kEps = 1e-4;
    constexpr double kErrorLimit = 5e-6;
    double max = 0.0;
    auto gradient = ComputeEloGradientWrtRater(
        settings, elos, rater_reliability, win_count, rater_correct_goldens,
        rater_incorrect_goldens);
    LOG(INFO) << "gradient rater " << gradient;
    for (size_t i = 0; i < rater_reliability.size(); i++) {
      Eigen::VectorXd ratereps = rater_reliability;
      ratereps(i) += kEps;
      double up = ComputeEloLogLikelihood(settings, elos, ratereps, win_count,
                                          rater_correct_goldens,
                                          rater_incorrect_goldens);
      ratereps = rater_reliability;
      ratereps(i) -= kEps;
      double down = ComputeEloLogLikelihood(settings, elos, ratereps, win_count,
                                            rater_correct_goldens,
                                            rater_incorrect_goldens);
      // ComputeEloGradientWrtRater includes a - sign.
      double estgrad = (down - up) / (2 * kEps);
      double diff = std::abs(estgrad - gradient(i));
      double relerr = diff / std::max(std::abs(estgrad), std::abs(gradient(i)));
      max = std::max(max, diff);
      EXPECT_LT(diff, kErrorLimit)
          << "Rater gradient " << i << " rel error " << relerr;
    }
    fprintf(stderr, "max error rater1: %e\n", max);
  }

  {
    // Finite differences seem to be a fairly bad approximation for rater
    // second derivative.
    constexpr double kEps = 1e-3;
    constexpr double kErrorLimit = 2e-4;
    double max = 0.0;
    auto hessian_diagonal_rater = ComputeEloHessianDiagonalWrtRater(
        settings, elos, rater_reliability, win_count, rater_correct_goldens,
        rater_incorrect_goldens);
    LOG(INFO) << "hessian rater " << hessian_diagonal_rater;
    for (size_t i = 0; i < rater_reliability.size(); i++) {
      Eigen::VectorXd ratereps = rater_reliability;
      ratereps(i) += kEps;
      double up = ComputeEloGradientWrtRater(settings, elos, ratereps,
                                             win_count, rater_correct_goldens,
                                             rater_incorrect_goldens)[i];
      ratereps = rater_reliability;
      ratereps(i) -= kEps;
      double down = ComputeEloGradientWrtRater(settings, elos, ratereps,
                                               win_count, rater_correct_goldens,
                                               rater_incorrect_goldens)[i];
      double estgrad = (up - down) / (2 * kEps);
      double diff = std::abs(estgrad - hessian_diagonal_rater(i));
      double relerr = diff / std::max(std::abs(estgrad),
                                      std::abs(hessian_diagonal_rater(i)));
      max = std::max(max, diff);
      EXPECT_LT(diff, kErrorLimit)
          << "Rater second derivative " << i << " rel error " << relerr
          << " abs " << std::abs(estgrad);
    }
    fprintf(stderr, "max error rater2: %e\n", max);
  }
}

TEST(ELOFastMath, ELOComputation) {
  absl::BitGen gen;

  constexpr double kS = kLog10 / kEloN;

  for (size_t _ = 0; _ < 1000000; _++) {
    float elo_a = absl::Uniform(gen, 0.0, 4000.0);
    float elo_b = absl::Uniform(gen, 0.0, 4000.0);

    float probability =
        std::exp(elo_a * kS) / (std::exp(elo_a * kS) + std::exp(elo_b * kS));

    EXPECT_NEAR(probability, WinProbability(elo_a, elo_b), 1e-5)
        << "Probability " << elo_a << " " << elo_b;
    EXPECT_NEAR(probability, std::exp(LogWinProbability(elo_a, elo_b)), 1e-5)
        << "LogProbability " << elo_a << " " << elo_b;
  }
}

TEST(ELOFastMath, WinMatrix) {
  absl::BitGen gen;

  for (size_t _ = 0; _ < 10000; _++) {
    // avoid powers of 2 to trigger edge cases
    size_t kNumElos = 13;

    Eigen::VectorXd elos(kNumElos);
    for (size_t i = 0; i < kNumElos; i++) {
      elos(i) = absl::Uniform(gen, 0.0, 4000.0);
    }

    Eigen::MatrixXd win_probabilities = WinProbabilityMatrix(elos);

    for (size_t i = 0; i < kNumElos; i++) {
      for (size_t j = 0; j < kNumElos; j++) {
        EXPECT_NEAR(win_probabilities(i, j), WinProbability(elos(i), elos(j)),
                    1e-5);
      }
    }
  }
}

TEST(ELOFastMath, LogWinMatrix) {
  absl::BitGen gen;

  for (size_t _ = 0; _ < 10000; _++) {
    // avoid powers of 2 to trigger edge cases
    size_t kNumElos = 17;

    Eigen::VectorXd elos(kNumElos);
    for (size_t i = 0; i < kNumElos; i++) {
      elos(i) = absl::Uniform(gen, 0.0, 4000.0);
    }

    Eigen::MatrixXd log_win_probabilities = LogWinProbabilityMatrix(elos);

    for (size_t i = 0; i < kNumElos; i++) {
      for (size_t j = 0; j < kNumElos; j++) {
        EXPECT_NEAR(log_win_probabilities(i, j),
                    LogWinProbability(elos(i), elos(j)), 1e-5);
      }
    }
  }
}

TEST(ELOFastMath, AccumulateH2) {
  absl::BitGen gen;

  for (size_t _ = 0; _ < 10000; _++) {
    // avoid powers of 2 to trigger edge cases
    size_t kSize = 15;

    Eigen::MatrixXd p(kSize, kSize);
    for (size_t i = 0; i < kSize; i++) {
      for (size_t j = 0; j < kSize; j++) {
        p(i, j) = absl::Uniform(gen, 0.1, 0.9);
      }
    }

    constexpr float kMul = 1.273;
    Eigen::MatrixXd h2 = Eigen::MatrixXd::Zero(kSize, kSize);
    AccumulateH2(p, kMul, h2);

    for (size_t i = 0; i < kSize; i++) {
      for (size_t j = 0; j < kSize; j++) {
        float pv = p(i, j);
        EXPECT_NEAR(
            h2(i, j),
            -kMul * (pv * std::log2(pv) + (1.0 - pv) * std::log2(1.0 - pv)),
            1e-5);
      }
    }
  }
}

TEST(ELOFastMath, KLDivergence) {
  absl::BitGen gen;

  for (size_t _ = 0; _ < 10000; _++) {
    // avoid powers of 2 to trigger edge cases
    size_t kSize = 15;

    Eigen::MatrixXd p(kSize, kSize);
    Eigen::MatrixXd true_p(kSize, kSize);
    for (size_t i = 0; i < kSize; i++) {
      for (size_t j = 0; j < kSize; j++) {
        p(i, j) = absl::Uniform(gen, 0.1, 0.9);
        true_p(i, j) = absl::Uniform(gen, 0.1, 0.9);
      }
    }

    double kl = KLDivergence(true_p, p);
    double kl_naive = 0;
    for (size_t i = 0; i < kSize; i++) {
      for (size_t j = 0; j < kSize; j++) {
        double tp = true_p(i, j);
        double pp = p(i, j);
        kl_naive += tp * (std::log(tp) - std::log(pp));
        tp = 1.0 - tp;
        pp = 1.0 - pp;
        kl_naive += tp * (std::log(tp) - std::log(pp));
      }
    }

    EXPECT_NEAR(kl, kl_naive, 1e-5);
  }
}

}  // namespace
