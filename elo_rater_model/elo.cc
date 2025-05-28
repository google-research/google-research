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

#include "elo.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "elo_fast_math.h"

// We define the optimal set of ELO scores as the one that achieves the maximum
// log-likelihood of the observed set of answers under the ELO/BTL model.
//
// Given s = ln 10 / ELOSettings::kEloN, this writes as minimizing the
// following:
//
//   -sum (a wins against b) log [exp(elo_a) / (exp(elo_a) + exp(elo_b))]   (1)
//
// Note that the "standard" ELO update rule can be seen as a form of gradient
// descent on this function, using ELO's `k` factor as a learning rate.
//
// We use the following terms as a prior:
//
//  sum 0.5 * RegularizerFactor() * (elo_i - elo_center)**2                 (2)
//   + 0.5 * sum_regularizer * (sum elo_i - n * elo_center)**2
//
// To model rater behaviour, we assume that each rater can answer randomly
// (50-50) with rater-dependent probability q_r. We then use
// expectation-maximization to jointly optimize rater random answer probability
// and ELO scores. We assume that golden questions have a probability of correct
// answer of 1 under the ELO model.
//
// More precisely, calling E_i the probability of answer i under the ELO
// model, q_r the probability of rater r to answer randomly, and p_i the
// probability of question i being answered by following the ELO model, the loss
// function we optimize, instead of (1)+(2), is the following plus (2):
//
//   -sum (a wins against b for r)  log [q_r*0.5 + (1.0-q_r) * E_i]
//   -sum (correct goldens for r)   log [q_r*0.5 + (1.0-q_r) * 1.0]
//   -sum (incorrect goldens for r) log [q_r*0.5]

namespace {

constexpr double kS = kLog10 / kEloN;

float RegularizerFactor(const Eigen::VectorXd& elos, float alpha) {
  return alpha / elos.size();
}

float SumRegularizer(const Eigen::VectorXd& elos) {
  // This value effectively does not matter, as long as it is large enough to
  // suppress variance due to mean shifts. We pick this value as it is
  // a large value that ensures that the GD fallback still converges.
  return 0.9 / elos.size();
}
}  // namespace

// Computes the log-likelihood function.
double ComputeEloLogLikelihood(const ELOSettings& settings,
                               const Eigen::VectorXd& elos,
                               const Eigen::VectorXd& rater_reliability,
                               const std::vector<Eigen::MatrixXd>& win_count,
                               const Eigen::VectorXd& rater_correct_goldens,
                               const Eigen::VectorXd& rater_incorrect_goldens) {
  auto shifted_elos = elos.array() - ELOSettings::kInitialElo;
  double shifted_elos_sum = shifted_elos.sum();
  // Prior.
  double accum =
      -0.5 * RegularizerFactor(elos, settings.alpha) *
          shifted_elos.square().sum() -
      0.5 * SumRegularizer(elos) * shifted_elos_sum * shifted_elos_sum;
  Eigen::MatrixXd loss = Eigen::MatrixXd::Zero(elos.size(), elos.size());
  Eigen::MatrixXd win_probability = WinProbabilityMatrix(elos);
  // Prior on raters
  auto rater_random_probability = 1.0 / (1.0 + rater_reliability.array().exp());
  double rater_regularizer =
      -(1.0 + rater_random_probability.square() * settings.rater_prior_strength)
           .log()
           .sum();
  for (size_t r = 0; r < rater_reliability.size(); r++) {
    double p_random = 1.0 / (1.0 + exp(rater_reliability[r]));
    double p_nonrandom = 1.0 / (1.0 + exp(-rater_reliability[r]));
    loss.array() +=
        win_count[r].array() *
        (p_random * 0.5 + p_nonrandom * win_probability.array()).log();
    rater_regularizer +=
        rater_correct_goldens[r] * std::log(p_random * 0.5 + p_nonrandom);
    rater_regularizer += rater_incorrect_goldens[r] * std::log(p_random * 0.5);
  }
  return accum + rater_regularizer + loss.sum();
}

// Computes the gradient of the negative log-likelihood function.
Eigen::VectorXd ComputeEloGradientWrtElo(
    const ELOSettings& settings, const Eigen::VectorXd& elos,
    const Eigen::VectorXd& rater_reliability,
    const std::vector<Eigen::MatrixXd>& win_count) {
  auto win_probability = WinProbabilityMatrix(elos);
  Eigen::VectorXd result = Eigen::VectorXd::Zero(elos.size());
  for (size_t r = 0; r < win_count.size(); r++) {
    double p_random = 1.0 / (1.0 + exp(rater_reliability[r]));
    double p_nonrandom = 1.0 / (1.0 + exp(-rater_reliability[r]));
    auto exp_loss = p_random * 0.5 + p_nonrandom * win_probability.array();
    auto numerator = win_probability.array() * (1.0 - win_probability.array()) *
                     kS * p_nonrandom;
    auto rater_matrix = win_count[r].array() * numerator / exp_loss;
    Eigen::VectorXd rater_result =
        (rater_matrix - rater_matrix.transpose()).rowwise().sum();
    result = result - rater_result;
  }
  // Add the gradient of the prior.
  return result.array() +
         RegularizerFactor(elos, settings.alpha) * elos.array() +
         SumRegularizer(elos) * elos.sum();
}

// Computes the gradient of the negative log-likelihood function.
Eigen::VectorXd ComputeEloGradientWrtRater(
    const ELOSettings& settings, const Eigen::VectorXd& elos,
    const Eigen::VectorXd& rater_reliability,
    const std::vector<Eigen::MatrixXd>& win_count,
    const Eigen::VectorXd& rater_correct_goldens,
    const Eigen::VectorXd& rater_incorrect_goldens) {
  auto win_probability = WinProbabilityMatrix(elos);
  Eigen::VectorXd result = Eigen::VectorXd::Zero(rater_reliability.size());
  for (size_t r = 0; r < rater_reliability.size(); r++) {
    double p_random = 1.0 / (1.0 + exp(rater_reliability[r]));
    double p_nonrandom = 1.0 / (1.0 + exp(-rater_reliability[r]));
    auto exp_loss = p_random * 0.5 + p_nonrandom * win_probability.array();
    float rater_prob_log_likelihood_gradient =
        (win_count[r].array() * (win_probability.array() - 0.5) /
         exp_loss.array())
            .sum();
    rater_prob_log_likelihood_gradient +=
        rater_correct_goldens[r] / (2.0 * p_nonrandom + p_random);
    rater_prob_log_likelihood_gradient -= rater_incorrect_goldens[r] / p_random;

    // Prior on raters
    rater_prob_log_likelihood_gradient +=
        2 * p_random * settings.rater_prior_strength /
        (settings.rater_prior_strength * p_random * p_random + 1.0);

    result[r] = -rater_prob_log_likelihood_gradient * p_nonrandom * p_random;
  }
  return result;
}

// Computes the Hessian of the negative log-likelihood function.
Eigen::MatrixXd ComputeEloHessianWrtElo(
    const ELOSettings& settings, const Eigen::VectorXd& elos,
    const Eigen::VectorXd& rater_reliability,
    const std::vector<Eigen::MatrixXd>& win_count) {
  auto win_probability = WinProbabilityMatrix(elos);
  Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(elos.size(), elos.size());
  for (size_t r = 0; r < win_count.size(); r++) {
    double p_random = 1.0 / (1.0 + exp(rater_reliability[r]));
    double p_nonrandom = 1.0 / (1.0 + exp(-rater_reliability[r]));
    auto exp_loss = p_random * 0.5 + p_nonrandom * win_probability.array();
    Eigen::MatrixXd first_term = p_nonrandom * win_probability.array() *
                                 win_probability.transpose().array();
    Eigen::MatrixXd numerator =
        first_term.array() * ((0.5 - win_probability.array()) * p_random -
                              p_nonrandom * win_probability.array().square());
    Eigen::MatrixXd rater_matrix = win_count[r].array() * numerator.array() /
                                   (exp_loss * exp_loss) * kS * kS;
    hessian = hessian.array() + rater_matrix.array() +
              rater_matrix.array().transpose();
  }
  // The diagonal is wrong, so fix it up.
  hessian.diagonal().array() = 0;
  Eigen::VectorXd diagonal = -hessian.colwise().sum();
  hessian.diagonal().array() = diagonal;
  // Add the hessian of the prior.
  hessian.diagonal().array() += RegularizerFactor(elos, settings.alpha);
  hessian.array() += SumRegularizer(elos);
  return hessian;
}

// Computes the diagonal of the Hessian of the negative log-likelihood function
// with respect to rater reliabilities.
Eigen::VectorXd ComputeEloHessianDiagonalWrtRater(
    const ELOSettings& settings, const Eigen::VectorXd& elos,
    const Eigen::VectorXd& rater_reliability,
    const std::vector<Eigen::MatrixXd>& win_count,
    const Eigen::VectorXd& rater_correct_goldens,
    const Eigen::VectorXd& rater_incorrect_goldens) {
  auto win_probability = WinProbabilityMatrix(elos);
  Eigen::VectorXd result = Eigen::VectorXd::Zero(rater_reliability.size());
  for (size_t r = 0; r < rater_reliability.size(); r++) {
    double p_random = 1.0 / (1.0 + exp(rater_reliability[r]));
    double p_nonrandom = 1.0 / (1.0 + exp(-rater_reliability[r]));
    auto exp_loss = p_random * 0.5 + p_nonrandom * win_probability.array();
    auto numerator = (0.5 - win_probability.array()) *
                     (win_probability.array() * p_nonrandom * p_nonrandom -
                      0.5 * p_random * p_random);
    auto denominator = exp_loss.array() * exp_loss.array();
    float rater_prob_log_likelihood_hessian =
        (win_count[r].array() * (numerator / denominator)).sum();
    rater_prob_log_likelihood_hessian -=
        rater_correct_goldens[r] *
        (1.0 - 2.0 / ((1.0 + p_nonrandom) * (1.0 + p_nonrandom)));
    rater_prob_log_likelihood_hessian -= rater_incorrect_goldens[r];

    // Prior on raters
    double scaled_p_random2 =
        settings.rater_prior_strength * p_random * p_random;

    rater_prob_log_likelihood_hessian +=
        2.0 * settings.rater_prior_strength * p_random *
        (scaled_p_random2 * p_random + 3.0 * p_random - 2.0) /
        ((scaled_p_random2 + 1.0) * (scaled_p_random2 + 1.0));

    result[r] = -rater_prob_log_likelihood_hessian * p_nonrandom * p_random;
  }
  return result;
}

namespace {

struct EloSampleData {
  // Sampled ELO scores
  Eigen::VectorXd elo_sample;
  // Matrix used by SuggestQuestions(), cached here to avoid additional
  // allocations.
  Eigen::MatrixXd suggestion_p;
  // Relative weight of this sample. It is guaranteed that all weights in a set
  // of samples sum to 1.
  float weight;
};

struct EloDetailedOutputs {
  std::vector<Eigen::MatrixXd> win_count;
  Eigen::MatrixXd covariance_matrix;
  Eigen::VectorXd elos;
  Eigen::VectorXd rater_reliability;
  // Not an Eigen::VectorXd, as it does not 0-initialize after resizing.
  std::vector<float> rater_incorrect_goldens;
  std::vector<float> rater_total_goldens;

  std::vector<EloSampleData> samples;
  bool dirty_samples = true;

  void Compute(const ELOSettings& settings) {
    dirty_samples = true;

    Eigen::VectorXd rater_correct_goldens(rater_total_goldens.size());
    Eigen::VectorXd rater_incorrect_goldens(rater_total_goldens.size());

    for (size_t i = 0; i < win_count.size(); i++) {
      rater_incorrect_goldens[i] = this->rater_incorrect_goldens[i];
      rater_correct_goldens[i] =
          rater_total_goldens[i] - rater_incorrect_goldens[i];
    }

    Eigen::VectorXd new_rater_probabilities =
        Eigen::VectorXd::Constant(rater_reliability.size(), 0.5);
    Eigen::VectorXd new_elos = Eigen::VectorXd::Constant(elos.size(), 0);
    // Initialize rater random probabilities with a reasonable guess, and ELO
    // scores with 0. This has proven to converge faster in practice.
    elos = new_elos;
    for (size_t r = 0; r < win_count.size(); r++) {
      if (rater_total_goldens[r] > 0) {
        float incorrect_gq_fraction =
            rater_incorrect_goldens[r] / rater_total_goldens[r];
        float random_p =
            std::max(1e-4, std::min(2.0 * incorrect_gq_fraction, 1.0 - 1e-4));
        rater_reliability[r] = -std::log(random_p / (1.0 - random_p));
      } else {
        rater_reliability[r] = 0.0;
      }
    }

    auto start = absl::Now();
    size_t total_elo_iters = 0;
    size_t total_prob_iters = 0;
    size_t outer_iters = 0;

    // Returns true if ELO values have changed.
    auto elo_step = [&]() {
      constexpr double kConvergenceThr = 1e-8;
      Eigen::VectorXd original_elos = elos;
      for (size_t iter = 0; iter < 100; iter++) {
        auto gradient = ComputeEloGradientWrtElo(settings, elos,
                                                 rater_reliability, win_count);
        double norm = gradient.squaredNorm();
        if (norm < kConvergenceThr) break;
        total_elo_iters += 1;
        auto hessian = ComputeEloHessianWrtElo(settings, elos,
                                               rater_reliability, win_count);
        auto hessian_svd =
            hessian.bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>();
        elos = elos - hessian_svd.solve(gradient);
      }
      constexpr double kRecomputeProbabilityThres = 1.0;
      double diff = (elos - original_elos).array().abs().maxCoeff();
      return diff > kRecomputeProbabilityThres;
    };

    std::mt19937 rng;
    std::normal_distribution<> dist;

    auto prob_step = [&]() {
      constexpr float kConvergenceThr = 1e-3;

      constexpr size_t kNumSamples = 100;

      Eigen::MatrixXd transform =
          ComputeEloHessianWrtElo(settings, elos, rater_reliability, win_count)
              .inverse()
              .llt()
              .matrixL();

      // We draw samples from the Gaussian approximation because that seems to
      // work more reliably in the low data regime.
      std::vector<Eigen::VectorXd> samples(kNumSamples);
      for (size_t i = 0; i < kNumSamples; i++) {
        Eigen::VectorXd basesample =
            Eigen::VectorXd(elos.size()).unaryExpr([&](auto _) {
              return dist(rng);
            });
        samples[i] = elos + transform * basesample;
      }

      bool changed = false;
      for (size_t iter = 0; iter < 100; iter++) {
        Eigen::VectorXd gradient =
            Eigen::VectorXd::Zero(rater_reliability.size());
        Eigen::VectorXd hessian_diagonal =
            Eigen::VectorXd::Zero(rater_reliability.size());

        for (const auto& sample : samples) {
          float weight = 1.0 / kNumSamples;
          gradient =
              gradient + weight * ComputeEloGradientWrtRater(
                                      settings, sample, rater_reliability,
                                      win_count, rater_correct_goldens,
                                      rater_incorrect_goldens);
          hessian_diagonal =
              hessian_diagonal +
              weight * ComputeEloHessianDiagonalWrtRater(
                           settings, sample, rater_reliability, win_count,
                           rater_correct_goldens, rater_incorrect_goldens);
        }
        if (gradient.squaredNorm() < kConvergenceThr) break;
        total_prob_iters += 1;
        changed = true;
        rater_reliability =
            rater_reliability.array() -
            gradient.array() / hessian_diagonal.array().max(1.0);
      }
      return changed;
    };

    constexpr float kMinBestLogReliability = 1.0;

    for (size_t i = 0; i < 100; i++) {
      outer_iters++;
      bool changed = false;
      float best_log_reliability = rater_reliability.size()
                                       ? rater_reliability.maxCoeff()
                                       : kMinBestLogReliability;
      if (best_log_reliability < kMinBestLogReliability) {
        LOG(INFO) << "low reliabilities, increasing...";
        rater_reliability.array() +=
            kMinBestLogReliability - best_log_reliability;
        changed = true;
      }
      changed |= elo_step();
      changed |= prob_step();
      if (!changed) break;
    }

    auto stop = absl::Now();

    LOG(INFO) << "Converged in " << outer_iters << " outer iterations, "
              << total_elo_iters << " ELO iterations, " << total_prob_iters
              << " probability iterations and "
              << (absl::FDivDuration(stop - start, absl::Milliseconds(1)) *
                  0.001)
              << " seconds.";

    // Center ELO scores around the initial ELO.
    float offset = elos.size() > 0 ? elos.mean() - ELOSettings::kInitialElo : 0;
    elos.array() -= offset;

    auto hessian =
        ComputeEloHessianWrtElo(settings, elos, rater_reliability, win_count);
    covariance_matrix = hessian.inverse();
  }

  template <typename RNG>
  void ComputeSamples(RNG& rng) {
    constexpr size_t kNumSamples = 1024 * 2;
    if (samples.size() == kNumSamples && !dirty_samples) {
      return;
    }
    samples.resize(kNumSamples);
    dirty_samples = false;
    std::normal_distribution<> dist;

    Eigen::VectorXd rater_correct_goldens(rater_total_goldens.size());
    Eigen::VectorXd rater_incorrect_goldens(rater_total_goldens.size());

    for (size_t i = 0; i < win_count.size(); i++) {
      rater_incorrect_goldens[i] = this->rater_incorrect_goldens[i];
      rater_correct_goldens[i] =
          rater_total_goldens[i] - rater_incorrect_goldens[i];
    }

    Eigen::MatrixXd transform = covariance_matrix.llt().matrixL();

    // Naive one pass algorithm on shifted data to compute the variance.
    double total_log_weight = 0;
    double total_log_weight_minus_shift_squared = 0;
    double shift = 0;
    Eigen::VectorXd basesample;

    std::vector<float> log_weight(samples.size());
    for (size_t iter = 0; iter < samples.size(); iter++) {
      basesample = Eigen::VectorXd(elos.size()).unaryExpr([&](auto _) {
        return dist(rng);
      });
      samples[iter].elo_sample = elos + transform * basesample;
      samples[iter].elo_sample.array() -=
          elos.mean() - ELOSettings::kInitialElo;
      float gaussian_log_pdf = -0.5 * basesample.squaredNorm();
      float log_actual_pdf = ComputeEloLogLikelihood(
          ELOSettings(), samples[iter].elo_sample, rater_reliability, win_count,
          rater_correct_goldens, rater_incorrect_goldens);
      log_weight[iter] = log_actual_pdf - gaussian_log_pdf;

      if (iter == 0) shift = log_weight[iter];
      total_log_weight += log_weight[iter];
      total_log_weight_minus_shift_squared +=
          (log_weight[iter] - shift) * (log_weight[iter] - shift);
    }
    float max_log_weight =
        *std::max_element(log_weight.begin(), log_weight.end());
    float total_weight = 0.0;
    float total_w2 = 0.0;
    for (size_t iter = 0; iter < samples.size(); iter++) {
      samples[iter].weight = std::exp(log_weight[iter] - max_log_weight);
      total_weight += samples[iter].weight;
      total_w2 += samples[iter].weight * samples[iter].weight;
    }

    double log_weight_mean = total_log_weight / samples.size();
    double log_weight_variance =
        total_log_weight_minus_shift_squared / samples.size() -
        (log_weight_mean - shift) * (log_weight_mean - shift);
    LOG(INFO) << "Sampled " << samples.size() << " samples, log weight "
              << log_weight_mean << " ± " << std::sqrt(log_weight_variance)
              << ", effective sample size "
              << total_weight * total_weight / total_w2;

    for (auto& sample : samples) {
      sample.weight /= total_weight;
    }
  }
};
}  // namespace

struct ELOData::Data {
  absl::flat_hash_map<std::string, size_t> method_name_to_index;
  std::vector<std::string> methods;
  std::vector<EloDetailedOutputs> info_for_question_kinds;
  std::vector<Suggestion> suggestions;
  std::vector<std::vector<ELOScore>> scores;
  std::vector<std::string> rater_ids;
  absl::flat_hash_map<std::string, size_t> rater_indices;
  bool dirty = true;
  size_t num_questions = 0;

  void Compute(const ELOSettings& settings) {
    if (!dirty) return;
    for (EloDetailedOutputs& out : info_for_question_kinds) {
      out.Compute(settings);
    }
    dirty = false;
  }
};

ELOData::ELOData(ELOSettings settings, size_t num_question_kinds,
                 const std::vector<std::string>& methods)
    : data_(std::make_unique<ELOData::Data>()), settings_(settings) {
  data_->info_for_question_kinds.resize(num_question_kinds);
  data_->methods = methods;
  for (auto& out : data_->info_for_question_kinds) {
    out.elos = Eigen::VectorXd::Constant(methods.size(), 0);
  }
  for (size_t i = 0; i < methods.size(); i++) {
    data_->method_name_to_index[methods[i]] = i;
  }
  data_->scores.resize(num_question_kinds,
                       std::vector<ELOScore>(methods.size()));
  for (size_t a = 0; a < methods.size(); a++) {
    for (size_t b = 0; b < methods.size(); b++) {
      if (a <= b) continue;
      data_->suggestions.push_back(Suggestion{0.0, methods[a], methods[b]});
    }
  }
}

void ELOData::AddQuestion(const std::optional<std::vector<bool>>&
                              golden_question_correct_answers_are_a,
                          absl::string_view method_a,
                          absl::string_view method_b,
                          const std::vector<std::string>& choices,
                          size_t rater_index) {
  auto aiter = data_->method_name_to_index.find(method_a);
  auto biter = data_->method_name_to_index.find(method_b);
  for (size_t i = 0; i < data_->info_for_question_kinds.size(); i++) {
    auto& info = data_->info_for_question_kinds[i];
    if (rater_index >= info.win_count.size()) {
      info.win_count.resize(
          rater_index + 1,
          Eigen::MatrixXd::Zero(info.elos.size(), info.elos.size()));
      info.rater_incorrect_goldens.resize(rater_index + 1);
      info.rater_total_goldens.resize(rater_index + 1);
      info.rater_reliability.resize(rater_index + 1);
    }
    if (golden_question_correct_answers_are_a.has_value()) {
      info.rater_total_goldens[rater_index] += 1;
      bool answer_is_a = method_a == choices[i];
      info.rater_incorrect_goldens[rater_index] +=
          golden_question_correct_answers_are_a.value()[i] != answer_is_a;
    } else {
      if (aiter == data_->method_name_to_index.end() ||
          biter == data_->method_name_to_index.end()) {
        return;
      }
      size_t a = aiter->second;
      size_t b = biter->second;
      if (method_a != choices[i]) {
        std::swap(a, b);
      }
      if (choices[i].empty()) {
        info.win_count[rater_index](a, b) += 0.5;
        info.win_count[rater_index](b, a) += 0.5;
      } else {
        info.win_count[rater_index](a, b) += 1;
      }
    }
  }
  data_->num_questions += 1;
  data_->dirty = true;
}

size_t ELOData::GetNumQuestions() const { return data_->num_questions; }
const std::vector<std::string>& ELOData::GetMethods() const {
  return data_->methods;
}

const std::vector<std::vector<ELOScore>>& ELOData::ComputeELOScores() {
  data_->Compute(settings_);
  for (size_t i = 0; i < data_->info_for_question_kinds.size(); i++) {
    const auto& info = data_->info_for_question_kinds[i];
    // Compute confidence intervals with a second-order Taylor expansion in a
    // neighbourhood of the optimum.
    // TODO(veluca): in principle, sampling should be more accurate if the
    // second-order approximation is not quite accurate. It is also more
    // resistant to numerical shenanigans in the Hessian. However, using
    // sampling introduces its own challenges, such as the fact that the mean of
    // the samples might not match the most likely ELO scores.
    constexpr float kErfc0_01 = 1.821386367;
    constexpr float kSqrt2 = 1.414213562;

    Eigen::VectorXd p99 =
        info.covariance_matrix.diagonal().array().sqrt() * kErfc0_01 * kSqrt2;

    auto& scores = data_->scores[i];
    for (size_t j = 0; j < scores.size(); j++) {
      scores[j].elo = info.elos[j];
      scores[j].p99_low = info.elos[j] - p99[j];
      scores[j].p99_hi = info.elos[j] + p99[j];
    }
  }
  return data_->scores;
}

Eigen::VectorXd ELOData::RaterRandomProbability(size_t question_kind) {
  data_->Compute(settings_);
  const auto& rater_reliability =
      data_->info_for_question_kinds[question_kind].rater_reliability;
  return 1.0 / (1.0 + rater_reliability.array().exp());
}

const std::vector<Suggestion>& ELOData::SuggestQuestions() {
  data_->Compute(settings_);

  std::mt19937_64 rng;

  for (auto& suggestion : data_->suggestions) {
    suggestion.weight = 0;
  }

  for (EloDetailedOutputs& outs : data_->info_for_question_kinds) {
    outs.ComputeSamples(rng);
    Eigen::MatrixXd total_prob =
        Eigen::MatrixXd::Zero(outs.elos.size(), outs.elos.size());
    Eigen::MatrixXd total_h2 =
        Eigen::MatrixXd::Zero(outs.elos.size(), outs.elos.size());

    for (auto& sample : outs.samples) {
      sample.suggestion_p = WinProbabilityMatrix(sample.elo_sample);
      total_prob += sample.suggestion_p * sample.weight;
      AccumulateH2(sample.suggestion_p, sample.weight, total_h2);
    }
    Eigen::MatrixXd question_weight = -total_h2;
    AccumulateH2(total_prob, 1.0, question_weight);
    size_t index = 0;
    for (size_t a = 0; a < outs.elos.size(); a++) {
      for (size_t b = 0; b < outs.elos.size(); b++) {
        if (a <= b) continue;
        data_->suggestions[index++].weight += question_weight(a, b);
      }
    }
  }
  return data_->suggestions;
}

double Log1MinusExpX(double x) { return std::log1p(-std::exp(x)); }

float ELOData::ComputeEloDivergence(const std::vector<float>& ground_truth_elos,
                                    size_t question_index) {
  data_->Compute(settings_);
  CHECK_GT(data_->info_for_question_kinds.size(), question_index);
  auto& info = data_->info_for_question_kinds[question_index];
  CHECK_EQ(ground_truth_elos.size(), info.elos.size());  // Crash OK
  std::mt19937_64 rng;
  info.ComputeSamples(rng);

  size_t n = ground_truth_elos.size();
  Eigen::VectorXd true_elos(n);
  for (size_t i = 0; i < n; i++) true_elos[i] = ground_truth_elos[i];

  Eigen::MatrixXd true_win_probabilities = WinProbabilityMatrix(true_elos);

  double weight_sum = 0.0;
  Eigen::MatrixXd probabilities = Eigen::MatrixXd::Zero(n, n);

  for (auto& sample : info.samples) {
    weight_sum += sample.weight;
    sample.suggestion_p = WinProbabilityMatrix(sample.elo_sample);
    probabilities += sample.suggestion_p * sample.weight;
  }

  return KLDivergence(true_win_probabilities, probabilities / weight_sum) /
         (n * (n - 1));
}

ELOData::ELOData(ELOData&&) = default;
ELOData& ELOData::operator=(ELOData&&) = default;
ELOData::~ELOData() = default;
