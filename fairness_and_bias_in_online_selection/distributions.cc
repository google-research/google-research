// Copyright 2021 The Google Research Authors.
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

#include "distributions.h"

namespace fair_secretary {

using std::vector;

void UniformDistribution::Init(int n, double range) {
  range_ = range;
  p_th_ = vector<double>(n, 0);
  vector<double> V(n, 0.0);
  V[n - 1] = range / 2;
  for (int i = n - 2; i >= 0; i--) {
    p_th_[i] = V[i + 1];
    V[i] = (range + p_th_[i]) / 2;
  }
}

double UniformDistribution::PThreshold(int index) { return p_th_[index]; }

double UniformDistribution::Middle(int n) {
  return range_ * pow(1.0 / 2, 1.0 / n);
}

double UniformDistribution::Reverse(double x) { return x * range_; }

double UniformDistribution::Sample() {
  return (static_cast<double>(RandomHandler::eng_()) /
          std::numeric_limits<uint64_t>::max()) *
         range_;
}

void BinomialDistribution::Init(int n, double p) {
  choose_ = vector<vector<double>>(n + 1, vector<double>(n + 1, 0));
  probability_ = vector<double>(n + 1, 1.0);
  r_probability_ = vector<double>(n + 1, 1.0);
  for (int i = 0; i <= n; i++) {
    choose_[i][0] = 1;
  }
  for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= n; j++) {
      choose_[i][j] = choose_[i - 1][j - 1] + choose_[i - 1][j];
    }
  }
  for (int i = 1; i <= n; i++) {
    probability_[i] = probability_[i - 1] * p;
    r_probability_[i] = r_probability_[i - 1] * (1 - p);
  }
  distribution_ =
      std::binomial_distribution<int>(probability_.size() - 1, probability_[1]);
}

double BinomialDistribution::Expected(double lower_bound) {
  double ans = 0;
  double range = 0;
  int n = probability_.size() - 1;
  for (int i = ceil(lower_bound * n); i <= n; i++) {
    ans += probability_[i] * r_probability_[n - i] * choose_[n][i] * i / n;
    range += probability_[i] * r_probability_[n - i] * choose_[n][i];
  }
  return ans / range;
}

double BinomialDistribution::Reverse(double x) {
  int n = probability_.size() - 1;
  for (int i = 0; i <= n; i++) {
    x -= choose_[n][i] * probability_[i] * r_probability_[n - i];
    if (x <= 0) {
      return static_cast<double>(i) / n;
    }
  }
  return 1.0;
}

double BinomialDistribution::Sample() {
  double rand = (static_cast<double>(RandomHandler::eng_()) /
                 std::numeric_limits<uint64_t>::max());
  return (static_cast<double>(distribution_(RandomHandler::generator_)) +
          rand) /
         (probability_.size() - 1);
}

double BinomialDistribution::Middle(int n) {
  for (int i = max_dist_.size() - 1; i >= 0; i--) {
    if (max_dist_[i] >= 0.5) {
      return static_cast<double>(i) / (max_dist_.size() - 1);
    }
  }
  return 0.0;
}

void BinomialDistribution::ComputeMaxDist(double num_dists) {
  max_dist_ = vector<double>(probability_.size(), 0.0);
  int n = probability_.size() - 1;
  double x = 0;
  for (int i = n; i >= 0; i--) {
    x += choose_[n][i] * probability_[i] * r_probability_[n - i];
    max_dist_[i] = (1 - pow(1 - x, num_dists));
  }
  // Computing PThreshold.
  p_th_ = vector<double>(num_dists, 0);
  vector<double> V(num_dists, 0.0);
  V[num_dists - 1] = Expected(0);
  for (int i = num_dists - 2; i >= 0; i--) {
    p_th_[i] = V[i + 1];
    V[i] = Expected(p_th_[i]);
  }
}

double BinomialDistribution::PThreshold(int index) { return p_th_[index]; }

}  // namespace fair_secretary
