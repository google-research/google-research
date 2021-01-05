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

#include "estimate_optimum_range.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "io.h"

namespace sliding_window {

namespace {

// Given a series of instances of the problem, runs the k-means++ algorithm on
// each instance and outputs the vector of costs of the solutions found. Any
// other algorithm could be used instead of k-means++.
vector<double> cost_samples(const vector<vector<TimePointPair>>& samples,
                            const int32_t k, absl::BitGen* gen) {
  vector<double> costs;
  costs.reserve(samples.size());
  for (int i = 0; i < samples.size(); i++) {
    std::vector<std::pair<TimePointPair, double>> instance;
    for (const auto& point : samples.at(i)) {
      instance.push_back(make_pair(point, 1.0));
    }
    CHECK(!samples.at(i).empty());
    std::vector<int32_t> ingnored_centers;
    double cost;
    k_means_plus_plus(instance, k, gen, &ingnored_centers, &cost);
    if (cost > 0) {
      costs.push_back(cost);
    }
  }
  return costs;
}

// Given a vector of costs, guesses a range for the optimum value. This is done
// by using a heuristic based on the min/max value and the standard deviation.
// Outputs the min and max bounds as a pair.
std::pair<double, double> guess_bounds(const vector<double>& costs) {
  CHECK(!costs.empty());
  auto min_max_it = std::minmax_element(costs.begin(), costs.end());
  double mean = std::accumulate(std::begin(costs), std::end(costs), double{0}) /
                costs.size();
  std::vector<double> costs_sq;
  costs_sq.resize(costs.size());
  for (const auto& cost : costs) {
    costs_sq.push_back(cost * cost);
  }
  double mean_sq =
      std::accumulate(std::begin(costs_sq), std::end(costs_sq), double{0}) /
      costs.size();
  double stddev = std::sqrt(mean_sq - mean * mean);

  double lowerbound = std::max(*min_max_it.first / 3, mean - 3.0 * stddev);
  double upperbound = std::max(mean + 3.0 * stddev, *min_max_it.second * 3);

  return std::make_pair(lowerbound, upperbound);
}

}  // namespace

std::pair<double, double> guess_optimum_range_bounds(
    const string& stream_file_path, int32_t window_size, int32_t num_samples,
    int32_t k, absl::BitGen* gen) {
  int32_t stream_size = 0;
  auto samples = get_windows_samples(stream_file_path, window_size, num_samples,
                                     0, &stream_size);
  CHECK_LE(window_size, 0.8 * stream_size);

  while (samples.size() < num_samples) {
    int32_t to_skip = absl::Uniform(*gen, 0u, 0.2 * window_size);
    auto new_samples = get_windows_samples(stream_file_path, window_size,
                                           num_samples - samples.size(),
                                           to_skip, &stream_size);
    samples.insert(samples.end(), new_samples.begin(), new_samples.end());
  }

  auto costs = cost_samples(samples, k, gen);
  return guess_bounds(costs);
}

}  // namespace sliding_window
