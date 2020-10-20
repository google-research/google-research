// Copyright 2020 The Google Research Authors.
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

#include "base.h"

#include "absl/random/random.h"

namespace sliding_window {

double l2_distance(const vector<double>& a, const vector<double>& b) {
  ++CALLS_DIST_FN;
  double l2 = 0.0;
  CHECK_EQ(a.size(), b.size());
  for (int i = 0; i < a.size(); i++) {
    l2 += (a[i] - b[i]) * (a[i] - b[i]);
  }
  return std::sqrt(l2);
}

// Implementation of the L2 Distance function.
double l2_distance(const TimePointPair& a, const TimePointPair& b) {
  return l2_distance(a.second, b.second);
}

void k_means_plus_plus(
    const std::vector<std::pair<TimePointPair, double>>& instance,
    const int32_t k, absl::BitGen* gen, std::vector<int32_t>* centers,
    double* cost) {
  centers->clear();

  std::vector<double> min_distance_to_centers(
      instance.size(), std::numeric_limits<double>::max());

  if (k >= instance.size()) {
    for (int32_t i = 0; i < instance.size(); i++) {
      centers->push_back(i);
    }
  } else {
    // add u.a.r. center.
    int32_t index = absl::Uniform(*gen, 0u, instance.size());
    centers->push_back(index);
    while (centers->size() < k) {
      double sum_pow_min_distances = 0.0;
      vector<double> min_dist_powers;
      min_dist_powers.reserve(instance.size());

      for (int pos = 0; pos < instance.size(); ++pos) {
        double min_distance =
            std::min(min_distance_to_centers.at(pos),
                     l2_distance(instance.at(pos).first.second,
                                 instance.at(centers->back()).first.second));
        min_distance_to_centers[pos] = min_distance;
        sum_pow_min_distances +=
            std::pow(min_distance, 2) * instance.at(pos).second;
        min_dist_powers.push_back(std::pow(min_distance, 2) *
                                  instance.at(pos).second);
      }

      double random_place = absl::Uniform(*gen, 0, sum_pow_min_distances);
      for (int32_t i = 0; i < instance.size(); i++) {
        if (random_place <= min_dist_powers[i]) {
          centers->push_back(i);
          break;
        }
        random_place -= min_dist_powers[i];
      }
    }
  }

  *cost = 0;
  for (int pos = 0; pos < instance.size(); ++pos) {
    double min_distance =
        std::min(min_distance_to_centers.at(pos),
                 l2_distance(instance.at(pos).first.second,
                             instance.at(centers->back()).first.second));
    *cost += std::pow(min_distance, 2) * instance.at(pos).second;
  }
}

double cost_solution(const std::vector<TimePointPair>& instance,
                     const std::vector<TimePointPair>& centers) {
  double cost = 0;
  for (const auto& point : instance) {
    double min_distance = std::numeric_limits<double>::max();
    for (const auto& center : centers) {
      min_distance =
          std::min(min_distance, l2_distance(point.second, center.second));
    }
    cost += std::pow(min_distance, 2);
  }

  return cost;
}

void cluster_assignment(const std::vector<TimePointPair>& instance,
                        const std::vector<TimePointPair>& centers,
                        std::vector<int32_t>* assignment) {
  assignment->clear();

  for (const auto& point : instance) {
    double min_distance = std::numeric_limits<double>::max();
    int32_t id = 0;
    for (int32_t i = 0; i < centers.size(); i++) {
      double d = l2_distance(point.second, centers[i].second);
      if (d < min_distance) {
        min_distance = d;
        id = i;
      }
    }
    assignment->push_back(id);
  }
}

}  //  namespace sliding_window
