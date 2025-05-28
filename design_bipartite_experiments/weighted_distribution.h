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

#pragma once

#include <algorithm>
#include <cassert>
#include <random>
#include <utility>
#include <vector>

namespace exposure_design {

template <typename T>
class WeightedDistribution {
 public:
  // operator() returns t with probability proportional to weight. weight must
  // be positive. Must not be called after operator().
  void Add(T t, double weight) {
    assert(weight > 0);
    assert(!frozen_);
    weights_.push_back({t, weight});
  }

  template <typename Generator>
  T operator()(Generator& gen) {
    if (!frozen_) {
      frozen_ = true;
      weights_.shrink_to_fit();
      // Sort for numerical stability and so that the hot elements are last.
      std::sort(
          weights_.begin(), weights_.end(),
          [](const std::pair<T, double>& a, const std::pair<T, double>& b) {
            return a.second < b.second;
          });
      // Switch to cumulative weights.
      double sum{0};
      for (auto& [t, w_t] : weights_) {
        double new_sum{sum + w_t};
        w_t = sum;
        sum = new_sum;
      }
      sum_ = sum;
    }
    assert(sum_ > 0);
    assert(!weights_.empty());
    double x{std::uniform_real_distribution<double>{0, sum_}(gen)};
    for (auto it{weights_.rbegin()}; it != weights_.rend(); ++it)
      if (it->second <= x) return it->first;
    assert(false);
  }

 private:
  bool frozen_{false};
  // When not frozen, .second is the weight. When frozen, .second is the
  // cumulative weight.
  std::vector<std::pair<T, double>> weights_;
  // When frozen, sum of weights.
  double sum_{0};
};

}  // namespace exposure_design
