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

#ifndef SCANN_UTILS_RANDOM_RESERVOIR_SAMPLING_H_
#define SCANN_UTILS_RANDOM_RESERVOIR_SAMPLING_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "absl/random/distributions.h"
#include "absl/random/uniform_int_distribution.h"
#include "absl/random/uniform_real_distribution.h"

namespace research_scann {

template <typename T>
std::vector<size_t> ReservoirSampleIdxs(T&& rng, size_t n, size_t sample_size) {
  std::vector<size_t> result(std::min(n, sample_size));
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = i;
  }

  absl::uniform_int_distribution<size_t> uniform_idx(0, sample_size - 1);
  double inv_sample_size = 1.0 / sample_size;

  double u = absl::Uniform(absl::IntervalOpenClosed, rng, 0.0, 1.0);
  double w = std::exp(std::log(u) * inv_sample_size);
  for (size_t i = result.size(); i < n; ++i) {
    u = absl::Uniform(absl::IntervalOpenClosed, rng, 0.0, 1.0);
    double skip_d = std::floor(std::log(u) / std::log1p(-w));
    size_t skip;
    if (std::isfinite(skip_d) &&
        skip_d < static_cast<double>(std::numeric_limits<size_t>::max())) {
      skip = skip_d;
    } else {
      skip = std::numeric_limits<size_t>::max();
    }
    i += skip;
    if (i >= n) break;
    size_t replace_idx = uniform_idx(rng);
    result[replace_idx] = i;

    u = absl::Uniform(absl::IntervalOpenClosed, rng, 0.0, 1.0);
    w *= std::exp(std::log(u) * inv_sample_size);
  }
  return result;
}
}  // namespace research_scann

#endif
