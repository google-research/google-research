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

#ifndef SCANN_UTILS_NOISE_SHAPING_UTILS_H_
#define SCANN_UTILS_NOISE_SHAPING_UTILS_H_

#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
T Square(T x) {
  return x * x;
}

inline double ComputeParallelCostMultiplier(double threshold,
                                            double squared_l2_norm,
                                            DimensionIndex dims) {
  const double parallel_cost = Square(threshold) / squared_l2_norm;
  const double perpendicular_cost =
      (1.0 - Square(threshold) / squared_l2_norm) / (dims - 1.0);
  return parallel_cost / perpendicular_cost;
}

}  // namespace research_scann

#endif
