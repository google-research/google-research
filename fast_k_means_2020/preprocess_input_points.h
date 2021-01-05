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

// Preprocessing the input before tree construction.

#ifndef FAST_K_MEANS_2020_PREPROCESS_INPUT_POINTS_H_
#define FAST_K_MEANS_2020_PREPROCESS_INPUT_POINTS_H_

#include <cstdint>
#include <vector>

#include "random_handler.h"

namespace fast_k_means {

class PreProcessInputPoints {
 public:
  // Shifts the minimum coordinate to zero for each coordinate independently.
  static void ShiftToDimensionsZero(
      std::vector<std::vector<int>>* input_points);

  // Scales the input points to integers by multiplying the coordinates by
  // scaling_factor, and then removing the fractional part.
  static std::vector<std::vector<int>> ScaleToIntSpace(
      const std::vector<std::vector<double>>& input_point_double,
      double scaling_factor);

  // Adds a value to all the input points.
  // Notice that this does not affect the objective value.
  static void RandomShiftSpace(std::vector<std::vector<int>>* input_points);
};

}  //  namespace fast_k_means
#endif  // FAST_K_MEANS_2020_PREPROCESS_INPUT_POINTS_H_
