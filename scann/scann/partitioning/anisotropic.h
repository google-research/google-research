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

#ifndef SCANN_PARTITIONING_ANISOTROPIC_H_
#define SCANN_PARTITIONING_ANISOTROPIC_H_

#include <utility>

#include "Eigen/Dense"
#include "gtest/gtest_prod.h"
#include "scann/utils/common.h"
#include "scann/utils/linear_algebra/types.h"

namespace research_scann {

class AvqAccumulator {
 public:
  AvqAccumulator(size_t dimensionality, float eta);

  AvqAccumulator& AddVectors(ConstSpan<float> vecs);

  EVectorXf GetCenter();

 private:
  const size_t dimensionality_;
  const float eta_;

  EMatrixXf xtx_matrix_;

  EVectorXf weighted_vector_sum_;

  float total_weight_;

  FRIEND_TEST(AVQTest, TestReduction);
};

inline EVectorXf ComputeAVQPartition(ConstSpan<float> partition,
                                     size_t dimensionality, float eta) {
  return AvqAccumulator(dimensionality, eta).AddVectors(partition).GetCenter();
}

std::pair<double, double> ComputeRescaleFraction(
    ConstSpan<float> partition_center, ConstSpan<float> partition_data);

}  // namespace research_scann

#endif
