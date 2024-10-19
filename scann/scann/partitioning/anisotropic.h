// Copyright 2024 The Google Research Authors.
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
#include "scann/utils/common.h"
#include "scann/utils/linear_algebra/types.h"

namespace research_scann {

struct AccumulatedAVQData {
  EMatrixXf xtx;

  EVectorXf norm_weighted_dp_sum;

  float dp_norm_sum;

  float eta;
};

AccumulatedAVQData ReducePartition(ConstSpan<float> partition,
                                   size_t dimensionality, float eta);

EVectorXf ComputeAVQPartition(const AccumulatedAVQData& avq_data);

EVectorXf ComputeAVQPartition(ConstSpan<float> partition, size_t dimensionality,
                              float eta);

std::pair<double, double> ComputeRescaleFraction(
    ConstSpan<float> partition_center, ConstSpan<float> partition_data);

}  // namespace research_scann

#endif
