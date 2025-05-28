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

#ifndef SCANN_PARTITIONING_ORTHOGONALITY_AMPLIFICATION_UTILS_H_
#define SCANN_PARTITIONING_ORTHOGONALITY_AMPLIFICATION_UTILS_H_

#include <algorithm>
#include <cmath>

#include "scann/data_format/datapoint.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
void ComputeNormalizedResidual(DatapointPtr<T> dptr,
                               DatapointPtr<float> centroid,
                               MutableSpan<float> result) {
  DCHECK_EQ(dptr.dimensionality(), centroid.dimensionality());
  DCHECK_EQ(centroid.dimensionality(), result.size());
  double sqnorm = 0.0;
  for (size_t i : Seq(dptr.dimensionality())) {
    result[i] = static_cast<float>(dptr.values()[i]) - centroid.values()[i];
    sqnorm += double{result[i]} * double{result[i]};
  }
  if (sqnorm < 1e-7) {
    std::fill(result.begin(), result.end(), 0.0);
    return;
  }
  const float inv_norm = 1.0 / std::sqrt(sqnorm);
  for (float& f : result) {
    f *= inv_norm;
  }
}

}  // namespace research_scann

#endif
