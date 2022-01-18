// Copyright 2022 The Google Research Authors.
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

#ifndef SCANN_DISTANCE_MEASURES_ONE_TO_ONE_LIMITED_INNER_PRODUCT_H_
#define SCANN_DISTANCE_MEASURES_ONE_TO_ONE_LIMITED_INNER_PRODUCT_H_

#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/one_to_one/common.h"
#include "scann/distance_measures/one_to_one/dot_product.h"
#include "scann/distance_measures/one_to_one/l2_distance.h"
#include "scann/utils/types.h"

namespace research_scann {

class LimitedInnerProductDistance final : public DistanceMeasure {
 public:
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(LIMITED_INNER_PRODUCT);

 private:
  template <typename T>
  SCANN_INLINE double GetDistanceDenseImpl(const DatapointPtr<T>& a,
                                           const DatapointPtr<T>& b) const {
    const double sq_norm_a2 = SquaredL2Norm(a);
    const double sq_norm_b2 = SquaredL2Norm(b);
    const double denom =
        std::sqrt(sq_norm_a2 * std::max(sq_norm_a2, sq_norm_b2));
    if (denom == 0.0f) {
      return 0.0f;
    }
    return -DenseDotProduct(a, b) / denom;
  }

  template <typename T>
  SCANN_INLINE double GetDistanceSparseImpl(const DatapointPtr<T>& a,
                                            const DatapointPtr<T>& b) const {
    const double sq_norm_a2 = SquaredL2Norm(a);
    const double sq_norm_b2 = SquaredL2Norm(b);
    const double denom =
        std::sqrt(sq_norm_a2 * std::max(sq_norm_a2, sq_norm_b2));
    if (denom == 0.0f) {
      return 0.0f;
    }
    return -SparseDotProduct(a, b) / denom;
  }

  template <typename T>
  SCANN_INLINE double GetDistanceHybridImpl(const DatapointPtr<T>& a,
                                            const DatapointPtr<T>& b) const {
    const double sq_norm_a2 = SquaredL2Norm(a);
    const double sq_norm_b2 = SquaredL2Norm(b);
    const double denom =
        std::sqrt(sq_norm_a2 * std::max(sq_norm_a2, sq_norm_b2));
    if (denom == 0.0f) {
      return 0.0f;
    }
    return -HybridDotProduct(a, b) / denom;
  }
};

}  // namespace research_scann

#endif
