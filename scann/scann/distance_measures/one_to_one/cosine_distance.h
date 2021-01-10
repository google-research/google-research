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

#ifndef SCANN_DISTANCE_MEASURES_ONE_TO_ONE_COSINE_DISTANCE_H_
#define SCANN_DISTANCE_MEASURES_ONE_TO_ONE_COSINE_DISTANCE_H_

#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/one_to_one/binary_distance_measure_base.h"
#include "scann/distance_measures/one_to_one/common.h"
#include "scann/distance_measures/one_to_one/dot_product.h"
#include "scann/utils/types.h"

namespace research_scann {

class CosineDistance final : public DistanceMeasure {
 public:
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(COSINE);

  Normalization NormalizationRequired() const final { return UNITL2NORM; }

 private:
  template <typename T>
  SCANN_INLINE double GetDistanceDenseImpl(const DatapointPtr<T>& a,
                                           const DatapointPtr<T>& b) const {
    return 1.0 - DenseDotProduct(a, b);
  }

  template <typename T>
  SCANN_INLINE double GetDistanceSparseImpl(const DatapointPtr<T>& a,
                                            const DatapointPtr<T>& b) const {
    return 1.0 - SparseDotProduct(a, b);
  }

  template <typename T>
  SCANN_INLINE double GetDistanceHybridImpl(const DatapointPtr<T>& a,
                                            const DatapointPtr<T>& b) const {
    return 1.0 - HybridDotProduct(a, b);
  }
};

class BinaryCosineDistance final : public BinaryDistanceMeasureBase {
 public:
  string_view name() const final;

  using BinaryDistanceMeasureBase::GetDistanceDense;
  using BinaryDistanceMeasureBase::GetDistanceHybrid;
  using BinaryDistanceMeasureBase::GetDistanceSparse;

  double GetDistanceDense(const DatapointPtr<uint8_t>& a,
                          const DatapointPtr<uint8_t>& b) const final;
  double GetDistanceSparse(const DatapointPtr<uint8_t>& a,
                           const DatapointPtr<uint8_t>& b) const final;
  double GetDistanceHybrid(const DatapointPtr<uint8_t>& a,
                           const DatapointPtr<uint8_t>& b) const final;
};

}  // namespace research_scann

#endif
