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

#ifndef SCANN_DISTANCE_MEASURES_ONE_TO_ONE_NONZERO_INTERSECT_DISTANCE_H_
#define SCANN_DISTANCE_MEASURES_ONE_TO_ONE_NONZERO_INTERSECT_DISTANCE_H_

#include "scann/data_format/datapoint.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/one_to_one/common.h"
#include "scann/distance_measures/one_to_one/dot_product.h"
#include "scann/utils/reduction.h"
#include "tensorflow/core/platform/logging.h"

namespace research_scann {

class NonzeroIntersectDistance final : public DistanceMeasure {
 public:
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(NOT_SPECIALLY_OPTIMIZED);

 private:
  template <typename T>
  SCANN_INLINE double GetDistanceDenseImpl(const DatapointPtr<T>& a,
                                           const DatapointPtr<T>& b) const {
    DCHECK_EQ(a.dimensionality(), b.dimensionality());
    DimensionIndex result = 0;
    for (DimensionIndex i : Seq(a.nonzero_entries())) {
      result += a.values()[i] && b.values()[i];
    }
    return -static_cast<double>(result);
  }

  template <typename T>
  SCANN_INLINE double GetDistanceSparseImpl(const DatapointPtr<T>& a,
                                            const DatapointPtr<T>& b) const {
    struct Sub1Reduce {
      void operator()(AccumulatorTypeFor<T>* acc, const T a, const T b) {
        *acc += (a && b);
      }
    };
    return -static_cast<double>(
        SparsePairAccumulate(a, b, Sub1Reduce(), DoNothingReduce()));
  }

  template <typename T>
  SCANN_INLINE double GetDistanceHybridImpl(const DatapointPtr<T>& a,
                                            const DatapointPtr<T>& b) const {
    DCHECK(a.IsSparse() != b.IsSparse());
    if (b.IsSparse()) return GetDistanceHybridImpl(b, a);
    DCHECK(a.IsSparse());
    DCHECK(b.IsDense());
    DimensionIndex result = 0;
    for (DimensionIndex i : Seq(a.nonzero_entries())) {
      const DimensionIndex dim = a.indices()[i];
      result += (a.values()[i] && b.values()[dim]);
    }
    return -static_cast<double>(result);
  }
};

}  // namespace research_scann

#endif
