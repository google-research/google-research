// Copyright 2020 The Google Research Authors.
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

#ifndef SCANN__DISTANCE_MEASURES_DISTANCE_MEASURE_BASE_H_
#define SCANN__DISTANCE_MEASURES_DISTANCE_MEASURE_BASE_H_

#include "scann/data_format/datapoint.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {

class DistanceMeasure : public VirtualDestructor {
 public:
  virtual string_view name() const = 0;

  virtual Normalization NormalizationRequired() const { return NONE; }

  enum SpeciallyOptimizedDistanceTag {
    L1,
    L2,
    SQUARED_L2,
    COSINE,
    DOT_PRODUCT,
    ABS_DOT_PRODUCT,
    LIMITED_INNER_PRODUCT,
    GENERAL_HAMMING,
    NEGATED_SQUARED_L2,
    NOT_SPECIALLY_OPTIMIZED
  };

  virtual SpeciallyOptimizedDistanceTag specially_optimized_distance_tag()
      const {
    return NOT_SPECIALLY_OPTIMIZED;
  }

  template <typename T>
  double GetDistance(const DatapointPtr<T>& a, const DatapointPtr<T>& b) const {
    const bool a_is_dense = a.IsDense();
    const bool b_is_dense = b.IsDense();
    const int n_dense = a_is_dense + b_is_dense;
    if (n_dense == 0) {
      return GetDistanceSparse(a, b);
    } else if (n_dense == 1) {
      return GetDistanceHybrid(a, b);
    } else {
      DCHECK_EQ(n_dense, 2);
      return GetDistanceDense(a, b);
    }
  }

  template <typename T>
  double GetDistance(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                     double threshold) const {
    const bool a_is_dense = a.IsDense();
    const bool b_is_dense = b.IsDense();
    const int n_dense = a_is_dense + b_is_dense;
    if (n_dense == 0) {
      return GetDistanceSparse(a, b);
    } else if (n_dense == 1) {
      return GetDistanceHybrid(a, b);
    } else {
      DCHECK_EQ(n_dense, 2);
      return GetDistanceDense(a, b, threshold);
    }
  }

#define SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(T)                     \
  virtual double GetDistanceDense(const DatapointPtr<T>& a,                   \
                                  const DatapointPtr<T>& b) const = 0;        \
  virtual double GetDistanceDense(const DatapointPtr<T>& a,                   \
                                  const DatapointPtr<T>& b, double threshold) \
      const = 0;                                                              \
  virtual double GetDistanceSparse(const DatapointPtr<T>& a,                  \
                                   const DatapointPtr<T>& b) const = 0;       \
  virtual double GetDistanceHybrid(const DatapointPtr<T>& a,                  \
                                   const DatapointPtr<T>& b) const = 0;

  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(int8_t);
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(uint8_t);

  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(int16_t);
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(uint16_t);

  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(int32_t);
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(uint32_t);

  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(int64_t);
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(uint64_t);

  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(float);
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(double);

#undef SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS

 private:
  virtual void UnusedKeyMethod();
};

}  // namespace scann_ops
}  // namespace tensorflow

#endif
