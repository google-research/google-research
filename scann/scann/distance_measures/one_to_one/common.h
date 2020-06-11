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

/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SCANN__DISTANCE_MEASURES_ONE_TO_ONE_COMMON_H_
#define SCANN__DISTANCE_MEASURES_ONE_TO_ONE_COMMON_H_

#include "scann/data_format/datapoint.h"
#include "scann/oss_wrappers/scann_bits.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {

template <typename T, typename U>
inline double SparseDenseDispatch(
    const DatapointPtr<T>& a, const DatapointPtr<U>& b,
    double (*sparse_fun)(const DatapointPtr<T>&, const DatapointPtr<U>&),
    double (*dense_fun)(const DatapointPtr<T>&, const DatapointPtr<U>&),
    double (*hybrid_fun)(const DatapointPtr<T>&, const DatapointPtr<U>&)) {
  const bool a_is_dense = a.IsDense();
  const bool b_is_dense = b.IsDense();
  const int n_dense = a_is_dense + b_is_dense;
  if (n_dense == 0) {
    return sparse_fun(a, b);
  } else if (n_dense == 1) {
    return hybrid_fun(a, b);
  } else {
    DCHECK_EQ(n_dense, 2);
    return dense_fun(a, b);
  }
}

inline DimensionIndex SparseDenseDispatch(
    const DatapointPtr<uint8_t>& a, const DatapointPtr<uint8_t>& b,
    DimensionIndex (*sparse_fun)(const DatapointPtr<uint8_t>&,
                                 const DatapointPtr<uint8_t>&),
    DimensionIndex (*dense_fun)(const DatapointPtr<uint8_t>&,
                                const DatapointPtr<uint8_t>&),
    DimensionIndex (*hybrid_fun)(const DatapointPtr<uint8_t>&,
                                 const DatapointPtr<uint8_t>&)) {
  const bool a_is_dense = a.IsDense();
  const bool b_is_dense = b.IsDense();
  const int n_dense = a_is_dense + b_is_dense;
  if (n_dense == 0) {
    return sparse_fun(a, b);
  } else if (n_dense == 1) {
    return hybrid_fun(a, b);
  } else {
    DCHECK_EQ(n_dense, 2);
    return dense_fun(a, b);
  }
}

#define SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(T)                    \
  double GetDistanceDense(const DatapointPtr<T>& a, const DatapointPtr<T>& b)  \
      const final;                                                             \
  double GetDistanceDense(const DatapointPtr<T>& a, const DatapointPtr<T>& b,  \
                          double threshold) const final;                       \
  double GetDistanceSparse(const DatapointPtr<T>& a, const DatapointPtr<T>& b) \
      const final;                                                             \
  double GetDistanceHybrid(const DatapointPtr<T>& a, const DatapointPtr<T>& b) \
      const final;

#define SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(TAG_NAME)   \
  string_view name() const final;                                  \
                                                                   \
  SpeciallyOptimizedDistanceTag specially_optimized_distance_tag() \
      const final {                                                \
    return TAG_NAME;                                               \
  }                                                                \
                                                                   \
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(int8_t)         \
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(uint8_t)        \
                                                                   \
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(int16_t)        \
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(uint16_t)       \
                                                                   \
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(int32_t)        \
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(uint32_t)       \
                                                                   \
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(int64_t)        \
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(unsigned long long_t)       \
                                                                   \
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(float)          \
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(double)

template <typename DistanceT, typename T>
SCANN_INLINE double GetDistanceDenseEarlyStopping(const DistanceT& dist,
                                                  const DatapointPtr<T>& a,
                                                  const DatapointPtr<T>& b,
                                                  double threshold,
                                                  size_t min_stop);

#define SCANN_DEFINE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(                     \
    CLASS, EARLY_STOPPING_GRANULARITY, T)                                    \
  double CLASS::GetDistanceDense(const DatapointPtr<T>& a,                   \
                                 const DatapointPtr<T>& b) const {           \
    return GetDistanceDenseImpl(a, b);                                       \
  }                                                                          \
  double CLASS::GetDistanceSparse(const DatapointPtr<T>& a,                  \
                                  const DatapointPtr<T>& b) const {          \
    return GetDistanceSparseImpl(a, b);                                      \
  }                                                                          \
  double CLASS::GetDistanceHybrid(const DatapointPtr<T>& a,                  \
                                  const DatapointPtr<T>& b) const {          \
    return GetDistanceHybridImpl(a, b);                                      \
  }                                                                          \
  double CLASS::GetDistanceDense(const DatapointPtr<T>& a,                   \
                                 const DatapointPtr<T>& b, double threshold) \
      const {                                                                \
    return GetDistanceDenseEarlyStopping(*this, a, b, threshold,             \
                                         EARLY_STOPPING_GRANULARITY);        \
  }

#define SCANN_DEFINE_DISTANCE_MEASURE_VIRTUAL_METHODS( \
    CLASS, EARLY_STOPPING_GRANULARITY)                 \
                                                       \
  SCANN_DEFINE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(     \
      CLASS, EARLY_STOPPING_GRANULARITY, int8_t)       \
  SCANN_DEFINE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(     \
      CLASS, EARLY_STOPPING_GRANULARITY, uint8_t)      \
                                                       \
  SCANN_DEFINE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(     \
      CLASS, EARLY_STOPPING_GRANULARITY, int16_t)      \
  SCANN_DEFINE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(     \
      CLASS, EARLY_STOPPING_GRANULARITY, uint16_t)     \
                                                       \
  SCANN_DEFINE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(     \
      CLASS, EARLY_STOPPING_GRANULARITY, int32_t)      \
  SCANN_DEFINE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(     \
      CLASS, EARLY_STOPPING_GRANULARITY, uint32_t)     \
                                                       \
  SCANN_DEFINE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(     \
      CLASS, EARLY_STOPPING_GRANULARITY, int64_t)      \
  SCANN_DEFINE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(     \
      CLASS, EARLY_STOPPING_GRANULARITY, unsigned long long_t)     \
                                                       \
  SCANN_DEFINE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(     \
      CLASS, EARLY_STOPPING_GRANULARITY, float)        \
  SCANN_DEFINE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(     \
      CLASS, EARLY_STOPPING_GRANULARITY, double)

enum : size_t {
  kEarlyStoppingNotSupported = numeric_limits<size_t>::max(),
};

template <typename DistanceT, typename T>
SCANN_INLINE double GetDistanceDenseEarlyStopping(const DistanceT& dist,
                                                  const DatapointPtr<T>& a,
                                                  const DatapointPtr<T>& b,
                                                  double threshold,
                                                  size_t min_stop) {
  if (min_stop == kEarlyStoppingNotSupported) {
    return dist.GetDistanceDense(a, b);
  }

  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());
  if (a.nonzero_entries() < min_stop) {
    return dist.GetDistanceDense(a, b);
  } else {
    const T* a_ptr = a.values();
    const T* b_ptr = b.values();
    size_t nonzero_entries = a.nonzero_entries();
    size_t half = nonzero_entries / 2;
    double sum = 0.0;
    while (nonzero_entries >= min_stop) {
      sum += dist.GetDistanceDense(MakeDatapointPtr(a_ptr, half),
                                   MakeDatapointPtr(b_ptr, half));
      if (sum > threshold) return sum;
      nonzero_entries -= half;
      a_ptr += half;
      b_ptr += half;
      half = nonzero_entries / 2;
    }

    return sum +
           dist.GetDistanceDense(MakeDatapointPtr(a_ptr, nonzero_entries),
                                 MakeDatapointPtr(b_ptr, nonzero_entries));
  }
}

#define SCANN_REGISTER_DISTANCE_MEASURE(CLASS) \
  string_view CLASS::name() const { return #CLASS; }

}  // namespace scann_ops
}  // namespace tensorflow

#endif
