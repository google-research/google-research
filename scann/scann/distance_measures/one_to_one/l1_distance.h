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

#ifndef SCANN__DISTANCE_MEASURES_ONE_TO_ONE_L1_DISTANCE_H_
#define SCANN__DISTANCE_MEASURES_ONE_TO_ONE_L1_DISTANCE_H_

#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/one_to_one/common.h"
#include "scann/distance_measures/one_to_one/l1_distance_sse4.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann/utils/reduction.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {

template <typename T, typename U>
double SparseL1Norm(const DatapointPtr<T>& a, const DatapointPtr<U>& b);

template <typename T, typename U>
double DenseL1Norm(const DatapointPtr<T>& a, const DatapointPtr<U>& b);

template <typename T, typename U>
double HybridL1Norm(const DatapointPtr<T>& a, const DatapointPtr<U>& b);

template <typename T, typename U>
inline double L1Norm(const DatapointPtr<T>& a, const DatapointPtr<U>& b) {
  return SparseDenseDispatch(a, b, &SparseL1Norm<T, U>, &DenseL1Norm<T, U>,
                             &HybridL1Norm<T, U>);
}

struct L1ReduceOne {
  template <typename Accumulator, typename T>
  void operator()(Accumulator* acc, const T x) {
    *acc += ScannAbs(static_cast<Accumulator>(x));
  }

  bool IsNoop() { return false; }
};

template <typename T>
inline double L1Norm(const DatapointPtr<T>& a) {
  return DenseSingleAccumulate(a.values_slice(), L1ReduceOne());
}

class L1Distance final : public DistanceMeasure {
 public:
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(L1);

 private:
  template <typename T>
  SCANN_INLINE double GetDistanceDenseImpl(const DatapointPtr<T>& a,
                                           const DatapointPtr<T>& b) const {
    return DenseL1Norm(a, b);
  }

  template <typename T>
  SCANN_INLINE double GetDistanceSparseImpl(const DatapointPtr<T>& a,
                                            const DatapointPtr<T>& b) const {
    return SparseL1Norm(a, b);
  }

  template <typename T>
  SCANN_INLINE double GetDistanceHybridImpl(const DatapointPtr<T>& a,
                                            const DatapointPtr<T>& b) const {
    return HybridL1Norm(a, b);
  }
};

struct L1ReduceTwo {
  template <typename Accumulator, typename T, typename U>
  void operator()(Accumulator* acc, const T a, const U b) {
    *acc += ScannAbs(static_cast<Accumulator>(a) - static_cast<Accumulator>(b));
  }
};

template <typename T, typename U>
double SparseL1Norm(const DatapointPtr<T>& a, const DatapointPtr<U>& b) {
  return SparsePairAccumulate(a, b, L1ReduceTwo(), L1ReduceOne());
}

template <typename T, typename U>
double DenseL1NormFallback(const DatapointPtr<T>& a, const DatapointPtr<U>& b) {
  return DensePairAccumulate(a.values(), b.values(), a.nonzero_entries(),
                             L1ReduceTwo());
}

template <typename T, typename U>
double DenseL1Norm(const DatapointPtr<T>& a, const DatapointPtr<U>& b) {
  return DenseL1NormFallback(a, b);
}

#ifdef __x86_64__

template <>
inline double DenseL1Norm<float, float>(const DatapointPtr<float>& a,
                                        const DatapointPtr<float>& b) {
  if (RuntimeSupportsSse4()) {
    return l1_internal::DenseL1NormSse4(a, b);
  } else {
    return DenseL1NormFallback(a, b);
  }
}

template <>
inline double DenseL1Norm<double, double>(const DatapointPtr<double>& a,
                                          const DatapointPtr<double>& b) {
  if (RuntimeSupportsSse4()) {
    return l1_internal::DenseL1NormSse4(a, b);
  } else {
    return DenseL1NormFallback(a, b);
  }
}

#endif

template <typename T, typename U>
double HybridL1Norm(const DatapointPtr<T>& a, const DatapointPtr<U>& b) {
  return HybridPairAccumulate(a, b, L1ReduceTwo(), L1ReduceOne());
}

}  // namespace scann_ops
}  // namespace tensorflow

#endif
