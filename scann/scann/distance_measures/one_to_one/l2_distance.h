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

#ifndef SCANN__DISTANCE_MEASURES_ONE_TO_ONE_L2_DISTANCE_H_
#define SCANN__DISTANCE_MEASURES_ONE_TO_ONE_L2_DISTANCE_H_

#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/one_to_one/common.h"
#include "scann/distance_measures/one_to_one/l2_distance_avx1.h"
#include "scann/distance_measures/one_to_one/l2_distance_sse4.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann/utils/reduction.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {

template <typename T, typename U>
double DenseSquaredL2Distance(const DatapointPtr<T>& a,
                              const DatapointPtr<U>& b);

template <typename T, typename U>
double SparseSquaredL2Distance(const DatapointPtr<T>& a,
                               const DatapointPtr<U>& b);

template <typename T, typename U>
double HybridSquaredL2Distance(const DatapointPtr<T>& a,
                               const DatapointPtr<U>& b);

template <typename T, typename U>
inline double SquaredL2DistanceBetween(const DatapointPtr<T>& a,
                                       const DatapointPtr<U>& b) {
  return SparseDenseDispatch(a, b, &SparseSquaredL2Distance<T, U>,
                             &DenseSquaredL2Distance<T, U>,
                             &HybridSquaredL2Distance<T, U>);
}

namespace l2_distance_internal {

struct Square {
  template <typename AT, typename T>
  void operator()(AT* acc, const T a) {
    const AT x = static_cast<AT>(a);
    *acc += x * x;
  }
};

}  // namespace l2_distance_internal

template <typename T>
inline double SquaredL2Norm(const DatapointPtr<T>& a) {
  return DenseSingleAccumulate(a.values_slice(),
                               l2_distance_internal::Square());
}

class SquaredL2Distance final : public DistanceMeasure {
 public:
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(SQUARED_L2);

 private:
  template <typename T>
  SCANN_INLINE double GetDistanceDenseImpl(const DatapointPtr<T>& a,
                                           const DatapointPtr<T>& b) const {
    return DenseSquaredL2Distance(a, b);
  }

  template <typename T>
  SCANN_INLINE double GetDistanceSparseImpl(const DatapointPtr<T>& a,
                                            const DatapointPtr<T>& b) const {
    return SparseSquaredL2Distance(a, b);
  }

  template <typename T>
  SCANN_INLINE double GetDistanceHybridImpl(const DatapointPtr<T>& a,
                                            const DatapointPtr<T>& b) const {
    return HybridSquaredL2Distance(a, b);
  }
};

class L2Distance final : public DistanceMeasure {
 public:
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(L2);

 private:
  template <typename T>
  SCANN_INLINE double GetDistanceDenseImpl(const DatapointPtr<T>& a,
                                           const DatapointPtr<T>& b) const {
    return std::sqrt(DenseSquaredL2Distance(a, b));
  }

  template <typename T>
  SCANN_INLINE double GetDistanceSparseImpl(const DatapointPtr<T>& a,
                                            const DatapointPtr<T>& b) const {
    return std::sqrt(SparseSquaredL2Distance(a, b));
  }

  template <typename T>
  SCANN_INLINE double GetDistanceHybridImpl(const DatapointPtr<T>& a,
                                            const DatapointPtr<T>& b) const {
    return std::sqrt(HybridSquaredL2Distance(a, b));
  }
};

class NegatedSquaredL2Distance final : public DistanceMeasure {
 public:
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(NEGATED_SQUARED_L2);

 private:
  template <typename T>
  SCANN_INLINE double GetDistanceDenseImpl(const DatapointPtr<T>& a,
                                           const DatapointPtr<T>& b) const {
    return -DenseSquaredL2Distance(a, b);
  }

  template <typename T>
  SCANN_INLINE double GetDistanceSparseImpl(const DatapointPtr<T>& a,
                                            const DatapointPtr<T>& b) const {
    return -SparseSquaredL2Distance(a, b);
  }

  template <typename T>
  SCANN_INLINE double GetDistanceHybridImpl(const DatapointPtr<T>& a,
                                            const DatapointPtr<T>& b) const {
    return -HybridSquaredL2Distance(a, b);
  }
};

struct SquaredL2ReduceTwo {
  template <typename Accumulator, typename T, typename U>
  void operator()(Accumulator* acc, const T a, const U b) {
    const Accumulator diff =
        static_cast<Accumulator>(a) - static_cast<Accumulator>(b);
    *acc += diff * diff;
  }
};

struct SquaredL2ReduceOne {
  template <typename Accumulator, typename T>
  void operator()(Accumulator* acc, const T a) {
    const Accumulator x = static_cast<Accumulator>(a);
    *acc += x * x;
  }

  bool IsNoop() { return false; }
};

template <typename T, typename U>
double HybridSquaredL2Distance(const DatapointPtr<T>& a,
                               const DatapointPtr<U>& b) {
  return HybridPairAccumulate(a, b, SquaredL2ReduceTwo(), SquaredL2ReduceOne());
}

template <typename T, typename U>
double SparseSquaredL2Distance(const DatapointPtr<T>& a,
                               const DatapointPtr<U>& b) {
  return SparsePairAccumulate(a, b, SquaredL2ReduceTwo(), SquaredL2ReduceOne());
}

template <typename T, typename U>
double DenseSquaredL2DistanceFallback(const DatapointPtr<T>& a,
                                      const DatapointPtr<U>& b) {
  return DensePairAccumulate(a.values(), b.values(), a.nonzero_entries(),
                             SquaredL2ReduceTwo());
}

template <typename T, typename U>
double DenseSquaredL2Distance(const DatapointPtr<T>& a,
                              const DatapointPtr<U>& b) {
  return DenseSquaredL2DistanceFallback(a, b);
}

#ifdef __x86_64__

template <>
inline double DenseSquaredL2Distance<uint8_t, uint8_t>(
    const DatapointPtr<uint8_t>& a, const DatapointPtr<uint8_t>& b) {
  if (RuntimeSupportsSse4()) {
    return l2_internal::DenseSquaredL2DistanceSse4(a, b);
  } else {
    return DenseSquaredL2DistanceFallback(a, b);
  }
}

template <>
inline double DenseSquaredL2Distance<int8_t, int8_t>(
    const DatapointPtr<int8_t>& a, const DatapointPtr<int8_t>& b) {
  if (RuntimeSupportsSse4()) {
    return l2_internal::DenseSquaredL2DistanceSse4(a, b);
  } else {
    return DenseSquaredL2DistanceFallback(a, b);
  }
}

template <>
inline double DenseSquaredL2Distance<float, float>(
    const DatapointPtr<float>& a, const DatapointPtr<float>& b) {
  if (RuntimeSupportsSse4()) {
    return l2_internal::DenseSquaredL2DistanceSse4(a, b);
  } else {
    return DenseSquaredL2DistanceFallback(a, b);
  }
}

template <>
inline double DenseSquaredL2Distance<double, double>(
    const DatapointPtr<double>& a, const DatapointPtr<double>& b) {
  if (RuntimeSupportsAvx1()) {
    return l2_internal::DenseSquaredL2DistanceAvx1(a, b);
  } else if (RuntimeSupportsSse4()) {
    return l2_internal::DenseSquaredL2DistanceSse4(a, b);
  } else {
    return DenseSquaredL2DistanceFallback(a, b);
  }
}

#endif

template <typename T, typename U>
double DenseSquaredL2Norm(const DatapointPtr<T>& a, const DatapointPtr<U>& b) {
  return DenseSquaredL2Distance(a, b);
}

}  // namespace scann_ops
}  // namespace tensorflow

#endif
