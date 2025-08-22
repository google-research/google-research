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

#ifndef SCANN_DISTANCE_MEASURES_ONE_TO_ONE_L2_DISTANCE_H_
#define SCANN_DISTANCE_MEASURES_ONE_TO_ONE_L2_DISTANCE_H_

#include <cstddef>
#include <cstdint>

#include "hwy/highway.h"
#include "scann/data_format/datapoint.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/one_to_one/common.h"
#include "scann/distance_measures/one_to_one/l2_distance_avx1.h"
#include "scann/distance_measures/one_to_one/l2_distance_sse4.h"
#include "scann/utils/common.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann/utils/reduction.h"
#include "scann/utils/types.h"

namespace research_scann {

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

template <typename FloatT>
double SquaredL2NormImplFloat(ConstSpan<FloatT> vec) {
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<FloatT>;
  const D d;
  using D2 = hn::ScalableTag<FloatT, -1>;
  const D2 d2;
  hn::Vec<D> a0 = hn::Zero(d);
  hn::Vec<D> a1 = hn::Zero(d);
  size_t i = 0;
  for (; i + 2 * hn::Lanes(d) < vec.size(); i += 2 * hn::Lanes(d)) {
    auto v0 = hn::LoadU(d, &vec[i]);
    auto v1 = hn::LoadU(d, &vec[i + hn::Lanes(d)]);
    a0 = hn::MulAdd(v0, v0, a0);
    a1 = hn::MulAdd(v1, v1, a1);
  }
  if (i + hn::Lanes(d) < vec.size()) {
    auto v0 = hn::LoadU(d, &vec[i]);
    a0 = hn::MulAdd(v0, v0, a0);
    i += hn::Lanes(d);
  }
  if (i + hn::Lanes(d2) < vec.size()) {
    auto v1 = hn::ZeroExtendVector(d, hn::LoadU(d2, &vec[i]));
    a1 = hn::MulAdd(v1, v1, a1);
    i += hn::Lanes(d2);
  }
  FloatT result = hn::ReduceSum(d, a0 + a1);
  for (; i < vec.size(); ++i) {
    result += vec[i] * vec[i];
  }
  return result;
}

}  // namespace l2_distance_internal

template <typename T>
inline double SquaredL2NormFast(ConstSpan<T> vec) {
  if constexpr (std::is_floating_point_v<T>) {
    return l2_distance_internal::SquaredL2NormImplFloat<T>(vec);
  } else {
    return DenseSingleAccumulate(vec, l2_distance_internal::Square());
  }
}

template <typename T>
inline double SquaredL2Norm(ConstSpan<T> vec) {
  return DenseSingleAccumulate(vec, l2_distance_internal::Square());
}

template <typename T>
inline double SquaredL2NormFast(const DatapointPtr<T>& a) {
  return SquaredL2NormFast(a.values_span());
}
template <typename T>
inline double SquaredL2Norm(const DatapointPtr<T>& a) {
  return SquaredL2Norm(a.values_span());
}

inline double SquaredL2NormFast(ConstSpan<float> vec) {
  return SquaredL2NormFast<float>(vec);
}
inline double SquaredL2Norm(ConstSpan<float> vec) {
  return SquaredL2Norm<float>(vec);
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

}  // namespace research_scann

#endif
