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

#ifndef SCANN__DISTANCE_MEASURES_ONE_TO_ONE_DOT_PRODUCT_H_
#define SCANN__DISTANCE_MEASURES_ONE_TO_ONE_DOT_PRODUCT_H_

#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/one_to_one/binary_distance_measure_base.h"
#include "scann/distance_measures/one_to_one/common.h"
#include "scann/distance_measures/one_to_one/dot_product_avx1.h"
#include "scann/distance_measures/one_to_one/dot_product_avx2.h"
#include "scann/distance_measures/one_to_one/dot_product_sse4.h"
#include "scann/utils/intrinsics/flags.h"
#include "scann/utils/reduction.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {

template <typename T, typename U>
double SparseDotProduct(const DatapointPtr<T>& a, const DatapointPtr<U>& b);

template <typename T, typename U>
double DenseDotProduct(const DatapointPtr<T>& a, const DatapointPtr<U>& b);

template <typename T, typename U>
double HybridDotProduct(const DatapointPtr<T>& a, const DatapointPtr<U>& b);

template <typename T, typename U>
double DotProduct(const DatapointPtr<T>& a, const DatapointPtr<U>& b) {
  return SparseDenseDispatch(a, b, &SparseDotProduct<T>, &DenseDotProduct<T>,
                             &HybridDotProduct<T>);
}

inline DimensionIndex BinaryDotProduct(const DatapointPtr<uint8_t>& a,
                                       const DatapointPtr<uint8_t>& b);
inline DimensionIndex DenseBinaryDotProduct(const DatapointPtr<uint8_t>& a,
                                            const DatapointPtr<uint8_t>& b);
inline DimensionIndex SparseBinaryDotProduct(const DatapointPtr<uint8_t>& a,
                                             const DatapointPtr<uint8_t>& b);
inline DimensionIndex HybridBinaryDotProduct(const DatapointPtr<uint8_t>& a,
                                             const DatapointPtr<uint8_t>& b);

class DotProductDistance final : public DistanceMeasure {
 public:
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(DOT_PRODUCT);

 private:
  template <typename T>
  SCANN_INLINE double GetDistanceDenseImpl(const DatapointPtr<T>& a,
                                           const DatapointPtr<T>& b) const {
    return -DenseDotProduct(a, b);
  }

  template <typename T>
  SCANN_INLINE double GetDistanceSparseImpl(const DatapointPtr<T>& a,
                                            const DatapointPtr<T>& b) const {
    return -SparseDotProduct(a, b);
  }

  template <typename T>
  SCANN_INLINE double GetDistanceHybridImpl(const DatapointPtr<T>& a,
                                            const DatapointPtr<T>& b) const {
    return -HybridDotProduct(a, b);
  }
};

class AbsDotProductDistance final : public DistanceMeasure {
 public:
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(ABS_DOT_PRODUCT);

 private:
  template <typename T>
  SCANN_INLINE double GetDistanceDenseImpl(const DatapointPtr<T>& a,
                                           const DatapointPtr<T>& b) const {
    return -std::abs(DenseDotProduct(a, b));
  }

  template <typename T>
  SCANN_INLINE double GetDistanceSparseImpl(const DatapointPtr<T>& a,
                                            const DatapointPtr<T>& b) const {
    return -std::abs(SparseDotProduct(a, b));
  }

  template <typename T>
  SCANN_INLINE double GetDistanceHybridImpl(const DatapointPtr<T>& a,
                                            const DatapointPtr<T>& b) const {
    return -std::abs(HybridDotProduct(a, b));
  }
};

class BinaryDotProductDistance final : public BinaryDistanceMeasureBase {
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
                           const DatapointPtr<uint8_t>& b) const final {
    return GetDistanceSparse(a, b);
  }
};

struct DotProductReduce {
  template <typename Accumulator, typename T, typename U>
  void operator()(Accumulator* acc, const T a, const U b) {
    *acc += static_cast<Accumulator>(a) * static_cast<Accumulator>(b);
  }
};

template <typename T, typename U>
double SparseDotProduct(const DatapointPtr<T>& a, const DatapointPtr<U>& b) {
  return SparsePairAccumulate(a, b, DotProductReduce(), DoNothingReduce());
}

template <typename T, typename U>
double DenseDotProductFallback(const DatapointPtr<T>& a,
                               const DatapointPtr<U>& b) {
  return DensePairAccumulate(a.values(), b.values(), a.nonzero_entries(),
                             DotProductReduce());
}

template <typename T, typename U>
double DenseDotProduct(const DatapointPtr<T>& a, const DatapointPtr<U>& b) {
  return DenseDotProductFallback(a, b);
}

template <typename T, typename U, typename V>
double DenseDotProductFallback(const DatapointPtr<T>& a,
                               const DatapointPtr<U>& b,
                               const DatapointPtr<V>& c) {
  using AT = AccumulatorTypeFor<T, U, V>;
  AT acc = 0;
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());
  DCHECK_EQ(a.nonzero_entries(), c.nonzero_entries());
  const T* aptr = a.values();
  const U* bptr = b.values();
  const V* cptr = c.values();

  for (size_t i : Seq(a.nonzero_entries())) {
    acc += static_cast<AT>(aptr[i]) * static_cast<AT>(bptr[i]) *
           static_cast<AT>(cptr[i]);
  }
  return static_cast<double>(acc);
}

template <typename T, typename U, typename V>
double DenseDotProduct(const DatapointPtr<T>& a, const DatapointPtr<U>& b,
                       const DatapointPtr<V>& c) {
  return DenseDotProductFallback(a, b, c);
}

#ifdef __x86_64__

template <>
inline double DenseDotProduct<uint8_t, uint8_t>(
    const DatapointPtr<uint8_t>& a, const DatapointPtr<uint8_t>& b) {
  if (RuntimeSupportsSse4()) {
    return dp_internal::DenseDotProductSse4(a, b);
  } else {
    return DenseDotProductFallback(a, b);
  }
}

template <>
inline double DenseDotProduct<int8_t, int8_t>(const DatapointPtr<int8_t>& a,
                                              const DatapointPtr<int8_t>& b) {
  if (RuntimeSupportsSse4()) {
    return dp_internal::DenseDotProductSse4(a, b);
  } else {
    return DenseDotProductFallback(a, b);
  }
}

template <>
inline double DenseDotProduct<float, float>(const DatapointPtr<float>& a,
                                            const DatapointPtr<float>& b) {
  if (RuntimeSupportsSse4()) {
    return dp_internal::DenseDotProductSse4(a, b);
  } else {
    return DenseDotProductFallback(a, b);
  }
}

template <>
inline double DenseDotProduct<double, double>(const DatapointPtr<double>& a,
                                              const DatapointPtr<double>& b) {
  if (RuntimeSupportsSse4()) {
    return dp_internal::DenseDotProductSse4(a, b);
  } else {
    return DenseDotProductFallback(a, b);
  }
}

template <>
inline double DenseDotProduct<int8_t, float>(const DatapointPtr<int8_t>& a,
                                             const DatapointPtr<float>& b) {
  if (RuntimeSupportsAvx2()) {
    return dp_internal::DenseDotProductAvx2(a, b);
  } else if (RuntimeSupportsAvx1()) {
    return dp_internal::DenseDotProductAvx1(a, b);
  } else if (RuntimeSupportsSse4()) {
    return dp_internal::DenseDotProductSse4(a, b);
  } else {
    return DenseDotProductFallback(a, b);
  }
}

template <>
inline double DenseDotProduct<float, int8_t>(const DatapointPtr<float>& a,
                                             const DatapointPtr<int8_t>& b) {
  return DenseDotProduct(b, a);
}

template <>
inline double DenseDotProduct<int8_t, float, float>(
    const DatapointPtr<int8_t>& a, const DatapointPtr<float>& b,
    const DatapointPtr<float>& c) {
  if (RuntimeSupportsAvx2()) {
    return dp_internal::DenseDotProductAvx2(a, b, c);
  } else if (RuntimeSupportsAvx1()) {
    return dp_internal::DenseDotProductAvx1(a, b, c);
  } else if (RuntimeSupportsSse4()) {
    return dp_internal::DenseDotProductSse4(a, b, c);
  } else {
    return DenseDotProductFallback(a, b, c);
  }
}

template <>
inline double DenseDotProduct<float, int8_t, float>(
    const DatapointPtr<float>& a, const DatapointPtr<int8_t>& b,
    const DatapointPtr<float>& c) {
  return DenseDotProduct(b, a, c);
}

template <>
inline double DenseDotProduct<float, float, int8_t>(
    const DatapointPtr<float>& a, const DatapointPtr<float>& b,
    const DatapointPtr<int8_t>& c) {
  return DenseDotProduct(c, a, b);
}

template <>
inline double DenseDotProduct<int8_t, int8_t, float>(
    const DatapointPtr<int8_t>& a, const DatapointPtr<int8_t>& b,
    const DatapointPtr<float>& c) {
  if (RuntimeSupportsAvx2()) {
    return dp_internal::DenseDotProductAvx2(a, b, c);
  } else if (RuntimeSupportsAvx1()) {
    return dp_internal::DenseDotProductAvx1(a, b, c);
  } else if (RuntimeSupportsSse4()) {
    return dp_internal::DenseDotProductSse4(a, b, c);
  } else {
    return DenseDotProductFallback(a, b, c);
  }
}

template <>
inline double DenseDotProduct<int8_t, float, int8_t>(
    const DatapointPtr<int8_t>& a, const DatapointPtr<float>& b,
    const DatapointPtr<int8_t>& c) {
  return DenseDotProduct(a, c, b);
}

template <>
inline double DenseDotProduct<float, int8_t, int8_t>(
    const DatapointPtr<float>& a, const DatapointPtr<int8_t>& b,
    const DatapointPtr<int8_t>& c) {
  return DenseDotProduct(c, b, a);
}

#endif

template <typename T, typename U>
double HybridDotProduct(const DatapointPtr<T>& a, const DatapointPtr<U>& b) {
  return HybridPairAccumulate(a, b, DotProductReduce(), DoNothingReduce());
}

inline DimensionIndex BinaryDotProduct(const DatapointPtr<uint8_t>& a,
                                       const DatapointPtr<uint8_t>& b) {
  return SparseDenseDispatch(a, b, &SparseBinaryDotProduct,
                             &DenseBinaryDotProduct, &HybridBinaryDotProduct);
}

class DenseBinaryDotProductAnd {
 public:
  template <typename T>
  T operator()(T a, T b) {
    return a & b;
  }
};

inline DimensionIndex DenseBinaryDotProduct(const DatapointPtr<uint8_t>& a,
                                            const DatapointPtr<uint8_t>& b) {
  return DenseBinaryMergeAndPopcnt(a, b, DenseBinaryDotProductAnd());
}

inline DimensionIndex SparseBinaryDotProduct(const DatapointPtr<uint8_t>& a,
                                             const DatapointPtr<uint8_t>& b) {
  if (a.nonzero_entries() == 0 || b.nonzero_entries() == 0) {
    return 0;
  }
  DimensionIndex intersection = 0;
  size_t a_front = 0, b_front = 0;
  size_t a_back = a.nonzero_entries() - 1, b_back = b.nonzero_entries() - 1;
  while (a_front < a_back && b_front < b_back) {
    const size_t to_add_front1 = a.indices()[a_front] <= b.indices()[b_front];
    const size_t to_add_front2 = a.indices()[a_front] >= b.indices()[b_front];
    const size_t to_sub_back2 = a.indices()[a_back] <= b.indices()[b_back];
    const size_t to_sub_back1 = a.indices()[a_back] >= b.indices()[b_back];
    intersection += a.indices()[a_front] == b.indices()[b_front];
    intersection += a.indices()[a_back] == b.indices()[b_back];
    a_front += to_add_front1;
    b_front += to_add_front2;
    a_back -= to_sub_back1;
    b_back -= to_sub_back2;
  }
  if (a_front == a_back) {
    for (; b_front <= b_back; ++b_front) {
      if (ABSL_PREDICT_FALSE(a.indices()[a_front] == b.indices()[b_front])) {
        intersection++;
        break;
      }
    }
  } else if (b_front == b_back) {
    for (; a_front <= a_back; ++a_front) {
      if (ABSL_PREDICT_FALSE(a.indices()[a_front] == b.indices()[b_front])) {
        intersection++;
        break;
      }
    }
  }
  return intersection;
}

inline DimensionIndex HybridBinaryDotProduct(const DatapointPtr<uint8_t>& a,
                                             const DatapointPtr<uint8_t>& b) {
  LOG(FATAL) << "Hybrid binary dot product not yet implemented.";
}

}  // namespace scann_ops
}  // namespace tensorflow

#endif
