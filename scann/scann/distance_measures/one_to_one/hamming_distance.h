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

#ifndef SCANN_DISTANCE_MEASURES_ONE_TO_ONE_HAMMING_DISTANCE_H_
#define SCANN_DISTANCE_MEASURES_ONE_TO_ONE_HAMMING_DISTANCE_H_

#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/one_to_one/binary_distance_measure_base.h"
#include "scann/distance_measures/one_to_one/common.h"
#include "scann/utils/reduction.h"
#include "scann/utils/types.h"

namespace research_scann {

inline DimensionIndex DenseBinaryHammingDistance(
    const DatapointPtr<uint8_t>& a, const DatapointPtr<uint8_t>& b);

class GeneralHammingDistance final : public DistanceMeasure {
 public:
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(GENERAL_HAMMING);

 private:
  template <typename T, typename Accumulator>
  SCANN_INLINE static DimensionIndex GetDistanceDenseAutoVectorized(
      const T* a, const T* b, size_t size);

  template <typename T>
  SCANN_INLINE static DimensionIndex GetDistanceUnrolled(const T* a, const T* b,
                                                         size_t size);

  template <typename T>
  SCANN_INLINE static double GetDistanceDenseImpl(const DatapointPtr<T>& a,
                                                  const DatapointPtr<T>& b);

  template <typename T>
  SCANN_INLINE double GetDistanceSparseImpl(const DatapointPtr<T>& a,
                                            const DatapointPtr<T>& b) const {
    LOG(FATAL) << "Sparse general Hamming distance not implemented yet.";
  }

  template <typename T>
  SCANN_INLINE double GetDistanceHybridImpl(const DatapointPtr<T>& a,
                                            const DatapointPtr<T>& b) const {
    LOG(FATAL) << "Hybrid general Hamming distance not implemented yet.";
  }
};

class BinaryHammingDistance final : public BinaryDistanceMeasureBase {
  string_view name() const final;

  using BinaryDistanceMeasureBase::GetDistanceDense;
  using BinaryDistanceMeasureBase::GetDistanceHybrid;
  using BinaryDistanceMeasureBase::GetDistanceSparse;

  double GetDistanceDense(const DatapointPtr<uint8_t>& a,
                          const DatapointPtr<uint8_t>& b) const final;
  double GetDistanceSparse(const DatapointPtr<uint8_t>& a,
                           const DatapointPtr<uint8_t>& b) const final {
    LOG(FATAL) << "Sparse binary Hamming distance not implemented yet.";
  }
  double GetDistanceHybrid(const DatapointPtr<uint8_t>& a,
                           const DatapointPtr<uint8_t>& b) const final {
    LOG(FATAL) << "Hybrid binary Hamming distance not implemented yet.";
  }
};

template <typename T, typename Accumulator>
DimensionIndex GeneralHammingDistance::GetDistanceDenseAutoVectorized(
    const T* a, const T* b, size_t size) {
  DimensionIndex result = 0;
  constexpr size_t block_size = numeric_limits<Accumulator>::max();
  size_t remaining_size = size;

  while (sizeof(Accumulator) < sizeof(result) && remaining_size >= block_size) {
    remaining_size -= block_size;
    Accumulator acc = 0;
    const T* block_end = a + block_size;
    for (; a < block_end; ++a, ++b) {
      acc += *a != *b;
    }
    result += acc;
  }

  Accumulator acc_final = 0;
  const T* block_end = a + remaining_size;
  for (; a < block_end; ++a, ++b) {
    acc_final += *a != *b;
  }
  return result + acc_final;
}

template <typename T>
DimensionIndex GeneralHammingDistance::GetDistanceUnrolled(const T* a,
                                                           const T* b,
                                                           size_t size) {
  DimensionIndex result = 0;
  uint8_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
  size_t remaining_size = size;
  for (; remaining_size >= 1008; remaining_size -= 1008) {
    const T* enda = a + 1008;
    for (; a + 3 < enda; a += 4, b += 4) {
      acc0 += a[0] != b[0];
      acc1 += a[1] != b[1];
      acc2 += a[2] != b[2];
      acc3 += a[3] != b[3];
    }

    result += static_cast<DimensionIndex>(acc0);
    result += static_cast<DimensionIndex>(acc1);
    result += static_cast<DimensionIndex>(acc2);
    result += static_cast<DimensionIndex>(acc3);
    acc0 = 0;
    acc1 = 0;
    acc2 = 0;
    acc3 = 0;
  }

  const T* enda = a + remaining_size;
  for (; a + 3 < enda; a += 4, b += 4) {
    acc0 += a[0] != b[0];
    acc1 += a[1] != b[1];
    acc2 += a[2] != b[2];
    acc3 += a[3] != b[3];
  }

  if (a + 1 < enda) {
    acc0 += a[0] != b[0];
    acc1 += a[1] != b[1];
    a += 2;
    b += 2;
  }

  if (a < enda) {
    acc3 += a[0] != b[0];
  }

  result += static_cast<DimensionIndex>(acc0);
  result += static_cast<DimensionIndex>(acc1);
  result += static_cast<DimensionIndex>(acc2);
  result += static_cast<DimensionIndex>(acc3);
  return result;
}

template <typename T>
double GeneralHammingDistance::GetDistanceDenseImpl(const DatapointPtr<T>& a,
                                                    const DatapointPtr<T>& b) {
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());

  if (sizeof(T) == 1) {
    return GetDistanceDenseAutoVectorized<T, uint8_t>(a.values(), b.values(),
                                                      a.nonzero_entries());
  } else if (sizeof(T) == 2) {
    return GetDistanceDenseAutoVectorized<T, uint16_t>(a.values(), b.values(),
                                                       a.nonzero_entries());
  } else if (sizeof(T) == 4) {
    return GetDistanceDenseAutoVectorized<T, uint32_t>(a.values(), b.values(),
                                                       a.nonzero_entries());
  } else {
    return GetDistanceUnrolled<T>(a.values(), b.values(), a.nonzero_entries());
  }
}

class DenseBinaryHammingXor {
 public:
  template <typename T>
  T operator()(T a, T b) {
    return a ^ b;
  }
};

inline DimensionIndex DenseBinaryHammingDistance(
    const DatapointPtr<uint8_t>& a, const DatapointPtr<uint8_t>& b) {
  return DenseBinaryMergeAndPopcnt(a, b, DenseBinaryHammingXor());
}

}  // namespace research_scann

#endif
