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

#ifndef SCANN__DISTANCE_MEASURES_ONE_TO_ONE_BINARY_DISTANCE_MEASURE_BASE_H_
#define SCANN__DISTANCE_MEASURES_ONE_TO_ONE_BINARY_DISTANCE_MEASURE_BASE_H_

#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/one_to_one/common.h"
#include "scann/oss_wrappers/scann_bits.h"

namespace tensorflow {
namespace scann_ops {

class BinaryDistanceMeasureBase : public DistanceMeasure {
 public:
  using DistanceMeasure::GetDistanceDense;
  using DistanceMeasure::GetDistanceHybrid;
  using DistanceMeasure::GetDistanceSparse;

  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(int8_t);
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(int16_t);
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(uint16_t);
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(int32_t);
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(uint32_t);
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(int64_t);
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(uint64_t);
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(float);
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS_1(double);

  double GetDistanceDense(const DatapointPtr<uint8_t>& a,
                          const DatapointPtr<uint8_t>& b,
                          double threshold) const final {
    return GetDistanceDense(a, b);
  }
};

template <typename Merge>
inline DimensionIndex DenseBinaryMergeAndPopcnt(const DatapointPtr<uint8_t>& v1,
                                                const DatapointPtr<uint8_t>& v2,
                                                Merge merge) {
  DCHECK_EQ(v1.nonzero_entries(), v2.nonzero_entries());

  DimensionIndex result = 0;
  DimensionIndex i = 0;
  for (; i + 8 <= v1.nonzero_entries(); i += 8) {
    result += bits::CountOnes64(
        merge(ABSL_INTERNAL_UNALIGNED_LOAD64(v1.values() + i),
              ABSL_INTERNAL_UNALIGNED_LOAD64(v2.values() + i)));
  }

  if (i + 4 <= v1.nonzero_entries()) {
    result +=
        bits::CountOnes(merge(ABSL_INTERNAL_UNALIGNED_LOAD32(v1.values() + i),
                              ABSL_INTERNAL_UNALIGNED_LOAD32(v2.values() + i)));
    i += 4;
  }

  if (i + 2 <= v1.nonzero_entries()) {
    result +=
        bits::CountOnes(merge(ABSL_INTERNAL_UNALIGNED_LOAD16(v1.values() + i),
                              ABSL_INTERNAL_UNALIGNED_LOAD16(v2.values() + i)));
    i += 2;
  }

  if (i < v1.nonzero_entries()) {
    result += bits::CountOnes(merge(v1.values()[i], v2.values()[i]));
  }

  return result;
}

}  // namespace scann_ops
}  // namespace tensorflow

#endif
