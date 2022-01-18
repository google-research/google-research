// Copyright 2022 The Google Research Authors.
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

#include "scann/distance_measures/one_to_one/cosine_distance.h"

#include <cstdint>

#include "absl/numeric/bits.h"
#include "scann/oss_wrappers/scann_bits.h"

namespace research_scann {

SCANN_DEFINE_DISTANCE_MEASURE_VIRTUAL_METHODS(CosineDistance,
                                              kEarlyStoppingNotSupported);
SCANN_REGISTER_DISTANCE_MEASURE(CosineDistance);

double BinaryCosineDistance::GetDistanceDense(
    const DatapointPtr<uint8_t>& a, const DatapointPtr<uint8_t>& b) const {
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());
  DimensionIndex num_intersect = 0;
  DimensionIndex a_num_ones = 0, b_num_ones = 0;
  for (size_t i = 0; i < a.nonzero_entries(); ++i) {
    a_num_ones += absl::popcount(a.values()[i]);
    b_num_ones += absl::popcount(b.values()[i]);
    num_intersect += absl::popcount(
        static_cast<unsigned char>(a.values()[i] & b.values()[i]));
  }

  return 1.0 - (num_intersect / sqrt(static_cast<uint64_t>(a_num_ones) *
                                     static_cast<uint64_t>(b_num_ones)));
}

double BinaryCosineDistance::GetDistanceSparse(
    const DatapointPtr<uint8_t>& a, const DatapointPtr<uint8_t>& b) const {
  const DimensionIndex num_intersect = SparseBinaryDotProduct(a, b);

  return 1.0 -
         (num_intersect / sqrt(static_cast<uint64_t>(a.nonzero_entries()) *
                               static_cast<uint64_t>(b.nonzero_entries())));
}

double BinaryCosineDistance::GetDistanceHybrid(
    const DatapointPtr<uint8_t>& a, const DatapointPtr<uint8_t>& b) const {
  DCHECK_EQ(a.dimensionality(), b.dimensionality());
  DimensionIndex num_intersect = 0;
  const DatapointPtr<uint8_t>& dense = (a.IsDense()) ? a : b;
  const DatapointPtr<uint8_t>& sparse = (a.IsDense()) ? b : a;
  DCHECK(sparse.IsSparse());
  for (size_t i = 0; i < sparse.nonzero_entries(); ++i) {
    num_intersect += dense.GetElementPacked(sparse.indices()[i]);
  }

  const auto num_ones_sparse = sparse.nonzero_entries();
  const auto num_ones_dense =
      bits::Count(dense.values(), dense.nonzero_entries());

  return 1.0 - (num_intersect / sqrt(static_cast<uint64_t>(num_ones_sparse) *
                                     static_cast<uint64_t>(num_ones_dense)));
}

SCANN_REGISTER_DISTANCE_MEASURE(BinaryCosineDistance)

}  // namespace research_scann
