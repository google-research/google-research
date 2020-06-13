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

#include "scann/distance_measures/many_to_many/fp8_transposed.h"

#include "scann/utils/intrinsics/flags.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {

namespace {
uint8_t SimdBlockSize() {
  if (RuntimeSupportsAvx512()) {
    return 16;
  } else if (RuntimeSupportsAvx1()) {
    return 8;
  } else if (RuntimeSupportsSse4()) {
    return 4;
  } else {
    return 1;
  }
}
}  // namespace

FP8SimdBlockTransposedDatabase::FP8SimdBlockTransposedDatabase()
    : dimensionality_(0), simd_block_size_(SimdBlockSize()) {}

FP8SimdBlockTransposedDatabase::FP8SimdBlockTransposedDatabase(
    const DenseDataset<int8_t>& db, ConstSpan<float> inverse_fp8_multipliers)
    : FP8SimdBlockTransposedDatabase(db, SimdBlockSize(),
                                     inverse_fp8_multipliers) {}

FP8SimdBlockTransposedDatabase::FP8SimdBlockTransposedDatabase(
    const DenseDataset<int8_t>& db, uint8_t simd_block_size,
    ConstSpan<float> inverse_fp8_multipliers)
    : payload_(new int8_t[db.data().size()]),
      inverse_fp8_multipliers_(inverse_fp8_multipliers.data()),
      size_(db.size()),
      dimensionality_(db.dimensionality()),
      simd_block_size_(simd_block_size) {
  if (!inverse_fp8_multipliers.empty()) {
    CHECK_EQ(dimensionality_, inverse_fp8_multipliers.size());
  }

  const DatapointIndex sz = db.size();
  const int8_t* untransposed_ptr = db.data().data();
  for (DatapointIndex block_start = 0; block_start < sz;
       block_start += simd_block_size_) {
    const DatapointIndex block_size =
        std::min<DatapointIndex>(simd_block_size_, sz - block_start);
    TransposeOneBlock(untransposed_ptr + block_start * dimensionality_,
                      block_size,
                      payload_.get() + block_start * dimensionality_);
  }
}

void FP8SimdBlockTransposedDatabase::TransposeOneBlock(const int8_t* src,
                                                       size_t block_size,
                                                       int8_t* dest) {
  for (DatapointIndex dp_idx : Seq(block_size)) {
    const int8_t* dp_start = src + dimensionality_ * dp_idx;
    for (DimensionIndex dim_idx : Seq(dimensionality_)) {
      dest[dim_idx * block_size + dp_idx] = dp_start[dim_idx];
    }
  }
}

}  // namespace scann_ops
}  // namespace tensorflow
