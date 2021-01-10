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



#ifndef SCANN_DISTANCE_MEASURES_MANY_TO_MANY_FP8_TRANSPOSED_H_
#define SCANN_DISTANCE_MEASURES_MANY_TO_MANY_FP8_TRANSPOSED_H_

#include "scann/data_format/dataset.h"
#include "scann/utils/types.h"

namespace research_scann {

class FP8SimdBlockTransposedDatabase {
 public:
  FP8SimdBlockTransposedDatabase();

  explicit FP8SimdBlockTransposedDatabase(
      const DenseDataset<int8_t>& db,
      ConstSpan<float> inverse_fp8_multipliers = {});

  FP8SimdBlockTransposedDatabase(const DenseDataset<int8_t>& db,
                                 uint8_t simd_block_size,
                                 ConstSpan<float> inverse_fp8_multipliers = {});

  uint8_t simd_block_size() const { return simd_block_size_; }

  DimensionIndex dimensionality() const { return dimensionality_; }

  DatapointIndex size() const { return size_; }

  inline bool empty() const { return size_ == 0; }

  ConstSpan<int8_t> GetBlock(DatapointIndex block_idx) const {
    const size_t full_block_size = dimensionality_ * simd_block_size_;
    const size_t block_start = block_idx * dimensionality_ * simd_block_size_;
    const size_t block_size =
        std::min(full_block_size,
                 static_cast<size_t>(size_) * dimensionality_ - block_start);
    return {payload_.get() + block_start, block_size};
  }

  ConstSpan<float> inverse_fp8_multipliers() const {
    return {inverse_fp8_multipliers_, inverse_fp8_multipliers_
                                          ? static_cast<size_t>(dimensionality_)
                                          : 0};
  }

 private:
  void TransposeOneBlock(const int8_t* src, size_t block_size, int8_t* dest);

  unique_ptr<int8_t[]> payload_;

  const float* inverse_fp8_multipliers_ = nullptr;

  DatapointIndex size_ = 0;

  uint32_t dimensionality_ : 27;

  uint8_t simd_block_size_ : 5;
};

}  // namespace research_scann

#endif
