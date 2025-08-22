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

#ifndef SCANN_UTILS_BFLOAT16_HELPERS_H_
#define SCANN_UTILS_BFLOAT16_HELPERS_H_

#include <cmath>
#include <cstdint>
#include <vector>

#include "absl/base/casts.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/utils/common.h"

namespace research_scann {

SCANN_INLINE int16_t Bfloat16Quantize(float value) {
  uint32_t value_bits = absl::bit_cast<uint32_t>(value);

  if (std::isinf(value)) return value_bits >> 16;
  if (std::isnan(value)) return (value_bits >> 16) | 1;

  uint32_t rounded_bits = value_bits + 0x8000;

  if (((rounded_bits >> 23) & 0xff) == 255) {
    return ((value_bits >> 31) << 15) | 0b1111'1110'1111'111;
  }

  return rounded_bits >> 16;
}

SCANN_INLINE float Bfloat16Decompress(int16_t value) {
  int value32 = value << 16;
  return absl::bit_cast<float>(value32);
}

DenseDataset<int16_t> Bfloat16QuantizeFloatDataset(
    const DenseDataset<float>& dataset);

DatapointPtr<int16_t> Bfloat16QuantizeFloatDatapoint(
    const DatapointPtr<float>& dptr, std::vector<int16_t>* quantized_storage);

DatapointPtr<int16_t> Bfloat16QuantizeFloatDatapoint(
    const DatapointPtr<float>& dptr, MutableSpan<int16_t> quantized);

DatapointPtr<int16_t> Bfloat16QuantizeFloatDatapointWithNoiseShaping(
    const DatapointPtr<float>& dptr, double noise_shaping_threshold,
    vector<int16_t>* quantized_storage, int* num_changes = nullptr,
    double* residual_ptr = nullptr, double* parallel_residual_ptr = nullptr);

DatapointPtr<int16_t> Bfloat16QuantizeFloatDatapointWithNoiseShaping(
    const DatapointPtr<float>& dptr, double noise_shaping_threshold,
    MutableSpan<int16_t> quantized, int* num_changes = nullptr,
    double* residual_ptr = nullptr, double* parallel_residual_ptr = nullptr);

DatapointPtr<int16_t> Bfloat16QuantizeFloatDatapointWithNoiseShaping(
    const DatapointPtr<float>& dptr, double noise_shaping_threshold,
    MutableSpan<int16_t> quantized, MutableSpan<float> residuals,
    MutableSpan<uint32_t> dims, int* num_changes = nullptr,
    double* residual_ptr = nullptr, double* parallel_residual_ptr = nullptr);

DenseDataset<int16_t> Bfloat16QuantizeFloatDatasetWithNoiseShaping(
    const DenseDataset<float>& dataset, float noise_shaping_threshold,
    ThreadPool* pool = nullptr);

}  // namespace research_scann

#endif
