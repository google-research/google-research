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

#ifndef SCANN_UTILS_SCALAR_QUANTIZATION_HELPERS_H_
#define SCANN_UTILS_SCALAR_QUANTIZATION_HELPERS_H_

#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/types/span.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

struct ScalarQuantizationResults {
  DenseDataset<int8_t> quantized_dataset;

  vector<float> multiplier_by_dimension;

  vector<float> inverse_multiplier_by_dimension;
};

SCANN_INLINE int8_t Int8Quantize(float value) {
  const float fp_val = std::round(value);
  DCHECK(std::isfinite(fp_val)) << "Float value is not finite: " << value;
  if (ABSL_PREDICT_FALSE(fp_val > numeric_limits<int8_t>::max())) {
    return numeric_limits<int8_t>::max();
  }
  if (ABSL_PREDICT_FALSE(fp_val < numeric_limits<int8_t>::min())) {
    return numeric_limits<int8_t>::min();
  }
  return fp_val;
}

std::vector<float> ComputeMaxQuantizationMultipliers(
    const DenseDataset<float>& dataset);

std::vector<float> ComputeMaxQuantizationMultipliers(
    const DenseDatasetView<float>& dataset);

std::vector<float> ComputeQuantiledQuantizationMultipliers(
    const DenseDataset<float>& dataset, float multiplier_quantile);

std::vector<float> ComputeQuantiledQuantizationMultipliers(
    const DenseDatasetView<float>& dataset, float multiplier_quantile);

ScalarQuantizationResults ScalarQuantizeFloatDataset(
    const DenseDataset<float>& dataset, float multiplier_quantile = 1.0f,
    double noise_shaping_threshold = NAN, ThreadPool* pool = nullptr);

ScalarQuantizationResults ScalarQuantizeFloatDatasetWithMultipliers(
    DenseDatasetView<float>&& dataset, std::vector<float> multipliers,
    double noise_shaping_threshold = NAN, ThreadPool* pool = nullptr);

SCANN_INLINE ScalarQuantizationResults
ScalarQuantizeFloatDatasetWithMultipliers(const DenseDataset<float>& dataset,
                                          std::vector<float> multipliers,
                                          double noise_shaping_threshold = NAN,
                                          ThreadPool* pool = nullptr) {
  return ScalarQuantizeFloatDatasetWithMultipliers(
      DefaultDenseDatasetView<float>(dataset), multipliers,
      noise_shaping_threshold, pool);
}

DatapointPtr<int8_t> ScalarQuantizeFloatDatapoint(
    const DatapointPtr<float>& dptr, absl::Span<const float> multipliers,
    std::vector<int8_t>* quantized_storage);

DatapointPtr<int8_t> ScalarQuantizeFloatDatapoint(
    const DatapointPtr<float>& dptr, absl::Span<const float> multipliers,
    MutableSpan<int8_t> quantized);

DatapointPtr<int8_t> ScalarQuantizeFloatDatapointWithNoiseShaping(
    const DatapointPtr<float>& dptr, absl::Span<const float> multipliers,
    double noise_shaping_threshold, vector<int8_t>* quantized_storage,
    int* num_changes = nullptr, double* residual_ptr = nullptr,
    double* parallel_residual_ptr = nullptr);

DatapointPtr<int8_t> ScalarQuantizeFloatDatapointWithNoiseShaping(
    const DatapointPtr<float>& dptr, absl::Span<const float> multipliers,
    double noise_shaping_threshold, MutableSpan<int8_t> quantized,
    int* num_changes = nullptr, double* residual_ptr = nullptr,
    double* parallel_residual_ptr = nullptr);

DatapointPtr<int8_t> ScalarQuantizeFloatDatapointWithNoiseShaping(
    const DatapointPtr<float>& dptr, absl::Span<const float> multipliers,
    double noise_shaping_threshold, MutableSpan<int8_t> quantized,
    MutableSpan<float> residuals, MutableSpan<uint32_t> dims,
    int* num_changes = nullptr, double* residual_ptr = nullptr,
    double* parallel_residual_ptr = nullptr);

DatapointPtr<int8_t> ScalarQuantizeFloatDatapoint(
    const DatapointPtr<float>& dptr, float multiplier,
    vector<int8_t>* quantized_storage);

DatapointPtr<int8_t> ScalarQuantizeFloatDatapoint(
    const DatapointPtr<float>& dptr, float multiplier,
    MutableSpan<int8_t> quantized);

unique_ptr<float[]> PrepareForAsymmetricScalarQuantizedDotProduct(
    const DatapointPtr<float>& query,
    ConstSpan<float> inverse_multiplier_by_dimension);

static constexpr float kFP4Max = 7.5f;
static constexpr float kFP8Max = numeric_limits<int8_t>::max();

SCANN_INLINE uint8_t Int4Quantize(float value) {
  value += kFP4Max;
  const float fp_val = std::round(value);
  DCHECK(std::isfinite(fp_val)) << "Float value is not finite: " << value;
  if (ABSL_PREDICT_FALSE(fp_val > 15)) {
    return 15;
  }
  if (ABSL_PREDICT_FALSE(fp_val < 0)) {
    return 0;
  }
  return fp_val;
}
SCANN_INLINE float Int4Dequantize(uint8_t value) {
  DCHECK_LE(value, 15);
  float fp_value = value;
  return fp_value - kFP4Max;
}

std::vector<float> Int8ToInt4Multipliers(absl::Span<const float> multipliers);
std::vector<float> InverseInt8ToInt4Multipliers(
    absl::Span<const float> multipliers);

constexpr size_t kNumBottomBits = 32;

constexpr size_t kMinDimensionsForBottomBits = 64;

absl::StatusOr<DatapointPtr<int8_t>>
ScalarQuantizeFloatDatapointWithNoiseShaping(
    const DatapointPtr<float>& dptr, absl::Span<const float> multipliers,
    std::optional<uint32_t> bottom_bits_data, double noise_shaping_threshold,
    MutableSpan<int8_t> quantized);

absl::Status Int4QuantizePackFloatDatapointWithNoiseShaping(
    const DatapointPtr<float>& dptr, absl::Span<const float> multipliers,
    std::optional<uint32_t> bottom_bits_data, double noise_shaping_threshold,
    MutableSpan<uint8_t> packed);

uint32_t DecodeBottomBitsDataFromPackedInt4(ConstSpan<uint8_t> packed);

uint32_t DecodeBottomBitsDataFromInt8(ConstSpan<int8_t> quantized);

}  // namespace research_scann

#endif
