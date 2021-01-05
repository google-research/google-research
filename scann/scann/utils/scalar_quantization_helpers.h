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

#ifndef SCANN__UTILS_SCALAR_QUANTIZATION_HELPERS_H_
#define SCANN__UTILS_SCALAR_QUANTIZATION_HELPERS_H_

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {

struct ScalarQuantizationResults {
  DenseDataset<int8_t> quantized_dataset;

  vector<float> multiplier_by_dimension;

  vector<float> inverse_multiplier_by_dimension;
};

SCANN_INLINE int8_t Int8Quantize(float value) {
  const float fp_val = std::round(value);
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

std::vector<float> ComputeQuantiledQuantizationMultipliers(
    const DenseDataset<float>& dataset, float multiplier_quantile);

ScalarQuantizationResults ScalarQuantizeFloatDataset(
    const DenseDataset<float>& dataset, float multiplier_quantile = 1.0f,
    double noise_shaping_threshold = NAN);

ScalarQuantizationResults ScalarQuantizeFloatDatasetWithMultipliers(
    const DenseDataset<float>& dataset, std::vector<float> multipliers,
    double noise_shaping_threshold = NAN);

DatapointPtr<int8_t> ScalarQuantizeFloatDatapoint(
    const DatapointPtr<float>& dptr, absl::Span<const float> multipliers,
    std::vector<int8_t>* quantized_storage);

DatapointPtr<int8_t> ScalarQuantizeFloatDatapointWithNoiseShaping(
    const DatapointPtr<float>& dptr, absl::Span<const float> multipliers,
    const double noise_shaping_threshold, vector<int8_t>* quantized_storage,
    int* num_changes = nullptr, double* residual_ptr = nullptr,
    double* parallel_residual_ptr = nullptr);

DatapointPtr<int8_t> ScalarQuantizeFloatDatapointWithNoiseShaping(
    const DatapointPtr<float>& dptr, absl::Span<const float> multipliers,
    const double noise_shaping_threshold, MutableSpan<int8_t> quantized,
    int* num_changes = nullptr, double* residual_ptr = nullptr,
    double* parallel_residual_ptr = nullptr);

DatapointPtr<int8_t> ScalarQuantizeFloatDatapoint(
    const DatapointPtr<float>& dptr, float multiplier,
    vector<int8_t>* quantized_storage);

unique_ptr<float[]> PrepareForAsymmetricScalarQuantizedDotProduct(
    const DatapointPtr<float>& dptr,
    ConstSpan<float> inverse_multiplier_by_dimension);

}  // namespace scann_ops
}  // namespace tensorflow

#endif
