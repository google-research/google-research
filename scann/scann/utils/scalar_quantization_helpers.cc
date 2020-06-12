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



#include "scann/utils/scalar_quantization_helpers.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "scann/data_format/datapoint.h"
#include "scann/utils/common.h"
#include "scann/utils/datapoint_utils.h"
#include "scann/utils/noise_shaping_utils.h"
#include "scann/utils/top_n_amortized_constant.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"
#include "scann/utils/zip_sort.h"

namespace tensorflow {
namespace scann_ops {

std::vector<float> ComputeMaxQuantizationMultipliers(
    const DenseDataset<float>& dataset) {
  const size_t dimensionality = dataset.dimensionality();
  vector<float> multipliers(dimensionality, 0.0f);
  for (auto dptr : dataset) {
    const float* values = dptr.values();
    for (size_t j : Seq(dimensionality)) {
      multipliers[j] = std::max<float>(multipliers[j], std::abs(values[j]));
    }
  }
  for (float& f : multipliers) {
    if (f == 0.0f) {
      f = 1.0f;
    } else {
      f = numeric_limits<int8_t>::max() / f;
    }
  }
  return multipliers;
}

std::vector<float> ComputeQuantiledQuantizationMultipliers(
    const DenseDataset<float>& dataset, float multiplier_quantile) {
  const size_t dimensionality = dataset.dimensionality();
  const size_t k = dataset.size() * (1.0 - multiplier_quantile) + 1;
  if (k == 1) return ComputeMaxQuantizationMultipliers(dataset);
  std::vector<TopNAmortizedConstant<float>> top_ns(dimensionality);
  for (auto& elem : top_ns) {
    elem = TopNAmortizedConstant<float>(k);
  }
  for (auto dptr : dataset) {
    const float* values = dptr.values();
    for (size_t j : Seq(dimensionality)) {
      top_ns[j].push(std::abs(values[j]));
    }
  }
  std::vector<float> multipliers(dataset.dimensionality());
  for (size_t j : Seq(dimensionality)) {
    multipliers[j] = numeric_limits<int8_t>::max() / top_ns[j].exact_bottom();
  }
  return multipliers;
}

ScalarQuantizationResults ScalarQuantizeFloatDataset(
    const DenseDataset<float>& dataset, float multiplier_quantile,
    double noise_shaping_threshold) {
  DCHECK_LE(multiplier_quantile, 1.0f);
  DCHECK_GT(multiplier_quantile, 0.0f);

  std::vector<float> multipliers =
      (fabs(multiplier_quantile - 1.0f) < 0.001)
          ? ComputeMaxQuantizationMultipliers(dataset)
          : ComputeQuantiledQuantizationMultipliers(dataset,
                                                    multiplier_quantile);

  return ScalarQuantizeFloatDatasetWithMultipliers(
      dataset, std::move(multipliers), noise_shaping_threshold);
}

ScalarQuantizationResults ScalarQuantizeFloatDatasetWithMultipliers(
    const DenseDataset<float>& dataset, vector<float> multipliers,
    double noise_shaping_threshold) {
  const size_t dimensionality = dataset.dimensionality();
  DCHECK_EQ(multipliers.size(), dimensionality);

  DenseDataset<int8_t> fixed_point_dataset;
  fixed_point_dataset.set_dimensionality(dimensionality);
  fixed_point_dataset.Reserve(dataset.size());
  unique_ptr<int8_t[]> fixed_point_dp(new int8_t[dimensionality]);
  for (auto dptr : dataset) {
    if (std::isnan(noise_shaping_threshold)) {
      const float* values = dptr.values();
      for (size_t j : Seq(dimensionality)) {
        fixed_point_dp[j] = Int8Quantize(values[j] * multipliers[j]);
      }
    } else {
      ScalarQuantizeFloatDatapointWithNoiseShaping(
          dptr, multipliers, noise_shaping_threshold,
          MakeMutableSpan(fixed_point_dp.get(), dimensionality));
    }
    fixed_point_dataset.AppendOrDie(
        MakeDatapointPtr(fixed_point_dp.get(), dimensionality), "");
  }

  vector<float> inv_multipliers(dimensionality);
  for (size_t j : Seq(dimensionality)) {
    inv_multipliers[j] = 1.0f / multipliers[j];
  }

  ScalarQuantizationResults results;
  results.quantized_dataset = std::move(fixed_point_dataset);
  results.multiplier_by_dimension = std::move(multipliers);
  results.inverse_multiplier_by_dimension = std::move(inv_multipliers);

  return results;
}

DatapointPtr<int8_t> ScalarQuantizeFloatDatapoint(
    const DatapointPtr<float>& dptr, absl::Span<const float> multipliers,
    std::vector<int8_t>* quantized_storage) {
  const size_t dimensionality = dptr.dimensionality();
  DCHECK_EQ(multipliers.size(), dimensionality);
  DCHECK_EQ(quantized_storage->size(), dimensionality);
  for (size_t i : Seq(dimensionality)) {
    (*quantized_storage)[i] = Int8Quantize(dptr.values()[i] * multipliers[i]);
  }
  return MakeDatapointPtr(*quantized_storage);
}

DatapointPtr<int8_t> ScalarQuantizeFloatDatapoint(
    const DatapointPtr<float>& dptr, float multiplier,
    std::vector<int8_t>* quantized_storage) {
  const size_t dimensionality = dptr.dimensionality();
  DCHECK_EQ(quantized_storage->size(), dimensionality);
  for (size_t i : Seq(dimensionality)) {
    (*quantized_storage)[i] = Int8Quantize(dptr.values()[i] * multiplier);
  }
  return MakeDatapointPtr(*quantized_storage);
}

DatapointPtr<int8_t> ScalarQuantizeFloatDatapointWithNoiseShaping(
    const DatapointPtr<float>& dptr, absl::Span<const float> multipliers,
    const double noise_shaping_threshold,
    std::vector<int8_t>* quantized_storage, int* num_changes,
    double* residual_ptr, double* parallel_residual_ptr) {
  quantized_storage->resize(dptr.dimensionality());
  MutableSpan<int8_t> quantized(quantized_storage->data(),
                                quantized_storage->size());
  return ScalarQuantizeFloatDatapointWithNoiseShaping(
      dptr, multipliers, noise_shaping_threshold, quantized, num_changes,
      residual_ptr, parallel_residual_ptr);
}

DatapointPtr<int8_t> ScalarQuantizeFloatDatapointWithNoiseShaping(
    const DatapointPtr<float>& dptr, absl::Span<const float> multipliers,
    const double noise_shaping_threshold, MutableSpan<int8_t> quantized,
    int* num_changes, double* residual_ptr, double* parallel_residual_ptr) {
  DCHECK_EQ(quantized.size(), dptr.dimensionality());
  for (size_t i : IndicesOf(quantized)) {
    quantized[i] = Int8Quantize(dptr.values()[i] * multipliers[i]);
  }

  vector<float> residuals(quantized.size());
  double residual_sq_norm = 0.0;
  double parallel_residual_dot = 0.0;
  const double squared_dptr_norm = SquaredL2Norm(dptr);
  const float dptr_norm = std::sqrt(squared_dptr_norm);
  const float inv_dptr_norm = 1.0 / dptr_norm;
  for (size_t i : IndicesOf(residuals)) {
    residuals[i] = quantized[i] / multipliers[i] - dptr.values()[i];
    residual_sq_norm += residuals[i] * residuals[i];
    parallel_residual_dot += residuals[i] * dptr.values()[i] * inv_dptr_norm;
  }

  vector<uint32_t> dims(residuals.size());
  std::iota(dims.begin(), dims.end(), 0U);
  ZipSortBranchOptimized(
      [](float a, float b) { return std::abs(a) > std::abs(b); },
      residuals.begin(), residuals.end(), quantized.begin(), quantized.end(),
      dims.begin(), dims.end());
  const double relative_cost = ComputeParallelCostMultiplier(
      noise_shaping_threshold, squared_dptr_norm, dptr.dimensionality());

  if (num_changes) *num_changes = 0;
  bool cur_round_changes = true;
  enum { kMaxRounds = 10 };
  for (int round = 0; cur_round_changes && round < kMaxRounds; ++round) {
    cur_round_changes = false;
    for (size_t i : IndicesOf(residuals)) {
      int8_t quantized_delta = 0;
      if (residuals[i] > 0) {
        if (quantized[i] == numeric_limits<int8_t>::min()) continue;
        quantized_delta = -1;
      } else {
        if (quantized[i] == numeric_limits<int8_t>::max()) continue;
        quantized_delta = 1;
      }
      const float actual = dptr.values()[dims[i]];
      const float original = quantized[i] / multipliers[dims[i]];
      const float shifted =
          (quantized[i] + quantized_delta) / multipliers[dims[i]];
      const double residual_delta =
          Square(shifted - actual) - Square(original - actual);
      const float new_residual = shifted - actual;
      const double new_parallel_dot = parallel_residual_dot -
                                      (residuals[i]) * actual * inv_dptr_norm +
                                      (new_residual)*actual * inv_dptr_norm;
      const double parallel_delta =
          Square(new_parallel_dot) - Square(parallel_residual_dot);
      if (parallel_delta > 0) continue;
      const double perpendicular_delta = residual_delta - parallel_delta;
      const double cost_delta =
          relative_cost * parallel_delta + perpendicular_delta;
      if (cost_delta < 0.0) {
        quantized[i] += quantized_delta;
        parallel_residual_dot = new_parallel_dot;
        residuals[i] = new_residual;
        residual_sq_norm += residual_delta;
        if (num_changes) ++*num_changes;
        cur_round_changes = true;
      }
    }
  }

  PermuteInPlace<int8_t>(quantized, MakeMutableSpan(dims));
  if (residual_ptr) *residual_ptr = residual_sq_norm;
  if (parallel_residual_ptr) {
    *parallel_residual_ptr = Square(parallel_residual_dot);
  }
  return MakeDatapointPtr(quantized.data(), quantized.size());
}

unique_ptr<float[]> PrepareForAsymmetricScalarQuantizedDotProduct(
    const DatapointPtr<float>& query,
    ConstSpan<float> inverse_multiplier_by_dimension) {
  const size_t dimensionality = query.nonzero_entries();
  const float* query_ptr = query.values();
  unique_ptr<float[]> result(new float[dimensionality]);
  for (size_t j : Seq(dimensionality)) {
    result[j] = inverse_multiplier_by_dimension[j] * query_ptr[j];
  }
  return result;
}

}  // namespace scann_ops
}  // namespace tensorflow
