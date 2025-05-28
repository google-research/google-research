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

#ifndef SCANN_UTILS_NOISE_SHAPING_UTILS_H_
#define SCANN_UTILS_NOISE_SHAPING_UTILS_H_

#include <cmath>

#include "scann/data_format/datapoint.h"
#include "scann/utils/datapoint_utils.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
T Square(T x) {
  return x * x;
}

inline double ComputeParallelCostMultiplier(double threshold,
                                            double squared_l2_norm,
                                            DimensionIndex dims) {
  const double parallel_cost = Square(threshold) / squared_l2_norm;
  const double perpendicular_cost =
      (1.0 - Square(threshold) / squared_l2_norm) / (dims - 1.0);
  return parallel_cost / perpendicular_cost;
}

template <typename QuantizedT, typename Lambdas>
DatapointPtr<QuantizedT> ScalarQuantizeFloatDatapointWithNoiseShapingImpl(
    const DatapointPtr<float>& dptr, double noise_shaping_threshold,
    MutableSpan<QuantizedT> quantized, MutableSpan<float> residuals,
    MutableSpan<uint32_t> dims, const Lambdas& lambdas, int* num_changes,
    double* residual_ptr, double* parallel_residual_ptr) {
  DCHECK_EQ(quantized.size(), dptr.dimensionality());
  DCHECK_EQ(residuals.size(), quantized.size());
  DCHECK_EQ(dims.size(), quantized.size());
  for (size_t i : IndicesOf(quantized)) {
    quantized[i] = lambdas.QuantizeSingleDim(dptr.values()[i], i);
  }

  double residual_sq_norm = 0.0;
  double parallel_residual_dot = 0.0;
  const double squared_dptr_norm = SquaredL2Norm(dptr);
  const float dptr_norm = std::sqrt(squared_dptr_norm);
  const float inv_dptr_norm = 1.0 / dptr_norm;
  for (size_t i : IndicesOf(residuals)) {
    residuals[i] =
        lambdas.DequantizeSingleDim(quantized[i], i) - dptr.values()[i];
    residual_sq_norm += residuals[i] * residuals[i];
    parallel_residual_dot += residuals[i] * dptr.values()[i] * inv_dptr_norm;
  }

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
      QuantizedT quantized_delta = 0;
      if (residuals[i] > 0) {
        if (lambdas.AtMinimum(quantized[i])) continue;
        quantized_delta = lambdas.NextSmallerDelta(quantized[i]);
      } else {
        if (lambdas.AtMaximum(quantized[i])) continue;
        quantized_delta = lambdas.NextLargerDelta(quantized[i]);
      }
      const float actual = dptr.values()[dims[i]];
      const float original = lambdas.DequantizeSingleDim(quantized[i], dims[i]);
      DCHECK(std::isfinite(original)) << actual;
      const float shifted =
          lambdas.DequantizeSingleDim(quantized[i] + quantized_delta, dims[i]);
      DCHECK(std::isfinite(shifted)) << actual;
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

  PermuteInPlace<QuantizedT>(quantized, MakeMutableSpan(dims));
  if (residual_ptr) *residual_ptr = residual_sq_norm;
  if (parallel_residual_ptr) {
    *parallel_residual_ptr = Square(parallel_residual_dot);
  }
  return MakeDatapointPtr(quantized.data(), quantized.size());
}

}  // namespace research_scann

#endif
