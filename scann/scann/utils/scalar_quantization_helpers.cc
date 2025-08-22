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



#include "scann/utils/scalar_quantization_helpers.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/utils/common.h"
#include "scann/utils/noise_shaping_utils.h"
#include "scann/utils/parallel_for.h"
#include "scann/utils/top_n_amortized_constant.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"

namespace research_scann {

std::vector<float> ComputeMaxQuantizationMultipliers(
    const DenseDataset<float>& dataset) {
  return ComputeMaxQuantizationMultipliers(
      DefaultDenseDatasetView<float>(dataset));
}

std::vector<float> ComputeMaxQuantizationMultipliers(
    const DenseDatasetView<float>& dataset) {
  const size_t dimensionality = dataset.dimensionality();
  vector<float> multipliers(dimensionality, 0.0f);
  for (size_t i : Seq(dataset.size())) {
    const float* values = dataset.GetPtr(i);
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
  return ComputeQuantiledQuantizationMultipliers(
      DefaultDenseDatasetView<float>(dataset), multiplier_quantile);
}

std::vector<float> ComputeQuantiledQuantizationMultipliers(
    const DenseDatasetView<float>& dataset, float multiplier_quantile) {
  const size_t dimensionality = dataset.dimensionality();
  const size_t k = dataset.size() * (1.0 - multiplier_quantile) + 1;
  if (k == 1) return ComputeMaxQuantizationMultipliers(dataset);
  std::vector<TopNAmortizedConstant<float>> top_ns(dimensionality);
  for (auto& elem : top_ns) {
    elem = TopNAmortizedConstant<float>(k);
  }
  for (size_t i : Seq(dataset.size())) {
    const float* values = dataset.GetPtr(i);
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
    double noise_shaping_threshold, ThreadPool* pool) {
  DCHECK_LE(multiplier_quantile, 1.0f);
  DCHECK_GT(multiplier_quantile, 0.0f);

  std::vector<float> multipliers =
      (fabs(multiplier_quantile - 1.0f) < 0.001)
          ? ComputeMaxQuantizationMultipliers(dataset)
          : ComputeQuantiledQuantizationMultipliers(dataset,
                                                    multiplier_quantile);

  return ScalarQuantizeFloatDatasetWithMultipliers(
      dataset, std::move(multipliers), noise_shaping_threshold, pool);
}

ScalarQuantizationResults ScalarQuantizeFloatDatasetWithMultipliers(
    DenseDatasetView<float>&& dataset, vector<float> multipliers,
    double noise_shaping_threshold, ThreadPool* pool) {
  const size_t dimensionality = dataset.dimensionality();
  DCHECK_EQ(multipliers.size(), dimensionality);
  vector<int8_t> quantized_vec(dimensionality * dataset.size());

  ParallelFor<128>(IndicesOf(dataset), pool, [&](size_t dp_idx) {
    MutableSpan<int8_t> quantized_dp(
        quantized_vec.data() + dp_idx * dimensionality, dimensionality);
    const float* values = dataset.GetPtr(dp_idx);
    if (std::isnan(noise_shaping_threshold)) {
      for (size_t dim_idx : Seq(dimensionality)) {
        quantized_dp[dim_idx] =
            Int8Quantize(values[dim_idx] * multipliers[dim_idx]);
      }
    } else {
      ScalarQuantizeFloatDatapointWithNoiseShaping(
          MakeDatapointPtr(values, dimensionality), multipliers,
          noise_shaping_threshold, quantized_dp);
    }
  });

  vector<float> inv_multipliers(dimensionality);
  for (size_t j : Seq(dimensionality)) {
    inv_multipliers[j] = 1.0f / multipliers[j];
  }

  ScalarQuantizationResults results;
  results.quantized_dataset =
      DenseDataset<int8_t>(std::move(quantized_vec), dataset.size());
  results.multiplier_by_dimension = std::move(multipliers);
  results.inverse_multiplier_by_dimension = std::move(inv_multipliers);

  return results;
}

DatapointPtr<int8_t> ScalarQuantizeFloatDatapoint(
    const DatapointPtr<float>& dptr, absl::Span<const float> multipliers,
    std::vector<int8_t>* quantized_storage) {
  MutableSpan<int8_t> quantized(quantized_storage->data(),
                                quantized_storage->size());
  return ScalarQuantizeFloatDatapoint(dptr, multipliers, quantized);
}

DatapointPtr<int8_t> ScalarQuantizeFloatDatapoint(
    const DatapointPtr<float>& dptr, absl::Span<const float> multipliers,
    MutableSpan<int8_t> quantized) {
  const size_t dimensionality = dptr.dimensionality();
  DCHECK_EQ(multipliers.size(), dimensionality);
  DCHECK_EQ(quantized.size(), dimensionality);
  for (size_t i : Seq(dimensionality)) {
    quantized[i] = Int8Quantize(dptr.values()[i] * multipliers[i]);
  }
  return MakeDatapointPtr(quantized.data(), quantized.size());
}

DatapointPtr<int8_t> ScalarQuantizeFloatDatapoint(
    const DatapointPtr<float>& dptr, float multiplier,
    std::vector<int8_t>* quantized_storage) {
  MutableSpan<int8_t> quantized(quantized_storage->data(),
                                quantized_storage->size());
  return ScalarQuantizeFloatDatapoint(dptr, multiplier, quantized);
}

DatapointPtr<int8_t> ScalarQuantizeFloatDatapoint(
    const DatapointPtr<float>& dptr, float multiplier,
    MutableSpan<int8_t> quantized) {
  const size_t dimensionality = dptr.dimensionality();
  DCHECK_EQ(quantized.size(), dimensionality);
  for (size_t i : Seq(dimensionality)) {
    quantized[i] = Int8Quantize(dptr.values()[i] * multiplier);
  }
  return MakeDatapointPtr(quantized.data(), quantized.size());
}

DatapointPtr<int8_t> ScalarQuantizeFloatDatapointWithNoiseShaping(
    const DatapointPtr<float>& dptr, absl::Span<const float> multipliers,
    double noise_shaping_threshold, std::vector<int8_t>* quantized_storage,
    int* num_changes, double* residual_ptr, double* parallel_residual_ptr) {
  quantized_storage->resize(dptr.dimensionality());
  MutableSpan<int8_t> quantized(quantized_storage->data(),
                                quantized_storage->size());
  return ScalarQuantizeFloatDatapointWithNoiseShaping(
      dptr, multipliers, noise_shaping_threshold, quantized, num_changes,
      residual_ptr, parallel_residual_ptr);
}

DatapointPtr<int8_t> ScalarQuantizeFloatDatapointWithNoiseShaping(
    const DatapointPtr<float>& dptr, absl::Span<const float> multipliers,
    double noise_shaping_threshold, MutableSpan<int8_t> quantized,
    int* num_changes, double* residual_ptr, double* parallel_residual_ptr) {
  vector<float> residuals(quantized.size());
  vector<uint32_t> dims(quantized.size());
  return ScalarQuantizeFloatDatapointWithNoiseShaping(
      dptr, multipliers, noise_shaping_threshold, quantized,
      MakeMutableSpan(residuals), MakeMutableSpan(dims), num_changes,
      residual_ptr, parallel_residual_ptr);
}

class Fixed8NoiseShapingLambdas {
 public:
  explicit Fixed8NoiseShapingLambdas(ConstSpan<float> multipliers)
      : multipliers_(multipliers) {}

  int8_t QuantizeSingleDim(float f, DimensionIndex dim_idx) const {
    return Int8Quantize(f * multipliers_[dim_idx]);
  }

  float DequantizeSingleDim(int8_t f, DimensionIndex dim_idx) const {
    return f / multipliers_[dim_idx];
  }

  bool AtMaximum(int8_t i) const { return i == numeric_limits<int8_t>::max(); }
  bool AtMinimum(int8_t i) const { return i == numeric_limits<int8_t>::min(); }

  int8_t NextLargerDelta(int8_t) const { return 1; }
  int8_t NextSmallerDelta(int8_t) const { return -1; }

  size_t BottomBitIndex(DimensionIndex dim_idx) const {
    return dim_idx % kNumBottomBits;
  }

 private:
  ConstSpan<float> multipliers_;
};

class Fixed4NoiseShapingLambdas {
 public:
  explicit Fixed4NoiseShapingLambdas(ConstSpan<float> multipliers)
      : multipliers_(multipliers) {}

  uint8_t QuantizeSingleDim(float f, DimensionIndex dim_idx) const {
    return Int4Quantize(f * multipliers_[dim_idx]);
  }

  float DequantizeSingleDim(uint8_t f, DimensionIndex dim_idx) const {
    return Int4Dequantize(f) / multipliers_[dim_idx];
  }

  bool AtMaximum(uint8_t i) const { return i == 15; }
  bool AtMinimum(uint8_t i) const { return i == 0; }

  uint8_t NextLargerDelta(uint8_t) const { return 1; }
  uint8_t NextSmallerDelta(uint8_t) const { return -1; }

  size_t BottomBitIndex(DimensionIndex dim_idx) const {
    return (dim_idx / 2) % kNumBottomBits;
  }

 private:
  ConstSpan<float> multipliers_;
};

DatapointPtr<int8_t> ScalarQuantizeFloatDatapointWithNoiseShaping(
    const DatapointPtr<float>& dptr, absl::Span<const float> multipliers,
    double noise_shaping_threshold, MutableSpan<int8_t> quantized,
    MutableSpan<float> residuals, MutableSpan<uint32_t> dims, int* num_changes,
    double* residual_ptr, double* parallel_residual_ptr) {
  return ScalarQuantizeFloatDatapointWithNoiseShapingImpl(
      dptr, noise_shaping_threshold, quantized, residuals, dims,
      Fixed8NoiseShapingLambdas(multipliers), num_changes, residual_ptr,
      parallel_residual_ptr);
}

template <typename QuantizedT, typename Lambdas>
absl::Status EncodeBottomBits(uint32_t bottom_bits_data,
                              const DatapointPtr<float>& dptr,
                              const Lambdas& lambdas,
                              MutableSpan<QuantizedT> quantized) {
  SCANN_RET_CHECK_GE(quantized.size(), kNumBottomBits);
  struct PerBitData {
    float error_diff = std::numeric_limits<float>::infinity();
    int dim = -1;
    QuantizedT new_val;
    bool parity = false;
  };
  std::array<PerBitData, kNumBottomBits> per_bit_datas;
  for (size_t dim : Seq(quantized.size())) {
    PerBitData& per_bit_data = per_bit_datas[lambdas.BottomBitIndex(dim)];
    const QuantizedT cur_val = quantized[dim];
    per_bit_data.parity ^= cur_val & 1;
    auto get_error = [&](QuantizedT val) {
      return std::abs(lambdas.DequantizeSingleDim(val, dim) -
                      dptr.values()[dim]);
    };
    const float cur_error = get_error(cur_val);
    auto update_per_bit_data = [&](QuantizedT new_val) {
      const float error_diff = get_error(new_val) - cur_error;
      if (error_diff < per_bit_data.error_diff) {
        per_bit_data.error_diff = error_diff;
        per_bit_data.dim = dim;
        per_bit_data.new_val = new_val;
      }
    };
    if (!lambdas.AtMaximum(cur_val)) {
      update_per_bit_data(cur_val + lambdas.NextLargerDelta(cur_val));
    }
    if (!lambdas.AtMinimum(cur_val)) {
      update_per_bit_data(cur_val + lambdas.NextSmallerDelta(cur_val));
    }
  }

  for (size_t bit : Seq(kNumBottomBits)) {
    const PerBitData& per_bit_data = per_bit_datas[bit];
    if (per_bit_data.parity != ((bottom_bits_data >> bit) & 1)) {
      quantized[per_bit_data.dim] = per_bit_data.new_val;
    }
  }
  return OkStatus();
}
absl::StatusOr<DatapointPtr<int8_t>>
ScalarQuantizeFloatDatapointWithNoiseShaping(
    const DatapointPtr<float>& dptr, absl::Span<const float> multipliers,
    std::optional<uint32_t> bottom_bits_data, double noise_shaping_threshold,
    MutableSpan<int8_t> quantized) {
  DatapointPtr<int8_t> result;
  if (std::isnan(noise_shaping_threshold)) {
    result = ScalarQuantizeFloatDatapoint(dptr, multipliers, quantized);
  } else {
    result = ScalarQuantizeFloatDatapointWithNoiseShaping(
        dptr, multipliers, noise_shaping_threshold, quantized);
  }
  if (bottom_bits_data != std::nullopt) {
    SCANN_RETURN_IF_ERROR(EncodeBottomBits(
        *bottom_bits_data, dptr, Fixed8NoiseShapingLambdas(multipliers),
        MakeMutableSpan(quantized)));
  }
  return result;
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

std::vector<float> Int8ToInt4Multipliers(absl::Span<const float> multipliers) {
  std::vector<float> result;
  result.reserve(multipliers.size());
  for (float m : multipliers) {
    result.push_back(m * (kFP4Max / kFP8Max));
  }
  return result;
}

std::vector<float> InverseInt8ToInt4Multipliers(
    absl::Span<const float> multipliers) {
  std::vector<float> result;
  result.reserve(multipliers.size());
  for (float m : multipliers) {
    result.push_back(m * (kFP8Max / kFP4Max));
  }
  return result;
}

absl::Status Int4QuantizePackFloatDatapointWithNoiseShaping(
    const DatapointPtr<float>& dptr, absl::Span<const float> multipliers,
    std::optional<uint32_t> bottom_bits_data, double noise_shaping_threshold,
    MutableSpan<uint8_t> packed) {
  const size_t dimensionality = dptr.dimensionality();

  vector<uint8_t> quantized(dimensionality);
  if (std::isnan(noise_shaping_threshold)) {
    for (size_t i : Seq(dimensionality)) {
      quantized[i] = Int4Quantize(dptr.values()[i] * multipliers[i]);
    }
  } else {
    vector<float> residuals(dimensionality);
    vector<uint32_t> dims(dimensionality);
    ScalarQuantizeFloatDatapointWithNoiseShapingImpl(
        dptr, noise_shaping_threshold, MakeMutableSpan(quantized),
        MakeMutableSpan(residuals), MakeMutableSpan(dims),
        Fixed4NoiseShapingLambdas(multipliers), nullptr, nullptr, nullptr);
  }
  if (bottom_bits_data != std::nullopt) {
    SCANN_RETURN_IF_ERROR(EncodeBottomBits(
        *bottom_bits_data, dptr, Fixed4NoiseShapingLambdas(multipliers),
        MakeMutableSpan(quantized)));
  }
  PackNibblesDatapoint(quantized, packed);
  return OkStatus();
}

uint32_t DecodeBottomBitsDataFromPackedInt4(ConstSpan<uint8_t> packed) {
  DCHECK_GE(packed.size() * 2, kMinDimensionsForBottomBits);
  uint32_t result = 0;
  for (size_t i : Seq(packed.size())) {
    uint32_t bottom_bit = (packed[i] ^ packed[i] >> 4) & 1;
    result ^= bottom_bit << (i % kNumBottomBits);
  }
  return result;
}

uint32_t DecodeBottomBitsDataFromInt8(ConstSpan<int8_t> quantized) {
  DCHECK_GE(quantized.size(), kMinDimensionsForBottomBits);
  uint32_t result = 0;
  for (size_t i : Seq(quantized.size())) {
    uint32_t bottom_bit = quantized[i] & 1;
    result ^= bottom_bit << (i % kNumBottomBits);
  }
  return result;
}

}  // namespace research_scann
