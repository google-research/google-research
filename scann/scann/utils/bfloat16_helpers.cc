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



#include "scann/utils/bfloat16_helpers.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/utils/common.h"
#include "scann/utils/noise_shaping_utils.h"
#include "scann/utils/parallel_for.h"
#include "scann/utils/types.h"

namespace research_scann {

DenseDataset<int16_t> Bfloat16QuantizeFloatDataset(
    const DenseDataset<float>& dataset) {
  const size_t dimensionality = dataset.dimensionality();

  DenseDataset<int16_t> result;
  result.set_dimensionality(dimensionality);
  result.Reserve(dataset.size());

  unique_ptr<int16_t[]> bfloat16_dp(new int16_t[dimensionality]);
  MutableSpan<int16_t> dp_span(bfloat16_dp.get(), dimensionality);
  for (auto dptr : dataset) {
    result.AppendOrDie(Bfloat16QuantizeFloatDatapoint(dptr, dp_span), "");
  }
  return result;
}

DatapointPtr<int16_t> Bfloat16QuantizeFloatDatapoint(
    const DatapointPtr<float>& dptr, std::vector<int16_t>* quantized_storage) {
  MutableSpan<int16_t> quantized(quantized_storage->data(),
                                 quantized_storage->size());
  return Bfloat16QuantizeFloatDatapoint(dptr, quantized);
}

DatapointPtr<int16_t> Bfloat16QuantizeFloatDatapoint(
    const DatapointPtr<float>& dptr, MutableSpan<int16_t> quantized) {
  const size_t dimensionality = dptr.dimensionality();
  DCHECK_EQ(quantized.size(), dimensionality);
  for (size_t i : Seq(dimensionality)) {
    quantized[i] = Bfloat16Quantize(dptr.values()[i]);
  }
  return MakeDatapointPtr(quantized.data(), quantized.size());
}

class Bfloat16NoiseShapingLambdas {
 public:
  Bfloat16NoiseShapingLambdas() = default;

  int16_t QuantizeSingleDim(float f, DimensionIndex dim_idx) const {
    return Bfloat16Quantize(f);
  }

  float DequantizeSingleDim(int16_t f, DimensionIndex dim_idx) const {
    return Bfloat16Decompress(f);
  }

  bool AtMaximum(int16_t i) const {
    return i == static_cast<int16_t>(0b0'11111110'1111111);
  }
  bool AtMinimum(int16_t i) const {
    return i == static_cast<int16_t>(0b1'11111110'1111111);
  }

  int16_t NextLargerDelta(int16_t i) const { return (i & kSignBit) ? -1 : 1; }
  int16_t NextSmallerDelta(int16_t i) const { return -NextLargerDelta(i); }

 private:
  enum : int16_t { kSignBit = static_cast<int16_t>(0b1000000000000000) };
};

DatapointPtr<int16_t> Bfloat16QuantizeFloatDatapointWithNoiseShaping(
    const DatapointPtr<float>& dptr, double noise_shaping_threshold,
    MutableSpan<int16_t> quantized, MutableSpan<float> residuals,
    MutableSpan<uint32_t> dims, int* num_changes, double* residual_ptr,
    double* parallel_residual_ptr) {
  return ScalarQuantizeFloatDatapointWithNoiseShapingImpl(
      dptr, noise_shaping_threshold, quantized, residuals, dims,
      Bfloat16NoiseShapingLambdas(), num_changes, residual_ptr,
      parallel_residual_ptr);
}

DatapointPtr<int16_t> Bfloat16QuantizeFloatDatapointWithNoiseShaping(
    const DatapointPtr<float>& dptr, double noise_shaping_threshold,
    MutableSpan<int16_t> quantized, int* num_changes, double* residual_ptr,
    double* parallel_residual_ptr) {
  vector<float> residuals(quantized.size());
  vector<uint32_t> dims(quantized.size());
  return Bfloat16QuantizeFloatDatapointWithNoiseShaping(
      dptr, noise_shaping_threshold, quantized, MakeMutableSpan(residuals),
      MakeMutableSpan(dims), num_changes, residual_ptr, parallel_residual_ptr);
}

DatapointPtr<int16_t> Bfloat16QuantizeFloatDatapointWithNoiseShaping(
    const DatapointPtr<float>& dptr, double noise_shaping_threshold,
    std::vector<int16_t>* quantized_storage, int* num_changes,
    double* residual_ptr, double* parallel_residual_ptr) {
  quantized_storage->resize(dptr.dimensionality());
  MutableSpan<int16_t> quantized(quantized_storage->data(),
                                 quantized_storage->size());
  return Bfloat16QuantizeFloatDatapointWithNoiseShaping(
      dptr, noise_shaping_threshold, quantized, num_changes, residual_ptr,
      parallel_residual_ptr);
}

DenseDataset<int16_t> Bfloat16QuantizeFloatDatasetWithNoiseShaping(
    const DenseDataset<float>& dataset, float noise_shaping_threshold,
    ThreadPool* pool) {
  const size_t dimensionality = dataset.dimensionality();
  vector<int16_t> result(dimensionality * dataset.size());

  ParallelFor<128>(IndicesOf(dataset), pool, [&](size_t dp_idx) {
    MutableSpan<int16_t> quantized(result.data() + dp_idx * dimensionality,
                                   dimensionality);
    Bfloat16QuantizeFloatDatapointWithNoiseShaping(
        dataset[dp_idx], noise_shaping_threshold, quantized);
  });
  return DenseDataset<int16_t>(std::move(result), dataset.size());
}

}  // namespace research_scann
