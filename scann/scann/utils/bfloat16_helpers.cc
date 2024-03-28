// Copyright 2024 The Google Research Authors.
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
#include <vector>

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"

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

}  // namespace research_scann
