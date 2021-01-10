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



#include "scann/data_format/dataset.h"
#include "scann/utils/types.h"

#ifndef SCANN_UTILS_FIXED_POINT_PRE_QUANTIZED_FIXED_POINT_H_
#define SCANN_UTILS_FIXED_POINT_PRE_QUANTIZED_FIXED_POINT_H_

namespace research_scann {

struct PreQuantizedFixedPoint {
  shared_ptr<DenseDataset<int8_t>> fixed_point_dataset = nullptr;

  shared_ptr<vector<float>> multiplier_by_dimension = nullptr;

  shared_ptr<vector<float>> squared_l2_norm_by_datapoint = nullptr;
};

inline PreQuantizedFixedPoint CreatePreQuantizedFixedPoint(
    const DenseDataset<int8_t>& dataset, const vector<float>& multipliers,
    const vector<float>& norms, bool reciprocate = false) {
  PreQuantizedFixedPoint res;
  res.fixed_point_dataset = make_shared<DenseDataset<int8_t>>(
      vector<int8_t>(dataset.data().begin(), dataset.data().end()),
      dataset.docids()->Copy());
  res.multiplier_by_dimension =
      make_shared<vector<float>>(multipliers.begin(), multipliers.end());
  res.squared_l2_norm_by_datapoint =
      make_shared<vector<float>>(norms.begin(), norms.end());
  if (reciprocate) {
    for (float& m : *res.multiplier_by_dimension) m = 1.0f / m;
  }
  return res;
}

}  // namespace research_scann

#endif
