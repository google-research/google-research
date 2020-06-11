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

/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */



#include "scann/data_format/dataset.h"
#include "scann/utils/types.h"

#ifndef SCANN__UTILS_FIXED_POINT_PRE_QUANTIZED_FIXED_POINT_H_
#define SCANN__UTILS_FIXED_POINT_PRE_QUANTIZED_FIXED_POINT_H_

namespace tensorflow {
namespace scann_ops {

struct PreQuantizedFixedPoint {
  shared_ptr<DenseDataset<int8_t>> fixed_point_dataset = nullptr;

  shared_ptr<vector<float>> multiplier_by_dimension = nullptr;

  shared_ptr<vector<float>> squared_l2_norm_by_datapoint = nullptr;
};

}  // namespace scann_ops
}  // namespace tensorflow

#endif
