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

#include "sparse/ops/cc/sddmm_launcher.h"

namespace sgk {

// Simple CPU kernel launcher.
void LaunchSddmm(const Eigen::ThreadPoolDevice &d, int m, int k, int n,
                 int nonzeros, const int *row_indices, const int *row_offsets,
                 const int *column_indices, const float *lhs_matrix,
                 const float *rhs_matrix, float *output_values) {
  for (int i = 0; i < m; ++i) {
    for (int j = row_offsets[i]; j < row_offsets[i + 1]; ++j) {
      int idx_n = column_indices[j];
      float accumulator = 0.0f;
      for (int l = 0; l < k; ++l) {
        accumulator += lhs_matrix[i * k + l] * rhs_matrix[idx_n * k + l];
      }
      output_values[j] = accumulator;
    }
  }
}

}  // namespace sgk
