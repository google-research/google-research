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

#include "sparse/ops/cc/spmm_launcher.h"

namespace sgk {

// Simple CPU kernel launcher.
void LaunchSpmm(const Eigen::ThreadPoolDevice &d, int m, int k, int n,
                int nonzeros, const float *values, const int *row_indices,
                const int *row_offsets, const int *column_indices,
                const float *dense_matrix, const float *bias,
                float *output_matrix) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float accumulator = 0.0f;
      for (int l = row_offsets[i]; l < row_offsets[i + 1]; ++l) {
        int column_index = column_indices[l];
        accumulator += values[l] * dense_matrix[column_index * n + j];
      }
      // If we're passed a bias, we're running the fused kernel. Apply
      // the bias and ReLU.
      if (bias != nullptr) {
        accumulator += bias[i];
        accumulator = accumulator > 0 ? accumulator : 0;
      }
      output_matrix[i * n + j] = accumulator;
    }
  }
}

}  // namespace sgk
