// Copyright 2022 The Google Research Authors.
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

#include "sparse/ops/cc/csr2idx_launcher.h"

namespace sgk {

void LaunchCsr2idx(const Eigen::ThreadPoolDevice &d, int m, int n, int nonzeros,
                   const int *row_offsets, const int *column_indices,
                   int *linear_indices) {
  for (int i = 0; i < m; ++i) {
    for (int l = row_offsets[i]; l < row_offsets[i + 1]; ++l) {
      linear_indices[l] = column_indices[l] * i * n;
    }
  }
}

}  // namespace sgk
