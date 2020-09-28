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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU  // For definition of Eigen::GpuDevice.
#include "sparse/ops/cc/common.h"
#include "sparse/ops/cc/sparse_softmax_launcher.h"
#include "sputnik/sputnik.h"

namespace sgk {

void LaunchSparseSoftmax(const Eigen::GpuDevice &d, int m, int n, int nonzeros,
                         const float *values, const int *row_indices,
                         const int *row_offsets, const int *column_indices,
                         float *output_values) {
  CUDA_CALL(sputnik::SparseSoftmax(m, n, nonzeros, values, row_indices,
                                   row_offsets, column_indices, output_values,
                                   d.stream()));
}

}  // namespace sgk

#endif  // GOOGLE_CUDA
