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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU  // For definition of Eigen::GpuDevice.
#include "sparse/ops/cc/common.h"
#include "sparse/ops/cc/spmm_launcher.h"
#include "sputnik/sputnik.h"

namespace sgk {

// CUDA kernel launcher.
void LaunchSpmm(const Eigen::GpuDevice &d, int m, int k, int n, int nonzeros,
                const float *values, const int *row_indices,
                const int *row_offsets, const int *column_indices,
                const float *dense_matrix, const float *bias,
                float *output_matrix) {
  // TODO(tgale): There should be a TensorFlow approach to checking
  // cudaError_t objects correctly. Switch to this, whatever it is.
  //
  // NOTE: Passing nullptr as bias will execute the standard spmm w/
  // no bias or relu.
  CUDA_CALL(sputnik::CudaSpmmBiasRelu(m, k, n, nonzeros, row_indices, values,
                                      row_offsets, column_indices, dense_matrix,
                                      bias, output_matrix, d.stream()));
}

}  // namespace sgk

#endif  // GOOGLE_CUDA
