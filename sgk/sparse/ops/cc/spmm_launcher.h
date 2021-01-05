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

#ifndef SGK_SPARSE_OPS_CC_SPMM_LAUNCHER_H_
#define SGK_SPARSE_OPS_CC_SPMM_LAUNCHER_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace sgk {

/**
 * @brief Helper to launch spmm kernel on the device.
 */
void LaunchSpmm(const Eigen::ThreadPoolDevice &d, int m, int k, int n,
                int nonzeros, const float *values, const int *row_indices,
                const int *row_offsets, const int *column_indices,
                const float *dense_matrix, const float *bias,
                float *output_matrix);

#if GOOGLE_CUDA
void LaunchSpmm(const Eigen::GpuDevice &d, int m, int k, int n, int nonzeros,
                const float *values, const int *row_indices,
                const int *row_offsets, const int *column_indices,
                const float *dense_matrix, const float *bias,
                float *output_matrix);
#endif  // GOOGLE_CUDA

}  // namespace sgk
#endif  // SGK_SPARSE_OPS_CC_SPMM_LAUNCHER_H_
