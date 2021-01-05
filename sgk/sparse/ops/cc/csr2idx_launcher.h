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

#ifndef SGK_SPARSE_OPS_CC_CSR2IDX_LAUNCHER_H_
#define SGK_SPARSE_OPS_CC_CSR2IDX_LAUNCHER_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace sgk {

void LaunchCsr2idx(const Eigen::ThreadPoolDevice &d, int m, int n, int nonzeros,
                   const int *row_offsets, const int *column_indices,
                   int *linear_indices);

#if GOOGLE_CUDA
#define EIGEN_USE_GPU  // For definition of Eigen::GpuDevice.

void LaunchCsr2idx(const Eigen::GpuDevice &d, int m, int n, int nonzeros,
                   const int *row_offsets, const int *column_indices,
                   int *linear_indices);

#endif  // GOOGLE_CUDA
}  // namespace sgk

#endif  // SGK_SPARSE_OPS_CC_CSR2IDX_LAUNCHER_H_
