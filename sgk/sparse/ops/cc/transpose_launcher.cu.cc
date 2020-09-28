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
#include "sparse/ops/cc/cusparse_util.h"
#include "sparse/ops/cc/transpose_launcher.h"

namespace sgk {

void AllocateTransposeWorkspace(
    tensorflow::OpKernelContext *context, const Eigen::GpuDevice &d, int m,
    int n, int nonzeros, const float *values, const int *row_offsets,
    const int *column_indices, float *output_values, int *output_row_offsets,
    int *output_column_indices, tensorflow::Tensor *workspace) {
  // Calculate the buffer size.
  size_t buffer_size = 0;
  CUSPARSE_CALL(cusparseCsr2cscEx2_bufferSize(
      GetHandleForStream(d.stream()), m, n, nonzeros, values, row_offsets,
      column_indices, output_values, output_row_offsets, output_column_indices,
      CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
      CUSPARSE_CSR2CSC_ALG1, &buffer_size));

  // Allocate the temporary buffer. Round up to the nearest
  // float for the size of the buffer.
  int64 buffer_size_signed = (buffer_size + sizeof(float) - 1) / sizeof(float);
  tensorflow::TensorShape shape = {buffer_size_signed};
  OP_REQUIRES_OK(
      context, context->allocate_temp(tensorflow::DT_FLOAT, shape, workspace));
}

void LaunchTranspose(const Eigen::GpuDevice &d, int m, int n, int nonzeros,
                     const float *values, const int *row_offsets,
                     const int *column_indices, float *output_values,
                     int *output_row_offsets, int *output_column_indices,
                     float *workspace) {
  // Launch the kernel.
  CUSPARSE_CALL(cusparseCsr2cscEx2(
      GetHandleForStream(d.stream()), m, n, nonzeros, values, row_offsets,
      column_indices, output_values, output_row_offsets, output_column_indices,
      CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
      CUSPARSE_CSR2CSC_ALG1, workspace));
}

}  // namespace sgk
#endif  // GOOGLE_CUDA
