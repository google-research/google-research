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

#include "sparse/ops/cc/transpose_launcher.h"

#include <limits>

namespace sgk {

void AllocateTransposeWorkspace(
    tensorflow::OpKernelContext *context, const Eigen::ThreadPoolDevice &d,
    int m, int n, int nonzeros, const float *values, const int *row_offsets,
    const int *column_indices, float *output_values, int *output_row_offsets,
    int *output_column_indices, tensorflow::Tensor *workspace) {
  // To transpose the matrix, we blow up the tensor into it's
  // dense, transposed representation and compress it back down.
  tensorflow::TensorShape shape = {m * n};
  OP_REQUIRES_OK(
      context, context->allocate_temp(tensorflow::DT_FLOAT, shape, workspace));
}

void LaunchTranspose(const Eigen::ThreadPoolDevice &d, int m, int n,
                     int nonzeros, const float *values, const int *row_offsets,
                     const int *column_indices, float *output_values,
                     int *output_row_offsets, int *output_column_indices,
                     float *workspace) {
  // Expand the tensor into it's tranposed dense representation.
  //
  // NOTE: We set the invalid values in the tensor to infinity. This
  // This avoids issues with the case where we have zero valued weights
  // in the sparse matrix.
  for (int i = 0; i < m * n; ++i) {
    workspace[i] = std::numeric_limits<float>::infinity();
  }
  for (int i = 0; i < m; ++i) {
    for (int l = row_offsets[i]; l < row_offsets[i + 1]; ++l) {
      int j = column_indices[l];
      workspace[j * m + i] = values[l];
    }
  }

  // Compress the matrix back down to it's sparse representation. Note
  // that the matrix is transposed, so 'n' is the number of rows and
  // 'm' is the number of columns.
  int offset = 0;
  output_row_offsets[0] = 0;
  for (int i = 0; i < n; ++i) {    // loop over rows.
    for (int j = 0; j < m; ++j) {  // loop over columns.
      int idx = i * m + j;
      if (workspace[idx] == std::numeric_limits<float>::infinity()) {
        continue;
      }
      DCHECK_LT(offset, nonzeros);
      output_values[offset] = workspace[idx];
      output_column_indices[offset] = j;
      ++offset;
    }
    output_row_offsets[i + 1] = offset;
  }
}

}  // namespace sgk
