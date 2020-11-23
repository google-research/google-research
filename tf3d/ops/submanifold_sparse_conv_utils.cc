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

#include "submanifold_sparse_conv_utils.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tf3d {

Status ValidateConvInputs(bool is_grad_op, int dims, OpKernelContext* ctx) {
  // Sparse coordinates.
  const Tensor& coordinates = ctx->input(0);
  if (coordinates.dtype() != DT_INT32) {
    return errors::InvalidArgument(
        "Datatype of input coordinates must be DT_INT32.");
  }
  if (coordinates.dims() != 3) {
    return errors::InvalidArgument("The coordinates tensor must be of rank 3.");
  }

  const int batch_size = coordinates.dim_size(0);
  const int max_num_coords_per_batch = coordinates.dim_size(1);
  if (coordinates.dim_size(2) != dims) {
    return errors::InvalidArgument(
        "The last dimension of the coordinates tensor must be ", dims);
  }

  // Valid coordinate counters.
  const Tensor& num_valid_coordinates = ctx->input(1);
  if (num_valid_coordinates.dims() != 1 ||
      (num_valid_coordinates.dims() == 1 &&
       num_valid_coordinates.dim_size(0) != batch_size)) {
    return errors::InvalidArgument(
        "The num_valid_coordinates tensor must be of "
        "shape [batch_size], but got ",
        num_valid_coordinates.shape());
  }

  // Sparse features.
  const Tensor& input_features = ctx->input(2);
  if (input_features.dtype() != DT_FLOAT) {
    return errors::InvalidArgument(
        "Currently only float32 input features are supported.");
  }
  if (input_features.dims() != 3) {
    return errors::InvalidArgument(
        "The input feature tensor must be of rank 3.");
  }
  if (input_features.dim_size(0) != batch_size) {
    return errors::InvalidArgument(
        "The input feature batch size doesn't "
        "match the coordinates batch size.");
  }
  if (input_features.dim_size(1) != max_num_coords_per_batch) {
    return errors::InvalidArgument(
        "The number of input features doesn't "
        "match the number of coordinates.");
  }

  // Filter.
  const Tensor& filter = ctx->input(3);
  if (filter.dtype() != input_features.dtype()) {
    return errors::InvalidArgument("Filter dtype doesn't match feature dtype.");
  }
  if (filter.dims() != dims + 2) {
    return errors::InvalidArgument("Filter must be of rank ", dims + 2, ".");
  }

  const int in_channels = filter.dim_size(dims);
  const int out_channels = filter.dim_size(dims + 1);
  if (in_channels != input_features.dim_size(2)) {
    return errors::InvalidArgument("Number of input feature channels (",
                                   input_features.dim_size(2),
                                   ") must be the same as the "
                                   "input channel size of the filter (",
                                   in_channels, ").");
  }
  if ((filter.dim_size(0) & filter.dim_size(1) &
       (dims == 2 ? 1 : filter.dim_size(2)) & 1) == 0) {
    return errors::InvalidArgument(
        "The ", (dims == 2 ? "" : "depth, "),
        "height and width of the filter must be odd "
        "numbers for submanifold sparse convolutions.");
  }

  if (is_grad_op) {
    const Tensor& d_output_features = ctx->input(4);
    if (d_output_features.dtype() != DT_FLOAT) {
      return errors::InvalidArgument(
          "Currently only float32 output features are supported.");
    }
    if (d_output_features.dims() != 3) {
      return errors::InvalidArgument(
          "The output feature tensor must be of rank 3.");
    }
    if (d_output_features.dim_size(0) != batch_size) {
      return errors::InvalidArgument(
          "The output feature batch size doesn't "
          "match the coordinates batch size.");
    }
    if (d_output_features.dim_size(1) != max_num_coords_per_batch) {
      return errors::InvalidArgument(
          "The number of output features doesn't "
          "match the number of coordinates.");
    }
    if (d_output_features.dim_size(2) != out_channels) {
      return errors::InvalidArgument(
          "The output feature channels doesn't match the number of output "
          "channels of the filter.");
    }
  }
  return Status::OK();
}

}  // namespace tf3d
}  // namespace tensorflow
