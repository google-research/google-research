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

#ifndef TF3D_OPS_SUBMANIFOLD_SPARSE_CONV_LAUNCHER_H_
#define TF3D_OPS_SUBMANIFOLD_SPARSE_CONV_LAUNCHER_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tf3d {

struct SubmanifoldSparseConvLaunchOptions {
  // int[batch_size, max_num_coords_per_batch, 2 or 3], the padded 2D/3D
  // coordinates. The third dimension determines the type of the convolution to
  // run: 2 means 2D and 3 means 3D.
  //
  // max_num_coords_per_batch is the max number of coordinates in each batch
  // item.
  const Tensor& coordinates;

  // int[batch_size], the number of valid coordinates per batch item. Only the
  // top num_valid_coordinates[i] entries in coordinates[i], input_features[i],
  // and output_features[i] are valid. The rest of the entries are paddings.
  const Tensor& num_valid_coordinates;

  // float[batch_size, max_num_coords_per_batch, in_channels], the input feature
  // map for each coordinates, where in_channels is the channel size of the
  // feature map.
  const Tensor& input_features;

  // The convolution filter (kernel).
  //
  // For 2D convolution, the shape is:
  // float[filter_height, filter_width, in_channels, out_channels]
  //
  // For 3D convolution, the shape is:
  // float[filter_depth, filter_height, filter_width, in_channels, out_channels]
  const Tensor& filter;

  // float[batch_size, max_num_coords_per_batch, out_channels], the output
  // feature map for each coordinates, where out_channels is the channel size of
  // the output feature.
  Tensor* output_features;

  // The context under which to run the convolutions.
  OpKernelContext* ctx;

  int batch_size() const { return coordinates.dim_size(0); }
  int max_num_coords_per_batch() const { return coordinates.dim_size(1); }
  int in_channels() const { return filter.dim_size(filter.dims() - 2); }
  int out_channels() const { return filter.dim_size(filter.dims() - 1); }
};

// Launch submanifold sparse convolutions with specific device type.
template <typename Device>
Status LaunchSubmanifoldSparseConvolution(
    const SubmanifoldSparseConvLaunchOptions& opts);

}  // namespace tf3d
}  // namespace tensorflow

#endif  // TF3D_OPS_SUBMANIFOLD_SPARSE_CONV_LAUNCHER_H_
