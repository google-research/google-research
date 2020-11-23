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

#include "coordinates_hashmap_wrapper.h"

#if GOOGLE_CUDA
#define EIGEN_USE_GPU  // For definition of Eigen::GpuDevice.
#include <string>

#include "coordinates_hashmap_gpu.h"
#include "submanifold_sparse_conv_utils.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace tf3d {
namespace cuda {

template <int dims>
__global__ void GetCoordinatesNeighborIndices(
    const CoordinatesHashMapGpu<dims> hashmap, const int batch_size,
    const int max_num_coords_per_batch,
    const FilterSpatialDims<dims> filter_dims,
    const int32* __restrict__ coordinates,
    const int32* __restrict__ num_valid_coordinates,
    int32* __restrict__ neighbor_indices) {
  for (int i : GpuGridRangeX(batch_size * max_num_coords_per_batch)) {
    const int cur_batch = i / max_num_coords_per_batch;
    const int cur_coords_id = i - cur_batch * max_num_coords_per_batch;
    // Skip invalid coordinates.
    if (cur_coords_id >= num_valid_coordinates[cur_batch]) {
      for (int offset = 0; offset < filter_dims.Size(); ++offset) {
        neighbor_indices[i + offset * batch_size * max_num_coords_per_batch] =
            -1;
      }
      continue;
    }

    const int32* cur_coords = coordinates + dims * i;
    NeighborIterator<dims> iter(cur_coords, filter_dims);
    for (int offset = 0; iter.Next(); ++offset) {
      const int index = hashmap.Lookup(cur_batch, iter.Get());
      neighbor_indices[i + offset * batch_size * max_num_coords_per_batch] =
          index;
    }
  }
}

template <int dims>
Status CoordinatesHashMapWrapper<dims>::Initialize(
    const Tensor& coordinates, const Tensor& num_valid_coordinates,
    OpKernelContext* ctx) {
  if (coordinates.dtype() != DT_INT32) {
    return errors::InvalidArgument(
        "Datatype mismatch. Expected: ", DataTypeString(DT_INT32),
        ", but got: ", DataTypeString(coordinates.dtype()));
  }
  if (coordinates.dims() != 3) {
    return errors::InvalidArgument("The coordinates tensor must be of rank 3.");
  }
  if (coordinates.dim_size(2) != dims) {
    return errors::InvalidArgument("Only ", dims,
                                   "D coordinates are supported.");
  }
  if (num_valid_coordinates.dims() != 1 ||
      num_valid_coordinates.dim_size(0) != coordinates.dim_size(0)) {
    return errors::InvalidArgument(
        "The num_valid_coordinates tensor must be of shape [batch_size], but "
        "got ",
        num_valid_coordinates.shape());
  }

  // Initialize the gpu data.
  TF_RETURN_IF_ERROR(gpu_hash_map_.Initialize(
      coordinates, num_valid_coordinates, ctx, &hashmap_tensors_));

  return Status::OK();
}

template <int dims>
Status CoordinatesHashMapWrapper<dims>::GetNeighborIndices(
    const Tensor& coordinates, const Tensor& num_valid_coordinates,
    const TensorShape& filter_shape, OpKernelContext* ctx,
    Tensor* neighbor_indices) const {
  const int batch_size = coordinates.dim_size(0);
  const int max_num_coords_per_batch = coordinates.dim_size(1);
  FilterSpatialDims<dims> filter_dims;
  TF_RETURN_IF_ERROR(
      FilterSpatialDims<dims>::FromFilterShape(filter_shape, &filter_dims));

  // Allocate the fast lookup table.
  TF_RETURN_IF_ERROR(ctx->allocate_temp(
      DT_INT32,
      TensorShape({filter_dims.Size(), batch_size, max_num_coords_per_batch}),
      neighbor_indices));

  // Initialize the fast lookup table.
  Eigen::GpuDevice d = ctx->template eigen_device<Eigen::GpuDevice>();
  GpuLaunchConfig config =
      GetGpuLaunchConfig(batch_size * max_num_coords_per_batch, d);
  TF_CHECK_OK(GpuLaunchKernel(
      GetCoordinatesNeighborIndices<dims>, config.block_count,
      config.thread_per_block,
      /*shared_memory_size_bytes=*/0, d.stream(), gpu_hash_map_, batch_size,
      max_num_coords_per_batch, filter_dims, coordinates.flat<int32>().data(),
      num_valid_coordinates.flat<int32>().data(),
      neighbor_indices->flat<int32>().data()));
  return Status::OK();
}

template class CoordinatesHashMapWrapper<2>;
template class CoordinatesHashMapWrapper<3>;

}  // namespace cuda
}  // namespace tf3d
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
