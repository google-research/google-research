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

#include "coordinates_hashmap_gpu.h"

#if GOOGLE_CUDA
#include <algorithm>
#include <vector>

#define EIGEN_USE_GPU  // For definition of Eigen::GpuDevice.
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
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace tf3d {
namespace cuda {

// CUDA kernel to reset the hashmap buckets and linked list.
template <int dims>
__global__ void ResetKernel(const int map_size, const int linked_list_size,
                            CoordinatesHashMapGpu<dims> hashmap) {
  int32* __restrict__ map = hashmap.d_buckets_;
  auto* __restrict__ linked_list = hashmap.d_linked_list_;
  const int32 kInvalidIdx = -1;
  for (int i : GpuGridRangeX(map_size)) {
    map[i] = kInvalidIdx;
  }
  for (int i : GpuGridRangeX(linked_list_size)) {
    linked_list[i].next_entry_idx = kInvalidIdx;
  }
}

// CUDA kernel to initialize the hashmap using given input.
template <int dims>
__global__ void InitializeMapEntriesKernel(
    const int batch_size, const int max_num_coords_per_batch,
    const int32* __restrict__ coordinates,
    const int32* __restrict__ num_valid_coordinates,
    CoordinatesHashMapGpu<dims> hashmap) {
  int32* __restrict__ map = hashmap.d_buckets_;
  auto* __restrict__ linked_list = hashmap.d_linked_list_;
  for (int i : GpuGridRangeX(batch_size * max_num_coords_per_batch)) {
    const int cur_batch = i / max_num_coords_per_batch;
    // Skip invalid coordinates.
    const int cur_coords_id = i - cur_batch * max_num_coords_per_batch;
    if (num_valid_coordinates &&
        cur_coords_id >= num_valid_coordinates[cur_batch]) {
      continue;
    }

    const int32* cur_coords = coordinates + dims * i;
    Coordinates<dims> key(cur_coords);
    const int32 idx = hashmap.GetCoordinateHashIndex(cur_batch, key);
    auto* entry = linked_list + i;

    // Copy the key and value.
    entry->key = key;

    // Update `map` in an atomic way.
    int32* address = map + idx;
    int32 new_value = i;
    int32 cur_value = *address;
    int32 assumed;
    do {
      assumed = cur_value;
      cur_value = atomicCAS(address, assumed, new_value);
    } while (assumed != cur_value);
    entry->next_entry_idx = cur_value;
  }
}

template <int dims>
Status CoordinatesHashMapGpu<dims>::Initialize(
    std::vector<Tensor>* hashmap_tensors) {
  Tensor& map_tensor = hashmap_tensors->at(0);
  Tensor& linked_list_tensor = hashmap_tensors->at(1);

  const int batch_size = map_tensor.dim_size(0);
  QCHECK_EQ(0, linked_list_tensor.TotalBytes() % sizeof(HashEntry));
  const int linked_list_size =
      linked_list_tensor.TotalBytes() / sizeof(HashEntry);
  QCHECK_EQ(0, linked_list_size % batch_size);

  // Set the array sizes.
  map_size_mask_ = map_tensor.dim_size(1) - 1;

  // Set the data arrays.
  d_buckets_ = map_tensor.flat<int32>().data();
  d_linked_list_ =
      reinterpret_cast<HashEntry*>(linked_list_tensor.flat<uint8>().data());
  return Status::OK();
}

template <int dims>
Status CoordinatesHashMapGpu<dims>::InitializeInternal(
    const int32* d_coordinates, const int32* d_num_valid_coordinates,
    const int32 batch_size, const int32 max_num_coords_per_batch,
    OpKernelContext* ctx, std::vector<Tensor>* hashmap_tensors) {
  const int map_size_per_batch = MapSizePerBatch(max_num_coords_per_batch);
  const int map_size = batch_size * map_size_per_batch;
  const int linked_list_size = batch_size * max_num_coords_per_batch;
  hashmap_tensors->clear();

  // Allocate the map.
  Tensor map_tensor;
  TF_RETURN_IF_ERROR(ctx->allocate_temp(
      DT_INT32, TensorShape({batch_size, map_size_per_batch}), &map_tensor));
  hashmap_tensors->push_back(map_tensor);

  // Allocate the linkedlist.
  Tensor linked_list_tensor;
  const int linked_list_bytes = linked_list_size * sizeof(HashEntry);
  TF_RETURN_IF_ERROR(ctx->allocate_temp(
      DT_UINT8, TensorShape({linked_list_bytes}), &linked_list_tensor));
  hashmap_tensors->push_back(linked_list_tensor);

  // Initialize the data members.
  TF_RETURN_IF_ERROR(Initialize(hashmap_tensors));

  Eigen::GpuDevice d = ctx->template eigen_device<Eigen::GpuDevice>();

  // Reset the hashmap.
  const int reset_count = std::max(map_size, linked_list_size);
  GpuLaunchConfig config = GetGpuLaunchConfig(reset_count, d);
  TF_CHECK_OK(GpuLaunchKernel(ResetKernel<dims>, config.block_count,
                              config.thread_per_block,
                              /*shared_memory_size_bytes=*/0, d.stream(),
                              map_size, linked_list_size, *this));

  // Initialized the hashmap using given input.
  config = GetGpuLaunchConfig(batch_size * max_num_coords_per_batch, d);
  TF_CHECK_OK(GpuLaunchKernel(InitializeMapEntriesKernel<dims>,
                              config.block_count, config.thread_per_block,
                              /*shared_memory_size_bytes=*/0, d.stream(),
                              batch_size, max_num_coords_per_batch,
                              d_coordinates, d_num_valid_coordinates, *this));
  return Status::OK();
}

template <int dims>
Status CoordinatesHashMapGpu<dims>::Initialize(
    const Tensor& coordinates, const Tensor& num_valid_coordinates,
    OpKernelContext* ctx, std::vector<Tensor>* hashmap_tensors) {
  const int batch_size = coordinates.dim_size(0);
  const int max_num_coords_per_batch = coordinates.dim_size(1);
  return InitializeInternal(coordinates.flat<int32>().data(),
                            num_valid_coordinates.flat<int32>().data(),
                            batch_size, max_num_coords_per_batch, ctx,
                            hashmap_tensors);
}

template <int dims>
Status CoordinatesHashMapGpu<dims>::InitializeNoBatch(
    const Tensor& coordinates, OpKernelContext* ctx,
    std::vector<Tensor>* hashmap_tensors) {
  const int num_coords = coordinates.dim_size(0);
  return InitializeInternal(coordinates.flat<int32>().data(),
                            /*d_num_valid_coordinates=*/nullptr,
                            /*batch_size=*/1, num_coords, ctx, hashmap_tensors);
}

// static
template <int dims>
int32 CoordinatesHashMapGpu<dims>::MapSizePerBatch(
    const int32 max_num_coords_per_batch) {
  int32 map_size = 1;
  while (map_size < max_num_coords_per_batch) map_size <<= 1;
  map_size <<= 1;
  return map_size;
}

// Explicitly instantiate the templates.
template class CoordinatesHashMapGpu<1>;
template class CoordinatesHashMapGpu<2>;
template class CoordinatesHashMapGpu<3>;
template class CoordinatesHashMapGpu<4>;

}  // namespace cuda
}  // namespace tf3d
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
