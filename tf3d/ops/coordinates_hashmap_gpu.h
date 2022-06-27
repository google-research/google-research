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

#ifndef TF3D_OPS_COORDINATES_HASHMAP_GPU_H_
#define TF3D_OPS_COORDINATES_HASHMAP_GPU_H_

#if GOOGLE_CUDA
#include <vector>

#define EIGEN_USE_GPU  // For definition of Eigen::GpuDevice.
#include "submanifold_sparse_conv_utils.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class OpKernelContext;

namespace tf3d {
namespace cuda {

// A hashmap that maps Coordinates to their indices in the input tensor.
// The hashmap is immutable after Initialize().
template <int dims>
class CoordinatesHashMapGpu {
 public:
  // Initialize from input coordinates. Store all the created Tensors in
  // hashmap_tensors, whose lifetime needs to be longer than this hashmap.
  //
  // - coordinates: int32[batch_size, max_num_coords_per_batch, dims]
  // - num_valid_coordinates: int32[batch_size]. The number of valid coordinates
  //   for each batch.
  Status Initialize(const Tensor& coordinates,
                    const Tensor& num_valid_coordinates, OpKernelContext* ctx,
                    std::vector<Tensor>* hashmap_tensors);

  // Similar as the `Initialize` above but takes a `coordinates` tensor without
  // batch dimension.
  //
  // Initialize from coordinates. Store all the created Tensors in
  // hashmap_tensors, whose lifetime needs to be longer than this hashmap.
  //
  // - coordinates: int32[num_coords, dims]
  Status InitializeNoBatch(const Tensor& coordinates, OpKernelContext* ctx,
                           std::vector<Tensor>* hashmap_tensors);

  // Initialize from tensors created/initlized by another CoordinatesHashMapGpu.
  Status Initialize(std::vector<Tensor>* hashmap_tensors);

  // Lookup a Coordinate in a particular input batch determined by batch_id, and
  // return its index in the original coordinates array.
  __device__ __forceinline__ int32 Lookup(const int batch_id,
                                          const Coordinates<dims> key) const {
    int32 entry_idx = d_buckets_[GetCoordinateHashIndex(batch_id, key)];
    while (entry_idx >= 0) {
      HashEntry* entry = d_linked_list_ + entry_idx;
      if (entry->key == key) return entry_idx;
      entry_idx = entry->next_entry_idx;
    }
    return -1;
  }

 private:
  struct __align__(8) HashEntry {
    int32 next_entry_idx;
    Coordinates<dims> key;
  };

  // The internal version of the `Initialize` methods. It does the real work.
  // See the public `Initialize` for more details.
  // When batch_size is 1, d_num_valid_coordinates can be optionally set to
  // nullptr as the number of valid coordinates in `d_coordinates` is the same
  // as `max_num_coords_per_batch`.
  Status InitializeInternal(const int32* d_coordinates,
                            const int32* d_num_valid_coordinates,
                            const int32 batch_size,
                            const int32 max_num_coords_per_batch,
                            OpKernelContext* ctx,
                            std::vector<Tensor>* hashmap_tensors);

  // Given the max number of coordinates for each batch, return the number of
  // elements needed by d_buckets_ for the batch. This function is used to
  // determine the size of memory to allocate for the hashmap during
  // Initialize(). The returned value is always a power of two.
  static int32 MapSizePerBatch(const int32 max_num_coords_per_batch);

  // Same as above but is used after Initialize().
  __device__ __forceinline__ int32 map_size_per_batch() const {
    return map_size_mask_ + 1;
  }

  // Get the index of the coordinates in the d_buckets_ array.
  __device__ __forceinline__ int32 GetCoordinateHashIndex(
      const int batch_id, const Coordinates<dims> key) const {
    return batch_id * map_size_per_batch() + (key.hash() & map_size_mask_);
  }

  // We implement the hashmap using buckets and linked list, and this is the
  // buckets array. Its size is batch_size*map_size_per_batch().
  int32* d_buckets_ = nullptr;  // Not owned.

  // The linked list of the hashmap. Its size is
  // batch_size*max_num_coords_per_batch.
  HashEntry* d_linked_list_ = nullptr;  // Not owned.

  // map_size_per_batch()-1. A bit mask used for fast modulo operation.
  int32 map_size_mask_;

  // Note: CUDA kernels cannot be member function, so we make them friend
  // functions instead.
  template <int friend_dims>
  friend __global__ void ResetKernel(
      const int map_size, const int linked_list_size,
      CoordinatesHashMapGpu<friend_dims> hashmap);

  template <int friend_dims>
  friend __global__ void InitializeMapEntriesKernel(
      const int batch_size, const int max_num_coords_per_batch,
      const int32* __restrict__ coordinates,
      const int32* __restrict__ num_valid_coordinates,
      CoordinatesHashMapGpu<friend_dims> hashmap);
};

}  // namespace cuda
}  // namespace tf3d
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // TF3D_OPS_COORDINATES_HASHMAP_GPU_H_
