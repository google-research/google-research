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

#ifndef TF3D_OPS_COORDINATES_HASHMAP_WRAPPER_H_
#define TF3D_OPS_COORDINATES_HASHMAP_WRAPPER_H_

#if GOOGLE_CUDA
#include <vector>

#define EIGEN_USE_GPU  // For definition of Eigen::GpuDevice.
#include "coordinates_hashmap_gpu.h"
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
namespace tf3d {
namespace cuda {

// A wrapper on CoordinatesHashMapGpu that manages the underlying storage and
// provides additional utility methods.
template <int dims>
class CoordinatesHashMapWrapper {
 public:
  // Initialize the hashmap.
  //
  // - coordinates: int32[batch_size, max_num_coords_per_batch, dims]
  // - num_valid_coordinates: int32[batch_size]. The number of valid coordinates
  //   for each batch.
  Status Initialize(const Tensor& coordinates,
                    const Tensor& num_valid_coordinates, OpKernelContext* ctx);

  // Returns the underlying GPU hashmap instance.
  const CoordinatesHashMapGpu<dims>& GetGpuHashMap() const {
    return gpu_hash_map_;
  }

  // Returns an int32 tensor of shape
  // [filter_volume, batch_size, max_num_coords_per_batch] containing the
  // indices to the neighbor coordinates for each input coordinates, where
  // filter_volume is the number of neighbors we care about and is equal to
  // `filter_height * filter_width [* filter_depth (for 3D)].
  Status GetNeighborIndices(const Tensor& coordinates,
                            const Tensor& num_valid_coordinates,
                            const TensorShape& filter_shape,
                            OpKernelContext* ctx,
                            Tensor* neighbor_indices) const;

 private:
  // Owns all data used by `gpu_hash_map_`.
  std::vector<Tensor> hashmap_tensors_;

  CoordinatesHashMapGpu<dims> gpu_hash_map_;
};

}  // namespace cuda
}  // namespace tf3d
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // TF3D_OPS_COORDINATES_HASHMAP_WRAPPER_H_
