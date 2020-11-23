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

#include "submanifold_sparse_conv_grad_launcher.h"

#include "submanifold_sparse_conv_utils.h"
#include "absl/container/node_hash_map.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tf3d {

template <int dims>
Status RunSubmanifoldSparseConvBackpropFilter(
    const SubmanifoldSparseConvBackpropFilterLaunchOptions& opts) {
  const int batch_size = opts.batch_size();
  const int max_num_coords_per_batch = opts.max_num_coords_per_batch();
  const int in_channels = opts.in_channels();
  const int out_channels = opts.out_channels();

  const int* coordinates_ptr = opts.coordinates.tensor<int32, 3>().data();
  auto num_valid_coordinates_t = opts.num_valid_coordinates.vec<int32>();
  auto input_features_t = opts.input_features.tensor<float, 3>();
  auto d_output_features_t = opts.d_output_features.tensor<float, 3>();
  auto d_filter_t = opts.d_filter->tensor<float, dims + 2>();
  d_filter_t.setConstant(0.0f);
  float* d_filter_ptr = d_filter_t.data();

  FilterSpatialDims<dims> filter_size;
  TF_RETURN_IF_ERROR(FilterSpatialDims<dims>::FromFilterShape(
      opts.d_filter->shape(), &filter_size));

  for (int cur_batch = 0; cur_batch < batch_size; ++cur_batch) {
    // Build a hashmap mapping coordinate values to corresponding indices.
    absl::node_hash_map<Coordinates<dims>, int, CoordinatesHasher<dims>>
        coordinates_hashmap;
    for (int cur_coords_id = 0;
         cur_coords_id < num_valid_coordinates_t(cur_batch); ++cur_coords_id) {
      Coordinates<dims> key(
          coordinates_ptr +
          (cur_batch * max_num_coords_per_batch + cur_coords_id) * dims);
      coordinates_hashmap[key] = cur_coords_id;
    }

    // Compute the gradients w.r.t. the filter.
    for (int cur_coords_id = 0;
         cur_coords_id < num_valid_coordinates_t(cur_batch); ++cur_coords_id) {
      NeighborIterator<dims> iter(
          coordinates_ptr +
              (cur_batch * max_num_coords_per_batch + cur_coords_id) * dims,
          filter_size);
      while (iter.Next()) {
        const auto neighbor = coordinates_hashmap.find(iter.Get());
        if (neighbor == coordinates_hashmap.end()) continue;
        const int neighbor_index = neighbor->second;
        float* cur_d_filter =
            d_filter_ptr + iter.Offset() * in_channels * out_channels;

        for (int cur_inchan = 0; cur_inchan < in_channels; ++cur_inchan) {
          for (int cur_outchan = 0; cur_outchan < out_channels; ++cur_outchan) {
            cur_d_filter[cur_inchan * out_channels + cur_outchan] +=
                input_features_t(cur_batch, neighbor_index, cur_inchan) *
                d_output_features_t(cur_batch, cur_coords_id, cur_outchan);
          }
        }
      }
    }
  }
  return Status::OK();
}

template <>
Status LaunchSubmanifoldSparseConvBackpropFilter<Eigen::ThreadPoolDevice>(
    const SubmanifoldSparseConvBackpropFilterLaunchOptions& opts) {
  const int dims = opts.coordinates.dim_size(2);
  if (dims == 2) return RunSubmanifoldSparseConvBackpropFilter<2>(opts);
  if (dims == 3) return RunSubmanifoldSparseConvBackpropFilter<3>(opts);
  return errors::InvalidArgument("Only 2D and 3D convolutions are supported.");
}

}  // namespace tf3d
}  // namespace tensorflow
