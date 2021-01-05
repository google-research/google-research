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

#include "submanifold_sparse_conv_launcher.h"

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "coordinates_hashmap_gpu.h"
#include "coordinates_hashmap_wrapper.h"
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

// Common information for input data and configurations.
struct InputInfo {
  const int batch_size;
  const int max_num_coords_per_batch;
  const int in_channels;
  const int out_channels;
  const int filter_spatial_size;

  // Number of filter blocks to cache in shared memory, where each block
  // contains [in_channels, out_channels] filter values.
  int num_cached_filter_blocks = 0;

  // Number of filter rows to cache in shared memory, where each row contains
  // [out_channels] filter values.
  int num_cached_filter_rows = 0;

  InputInfo(int batch_size, int max_num_coords_per_batch, int in_channels,
            int out_channels, int filter_spatial_size)
      : batch_size(batch_size),
        max_num_coords_per_batch(max_num_coords_per_batch),
        in_channels(in_channels),
        out_channels(out_channels),
        filter_spatial_size(filter_spatial_size) {}
};

// General (not optimized) kernels for submanifold sparse convolutions.
//
// coordinates: int32[batch_size * max_num_coords_per_batch * 3]
// input_features: T[batch_size * max_num_coords_per_batch, in_channels]
// filter: T[depth, height, width, in_channels, out_channels]
// output_features: T[batch_size * max_num_coords_per_batch, out_channels]
template <int dims, typename FeatureType>
__global__ void SubmanifoldSparseConvKernel(
    const InputInfo info, const int32* __restrict__ coordinates,
    const int32* __restrict__ num_valid_coordinates,
    const FeatureType* __restrict__ input_features,
    const FilterSpatialDims<dims> filter_size,
    const FeatureType* __restrict__ filter,
    const CoordinatesHashMapGpu<dims> hashmap,
    FeatureType* __restrict__ output_features) {
  const int work_per_batch = info.max_num_coords_per_batch * info.out_channels;
  for (int i : GpuGridRangeX(info.batch_size * work_per_batch)) {
    // Contiguous threads work on the same coordinate but contiguous output
    // channels. As a result, Lookup() and all operations will have the same
    // pattern.
    const int cur_batch = i / work_per_batch;
    const int previous_coordinates = cur_batch * info.max_num_coords_per_batch;
    const int cur_coords_id =
        (i - previous_coordinates * info.out_channels) / info.out_channels;
    // Skip invalid coordinates.
    if (cur_coords_id >= num_valid_coordinates[cur_batch]) {
      output_features[i] = 0;
      continue;
    }
    const int cur_outchan =
        i - (previous_coordinates + cur_coords_id) * info.out_channels;

    NeighborIterator<dims> iter(
        coordinates + (previous_coordinates + cur_coords_id) * dims,
        filter_size);
    FeatureType value = 0;
    while (iter.Next()) {
      // Lookup the neighbor.
      const int neighbor_index = hashmap.Lookup(cur_batch, iter.Get());
      if (neighbor_index >= 0) {
        const FeatureType* cur_feature =
            input_features + neighbor_index * info.in_channels;
        const FeatureType* cur_filter =
            filter + (iter.Offset() * info.in_channels * info.out_channels +
                      cur_outchan);
        // Compute convolutions.
        for (int cur_inchan = 0; cur_inchan < info.in_channels; ++cur_inchan) {
          // For the same iteration, contiguous threads have the same
          // cur_inchan but cur_outchan is different (but contiguous), so
          // this results in coalesced memory access.
          value += cur_feature[cur_inchan] *
                   cur_filter[cur_inchan * info.out_channels];
        }
      }
    }
    output_features[i] = value;
  }
}

// Kernel for small convolutions, with all filter values cached in shared
// memory.
template <typename FeatureType>
__global__ void SubmanifoldSparseConvKernel_CacheAllFilters(
    const InputInfo info, const int32* __restrict__ num_valid_coordinates,
    const FeatureType* __restrict__ input_features,
    const FeatureType* __restrict__ filter,
    const int32* __restrict__ neighbor_indices,
    FeatureType* __restrict__ output_features) {
  extern __shared__ FeatureType cached_filter[];
  // Load the filter values into shared memory.
  for (int i = threadIdx.x;
       i < info.filter_spatial_size * info.in_channels * info.out_channels;
       i += blockDim.x) {
    cached_filter[i] = filter[i];
  }
  __syncthreads();

  const int work_per_batch = info.max_num_coords_per_batch * info.out_channels;
  for (int i : GpuGridRangeX(info.batch_size * work_per_batch)) {
    const int cur_batch = i / work_per_batch;
    const int previous_coordinates = cur_batch * info.max_num_coords_per_batch;
    const int cur_coords_id =
        (i - previous_coordinates * info.out_channels) / info.out_channels;
    // Skip invalid coordinates.
    if (cur_coords_id >= num_valid_coordinates[cur_batch]) {
      output_features[i] = 0;
      continue;
    }
    const int cur_outchan =
        i - (previous_coordinates + cur_coords_id) * info.out_channels;

    FeatureType value = 0;
    const int32* __restrict__ cur_neighbor_indices =
        neighbor_indices + previous_coordinates + cur_coords_id;

    for (int j = 0; j < info.filter_spatial_size; ++j) {
      const int neighbor_index =
          cur_neighbor_indices[j * info.batch_size *
                               info.max_num_coords_per_batch];
      if (neighbor_index < 0) continue;
      const FeatureType* cur_feature =
          input_features + neighbor_index * info.in_channels;
      const FeatureType* cur_filter =
          cached_filter +
          (j * info.in_channels * info.out_channels + cur_outchan);

      // Compute convolutions.
      for (int cur_inchan = 0; cur_inchan < info.in_channels; ++cur_inchan) {
        // For the same iteration, contiguous threads have the same cur_inchan
        // and contiguous cur_outchan, so this results in coalesced memory
        // access for `filter`.
        value += cur_feature[cur_inchan] *
                 cur_filter[cur_inchan * info.out_channels];
      }
    }
    output_features[i] = value;
  }
}

// Kernel for small convolutions, with num_cached_filter_blocks blocks of filter
// values cached in shared memory, where each block contains
// [in_channels, out_channels] filter values.
template <typename FeatureType>
__global__ void SubmanifoldSparseConvKernel_WithFilterBlockCache(
    const InputInfo info, const int32* __restrict__ num_valid_coordinates,
    const FeatureType* __restrict__ input_features,
    const FeatureType* __restrict__ filter,
    const int32* __restrict__ neighbor_indices,
    FeatureType* __restrict__ output_features) {
  const int work_per_batch = info.max_num_coords_per_batch * info.out_channels;
  for (int i : GpuGridRangeX(info.batch_size * work_per_batch)) {
    output_features[i] = 0;
  }
  // Don't need to sync here, since each output value is owned by one thread
  // only.

  extern __shared__ FeatureType cached_filter[];
  for (int offset_beg = 0; offset_beg < info.filter_spatial_size;
       offset_beg += info.num_cached_filter_blocks) {
    const int offset_end = min(offset_beg + info.num_cached_filter_blocks,
                                    info.filter_spatial_size);
    for (int i = threadIdx.x;
         i < (offset_end - offset_beg) * info.in_channels * info.out_channels;
         i += blockDim.x) {
      cached_filter[i] =
          filter[offset_beg * info.in_channels * info.out_channels + i];
    }
    __syncthreads();

    for (int i : GpuGridRangeX(info.batch_size * work_per_batch)) {
      const int cur_batch = i / work_per_batch;
      const int previous_coordinates =
          cur_batch * info.max_num_coords_per_batch;
      const int cur_coords_id =
          (i - previous_coordinates * info.out_channels) / info.out_channels;
      const int cur_outchan =
          i - (previous_coordinates + cur_coords_id) * info.out_channels;
      // Skip invalid coordinates.
      if (cur_coords_id >= num_valid_coordinates[cur_batch]) continue;

      FeatureType value = 0;
      for (int offset = offset_beg; offset < offset_end; ++offset) {
        const int neighbor_index =
            neighbor_indices[previous_coordinates + cur_coords_id +
                             offset * info.batch_size *
                                 info.max_num_coords_per_batch];
        if (neighbor_index < 0) continue;

        const FeatureType* cur_filter =
            cached_filter +
            ((offset - offset_beg) * info.in_channels * info.out_channels +
             cur_outchan);
        const FeatureType* cur_feature =
            input_features + neighbor_index * info.in_channels;
        // Compute convolutions.
        for (int cur_inchan = 0; cur_inchan < info.in_channels; ++cur_inchan) {
          value += cur_feature[cur_inchan] *
                   cur_filter[cur_inchan * info.out_channels];
        }
      }
      output_features[i] += value;
    }
    __syncthreads();
  }
}

// Like SubmanifoldSparseConvKernel_WithFilterBlockCache but only cache one
// block of filter values.
template <typename FeatureType>
__global__ void SubmanifoldSparseConvKernel_WithFilterBlockCache_OneBlock(
    const InputInfo info, const int32* __restrict__ num_valid_coordinates,
    const FeatureType* __restrict__ input_features,
    const FeatureType* __restrict__ filter,
    const int32* __restrict__ neighbor_indices,
    FeatureType* __restrict__ output_features) {
  const int work_per_batch = info.max_num_coords_per_batch * info.out_channels;
  for (int i : GpuGridRangeX(info.batch_size * work_per_batch)) {
    output_features[i] = 0;
  }
  // Don't need to sync here, since each output value is owned by one thread
  // only.

  extern __shared__ FeatureType cached_filter[];
  for (int offset = 0; offset < info.filter_spatial_size; ++offset) {
    for (int i = threadIdx.x; i < info.in_channels * info.out_channels;
         i += blockDim.x) {
      cached_filter[i] =
          filter[offset * info.in_channels * info.out_channels + i];
    }
    __syncthreads();

    for (int i : GpuGridRangeX(info.batch_size * work_per_batch)) {
      const int cur_batch = i / work_per_batch;
      const int previous_coordinates =
          cur_batch * info.max_num_coords_per_batch;
      const int cur_coords_id =
          (i - previous_coordinates * info.out_channels) / info.out_channels;
      const int cur_outchan =
          i - (previous_coordinates + cur_coords_id) * info.out_channels;
      // Skip invalid coordinates.
      if (cur_coords_id >= num_valid_coordinates[cur_batch]) continue;

      const int neighbor_index =
          neighbor_indices[previous_coordinates + cur_coords_id +
                           offset * info.batch_size *
                               info.max_num_coords_per_batch];
      if (neighbor_index < 0) continue;

      const FeatureType* cur_feature =
          input_features + neighbor_index * info.in_channels;
      FeatureType value = 0;
      // Compute convolutions.
      for (int cur_inchan = 0; cur_inchan < info.in_channels; ++cur_inchan) {
        value += cur_feature[cur_inchan] *
                 cached_filter[cur_inchan * info.out_channels + cur_outchan];
      }
      output_features[i] += value;
    }
    __syncthreads();
  }
}

// Kernel for small convolutions, with num_cached_filter_rows rows of filter
// values cached in shared memory, where each row contains [out_channels] filter
// values.
template <typename FeatureType>
__global__ void SubmanifoldSparseConvKernel_WithFilterRowCache(
    const InputInfo info, const int32* __restrict__ num_valid_coordinates,
    const FeatureType* __restrict__ input_features,
    const FeatureType* __restrict__ filter,
    const int32* __restrict__ neighbor_indices,
    FeatureType* __restrict__ output_features) {
  const int work_per_batch = info.max_num_coords_per_batch * info.out_channels;
  for (int i : GpuGridRangeX(info.batch_size * work_per_batch)) {
    output_features[i] = 0;
  }
  // Don't need to sync here, since each output value is owned by one thread
  // only.

  extern __shared__ FeatureType cached_filter[];
  for (int offset = 0; offset < info.filter_spatial_size; ++offset) {
    for (int cur_inchan_beg = 0; cur_inchan_beg < info.in_channels;
         cur_inchan_beg += info.num_cached_filter_rows) {
      const int cur_inchan_end = min(
          cur_inchan_beg + info.num_cached_filter_rows, info.in_channels);

      // Load the filter values into shared memory.
      for (int i = threadIdx.x;
           i < (cur_inchan_end - cur_inchan_beg) * info.out_channels;
           i += blockDim.x) {
        cached_filter[i] = filter[(offset * info.in_channels + cur_inchan_beg) *
                                      info.out_channels +
                                  i];
      }
      __syncthreads();

      for (int i : GpuGridRangeX(info.batch_size * work_per_batch)) {
        const int cur_batch = i / work_per_batch;
        const int previous_coordinates =
            cur_batch * info.max_num_coords_per_batch;
        const int cur_coords_id =
            (i - previous_coordinates * info.out_channels) / info.out_channels;
        const int cur_outchan =
            i - (previous_coordinates + cur_coords_id) * info.out_channels;
        // Skip invalid coordinates.
        if (cur_coords_id >= num_valid_coordinates[cur_batch]) continue;

        const int neighbor_index =
            neighbor_indices[previous_coordinates + cur_coords_id +
                             offset * info.batch_size *
                                 info.max_num_coords_per_batch];
        if (neighbor_index < 0) continue;

        const FeatureType* cur_feature =
            input_features + neighbor_index * info.in_channels;
        FeatureType value = 0;

        // Compute convolutions.
        for (int cur_inchan = cur_inchan_beg; cur_inchan < cur_inchan_end;
             ++cur_inchan) {
          value +=
              cur_feature[cur_inchan] *
              cached_filter[(cur_inchan - cur_inchan_beg) * info.out_channels +
                            cur_outchan];
        }
        output_features[i] += value;
      }
      __syncthreads();
    }
  }
}

// Launch convolution kernel of specific type.
template <typename KernelType>
static Status LaunchSubmanifoldSparseConvWithTypeSmallWindow(
    KernelType kernel, const SubmanifoldSparseConvLaunchOptions& opts,
    const InputInfo& info, const int shared_memory_size_bytes,
    const Tensor& neighbor_indices) {
  const int total_work =
      opts.batch_size() * opts.max_num_coords_per_batch() * opts.out_channels();
  Eigen::GpuDevice d = opts.ctx->template eigen_device<Eigen::GpuDevice>();
  auto config = GetGpuLaunchConfig(total_work, d, kernel,
                                   shared_memory_size_bytes, total_work);
  TF_CHECK_OK(GpuLaunchKernel(
      kernel, config.block_count, config.thread_per_block,
      shared_memory_size_bytes, d.stream(), info,
      opts.num_valid_coordinates.flat<int32>().data(),
      opts.input_features.flat<float>().data(),
      opts.filter.flat<float>().data(), neighbor_indices.flat<int32>().data(),
      opts.output_features->flat<float>().data()));
  return Status::OK();
}

// Launch convolution kernel with small convolution window.
static Status LaunchSubmanifoldSparseConvSmallWindow(
    const SubmanifoldSparseConvLaunchOptions& opts,
    const Tensor& neighbor_indices) {
  InputInfo info{opts.batch_size(), opts.max_num_coords_per_batch(),
                 opts.in_channels(), opts.out_channels(),
                 static_cast<int>(neighbor_indices.dim_size(0))};
  const int out_channel_bytes = opts.out_channels() * sizeof(float);
  Eigen::GpuDevice d = opts.ctx->template eigen_device<Eigen::GpuDevice>();

  // If the size of filter is smaller than the size of shared memory, cache all
  // filter values.
  if (info.filter_spatial_size * opts.in_channels() * out_channel_bytes <=
      d.sharedMemPerBlock()) {
    const int smem_bytes =
        info.filter_spatial_size * opts.in_channels() * out_channel_bytes;
    return LaunchSubmanifoldSparseConvWithTypeSmallWindow(
        SubmanifoldSparseConvKernel_CacheAllFilters<float>, opts, info,
        smem_bytes, neighbor_indices);
  }

  // Otherwise, if the size of a filter block of size
  // [in_channels, out_channels] is smaller than the size of shared memory,
  // cache the filter blocks instead.
  if (opts.in_channels() * out_channel_bytes <= d.sharedMemPerBlock()) {
    info.num_cached_filter_blocks =
        d.sharedMemPerBlock() / (opts.in_channels() * out_channel_bytes);
    const int smem_bytes =
        info.num_cached_filter_blocks * opts.in_channels() * out_channel_bytes;
    if (info.num_cached_filter_blocks == 1) {
      return LaunchSubmanifoldSparseConvWithTypeSmallWindow(
          SubmanifoldSparseConvKernel_WithFilterBlockCache_OneBlock<float>,
          opts, info, smem_bytes, neighbor_indices);
    } else {
      return LaunchSubmanifoldSparseConvWithTypeSmallWindow(
          SubmanifoldSparseConvKernel_WithFilterBlockCache<float>, opts, info,
          smem_bytes, neighbor_indices);
    }
  }

  // Otherwise, cache filter rows instead.
  if (out_channel_bytes > d.sharedMemPerBlock()) {
    // TODO(laigd): it should fallback to the general kernel.
    return errors::InvalidArgument(
        "Too many output channels (expected to be less than ",
        d.sharedMemPerBlock() / sizeof(float), ").");
  }
  info.num_cached_filter_rows = d.sharedMemPerBlock() / out_channel_bytes;
  const int smem_bytes = info.num_cached_filter_rows * out_channel_bytes;
  return LaunchSubmanifoldSparseConvWithTypeSmallWindow(
      SubmanifoldSparseConvKernel_WithFilterRowCache<float>, opts, info,
      smem_bytes, neighbor_indices);
}

// Implementation of the launcher function.
template <int dims>
static Status LaunchSubmanifoldSparseConvolutionImpl(
    const SubmanifoldSparseConvLaunchOptions& opts) {
  FilterSpatialDims<dims> filter_size;
  TF_RETURN_IF_ERROR(FilterSpatialDims<dims>::FromFilterShape(
      opts.filter.shape(), &filter_size));

  // Build the hashmap.
  CoordinatesHashMapWrapper<dims> hashmap;
  TF_RETURN_IF_ERROR(hashmap.Initialize(opts.coordinates,
                                        opts.num_valid_coordinates, opts.ctx));

  // If the convolution window is small, run optimized kernels.
  if (filter_size.Size() <= 27) {
    Tensor neighbor_indices;
    TF_RETURN_IF_ERROR(hashmap.GetNeighborIndices(
        opts.coordinates, opts.num_valid_coordinates, opts.filter.shape(),
        opts.ctx, &neighbor_indices));
    if (neighbor_indices.dim_size(0) != filter_size.Size()) {
      return errors::Internal("First dimension of neighbor indices (",
                              neighbor_indices.dim_size(0),
                              ") doesn't match the filter spatial size (",
                              filter_size.Size(), ").");
    }
    return LaunchSubmanifoldSparseConvSmallWindow(opts, neighbor_indices);
  }

  // Otherwise, run the unoptimized kernel.
  InputInfo info{opts.batch_size(), opts.max_num_coords_per_batch(),
                 opts.in_channels(), opts.out_channels(),
                 /*filter_spatial_size=*/filter_size.Size()};
  Eigen::GpuDevice d = opts.ctx->template eigen_device<Eigen::GpuDevice>();
  const int total_work =
      opts.batch_size() * opts.max_num_coords_per_batch() * opts.out_channels();
  GpuLaunchConfig config;
  const int shared_memory_size_bytes = 0;
  config = GetGpuLaunchConfig(total_work, d,
                              SubmanifoldSparseConvKernel<dims, float>,
                              shared_memory_size_bytes, total_work);

  TF_CHECK_OK(GpuLaunchKernel(
      SubmanifoldSparseConvKernel<dims, float>, config.block_count,
      config.thread_per_block, shared_memory_size_bytes, d.stream(), info,
      opts.coordinates.flat<int32>().data(),
      opts.num_valid_coordinates.flat<int32>().data(),
      opts.input_features.flat<float>().data(), filter_size,
      opts.filter.flat<float>().data(), hashmap.GetGpuHashMap(),
      opts.output_features->flat<float>().data()));
  return Status::OK();
}

}  // namespace cuda

template <>
Status LaunchSubmanifoldSparseConvolution<Eigen::GpuDevice>(
    const SubmanifoldSparseConvLaunchOptions& opts) {
  const int dims = opts.coordinates.dim_size(2);
  if (dims == 2) return cuda::LaunchSubmanifoldSparseConvolutionImpl<2>(opts);
  if (dims == 3) return cuda::LaunchSubmanifoldSparseConvolutionImpl<3>(opts);
  return errors::InvalidArgument("Only 2D and 3D convolutions are supported.");
}

}  // namespace tf3d
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
