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

#include "submanifold_sparse_conv_grad_launcher.h"

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "coordinates_hashmap_gpu.h"
#include "coordinates_hashmap_wrapper.h"
#include "submanifold_sparse_conv_utils.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "third_party/gpus/cuda/include/cooperative_groups.h"
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
namespace cg = ::cooperative_groups;

struct __align__(4) InputInfo {
  const int batch_size;
  const int max_num_coords_per_batch;
  const int in_channels;
  const int out_channels;
};

// General (not optimized) kernels for submanifold sparse convolutions, where
// each thread works on one output element.
template <int dims, typename FeatureType>
__global__ void SubmanifoldSparseConvBackpropFilterKernel(
    const InputInfo info, const FilterSpatialDims<dims> filter_size,
    const int32* __restrict__ coordinates,
    const int32* __restrict__ num_valid_coordinates,
    const CoordinatesHashMapGpu<dims> hashmap,
    const FeatureType* __restrict__ input_features,
    const FeatureType* __restrict__ d_output_features,
    FeatureType* __restrict__ d_filter) {
  for (int i : GpuGridRangeX(filter_size.Size() * info.in_channels *
                             info.out_channels)) {
    const int previous_outchans = i / info.out_channels;
    const int cur_outchan = i - previous_outchans * info.out_channels;
    const int neighbor_id = previous_outchans / info.in_channels;
    const int cur_inchan = previous_outchans - neighbor_id * info.in_channels;
    const Coordinates<dims> offset =
        NeighborIterator<dims>::GetOffset(filter_size, neighbor_id);

    FeatureType value = 0;
    for (int cur_batch = 0; cur_batch < info.batch_size; ++cur_batch) {
      const FeatureType* __restrict__ cur_d_output_features =
          d_output_features +
          cur_batch * info.max_num_coords_per_batch * info.out_channels;

      for (int cur_coords_id = 0;
           cur_coords_id < num_valid_coordinates[cur_batch]; ++cur_coords_id) {
        Coordinates<dims> coords(
            coordinates +
            (cur_batch * info.max_num_coords_per_batch + cur_coords_id) * dims);
        const int neighbor_index = hashmap.Lookup(cur_batch, coords + offset);
        if (neighbor_index >= 0) {
          value +=
              cur_d_output_features[cur_coords_id * info.out_channels +
                                    cur_outchan] *
              input_features[neighbor_index * info.in_channels + cur_inchan];
        }
      }
    }
    d_filter[i] = value;
  }
}

// Kernel for small filter window size. High level algorithm:
// - Each block works on a subset of filter values. To be specific, each block
//   works on `inchans_for_block` filter rows, starting at `inchan_beg`, where
//   each filter row contains [out_channels] contiguous filter values.
// - Each warp (contiguous 32 threads) process one coordinate, and each thread
//   inside the warp process one output channels and `inchans_for_block` input
//   channels. The result (delta) is stored in shared memory.
// - Synchronize the threads, and then merge the filter values produced by
//   different warps, and write the merged result to global memory.
template <typename FeatureType>
__global__ void SubmanifoldSparseConvBackpropFilterKernel_OneCoordPerWarp(
    const InputInfo info, const int filter_volume, int in_channels_per_block,
    const int32* __restrict__ num_valid_coordinates,
    const FeatureType* __restrict__ input_features,
    const int32* __restrict__ neighbor_indices,
    const FeatureType* __restrict__ d_output_features,
    FeatureType* __restrict__ d_filter) {
  const int kWarpSize = 32;
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<kWarpSize> warp = cg::tiled_partition<kWarpSize>(block);
  const int warps_per_block = block.size() >> 5;
  const int warp_id_in_block = (block.thread_rank() >> 5);
  const int filter_volume_offset = blockIdx.y;
  const int inchan_beg = blockIdx.x * in_channels_per_block;
  // This block will works on this many input channels.
  const int inchans_for_block =
      min(in_channels_per_block, info.in_channels - inchan_beg);

  extern __shared__ FeatureType cached_d_filter[];

  // Each warp iterates through all output channels.
  for (int cur_outchan_beg = 0; cur_outchan_beg < info.out_channels;
       cur_outchan_beg += kWarpSize) {
    // Reset the cache.
    for (int i = 0; i < inchans_for_block; ++i) {
      cached_d_filter[i * block.size() + block.thread_rank()] = 0;
    }
    // Don't need to sync since each thread works on its own cache elements.

    const int cur_outchan = cur_outchan_beg + warp.thread_rank();
    if (cur_outchan >= info.out_channels) continue;

    // Each warp works on different sets of coordinates.
    for (int cur_batch = 0; cur_batch < info.batch_size; ++cur_batch) {
      const int cur_num_valid_coords = num_valid_coordinates[cur_batch];
      for (int cur_coords_id = warp_id_in_block;
           cur_coords_id < cur_num_valid_coords;
           cur_coords_id += warps_per_block) {
        // Each thread in the same warp works on the same coordinate but
        // different output channels.
        const int cur_global_coords_id =
            cur_batch * info.max_num_coords_per_batch + cur_coords_id;
        const int neighbor_index =
            neighbor_indices[cur_global_coords_id +
                             filter_volume_offset * info.batch_size *
                                 info.max_num_coords_per_batch];
        if (neighbor_index < 0) continue;

        const FeatureType* __restrict__ cur_input_features =
            input_features + cur_global_coords_id * info.in_channels +
            inchan_beg;
        // Read the output feature once, and reuse for all input channels
        // handled by this block.
        const FeatureType d_out_feature =
            d_output_features[neighbor_index * info.out_channels + cur_outchan];
        for (int cur_inchan = 0; cur_inchan < inchans_for_block; ++cur_inchan) {
          // Compute the delta and save it to shared memory.
          cached_d_filter[cur_inchan * block.size() + block.thread_rank()] +=
              d_out_feature * cur_input_features[cur_inchan];
        }
      }
    }
    block.sync();

    // Merge the result and write it to global memory.
    const int reversed_filter_offset = filter_volume - 1 - filter_volume_offset;
    FeatureType* __restrict__ cur_d_filter =
        d_filter + (reversed_filter_offset * info.in_channels + inchan_beg) *
                       info.out_channels;
    // Each warp merges the result for one input channel.
    for (int i = warp_id_in_block; i < inchans_for_block;
         i += warps_per_block) {
      FeatureType value = 0;
      // Each thread merges the result for the same input and output channel.
      // There are `warps_per_block` elements to merge (since each warp works on
      // different sets of coordinates).
      for (int j = 0; j < warps_per_block; ++j) {
        value += cached_d_filter[i * block.size() + j * kWarpSize +
                                 warp.thread_rank()];
      }
      cur_d_filter[i * info.out_channels + cur_outchan_beg +
                   warp.thread_rank()] = value;
    }
    block.sync();
  }
}

// Launch convolution kernel with small convolution window.
template <int dims>
static Status LaunchSubmanifoldSparseConvBackpropFilterSmallWindow(
    const SubmanifoldSparseConvBackpropFilterLaunchOptions& opts,
    const int filter_volume, const InputInfo& info,
    CoordinatesHashMapWrapper<dims>* hashmap) {
  Eigen::GpuDevice device = opts.ctx->eigen_device<Eigen::GpuDevice>();
  const int total_smem_bytes = device.sharedMemPerBlock();
  const int warp_size = 32;
  const int total_coordinates =
      opts.batch_size() * opts.max_num_coords_per_batch();

  // Each warp works on different coordinates, so the total number of warps
  // should be smaller that the total number of coordinates.
  const int max_num_warps =
      min(device.maxGpuThreadsPerBlock() / warp_size, total_coordinates);

  // Calculates the shared memory settings under max_num_warps.
  // If total_smem_bytes is 48KB, each warp should get at least 1.5KB smem.
  const int max_smem_bytes_per_warp = total_smem_bytes / max_num_warps;

  // Calculates max number of input channels that can be handled by the block at
  // each iteration. All warps work on same number of input channels.
  const int smem_bytes_per_inchan = warp_size * sizeof(float);
  // When device.maxGpuThreadsPerBlock() == 1024, this number is at least 12.
  const int max_inchans_per_block =
      max_smem_bytes_per_warp / smem_bytes_per_inchan;
  // Heuristic: round to multiples of 4.
  const int in_channels_per_block =
      min(opts.in_channels(), std::max(1, max_inchans_per_block / 4 * 4));

  Gpu2DLaunchConfig config;
  // Partition the number of input channels to these many pieces.
  config.block_count.x =
      (opts.in_channels() + in_channels_per_block - 1) / in_channels_per_block;
  config.block_count.y = filter_volume;
  config.block_count.z = 1;

  // On each iteration it works on 32 output channels. I.e. one output
  // channel per thread in the warp.
  const int smem_bytes_per_warp = in_channels_per_block * smem_bytes_per_inchan;
  config.thread_per_block.x = min(
      // Each warp works on one coordinate, so #warps <= total_coordinates.
      warp_size * total_coordinates,
      // #warps * smem_bytes_per_warp should be <= total_smem_bytes.
      total_smem_bytes / smem_bytes_per_warp * warp_size);
  if (config.thread_per_block.x > device.maxGpuThreadsPerBlock()) {
    config.thread_per_block.x = device.maxGpuThreadsPerBlock();
  }
  config.thread_per_block.y = 1;
  config.thread_per_block.z = 1;
  const int shared_memory_size_bytes =
      config.thread_per_block.x / warp_size * smem_bytes_per_warp;

  Tensor neighbor_indices;
  TF_RETURN_IF_ERROR(hashmap->GetNeighborIndices(
      opts.coordinates, opts.num_valid_coordinates, opts.d_filter->shape(),
      opts.ctx, &neighbor_indices));
  return GpuLaunchKernel(
      SubmanifoldSparseConvBackpropFilterKernel_OneCoordPerWarp<float>,
      config.block_count, config.thread_per_block, shared_memory_size_bytes,
      device.stream(), info, filter_volume, in_channels_per_block,
      opts.num_valid_coordinates.flat<int32>().data(),
      opts.input_features.flat<float>().data(),
      neighbor_indices.flat<int32>().data(),
      opts.d_output_features.flat<float>().data(),
      opts.d_filter->flat<float>().data());
}

template <int dims>
static Status LaunchSubmanifoldSparseConvBackpropFilterImpl(
    const SubmanifoldSparseConvBackpropFilterLaunchOptions& opts) {
  // Reset the output.
  Eigen::GpuDevice device = opts.ctx->eigen_device<Eigen::GpuDevice>();
  const int max_threads = device.maxGpuThreadsPerBlock();
  const int total_work = opts.d_filter->NumElements();
  TF_CHECK_OK(GpuLaunchKernel(
      SetZero<float>, (total_work + max_threads - 1) / max_threads,
      min(total_work, max_threads), 0, device.stream(), total_work,
      opts.d_filter->flat<float>().data()));

  // Build the coordinates hashmap.
  CoordinatesHashMapWrapper<dims> hashmap;
  TF_RETURN_IF_ERROR(hashmap.Initialize(opts.coordinates,
                                        opts.num_valid_coordinates, opts.ctx));
  FilterSpatialDims<dims> filter_size;
  TF_RETURN_IF_ERROR(FilterSpatialDims<dims>::FromFilterShape(
      opts.d_filter->shape(), &filter_size));

  // Launch the kernel.
  InputInfo info{
      .batch_size = opts.batch_size(),
      .max_num_coords_per_batch = opts.max_num_coords_per_batch(),
      .in_channels = opts.in_channels(),
      .out_channels = opts.out_channels(),
  };

  // Use 27 as the threshold to allow at least 5x5 2D and 3x3x3 3D filter window
  // size.
  if (filter_size.Size() <= 27) {
    return LaunchSubmanifoldSparseConvBackpropFilterSmallWindow<dims>(
        opts, filter_size.Size(), info, &hashmap);
  }

  GpuLaunchConfig config = GetGpuLaunchConfig(
      total_work, device,
      SubmanifoldSparseConvBackpropFilterKernel<dims, float>, 0, total_work);
  return GpuLaunchKernel(
      SubmanifoldSparseConvBackpropFilterKernel<dims, float>,
      config.block_count, config.thread_per_block, 0, device.stream(), info,
      filter_size, opts.coordinates.flat<int32>().data(),
      opts.num_valid_coordinates.flat<int32>().data(), hashmap.GetGpuHashMap(),
      opts.input_features.flat<float>().data(),
      opts.d_output_features.flat<float>().data(),
      opts.d_filter->flat<float>().data());
}

}  // namespace cuda

template <>
Status LaunchSubmanifoldSparseConvBackpropFilter<Eigen::GpuDevice>(
    const SubmanifoldSparseConvBackpropFilterLaunchOptions& opts) {
  const int dims = opts.coordinates.dim_size(2);
  if (dims == 2) {
    return cuda::LaunchSubmanifoldSparseConvBackpropFilterImpl<2>(opts);
  }
  if (dims == 3) {
    return cuda::LaunchSubmanifoldSparseConvBackpropFilterImpl<3>(opts);
  }
  return errors::InvalidArgument("Only 2D and 3D convolutions are supported.");
}

}  // namespace tf3d
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
