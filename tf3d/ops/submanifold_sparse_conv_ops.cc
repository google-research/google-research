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

#define EIGEN_USE_THREADS
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include "submanifold_sparse_conv_launcher.h"
#include "submanifold_sparse_conv_utils.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tf3d {

// Template class for 2D/3D sparse conv ops with various device types.
template <typename DeviceType, int dims>
class SubmanifoldSparseConvOp : public OpKernel {
 public:
  explicit SubmanifoldSparseConvOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES_OK(
        ctx, ValidateConvInputs(/*is_grad_op=*/false, /*dims=*/dims, ctx));

    const Tensor& coordinates = ctx->input(0);
    const Tensor& num_valid_coordinates = ctx->input(1);
    const Tensor& input_features = ctx->input(2);
    const Tensor& filter = ctx->input(3);

    // Allocate output tensors.
    const int batch_size = coordinates.dim_size(0);
    const int max_num_coords_per_batch = coordinates.dim_size(1);
    const int out_channels = filter.dim_size(dims + 1);
    Tensor* output_features = nullptr;
    OP_REQUIRES_OK(
        ctx,
        ctx->allocate_output(
            0, TensorShape{batch_size, max_num_coords_per_batch, out_channels},
            &output_features));
    OP_REQUIRES_OK(ctx, LaunchSubmanifoldSparseConvolution<DeviceType>(
                            {coordinates, num_valid_coordinates, input_features,
                             filter, output_features, ctx}));
  }
};

// Register CPU kernels.
REGISTER_KERNEL_BUILDER(Name("SubmanifoldSparseConv2D").Device(DEVICE_CPU),
                        SubmanifoldSparseConvOp<Eigen::ThreadPoolDevice, 2>);
REGISTER_KERNEL_BUILDER(Name("SubmanifoldSparseConv3D").Device(DEVICE_CPU),
                        SubmanifoldSparseConvOp<Eigen::ThreadPoolDevice, 3>);

#if GOOGLE_CUDA
// Register GPU kernels.
REGISTER_KERNEL_BUILDER(Name("SubmanifoldSparseConv2D").Device(DEVICE_GPU),
                        SubmanifoldSparseConvOp<Eigen::GpuDevice, 2>);
REGISTER_KERNEL_BUILDER(Name("SubmanifoldSparseConv3D").Device(DEVICE_GPU),
                        SubmanifoldSparseConvOp<Eigen::GpuDevice, 3>);
#endif  // GOOGLE_CUDA

}  // namespace tf3d
}  // namespace tensorflow
