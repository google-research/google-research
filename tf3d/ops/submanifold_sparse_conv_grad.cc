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

#define EIGEN_USE_THREADS
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include "submanifold_sparse_conv_grad_launcher.h"
#include "submanifold_sparse_conv_launcher.h"
#include "submanifold_sparse_conv_utils.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/reverse_op.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tf3d {

template <typename DeviceType, int dims>
class SubmanifoldSparseConvBackpropInput : public OpKernel {
 public:
  explicit SubmanifoldSparseConvBackpropInput(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx,
                   ValidateConvInputs(/*is_grad_op=*/true, /*dims=*/dims, ctx));

    const Tensor& coordinates = ctx->input(0);
    const Tensor& num_valid_coordinates = ctx->input(1);
    const Tensor& input_features = ctx->input(2);
    const Tensor& filter = ctx->input(3);
    const Tensor& d_output_features = ctx->input(4);

    // Allocate output tensors.
    Tensor* d_input_features = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_features.shape(),
                                             &d_input_features));

    // Transpose the filter and switch the last two dimensions (in_channels and
    // out_channels).
    Tensor filter_transposed;
    TensorShape filter_transposed_shape(filter.shape());
    filter_transposed_shape.set_dim(dims, filter.dim_size(dims + 1));
    filter_transposed_shape.set_dim(dims + 1, filter.dim_size(dims));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, filter_transposed_shape,
                                           &filter_transposed));
    std::vector<int> perm(dims + 2);
    std::iota(perm.begin(), perm.begin() + dims, 0);
    perm[dims] = dims + 1;
    perm[dims + 1] = dims;

    OP_REQUIRES_OK(ctx, DoTranspose(ctx->eigen_device<DeviceType>(), filter,
                                    perm, &filter_transposed));

    // Reverse the transposed filter.
    Tensor filter_transposed_reversed;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, filter_transposed_shape,
                                           &filter_transposed_reversed));
    const int filter_dims = dims + 2;
    typename Eigen::array<bool, filter_dims> axes_di;
    for (int i = 0; i < dims; ++i) {
      axes_di.at(i) = true;
    }
    axes_di.at(dims) = false;
    axes_di.at(dims + 1) = false;
    functor::Reverse<DeviceType, float, filter_dims>()(
        ctx->eigen_device<DeviceType>(),
        const_cast<const Tensor*>(&filter_transposed)
            ->tensor<float, filter_dims>(),
        axes_di, filter_transposed_reversed.tensor<float, filter_dims>());

    // Run actual computation.
    OP_REQUIRES_OK(ctx,
                   LaunchSubmanifoldSparseConvolution<DeviceType>(
                       {coordinates, num_valid_coordinates, d_output_features,
                        filter_transposed_reversed, d_input_features, ctx}));
  }
};

// Register CPU kernels.
REGISTER_KERNEL_BUILDER(
    Name("SubmanifoldSparseConv2DBackpropInput").Device(DEVICE_CPU),
    SubmanifoldSparseConvBackpropInput<Eigen::ThreadPoolDevice, 2>);
REGISTER_KERNEL_BUILDER(
    Name("SubmanifoldSparseConv3DBackpropInput").Device(DEVICE_CPU),
    SubmanifoldSparseConvBackpropInput<Eigen::ThreadPoolDevice, 3>);

#if GOOGLE_CUDA
// Register GPU kernels.
REGISTER_KERNEL_BUILDER(
    Name("SubmanifoldSparseConv2DBackpropInput").Device(DEVICE_GPU),
    SubmanifoldSparseConvBackpropInput<Eigen::GpuDevice, 2>);
REGISTER_KERNEL_BUILDER(
    Name("SubmanifoldSparseConv3DBackpropInput").Device(DEVICE_GPU),
    SubmanifoldSparseConvBackpropInput<Eigen::GpuDevice, 3>);
#endif  // GOOGLE_CUDA

template <typename DeviceType, int dims>
class SubmanifoldSparseConvBackpropFilter : public OpKernel {
 public:
  explicit SubmanifoldSparseConvBackpropFilter(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx,
                   ValidateConvInputs(/*is_grad_op=*/true, /*dims=*/dims, ctx));

    const Tensor& coordinates = ctx->input(0);
    const Tensor& num_valid_coordinates = ctx->input(1);
    const Tensor& input_features = ctx->input(2);
    const Tensor& filter = ctx->input(3);
    const Tensor& d_output_features = ctx->input(4);

    // Allocate output tensors.
    Tensor* d_filter = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, filter.shape(), &d_filter));
    OP_REQUIRES_OK(ctx, LaunchSubmanifoldSparseConvBackpropFilter<DeviceType>(
                            {coordinates, num_valid_coordinates, input_features,
                             d_output_features, d_filter, ctx}));
  }
};

// Register CPU kernels.
REGISTER_KERNEL_BUILDER(
    Name("SubmanifoldSparseConv2DBackpropFilter").Device(DEVICE_CPU),
    SubmanifoldSparseConvBackpropFilter<Eigen::ThreadPoolDevice, 2>);
REGISTER_KERNEL_BUILDER(
    Name("SubmanifoldSparseConv3DBackpropFilter").Device(DEVICE_CPU),
    SubmanifoldSparseConvBackpropFilter<Eigen::ThreadPoolDevice, 3>);

#if GOOGLE_CUDA
// Register GPU kernels.
REGISTER_KERNEL_BUILDER(
    Name("SubmanifoldSparseConv2DBackpropFilter").Device(DEVICE_GPU),
    SubmanifoldSparseConvBackpropFilter<Eigen::GpuDevice, 2>);
REGISTER_KERNEL_BUILDER(
    Name("SubmanifoldSparseConv3DBackpropFilter").Device(DEVICE_GPU),
    SubmanifoldSparseConvBackpropFilter<Eigen::GpuDevice, 3>);
#endif  // GOOGLE_CUDA

}  // namespace tf3d
}  // namespace tensorflow
