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

#include "sparse/ops/cc/common.h"
#include "sparse/ops/cc/fused_depthwise_launcher.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/padding.h"

namespace sgk {

using ::tensorflow::Tensor;
using ::tensorflow::TensorFormat;
using ::tensorflow::TensorShape;
using ::tensorflow::TensorShapeUtils;
using ::tensorflow::errors::InvalidArgument;

template <typename Device, typename T>
class DepthwiseConvOp : public tensorflow::OpKernel {
 public:
  explicit DepthwiseConvOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));

    // NOTE: This op only supports NCHW format.
    std::string data_format = "NCHW";
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                InvalidArgument("Invalid data format"));

    // NOTE: This kernel only supports matching strides for the H & W
    // dimensions, and does not support stride in the channel and batch
    // dimensions.
    OP_REQUIRES(context, strides_.size() == 4,
                InvalidArgument("Sliding window strides field must "
                                "specify 4 dimensions"));
    stride_ = GetTensorDim(strides_, data_format_, 'H');
    const int64 stride_w = GetTensorDim(strides_, data_format_, 'W');
    const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
    const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');

    OP_REQUIRES(
        context, stride_ == stride_w,
        InvalidArgument("Current implementation only supports equal length "
                        "strides in the row and column dimensions."));
    OP_REQUIRES(context, (stride_n == 1 && stride_c == 1),
                InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("explicit_paddings", &padding_));
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    ComputeHelper(context, /*bias_ptr=*/nullptr);
  }

  void ComputeHelper(tensorflow::OpKernelContext* context,
                     const float* bias_ptr) {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, depth_multiplier]
    const Tensor& filter = context->input(1);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, input.dims() == 4,
                InvalidArgument("input must be 4-dimensional",
                                input.shape().DebugString()));
    OP_REQUIRES(context, filter.dims() == 4,
                InvalidArgument("filter must be 4-dimensional: ",
                                filter.shape().DebugString()));

    // in_depth for input and filter must match.
    const int64 in_depth = GetTensorDim(input, data_format_, 'C');
    OP_REQUIRES(context, in_depth == filter.dim_size(0),
                InvalidArgument("input and filter must have the same depth: ",
                                in_depth, " vs ", filter.dim_size(0)));

    // The last dimension for filter is depth multiplier.
    //
    // NOTE: We only support depth_multiplier == 1.
    const int32 depth_multiplier = filter.dim_size(3);
    OP_REQUIRES(context, depth_multiplier == 1,
                InvalidArgument("Depth multiplier must be 1."));

    // The output depth is input depth x depth multiplier
    const int32 out_depth = in_depth * depth_multiplier;

    // NOTE: We only support 3x3 kernels.
    const int32 input_rows = GetTensorDim(input, data_format_, 'H');
    const int32 filter_rows = filter.dim_size(1);
    const int32 input_cols = GetTensorDim(input, data_format_, 'W');
    const int32 filter_cols = filter.dim_size(2);
    OP_REQUIRES(context, input_rows == input_cols,
                InvalidArgument("Only supports square images."));
    OP_REQUIRES(context, filter_rows == 3,
                InvalidArgument("Only supports 3x3 kernels."));
    OP_REQUIRES(context, filter_cols == 3,
                InvalidArgument("Only supports 3x3 kernels."));

    // The first dimension for input is batch.
    const int32 batch = input.dim_size(0);

    // Get and validate the padding arguments.
    int64 pad_rows = GetTensorDim(padding_, data_format_, 'H');
    int64 pad_cols = GetTensorDim(padding_, data_format_, 'W');
    OP_REQUIRES(context, GetTensorDim(padding_, data_format_, 'C') == 0,
                InvalidArgument("Channel padding not supported."));
    OP_REQUIRES(context, GetTensorDim(padding_, data_format_, 'N') == 0,
                InvalidArgument("Batch padding not supported."));
    OP_REQUIRES(context, pad_rows == pad_cols,
                InvalidArgument("Height and width padding must match."));

    int64 out_rows = 0, out_cols = 0;
    OP_REQUIRES_OK(
        context, tensorflow::GetWindowedOutputSizeVerboseV2(
                     input_rows, filter_rows, /* dilation_rate = */ 1, stride_,
                     /* padding_type = */ tensorflow::EXPLICIT, &out_rows,
                     &pad_rows, &pad_rows));
    OP_REQUIRES_OK(
        context, tensorflow::GetWindowedOutputSizeVerboseV2(
                     input_cols, filter_cols, /* dilation_rate = */ 1, stride_,
                     /* padding_type = */ tensorflow::EXPLICIT, &out_cols,
                     &pad_cols, &pad_cols));

    // Setup and allocate the output tensor.
    TensorShape out_shape = {batch, out_depth, out_rows, out_cols};
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    LaunchFusedDepthwiseConv(
        context->eigen_device<Device>(), batch, in_depth, input_rows,
        input_cols, input.template flat<T>().data(), filter_rows, pad_rows,
        stride_, filter.template flat<T>().data(), bias_ptr,
        output->template flat<T>().data());
  }

 protected:
  std::vector<int32> strides_;
  std::vector<int64> padding_;
  TensorFormat data_format_;
  int64 stride_;  // in height/width dimension.
};

template <typename Device, typename T>
class FusedDepthwiseConvOp : public DepthwiseConvOp<Device, T> {
 public:
  explicit FusedDepthwiseConvOp(tensorflow::OpKernelConstruction* context)
      : DepthwiseConvOp<Device, T>(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    // Input bias is of the following dimensions:
    // [in_depth * depth_multiplier]
    const Tensor& bias = context->input(2);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(bias.shape()),
                InvalidArgument("Bias must be 1-dimensional."));

    const int64 out_depth =
        GetTensorDim(context->input(0), this->data_format_, 'C');
    OP_REQUIRES(context, out_depth == bias.dim_size(0),
                InvalidArgument("Bias must match output depth."));
    this->ComputeHelper(context, bias.template flat<T>().data());
  }
};

#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("DepthwiseConv")
                            .Device(tensorflow::DEVICE_GPU)
                            .TypeConstraint<float>("T"),
                        DepthwiseConvOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(Name("FusedDepthwiseConv")
                            .Device(tensorflow::DEVICE_GPU)
                            .TypeConstraint<float>("T"),
                        FusedDepthwiseConvOp<Eigen::GpuDevice, float>);
#endif  // GOOGLE_CUDA

}  // namespace sgk
