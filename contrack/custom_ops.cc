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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using tensorflow::int32;
using tensorflow::int64;
using tensorflow::OpInputList;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::TensorShapeUtils;

namespace contrack {

class SequenceConcatOp : public OpKernel {
 public:
  explicit SequenceConcatOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the sequences tensor
    OpInputList sequences;
    OP_REQUIRES_OK(context, context->input_list("sequences", &sequences));

    // Grab the sequence lengths tensor
    OpInputList lengths;
    OP_REQUIRES_OK(context, context->input_list("lengths", &lengths));

    // Create a concatenated lengths tensor
    Tensor* output_lengths_tensor = nullptr;
    TensorShape output_lengths_shape = lengths[0].shape();
    OP_REQUIRES_OK(context, context->allocate_output(1, output_lengths_shape,
                                                     &output_lengths_tensor));
    auto output_lengths_flat = output_lengths_tensor->vec<int32>();

    int64 max_length = 0;
    int64 num_seqs = lengths.size();
    int64 batch_size = output_lengths_shape.dim_size(0);
    std::vector<int64> lengths_vec(batch_size, 0);
    for (int batch_num = 0; batch_num < batch_size; batch_num++) {
      for (int seq_num = 0; seq_num < num_seqs; seq_num++) {
        lengths_vec[batch_num] += lengths[seq_num].vec<int32>()(batch_num);
      }
      max_length = std::max(max_length, lengths_vec[batch_num]);
      output_lengths_flat(batch_num) = lengths_vec[batch_num];
    }

    // Create a concatenated tensor
    const TensorShape& input_shape = sequences[0].shape();
    TensorShape output_shape(input_shape);
    output_shape.RemoveDim(1);
    output_shape.InsertDim(1, max_length);
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));

    int64 remainder_dim_size = 1;
    for (int i = 2; i < input_shape.dims(); i++) {
      remainder_dim_size *= input_shape.dim_size(i);
    }
    std::vector<tensorflow::TTypes<float, 3>::ConstTensor> reshaped_inputs;
    for (int i = 0; i < num_seqs; i++) {
      int seq_length = sequences[i].shape().dim_size(1);
      reshaped_inputs.push_back(sequences[i].shaped<float, 3>(
          {batch_size, seq_length, remainder_dim_size}));
    }
    auto reshaped_output = output_tensor->shaped<float, 3>(
        {batch_size, max_length, remainder_dim_size});
    // LOG(INFO) << "prepare compute seq";

    for (int64 batch_num = 0; batch_num < batch_size; batch_num++) {
      int64 concat_seq_index = 0;
      for (int64 seq_num = 0; seq_num < num_seqs; seq_num++) {
        for (int64 seq_index = 0;
             seq_index < lengths[seq_num].vec<int32>()(batch_num);
             seq_index++) {
          for (int i = 0; i < remainder_dim_size; i++) {
            reshaped_output(batch_num, concat_seq_index, i) =
                reshaped_inputs[seq_num](batch_num, seq_index, i);
          }
          concat_seq_index++;
        }
      }
      while (concat_seq_index < max_length) {
        for (int i = 0; i < remainder_dim_size; i++) {
          reshaped_output(batch_num, concat_seq_index, i) = 0.0;
        }
        concat_seq_index++;
      }
    }
  }
};

class NewIdOp : public OpKernel {
 public:
  explicit NewIdOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& ids_seq_tensor = context->input(0);
    auto ids_seq_data = ids_seq_tensor.tensor<float, 3>();

    const Tensor& ids_len_tensor = context->input(1);
    auto ids_len_data = ids_len_tensor.vec<int32>();

    const Tensor& is_new_tensor = context->input(2);
    auto is_new_data = is_new_tensor.matrix<float>();

    int64 batch_size = ids_seq_tensor.dim_size(0);
    int64 num_ids = ids_seq_tensor.dim_size(2);

    int64 is_new_seq_len = is_new_tensor.dim_size(1);

    // Create output tensor
    std::vector<int64> output_shape_dims = {batch_size, is_new_seq_len,
                                            num_ids};
    Tensor* output_tensor = nullptr;
    TensorShape output_shape;
    OP_REQUIRES_OK(
        context, TensorShapeUtils::MakeShape(output_shape_dims, &output_shape));
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));
    auto output_data = output_tensor->shaped<float, 3>(output_shape_dims);

    for (int64 batch_num = 0; batch_num < batch_size; batch_num++) {
      int64 max_id = 0;
      for (int64 seq_num = 0; seq_num < is_new_seq_len; seq_num++) {
        for (int i = 0; i < num_ids; i++) {
          output_data(batch_num, seq_num, i) = 0.0;
        }

        if (seq_num < ids_len_data(batch_num)) {
          int64 state_id = 0;
          for (int i = 0; i < num_ids; i++) {
            if (ids_seq_data(batch_num, seq_num, i) > 0.0) {
              state_id = i;
            }
          }

          max_id = std::max(max_id, state_id);
        } else {
          if (is_new_data(batch_num, seq_num) > 0.0 && max_id < num_ids - 1) {
            max_id++;
          }
        }
        output_data(batch_num, seq_num, max_id) = 1.0;
      }
    }
  }
};

}  // namespace contrack

REGISTER_OP("SequenceConcat")
    .Attr("N: int")
    .Input("sequences: N * float32")
    .Input("lengths: N * int32")
    .Output("concatenated: float32")
    .Output("concatenated_lengths: int32")
    .Doc(R"doc(
Concatenates the sequences in two batches of sequences. 
)doc")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle output_shape;
      std::vector<::tensorflow::shape_inference::ShapeHandle> input_shapes;
      TF_RETURN_IF_ERROR(c->input("sequences", &input_shapes));
      TF_RETURN_IF_ERROR(
          c->ReplaceDim(input_shapes[0], 1, c->UnknownDim(), &output_shape));
      c->set_output(0, output_shape);

      std::vector<::tensorflow::shape_inference::ShapeHandle> input_len_shapes;
      TF_RETURN_IF_ERROR(c->input("lengths", &input_len_shapes));
      c->set_output(1, input_len_shapes[0]);

      return Status::OK();
    });

REGISTER_OP("NewId")
    .Input("state_ids: float32")
    .Input("state_len: int32")
    .Input("is_new: float32")
    .Output("new_ids: float32")
    .Doc(R"doc(
Computes the next entity ID for each position in the contrack input sequence.
)doc")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle output_shape;
      auto seq_len = c->Dim(c->input(2), 1);
      TF_RETURN_IF_ERROR(c->ReplaceDim(c->input(0), 1, seq_len, &output_shape));
      c->set_output(0, output_shape);

      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("SequenceConcat").Device(tensorflow::DEVICE_CPU),
                        ::contrack::SequenceConcatOp);
REGISTER_KERNEL_BUILDER(Name("NewId").Device(tensorflow::DEVICE_CPU),
                        ::contrack::NewIdOp);
