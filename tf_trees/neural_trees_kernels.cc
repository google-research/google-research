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

#include "third_party/google_research/google_research/tf_trees/neural_trees_helpers.h"
#include "third_party/tensorflow/core/framework/op_kernel.h"
#include "third_party/tensorflow/core/lib/math/math_util.h"
#include "third_party/tensorflow/core/util/work_sharder.h"

namespace tensorflow {

class NTComputeInputAndInternalParamsGradientsOp : public OpKernel {
 public:
  explicit NTComputeInputAndInternalParamsGradientsOp(
      OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("output_logits_dim", &output_logits_dim_));
    OP_REQUIRES_OK(context, context->GetAttr("depth", &depth_));
    OP_REQUIRES_OK(context, context->GetAttr("parallelize_over_samples",
                                             &parallelize_over_samples_));
  }
  void Compute(OpKernelContext* context) override {
    // Read inputs
    const Tensor* grad_loss_wrt_tree_output_t;
    OP_REQUIRES_OK(context, context->input("grad_loss_wrt_tree_output",
                                           &grad_loss_wrt_tree_output_t));
    const int batch_size = grad_loss_wrt_tree_output_t->dim_size(0);
    const ConstMatrixMap grad_loss_wrt_tree_output =
        NTUtils::TensorToEigenMatrixReadOnly(grad_loss_wrt_tree_output_t,
                                             batch_size, output_logits_dim_);

    const Tensor* input_features_t;
    OP_REQUIRES_OK(context,
                   context->input("input_features", &input_features_t));
    const int input_dim = input_features_t->dim_size(1);
    const ConstMatrixMap input_features = NTUtils::TensorToEigenMatrixReadOnly(
        input_features_t, batch_size, input_dim);

    const Tensor* node_weights_t;
    OP_REQUIRES_OK(context, context->input("node_weights", &node_weights_t));
    const int num_internal_nodes = node_weights_t->dim_size(1);
    const ConstMatrixMap node_weights = NTUtils::TensorToEigenMatrixReadOnly(
        node_weights_t, input_dim, num_internal_nodes);

    const Tensor* leaf_weights_t;
    OP_REQUIRES_OK(context, context->input("leaf_weights", &leaf_weights_t));
    const int num_leaves = leaf_weights_t->dim_size(1);
    const ConstMatrixMap leaf_weights = NTUtils::TensorToEigenMatrixReadOnly(
        leaf_weights_t, output_logits_dim_, num_leaves);

    const Tensor* smooth_step_param_t;
    OP_REQUIRES_OK(context,
                   context->input("smooth_step_param", &smooth_step_param_t));
    const ConstMatrixMap smooth_step_param_map =
        NTUtils::TensorToEigenMatrixReadOnly(smooth_step_param_t, 1, 1);
    smooth_step_param_ = smooth_step_param_map(0, 0);

    // Allocate outputs
    Tensor* grad_loss_wrt_input_features_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("grad_loss_wrt_input_features",
                                            {batch_size, input_dim},
                                            &grad_loss_wrt_input_features_t));
    MatrixMap grad_loss_wrt_input_features = NTUtils::TensorToEigenMatrix(
        grad_loss_wrt_input_features_t, batch_size, input_dim);

    Tensor* grad_loss_wrt_node_weights_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("grad_loss_wrt_node_weights",
                                            {input_dim, num_internal_nodes},
                                            &grad_loss_wrt_node_weights_t));
    MatrixMap grad_loss_wrt_node_weights = NTUtils::TensorToEigenMatrix(
        grad_loss_wrt_node_weights_t, input_dim, num_internal_nodes);

    Tensor* grad_loss_wrt_leaf_weights_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("grad_loss_wrt_leaf_weights",
                                            {output_logits_dim_, num_leaves},
                                            &grad_loss_wrt_leaf_weights_t));
    MatrixMap grad_loss_wrt_leaf_weights = NTUtils::TensorToEigenMatrix(
        grad_loss_wrt_leaf_weights_t, output_logits_dim_, num_leaves);

    // Set the outputs to zero.
    grad_loss_wrt_input_features.setZero();
    grad_loss_wrt_node_weights.setZero();
    grad_loss_wrt_leaf_weights.setZero();

    if (parallelize_over_samples_) {
      // Temporary matrices for storing the gradients of every sample.
      std::vector<Eigen::MatrixXf> grad_loss_wrt_node_weights_all_samples(
          batch_size, Eigen::MatrixXf::Zero(input_dim, num_internal_nodes));
      std::vector<Eigen::MatrixXf> grad_loss_wrt_leaf_weights_all_samples(
          batch_size, Eigen::MatrixXf::Zero(output_logits_dim_, num_leaves));

      // Iterate over the samples, performing a backward pass on each.
      auto do_work = [&input_features, &node_weights, &leaf_weights,
                      &grad_loss_wrt_tree_output, &grad_loss_wrt_input_features,
                      &grad_loss_wrt_node_weights_all_samples,
                      &grad_loss_wrt_leaf_weights_all_samples,
                      num_internal_nodes, input_dim, num_leaves,
                      this](int32 start, int32 end) {
        for (int sample_index = start; sample_index < end; ++sample_index) {
          const Eigen::VectorXf grad_loss_wrt_tree_output_sample =
              grad_loss_wrt_tree_output.row(sample_index);
          const Eigen::VectorXf input_features_sample =
              input_features.row(sample_index);

          // Update the gradients per sample.
          Eigen::VectorXf grad_loss_wrt_input_features_sample(input_dim);
          Eigen::MatrixXf grad_loss_wrt_node_weights_sample(input_dim,
                                                            num_internal_nodes);
          Eigen::MatrixXf grad_loss_wrt_leaf_weights_sample(output_logits_dim_,
                                                            num_leaves);
          BackwardPassSingleSample(
              node_weights, leaf_weights, input_features_sample,
              grad_loss_wrt_tree_output_sample, depth_, smooth_step_param_,
              &grad_loss_wrt_input_features_sample,
              &grad_loss_wrt_node_weights_sample,
              &grad_loss_wrt_leaf_weights_sample);

          // Update the global gradients (i.e., for all samples).
          grad_loss_wrt_input_features.row(sample_index) =
              grad_loss_wrt_input_features_sample;
          grad_loss_wrt_node_weights_all_samples[sample_index] =
              grad_loss_wrt_node_weights_sample;
          grad_loss_wrt_leaf_weights_all_samples[sample_index] =
              grad_loss_wrt_leaf_weights_sample;
        }
      };
      const int64 cost = 10000 * std::log10(input_dim) * std::log2(num_leaves);
      thread::ThreadPool* const worker_threads =
          context->device()->tensorflow_cpu_worker_threads()->workers;
      Shard(worker_threads->NumThreads(), worker_threads, batch_size,
            /*cost_per_unit=*/cost, do_work);

      // Aggregate the gradients of the samples.
      for (int i = 0; i < batch_size; ++i) {
        grad_loss_wrt_node_weights += grad_loss_wrt_node_weights_all_samples[i];
        grad_loss_wrt_leaf_weights += grad_loss_wrt_leaf_weights_all_samples[i];
      }
    } else {
      for (int sample_index = 0; sample_index < batch_size; ++sample_index) {
        const Eigen::VectorXf grad_loss_wrt_tree_output_sample =
            grad_loss_wrt_tree_output.row(sample_index);
        const Eigen::VectorXf input_features_sample =
            input_features.row(sample_index);

        // Update the gradients per sample.
        Eigen::VectorXf grad_loss_wrt_input_features_sample(input_dim);
        Eigen::MatrixXf grad_loss_wrt_node_weights_sample(input_dim,
                                                          num_internal_nodes);
        Eigen::MatrixXf grad_loss_wrt_leaf_weights_sample(output_logits_dim_,
                                                          num_leaves);
        BackwardPassSingleSample(
            node_weights, leaf_weights, input_features_sample,
            grad_loss_wrt_tree_output_sample, depth_, smooth_step_param_,
            &grad_loss_wrt_input_features_sample,
            &grad_loss_wrt_node_weights_sample,
            &grad_loss_wrt_leaf_weights_sample);

        // Update the global gradients (i.e., for all samples).
        grad_loss_wrt_input_features.row(sample_index) =
            grad_loss_wrt_input_features_sample;
        grad_loss_wrt_node_weights += grad_loss_wrt_node_weights_sample;
        grad_loss_wrt_leaf_weights += grad_loss_wrt_leaf_weights_sample;
      }
    }
  }

 private:
  int output_logits_dim_;
  int depth_;
  float smooth_step_param_;
  bool parallelize_over_samples_;
};

REGISTER_KERNEL_BUILDER(
    Name("NTComputeInputAndInternalParamsGradientsOp").Device(DEVICE_CPU),
    NTComputeInputAndInternalParamsGradientsOp);

class NTComputeOutputOp : public OpKernel {
 public:
  explicit NTComputeOutputOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("output_logits_dim", &output_logits_dim_));
    OP_REQUIRES_OK(context, context->GetAttr("depth", &depth_));
    OP_REQUIRES_OK(context, context->GetAttr("parallelize_over_samples",
                                             &parallelize_over_samples_));
  }
  void Compute(OpKernelContext* context) override {
    // Read inputs
    const Tensor* input_features_t;
    OP_REQUIRES_OK(context,
                   context->input("input_features", &input_features_t));
    const int batch_size = input_features_t->dim_size(0);
    const int input_dim = input_features_t->dim_size(1);
    const ConstMatrixMap input_features = NTUtils::TensorToEigenMatrixReadOnly(
        input_features_t, batch_size, input_dim);

    const Tensor* node_weights_t;
    OP_REQUIRES_OK(context, context->input("node_weights", &node_weights_t));
    const int num_internal_nodes = node_weights_t->dim_size(1);
    const ConstMatrixMap node_weights = NTUtils::TensorToEigenMatrixReadOnly(
        node_weights_t, input_dim, num_internal_nodes);

    const Tensor* leaf_weights_t;
    OP_REQUIRES_OK(context, context->input("leaf_weights", &leaf_weights_t));
    const int num_leaves = leaf_weights_t->dim_size(1);
    const ConstMatrixMap leaf_weights = NTUtils::TensorToEigenMatrixReadOnly(
        leaf_weights_t, output_logits_dim_, num_leaves);

    const Tensor* smooth_step_param_t;
    OP_REQUIRES_OK(context,
                   context->input("smooth_step_param", &smooth_step_param_t));
    const ConstMatrixMap smooth_step_param_map =
        NTUtils::TensorToEigenMatrixReadOnly(smooth_step_param_t, 1, 1);
    smooth_step_param_ = smooth_step_param_map(0, 0);

    // Allocate outputs
    Tensor* output_logits_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("output_logits",
                                            {batch_size, output_logits_dim_},
                                            &output_logits_t));
    MatrixMap output_logits = NTUtils::TensorToEigenMatrix(
        output_logits_t, batch_size, output_logits_dim_);

    Tensor* average_num_reachable_leaves_t = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("average_num_reachable_leaves", {},
                                            &average_num_reachable_leaves_t));
    float average_num_reachable_leaves = 0;

    const int tree_num_nodes = MathUtil::IPow(2, depth_ + 1) - 1;

    if (parallelize_over_samples_) {
      // Iterate over the samples, performing a forward pass on each.
      auto do_work = [&input_features, &node_weights, &leaf_weights,
                      &output_logits, &average_num_reachable_leaves,
                      tree_num_nodes, batch_size,
                      this](int32 start, int32 end) {
        for (int sample_index = start; sample_index < end; ++sample_index) {
          const Eigen::VectorXf input_features_sample =
              input_features.row(sample_index);

          // This tree is specific to the current sample.
          std::vector<Node> sample_tree(tree_num_nodes);

          Eigen::VectorXf output_logits_sample(output_logits_dim_);
          std::vector<int> leaves;
          ForwardPassSingleSample(node_weights, leaf_weights,
                                  input_features_sample, depth_,
                                  smooth_step_param_, false,
                                  &output_logits_sample, &sample_tree, &leaves);

          // Update the tree output matrix.
          output_logits.row(sample_index) = output_logits_sample;
          average_num_reachable_leaves +=
              leaves.size() / static_cast<float>(batch_size);
        }
      };
      const int64 cost = 10000 * std::log10(input_dim) * std::log2(num_leaves);
      thread::ThreadPool* const worker_threads =
          context->device()->tensorflow_cpu_worker_threads()->workers;
      Shard(worker_threads->NumThreads(), worker_threads, batch_size,
            /*cost_per_unit=*/cost, do_work);
    } else {
      for (int sample_index = 0; sample_index < batch_size; ++sample_index) {
        const Eigen::VectorXf input_features_sample =
            input_features.row(sample_index);

        // This tree is specific to the current sample.
        std::vector<Node> sample_tree(tree_num_nodes);
        Eigen::VectorXf output_logits_sample(output_logits_dim_);
        std::vector<int> leaves;
        ForwardPassSingleSample(node_weights, leaf_weights,
                                input_features_sample, depth_,
                                smooth_step_param_, false,
                                &output_logits_sample, &sample_tree, &leaves);
        // Update the tree output matrix.
        output_logits.row(sample_index) = output_logits_sample;
        average_num_reachable_leaves += leaves.size();
      }
      average_num_reachable_leaves /= static_cast<float>(batch_size);
    }
    average_num_reachable_leaves_t->scalar<float>()() =
        average_num_reachable_leaves;
  }

 private:
  int output_logits_dim_;
  int depth_;
  float smooth_step_param_;
  bool parallelize_over_samples_;
};

REGISTER_KERNEL_BUILDER(Name("NTComputeOutputOp").Device(DEVICE_CPU),
                        NTComputeOutputOp);

}  // namespace tensorflow
