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

#ifndef NEURAL_TREES_HELPERS_H_
#define NEURAL_TREES_HELPERS_H_
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

using Matrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ConstMatrixMap = Eigen::Map<const Matrix>;
using MatrixMap = Eigen::Map<Matrix>;

// NTUtils contains helper functions for type conversion.
struct NTUtils {
  static ConstMatrixMap TensorToEigenMatrixReadOnly(const Tensor* tensor,
                                                    const int num_rows,
                                                    const int num_cols) {
    return ConstMatrixMap(tensor->flat<float>().data(), num_rows, num_cols);
  }

  static MatrixMap TensorToEigenMatrix(Tensor* tensor, const int num_rows,
                                       const int num_cols) {
    return MatrixMap(tensor->flat<float>().data(), num_rows, num_cols);
  }
};

// Stores a node in the binary decision tree.
struct Node {
  float root_to_node_prob;
  float weight_input_dot_product;
  float routing_left_prob;
  bool reachable_descendant_leaf = false;
  double sum_g = 0;
};

// A smooth approximation to the indicator function.
// smooth_step_param must be >= 0.
float SmoothIndicator(const float v, const float smooth_step_param);

// Derivative w.r.t. to SmoothIndicator's input.
// smooth_step_param must be >= 0.
float SmoothIndicatorDerivative(const float v, const float smooth_step_param);

// Performs a forward pass over the tree while identifying reachable leaves.
// Returns (i) the output vector, (ii) the updated tree, and (iii) a
// vector of reachable leaves (contains the indices of the reachable leaves in
// tree_nodes).
void ForwardPassSingleSample(const Eigen::MatrixXf& node_weights,
                             const Eigen::MatrixXf& leaf_weights,
                             const Eigen::VectorXf& input_features,
                             const int depth, const float smooth_step_param,
                             const bool training_mode, Eigen::VectorXf* output,
                             std::vector<Node>* tree_nodes,
                             std::vector<int>* reachable_leaves);

// Returns the gradients w.r.t. to the inputs of the tree, internal nodes, and
// leaves. Internally calls ForwardPassSingleSample to efficiently construct the
// tree.
void BackwardPassSingleSample(const Eigen::MatrixXf& node_weights,
                              const Eigen::MatrixXf& leaf_weights,
                              const Eigen::VectorXf& input_features,
                              const Eigen::VectorXf& grad_loss_wrt_tree_output,
                              const int depth, const float smooth_step_param,
                              Eigen::VectorXf* grad_loss_wrt_input_features,
                              Eigen::MatrixXf* grad_loss_wrt_node_weights,
                              Eigen::MatrixXf* grad_loss_wrt_leaf_weights);

}  // namespace tensorflow

#endif  // NEURAL_TREES_HELPERS_H_
