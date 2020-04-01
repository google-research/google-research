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

#include "neural_trees_helpers.h"

#include <stack>

#include "tensorflow/core/lib/math/math_util.h"

namespace tensorflow {

const float kTolerance = 1e-10;

float SmoothIndicator(const float v, const float smooth_step_param) {
  float out;
  if (v <= -0.5 / smooth_step_param) {
    out = 0;
  } else if (v >= 0.5 / smooth_step_param) {
    out = 1;
  } else {
    const float x = smooth_step_param * v + 0.5;
    const float x_squared = x * x;
    const float x_cubed = x_squared * x;
    out = -2 * x_cubed + 3 * x_squared;
  }
  return out;
}

float SmoothIndicatorDerivative(const float v, const float smooth_step_param) {
  float out;
  if (std::fabs(v) <= 0.5 / smooth_step_param) {
    const float x = smooth_step_param * v + 0.5;
    const float x_squared = x * x;
    out = 6 * smooth_step_param * (-x_squared + x);
  } else {
    out = 0;
  }
  return out;
}

void ForwardPassSingleSample(const Eigen::MatrixXf& node_weights,
                             const Eigen::MatrixXf& leaf_weights,
                             const Eigen::VectorXf& input_features,
                             const int depth, const float smooth_step_param,
                             const bool training_mode, Eigen::VectorXf* output,
                             std::vector<Node>* tree_nodes,
                             std::vector<int>* reachable_leaves) {
  DCHECK(tree_nodes != nullptr) << "Got a null ptr to tree!";
  // tree allows for more readable indexing (e.g., tree[i]).
  std::vector<Node>& tree = *tree_nodes;
  // Check tree size.
  DCHECK(tree.size() == MathUtil::IPow(2, depth + 1) - 1)
      << "Inconsistent tree size!";
  // Label of the first leaf (assuming a breadth-first order).
  const int first_leaf_label = (tree.size() + 1) / 2 - 1;
  // Stack of indices (of nodes) to traverse.
  std::stack<int> to_traverse;
  // Initialize root probability.
  tree[0].root_to_node_prob = 1;
  // Push the root index.
  to_traverse.push(0);

  // Fill the tree depth-first, while skipping unreachable nodes.
  // Note: tree is a perfect binary tree.
  while (!to_traverse.empty()) {
    const int current_index = to_traverse.top();
    to_traverse.pop();
    tree[current_index].weight_input_dot_product =
        input_features.dot(node_weights.col(current_index));
    const float probability_left = SmoothIndicator(
        tree[current_index].weight_input_dot_product, smooth_step_param);
    tree[current_index].routing_left_prob = probability_left;
    // Branch left if prob_left is non zero.
    if (tree[current_index].routing_left_prob > kTolerance) {
      const int left_index = 2 * current_index + 1;
      tree[left_index].root_to_node_prob =
          tree[current_index].root_to_node_prob *
          tree[current_index].routing_left_prob;
      // Push to the stack only if left child is an internal node.
      if (left_index < first_leaf_label) {
        to_traverse.push(left_index);
      } else {
        // This is a reachable leaf.
        reachable_leaves->push_back(left_index);
      }
    }

    // Branch right if prob_right is non zero.
    // Note: Not mutually exclusive with the previous if.
    if (1 - tree[current_index].routing_left_prob > kTolerance) {
      const int right_index = 2 * current_index + 2;
      tree[right_index].root_to_node_prob =
          tree[current_index].root_to_node_prob *
          (1 - tree[current_index].routing_left_prob);
      // Push to the stack only if left child is an internal node.
      if (right_index < first_leaf_label) {
        to_traverse.push(right_index);
      } else {
        // This is a reachable leaf.
        reachable_leaves->push_back(right_index);
      }
    }
  }

  output->setZero();

  // Iterate over the (reachable) leaves to update the tree output.
  for (auto i : *reachable_leaves) {
    // (i - first_leaf_label) is the index of the column of leaf i in
    // leaf_weights.
    *output +=
        tree[i].root_to_node_prob * leaf_weights.col(i - first_leaf_label);
  }

  if (training_mode) {
    // Mark all the reachable leaves and their ancestors.
    for (auto i : *reachable_leaves) {
      // Traverse up to the root starting from the current leaf.
      int current_index = i;
      tree[i].reachable_descendant_leaf = true;
      while (current_index != 0) {
        // The body below marks the parent.
        // Is the current_index a left child?
        const bool left_child = (current_index % 2 == 1);
        const int parent_index =
            left_child ? (current_index - 1) / 2 : (current_index - 2) / 2;
        if (tree[parent_index].reachable_descendant_leaf) {
          break;
        } else {
          tree[parent_index].reachable_descendant_leaf = true;
        }
        // Move to the parent.
        current_index = parent_index;
      }
    }
  }
}

void BackwardPassSingleSample(const Eigen::MatrixXf& node_weights,
                              const Eigen::MatrixXf& leaf_weights,
                              const Eigen::VectorXf& input_features,
                              const Eigen::VectorXf& grad_loss_wrt_tree_output,
                              const int depth, const float smooth_step_param,
                              Eigen::VectorXf* grad_loss_wrt_input_features,
                              Eigen::MatrixXf* grad_loss_wrt_node_weights,
                              Eigen::MatrixXf* grad_loss_wrt_leaf_weights) {
  const int tree_num_nodes = MathUtil::IPow(2, depth + 1) - 1;
  std::vector<Node> tree(tree_num_nodes);
  // Label of the first leaf (assuming a breadth-first order).
  const int first_leaf_label = (tree_num_nodes + 1) / 2 - 1;
  const int output_logits_dim = leaf_weights.rows();
  Eigen::VectorXf output_logits_sample(output_logits_dim);
  std::vector<int> reachable_leaves;
  // Do a forward pass to build the tree and obtain the reachable leaves.
  // TODO: Remove this forward pass and use the results from
  // the previous call to the forward pass.
  ForwardPassSingleSample(node_weights, leaf_weights, input_features, depth,
                          smooth_step_param, true, &output_logits_sample, &tree,
                          &reachable_leaves);

  grad_loss_wrt_input_features->setZero();
  grad_loss_wrt_node_weights->setZero();
  grad_loss_wrt_leaf_weights->setZero();

  // Stacks s1 and s2 are for post order traversal.
  std::stack<int> s1, s2;
  s1.push(0);
  while (!s1.empty()) {
    // Pop an item from s1 and push it to s2.
    const int current_index = s1.top();
    s1.pop();
    s2.push(current_index);
    // Push "reachable" left and right children to s1.
    const int left_index = 2 * current_index + 1;
    if (left_index < tree_num_nodes &&
        tree[left_index].reachable_descendant_leaf) {
      s1.push(left_index);
    }
    const int right_index = left_index + 1;
    if (right_index < tree_num_nodes &&
        tree[right_index].reachable_descendant_leaf) {
      s1.push(right_index);
    }
  }

  // Now do post order traversal by iterating over s2.
  while (!s2.empty()) {
    const int current_index = s2.top();
    s2.pop();
    // Process a leaf.
    if (current_index >= first_leaf_label) {
      // Update grad_loss_wrt_leaf_weights for leaf i.
      // index of current leaf starting from 0.
      const int leaf_zero_based_index = current_index - first_leaf_label;

      grad_loss_wrt_leaf_weights->col(leaf_zero_based_index) =
          grad_loss_wrt_tree_output * tree[current_index].root_to_node_prob;

      tree[current_index].sum_g =
          grad_loss_wrt_leaf_weights->col(leaf_zero_based_index)
              .dot(leaf_weights.col(leaf_zero_based_index));

    }
    // Process an internal node only if it's fractional (i.e., belongs to the
    // fractional tree).
    else if (tree[current_index].routing_left_prob > 0 &&
             tree[current_index].routing_left_prob < 1) {
      float activation_function_derivative = SmoothIndicatorDerivative(
          tree[current_index].weight_input_dot_product, smooth_step_param);
      // Notation below defined in Algorithm 2 of TEL's paper.
      const double mu_1 = activation_function_derivative /
                          tree[current_index].routing_left_prob;
      const double mu_2 = activation_function_derivative /
                          (1 - tree[current_index].routing_left_prob);
      const int left_index = 2 * current_index + 1;
      const int right_index = left_index + 1;
      const double a_minus_b =
          mu_1 * tree[left_index].sum_g - mu_2 * tree[right_index].sum_g;
      *grad_loss_wrt_input_features +=
          a_minus_b * node_weights.col(current_index);
      grad_loss_wrt_node_weights->col(current_index) =
          a_minus_b * input_features;
      tree[current_index].sum_g =
          tree[left_index].sum_g + tree[right_index].sum_g;
    } else {
      const int left_index = 2 * current_index + 1;
      const int right_index = left_index + 1;
      tree[current_index].sum_g =
          tree[left_index].sum_g + tree[right_index].sum_g;
    }
  }
}

}  // namespace tensorflow
