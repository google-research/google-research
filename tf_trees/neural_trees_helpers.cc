#include "third_party/google_research/google_research/tf_trees/neural_trees_helpers.h"

#include <queue>

#include "third_party/tensorflow/core/lib/math/math_util.h"

namespace tensorflow {

const float kTolerance = 1e-10;

float SmoothIndicator(const float v, const float smooth_step_param,
                      const float c_1, const float c_2) {
  float out;
  if (v <= -0.5 + smooth_step_param) {
    out = 0;
  } else if (v >= 0.5 - smooth_step_param) {
    out = 1;
  } else {
    const float x = v + 0.5;
    const float x_squared = x * x;
    const float x_cubed = x_squared * x;
    out = c_1 * (-2 * x_cubed + 3 * x_squared + c_2);
  }
  return out;
}

void SmoothIndicatorConstants(const float smooth_step_param, float* constant_1,
                              float* constant_2) {
  const float smooth_step_param_squared = smooth_step_param * smooth_step_param;
  const float smooth_step_param_cubed =
      smooth_step_param_squared * smooth_step_param;
  *constant_1 =
      1.0 / (4 * smooth_step_param_cubed - 6 * smooth_step_param_squared + 1);
  *constant_2 = smooth_step_param_squared * (2 * smooth_step_param - 3);
}

float SmoothIndicatorDerivative(const float v, const float smooth_step_param,
                                const float c_1) {
  float out;
  if (std::fabs(v) <= 0.5 - smooth_step_param) {
    const float x = v + 0.5;
    const float x_squared = x * x;
    out = 6 * c_1 * (-x_squared + x);
  } else {
    out = 0;
  }
  return out;
}

void ForwardPassSingleSample(const Eigen::MatrixXf& node_weights,
                             const Eigen::MatrixXf& leaf_weights,
                             const Eigen::VectorXf& input_features,
                             const int depth, const float smooth_step_param,
                             Eigen::VectorXf* output,
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
  // Compute the constants used by the smooth indicator.
  float smooth_indicator_const_1;
  float smooth_indicator_const_2;
  SmoothIndicatorConstants(smooth_step_param, &smooth_indicator_const_1,
                           &smooth_indicator_const_2);
  // Queue of indices (of nodes) to traverse.
  std::queue<int> queue;
  // Initialize root probability.
  tree[0].root_to_node_prob = 1;
  // Push the root index.
  queue.push(0);

  // Fill the tree breadth first, while skipping unreachable nodes.
  while (!queue.empty()) {
    int current_index = queue.front();
    queue.pop();
    tree[current_index].weight_input_dot_product =
        input_features.dot(node_weights.col(current_index));
    tree[current_index].routing_left_prob = SmoothIndicator(
        tree[current_index].weight_input_dot_product, smooth_step_param,
        smooth_indicator_const_1, smooth_indicator_const_2);

    // Branch left if prob_left is non zero.
    if (tree[current_index].routing_left_prob > kTolerance) {
      int left_index = 2 * current_index + 1;
      tree[left_index].root_to_node_prob =
          tree[current_index].root_to_node_prob *
          tree[current_index].routing_left_prob;
      // Push to the queue only if left child is an internal node.
      if (left_index < first_leaf_label) {
        queue.push(left_index);
      } else {
        // This is a reachable leaf.
        reachable_leaves->push_back(left_index);
      }
    }

    // Branch right if prob_right is non zero.
    // Note: Not mutually exclusive with the previous if.
    if (1 - tree[current_index].routing_left_prob > kTolerance) {
      int right_index = 2 * current_index + 2;
      tree[right_index].root_to_node_prob =
          tree[current_index].root_to_node_prob *
          (1 - tree[current_index].routing_left_prob);
      // Push to the queue only if left child is an internal node.
      if (right_index < first_leaf_label) {
        queue.push(right_index);
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
  // TODO(hazimeh): Remove this forward pass and use the results from
  // the previous call to the forward pass.
  ForwardPassSingleSample(node_weights, leaf_weights, input_features, depth,
                          smooth_step_param, &output_logits_sample, &tree,
                          &reachable_leaves);

  grad_loss_wrt_input_features->setZero();
  grad_loss_wrt_node_weights->setZero();
  grad_loss_wrt_leaf_weights->setZero();
  // Compute the constants used by the smooth indicator.
  float smooth_indicator_const_1;
  float smooth_indicator_const_2;
  SmoothIndicatorConstants(smooth_step_param, &smooth_indicator_const_1,
                           &smooth_indicator_const_2);
  // Traverse the tree starting from the reachable leaves up
  // to the root, while updating the 3 gradients.
  for (auto i : reachable_leaves) {
    // index of current leaf starting from 0.
    const int leaf_zero_based_index = i - first_leaf_label;
    // Update grad_loss_wrt_leaf_weights for leaf i.
    grad_loss_wrt_leaf_weights->col(leaf_zero_based_index) =
        grad_loss_wrt_tree_output * tree[i].root_to_node_prob;
    // Intermediate quantity used when computing the other two gradients.
    float delloss_by_delt_dot_ol =
        grad_loss_wrt_tree_output.dot(leaf_weights.col(leaf_zero_based_index));
    // Traverse up to the root starting from the current leaf.
    int current_index = i;
    while (current_index != 0) {
      // The body visits the parent of the current_index to update the grads.
      // Is the current_index a left child?
      const bool left_child = (current_index % 2 == 1);

      const int parent_index =
          left_child ? (current_index - 1) / 2 : (current_index - 2) / 2;

      const float prob_parent_route_to_leaf =
          left_child ? tree[parent_index].routing_left_prob
                     : 1 - tree[parent_index].routing_left_prob;

      const float prob_root_to_leaf_exclude_parent =
          tree[i].root_to_node_prob / prob_parent_route_to_leaf;

      const float smooth_indicator_derivative = SmoothIndicatorDerivative(
          tree[parent_index].weight_input_dot_product, smooth_step_param,
          smooth_indicator_const_1);

      Eigen::VectorXf gradient_routing_function_wrt_input_features =
          smooth_indicator_derivative * node_weights.col(parent_index);
      if (!left_child) gradient_routing_function_wrt_input_features *= -1;

      *grad_loss_wrt_input_features +=
          delloss_by_delt_dot_ol *
          gradient_routing_function_wrt_input_features *
          prob_root_to_leaf_exclude_parent;

      Eigen::VectorXf gradient_routing_function_wrt_current_node_weight =
          smooth_indicator_derivative * input_features;
      if (!left_child) gradient_routing_function_wrt_current_node_weight *= -1;

      grad_loss_wrt_node_weights->col(parent_index) +=
          delloss_by_delt_dot_ol *
          gradient_routing_function_wrt_current_node_weight *
          prob_root_to_leaf_exclude_parent;

      // Move to the parent.
      current_index = parent_index;
    }
  }
}

}  // namespace tensorflow
