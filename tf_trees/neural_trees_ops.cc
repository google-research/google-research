#include "third_party/tensorflow/core/framework/op.h"
#include "third_party/tensorflow/core/framework/shape_inference.h"
#include "third_party/tensorflow/core/lib/math/math_util.h"

namespace tensorflow {

using shape_inference::InferenceContext;

// This op performs a backward pass over the tree to
// compute the gradients w.r.t. tree's parameters and inputs
REGISTER_OP("NTComputeInputAndInternalParamsGradientsOp")
    // Gradient of loss w.r.t. tree's output
    // shape = [batch_size, output_logits_dim]
    .Input("grad_loss_wrt_tree_output: float32")
    // Logits predicted by the input layer preceding the tree.
    // shape = [batch_size, input_dim]
    .Input("input_features: float32")
    // Weights of the internal nodes in the tree
    // shape = [input_dim, num_internal_nodes]
    .Input("node_weights: float32")
    // Weights of the leaves in the tree
    // shape = [output_logits_dim, num_leaves]
    .Input("leaf_weights: float32")
    // Dimension of the tree output logits
    .Attr("output_logits_dim: int >= 1")
    // Depth of the tree
    .Attr("depth: int >= 1")
    // Smooth step activation function parameter
    .Attr("smooth_step_param: float")
    // Set to true to parallelize over the samples in the batch.
    .Attr("parallelize_over_samples: bool")
    // Gradient of loss w.r.t. the tree input features
    // shape = [batch_size, input_dim].
    .Output("grad_loss_wrt_input_features: float32")
    // Gradient of loss w.r.t. internal node weights
    // shape = [input_dim, num_internal_nodes].
    .Output("grad_loss_wrt_node_weights: float32")
    // Gradient of loss w.r.t. leaf weights
    // shape = [output_logits_dim, num_leaves].
    .Output("grad_loss_wrt_leaf_weights: float32")
    .SetShapeFn([](InferenceContext* c) {
      shape_inference::ShapeHandle unused_shape;
      // Extract the attributes
      int output_logits_dim;
      TF_RETURN_IF_ERROR(c->GetAttr("output_logits_dim", &output_logits_dim));
      int depth;
      TF_RETURN_IF_ERROR(c->GetAttr("depth", &depth));
      // Compute the number of leaves and internal nodes
      const int num_leaves = MathUtil::IPow(2, depth);
      const int num_internal_nodes = num_leaves - 1;

      // Check the input dims and shapes
      // grad_loss_wrt_tree_output
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused_shape));
      auto batch_size = c->Dim(c->input(0), 0);
      auto grad_loss_wrt_tree_output_shape =
          c->MakeShape({batch_size, output_logits_dim});
      TF_RETURN_IF_ERROR(c->Merge(c->input(0), grad_loss_wrt_tree_output_shape,
                                  &unused_shape));
      // input_features
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &unused_shape));
      auto input_dim = c->Dim(c->input(1), 1);
      auto input_features_shape = c->MakeShape({batch_size, input_dim});
      TF_RETURN_IF_ERROR(
          c->Merge(c->input(1), input_features_shape, &unused_shape));
      // node_weights
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &unused_shape));
      auto node_weights_shape = c->MakeShape({input_dim, num_internal_nodes});
      TF_RETURN_IF_ERROR(
          c->Merge(c->input(2), node_weights_shape, &unused_shape));
      // leaf_weights
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &unused_shape));
      auto leaf_weights_shape = c->MakeShape({output_logits_dim, num_leaves});
      TF_RETURN_IF_ERROR(
          c->Merge(c->input(3), leaf_weights_shape, &unused_shape));

      // Set the output shapes.
      // grad_loss_wrt_input_features
      c->set_output(0, input_features_shape);
      // grad_loss_wrt_node_weights
      c->set_output(1, node_weights_shape);
      // grad_loss_wrt_leaf_weights
      c->set_output(2, leaf_weights_shape);

      return Status::OK();
    });

// This op performs a forward pass over the tree and
// returns the predictions.
REGISTER_OP("NTComputeOutputOp")
    // Logits predicted by the input layer preceding the tree.
    // shape = [batch_size, input_dim]
    .Input("input_features: float32")
    // Weights of the internal nodes in the tree.
    // shape = [input_dim, num_internal_nodes]
    .Input("node_weights: float32")
    // Weights of the leaves in the tree.
    // shape = [output_logits_dim, num_leaves]
    .Input("leaf_weights: float32")
    // Dimension of the tree output logits.
    .Attr("output_logits_dim: int >= 1")
    // Depth of the tree
    .Attr("depth: int >= 1")
    // Smooth step activation function parameter.
    .Attr("smooth_step_param: float")
    // Set to true to parallelize over the samples in the batch.
    .Attr("parallelize_over_samples: bool")
    // Prediction of the tree.
    // shape = [batch_size, output_logits_dim].
    .Output("output_logits: float32")
    .SetShapeFn([](InferenceContext* c) {
      shape_inference::ShapeHandle unused_shape;
      // Extract the attributes
      int output_logits_dim;
      TF_RETURN_IF_ERROR(c->GetAttr("output_logits_dim", &output_logits_dim));
      int depth;
      TF_RETURN_IF_ERROR(c->GetAttr("depth", &depth));
      // Compute the number of leaves and internal nodes.
      const int num_leaves = MathUtil::IPow(2, depth);
      const int num_internal_nodes = num_leaves - 1;

      // Check the input dims and shapes
      // input_features
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &unused_shape));
      auto input_dim = c->Dim(c->input(0), 1);
      auto batch_size = c->Dim(c->input(0), 0);
      // node_weights
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &unused_shape));
      auto node_weights_shape = c->MakeShape({input_dim, num_internal_nodes});
      TF_RETURN_IF_ERROR(
          c->Merge(c->input(1), node_weights_shape, &unused_shape));
      // leaf_weights
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &unused_shape));
      auto leaf_weights_shape = c->MakeShape({output_logits_dim, num_leaves});
      TF_RETURN_IF_ERROR(
          c->Merge(c->input(2), leaf_weights_shape, &unused_shape));

      // Set the output shape
      // output_logits
      auto output_logits_shape = c->MakeShape({batch_size, output_logits_dim});
      c->set_output(0, output_logits_shape);

      return Status::OK();
    });

}  // namespace tensorflow
