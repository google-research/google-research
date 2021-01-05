# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Neural Tree Ops."""
import numpy as np

from tf_trees.gen_neural_trees_ops import nt_compute_input_and_internal_params_gradients_op
from tf_trees.gen_neural_trees_ops import nt_compute_output_op
# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest
# pylint:enable=g-direct-tensorflow-import


class NTTest(test_util.TensorFlowTestCase):
  """Tests for neural trees ops."""

  def test_nt_compute_output_op_depth_1(self):
    with self.cached_session():
      # Define tree params and weights.
      depth = 1
      output_logits_dim = 2
      smooth_step_param = 0
      parallelize_over_samples = False
      node_weights = array_ops.constant([[1], [1]], dtype=np.float32)
      leaf_weights = array_ops.constant([[0, 1], [1, 0]], dtype=np.float32)

      # Test 1: Single input sample which is soft routed with probability
      # 0.5 to the left and right.
      input_features = array_ops.constant([[0, 0]], dtype=np.float32)

      output_logits = nt_compute_output_op(input_features, node_weights,
                                           leaf_weights, output_logits_dim,
                                           depth, smooth_step_param,
                                           parallelize_over_samples)

      self.assertAllClose(0.5, output_logits[0, 0], atol=1e-6)
      self.assertAllClose(0.5, output_logits[0, 1], atol=1e-6)

      # Test 2: A batch of 3 input samples.
      # Sample 1: Soft routed w.p. 0.5.
      # Sample 2: Hard routed to the left leaf.
      # Sample 3: Hard routed to the right leaf.
      input_features = array_ops.constant([[0, 0], [0.5, 0], [-0.5, 0]],
                                          dtype=np.float32)

      output_logits = nt_compute_output_op(input_features, node_weights,
                                           leaf_weights, output_logits_dim,
                                           depth, smooth_step_param,
                                           parallelize_over_samples)

      # Check the output for sample 1.
      self.assertAllClose(0.5, output_logits[0, 0], atol=1e-6)
      self.assertAllClose(0.5, output_logits[0, 1], atol=1e-6)

      # Check the output for sample 2.
      self.assertAllClose(0, output_logits[1, 0], atol=1e-6)
      self.assertAllClose(1, output_logits[1, 1], atol=1e-6)

      # Check the output for sample 3.
      self.assertAllClose(1, output_logits[2, 0], atol=1e-6)
      self.assertAllClose(0, output_logits[2, 1], atol=1e-6)

  def test_nt_compute_output_op_depth_2(self):
    with self.cached_session():
      # Define tree params and weights.
      depth = 2
      output_logits_dim = 3
      smooth_step_param = 0
      parallelize_over_samples = False
      node_weights = array_ops.constant([[1, 0.1, 0], [2, 0.05, 1]],
                                        dtype=np.float32)
      leaf_weights = array_ops.constant(
          [[0.5, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=np.float32)

      # Sample 1: Hard routed to the left subtree and then soft routed to
      # the leaves.
      # Sample 2: Hard routed to the right most leaf.
      # Sample 3: Hard routed to the left most leaf.
      input_features = array_ops.constant([[1, 1], [-1, -1], [10, 10]],
                                          dtype=np.float32)

      output_logits = nt_compute_output_op(input_features, node_weights,
                                           leaf_weights, output_logits_dim,
                                           depth, smooth_step_param,
                                           parallelize_over_samples)

      # Check the output for sample 1.
      self.assertAllClose(0.359125, output_logits[0, 0], atol=1e-6)
      self.assertAllClose(0.28175, output_logits[0, 1], atol=1e-6)
      self.assertAllClose(0, output_logits[0, 2], atol=1e-6)

      # Check the output for sample 2.
      self.assertAllClose(1, output_logits[1, 0], atol=1e-6)
      self.assertAllClose(1, output_logits[1, 1], atol=1e-6)
      self.assertAllClose(0, output_logits[1, 2], atol=1e-6)

      # Check the output for sample 3.
      self.assertAllClose(0.5, output_logits[2, 0], atol=1e-6)
      self.assertAllClose(0, output_logits[2, 1], atol=1e-6)
      self.assertAllClose(0, output_logits[2, 2], atol=1e-6)

  def test_nt_compute_input_and_internal_params_gradients_op_depth_1(self):
    with self.cached_session():
      # Define tree params and weights.
      depth = 1
      output_logits_dim = 2
      smooth_step_param = 0
      parallelize_over_samples = False
      grad_loss_wrt_tree_output = array_ops.constant([[1, 1]], dtype=np.float32)
      node_weights = array_ops.constant([[1], [1]], dtype=np.float32)
      leaf_weights = array_ops.constant([[0, 1], [1, 0]], dtype=np.float32)

      # Test 1: Single input sample which is soft routed with probability
      # 0.5 to the left and right.
      input_features = array_ops.constant([[0, 0]], dtype=np.float32)

      (grad_loss_wrt_input_features,
       grad_loss_wrt_node_weights,
       grad_loss_wrt_leaf_weights) \
       = nt_compute_input_and_internal_params_gradients_op(
           grad_loss_wrt_tree_output, input_features, node_weights, leaf_weights,
           output_logits_dim, depth, smooth_step_param, parallelize_over_samples)

      self.assertAllClose(0, grad_loss_wrt_input_features[0, 0], atol=1e-6)
      self.assertAllClose(0, grad_loss_wrt_input_features[0, 1], atol=1e-6)

      self.assertAllClose(0, grad_loss_wrt_node_weights[0, 0], atol=1e-6)
      self.assertAllClose(0, grad_loss_wrt_node_weights[1, 0], atol=1e-6)

      self.assertAllClose(0.5, grad_loss_wrt_leaf_weights[0, 0], atol=1e-6)
      self.assertAllClose(0.5, grad_loss_wrt_leaf_weights[1, 0], atol=1e-6)
      self.assertAllClose(0.5, grad_loss_wrt_leaf_weights[0, 1], atol=1e-6)
      self.assertAllClose(0.5, grad_loss_wrt_leaf_weights[1, 1], atol=1e-6)

      # Test 2: A batch of 4 samples.
      # Sample 1: Soft routed w.p. 0.5.
      # Sample 2: Hard routed to the left.
      # Sample 3: Soft routed to the right.
      # Sample 4: Soft routed w.p. 0.648 to the left.
      input_features = array_ops.constant([[0, 0], [1, 1], [-1, -1], [0.1, 0]],
                                          dtype=np.float32)
      grad_loss_wrt_tree_output = array_ops.constant(
          [[1, 1], [1, 1], [1, 1], [0, 1]], dtype=np.float32)

      (grad_loss_wrt_input_features,
       grad_loss_wrt_node_weights,
       grad_loss_wrt_leaf_weights) \
       = nt_compute_input_and_internal_params_gradients_op(
           grad_loss_wrt_tree_output, input_features, node_weights, leaf_weights,
           output_logits_dim, depth, smooth_step_param, parallelize_over_samples)

      self.assertAllClose(0, grad_loss_wrt_input_features[0, 0], atol=1e-6)
      self.assertAllClose(0, grad_loss_wrt_input_features[0, 1], atol=1e-6)
      self.assertAllClose(0, grad_loss_wrt_input_features[1, 0], atol=1e-6)
      self.assertAllClose(0, grad_loss_wrt_input_features[1, 1], atol=1e-6)
      self.assertAllClose(0, grad_loss_wrt_input_features[2, 0], atol=1e-6)
      self.assertAllClose(0, grad_loss_wrt_input_features[2, 1], atol=1e-6)
      self.assertAllClose(1.44, grad_loss_wrt_input_features[3, 0], atol=1e-6)
      self.assertAllClose(1.44, grad_loss_wrt_input_features[3, 1], atol=1e-6)

      self.assertAllClose(0.144, grad_loss_wrt_node_weights[0, 0], atol=1e-6)
      self.assertAllClose(0, grad_loss_wrt_node_weights[1, 0], atol=1e-6)

      self.assertAllClose(1.5, grad_loss_wrt_leaf_weights[0, 0], atol=1e-6)
      self.assertAllClose(2.148, grad_loss_wrt_leaf_weights[1, 0], atol=1e-6)
      self.assertAllClose(1.5, grad_loss_wrt_leaf_weights[0, 1], atol=1e-6)
      self.assertAllClose(1.852, grad_loss_wrt_leaf_weights[1, 1], atol=1e-6)


if __name__ == "__main__":
  googletest.main()
