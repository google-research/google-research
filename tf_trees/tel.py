# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Main code for creating the tree ensemble layer."""
import os
import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.python.framework import ops
from tensorflow.keras.initializers import RandomUniform

# Assumes that neural_trees_ops.so is in tf_trees.
dir_path = os.path.dirname(os.path.realpath(__file__))
tf_trees_module = tf.load_op_library(dir_path + '/neural_trees_ops.so')


# Register the custom gradient for NTComputeOutputOp.
@ops.RegisterGradient('NTComputeOutputOp')
def _nt_compute_input_and_internal_params_gradients_op_cc(
    op, grad_loss_wrt_tree_output, _):
  """Associated a custom gradient with an op."""
  output_logits_dim = op.get_attr('output_logits_dim')
  depth = op.get_attr('depth')
  parallelize_over_samples = op.get_attr('parallelize_over_samples')
  return [
      tf_trees_module.nt_compute_input_and_internal_params_gradients_op(
          grad_loss_wrt_tree_output, op.inputs[0], op.inputs[1], op.inputs[2],
          op.inputs[3], output_logits_dim, depth, parallelize_over_samples),
      None
  ]


class TEL(keras.layers.Layer):
  """A custom layer containing additive differentiable decision trees.

    Each tree in the layer is composed of splitting (internal) nodes and leaves.
    A splitting node "routes" the samples left or right based on the
    corresponding activation. Samples can be routed in a hard way (i.e., sent
    to only one child) or in a soft way. The decision whether to hard or soft
    route is controlled by the smooth_step_param (see details below).
    The trees are modeled using smooth functions and can be optimized
    using standard continuous optimization methods (e.g., SGD).

    The layer can be combined with other Keras layers and can be used anywhere
    in the neural net.

    Attributes:
      output_logits_dim: Dimension of the output.
      trees_num: Number of trees in the layer.
      depth: Depth of each tree.
      smooth_step_param: A non-negative float. Larger values make the trees more
        likely to hard route samples (i.e., samples reach fewer leaves). Values
        >= 1 are recommended to exploit conditional computation. Note
        smooth_step_param = 1/gamma, where gamma is the parameter defined in the
        TEL paper.
      sum_outputs: Boolean. If true, the outputs of the trees will be added,
        leading to a 2D tensor of shape=[batch_size, output_logits_dim].
        Otherwise, the tree outputs are not added and the layer output is a 2D
        tensor of shape=[batch_size, trees_num * output_logits_dim].
      parallelize_over_samples: Boolean, If true, parallelizes the updates over
        the samples in the batch. Might lead to speedups when the number of
        trees is small (at the cost of extra memory consumption).
      split_initializer: A Keras initializer for the internal (splitting) nodes.
      leaf_initializer: A Keras initializer for the leaves.
      split_regularizer: A Keras regularizer for the internal (splitting) nodes.
      leaf_regularizer: A Keras regularizer for the leaves.
    Input shape: A tensor of shape=[batch_size, input_dim].
    Output shape: A tensor of shape=[batch_size, output_logits_dim] if
      sum_outputs=True. Otherwise, a tensor of shape=[batch_size, trees_num *
      output_logits_dim].
  """

  def __init__(self,
               output_logits_dim,
               trees_num=1,
               depth=3,
               smooth_step_param=1.0,
               sum_outputs=True,
               parallelize_over_samples=False,
               split_initializer=RandomUniform(-0.01, 0.01),
               leaf_initializer=RandomUniform(-0.01, 0.01),
               split_regularizer=None,
               leaf_regularizer=None,
               **kwargs):
    """Initializes neural trees layer."""
    self._trees_num = trees_num
    self._depth = depth
    self._output_logits_dim = output_logits_dim
    self._smooth_step_param = smooth_step_param
    self._parallelize_over_samples = parallelize_over_samples
    self._sum_outputs = sum_outputs
    self._split_initializer = keras.initializers.get(split_initializer)
    self._leaf_initializer = keras.initializers.get(leaf_initializer)
    self._split_regularizer = keras.regularizers.get(split_regularizer)
    self._leaf_regularizer = keras.regularizers.get(leaf_regularizer)
    super(TEL, self).__init__(**kwargs)

  # @override
  def build(self, input_shape):
    """Creates a keras layer."""
    num_leaves = 2**self._depth
    num_internal_nodes = num_leaves - 1
    input_dim = input_shape[1]
    # A list of node weights. Each element corresponds to one tree.
    self._node_weights = []
    # A list of leaf weights. Each element corresponds to one tree.
    self._leaf_weights = []
    # Create node/leaf weights for every tree.
    for _ in range(self._trees_num):
      self._node_weights.append(
          self.add_weight(
              name='node_weights',
              shape=[input_dim, num_internal_nodes],
              initializer=self._split_initializer,
              trainable=True,
              regularizer=self._split_regularizer))

      self._leaf_weights.append(
          self.add_weight(
              name='leaf_weights',
              shape=[self._output_logits_dim, num_leaves],
              initializer=self._leaf_initializer,
              trainable=True,
              regularizer=self._leaf_regularizer))

      self._smooth_step_param_var = self.add_weight(
          name='smooth_step_param_var',
          shape=[],
          initializer=tf.constant_initializer(float(self._smooth_step_param)),
          trainable=False)
    super(TEL, self).build(input_shape)

  # @override
  def call(self, inputs):
    """Predict op."""
    # A list of tree outputs. Each element corresponds to one tree.
    tree_logits = []
    for tree_index in range(self._trees_num):
      tree_out, _ = tf_trees_module.nt_compute_output_op(
          inputs, self._node_weights[tree_index],
          self._leaf_weights[tree_index], self._smooth_step_param_var,
          self._output_logits_dim, self._depth, self._parallelize_over_samples)
      tree_logits.append(tree_out)
    if self._trees_num == 1:
      return tree_logits[0]
    elif self._sum_outputs:
      return tf.math.accumulate_n(tree_logits)
    else:
      return tf.concat(tree_logits, axis=1)

  # @override
  def get_config(self):
    """Prepares a config with hyperparams."""
    config = ({
        'trees_num':
            self._trees_num,
        'depth':
            self._depth,
        'output_logits_dim':
            self._output_logits_dim,
        'smooth_step_param':
            self._smooth_step_param,
        'parallelize_over_samples':
            self._parallelize_over_samples,
        'sum_outputs':
            self._sum_outputs,
        'split_initializer':
            keras.initializers.serialize(self._split_initializer),
        'leaf_initializer':
            keras.initializers.serialize(self._leaf_initializer),
        'split_regularizer':
            keras.regularizers.serialize(self._split_regularizer),
        'leaf_regularizer':
            keras.regularizers.serialize(self._leaf_regularizer),
    })

    base_config = super(TEL, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  # @override
  def compute_output_shape(self, input_shape):
    """Defines an output shape."""
    if self._sum_outputs:
      return tf.TensorShape([input_shape[0], self._output_logits_dim])
    else:
      return tf.TensorShape(
          [input_shape[0], self._trees_num * self._output_logits_dim])
