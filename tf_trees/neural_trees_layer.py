# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Main code for creating neural tree layer."""
import tensorflow as tf
from tensorflow import keras
from tensorflow import ops
from tensorflow.keras.initializers import RandomUniform

from tf_trees.gen_neural_trees_ops import nt_compute_input_and_internal_params_gradients_op
from tf_trees.gen_neural_trees_ops import nt_compute_output_op


# Register the custom gradient for NTComputeOutputOp.
@ops.RegisterGradient('NTComputeOutputOp')
def _nt_compute_input_and_internal_params_gradients_op_cc(
    op, grad_loss_wrt_tree_output):
  """Associated a custom gradient with an op."""
  output_logits_dim = op.get_attr('output_logits_dim')
  depth = op.get_attr('depth')
  smooth_step_param = op.get_attr('smooth_step_param')
  parallelize_over_samples = op.get_attr('parallelize_over_samples')
  return nt_compute_input_and_internal_params_gradients_op(
      grad_loss_wrt_tree_output, op.inputs[0], op.inputs[1], op.inputs[2],
      output_logits_dim, depth, smooth_step_param, parallelize_over_samples)


class NeuralTrees(keras.layers.Layer):
  """A custom layer containing additive decision trees.

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
    smooth_step_param: A float taking values in the range (0, 0.5]. Values
      closer to 0.5 makes the trees more likely to hard route samples (i.e.,
      send samples to fewer leaves).
    sum_outputs: Boolean. If true, the outputs of the trees will be added,
      leading to a 2D tensor of shape=[batch_size, output_logits_dim].
      Otherwise, the outputs of the trees are not added and the layer output is
      a 2D tensor of shape=[batch_size, trees_num * output_logits_dim].
    parallelize_over_samples: Boolean, If true, parallelizes the updates over
      the samples in each tree. Might lead to speedups when the number of trees
      is small (at the cost of extra memory consumption).
    split_initializer: A Keras initializer for the internal (splitting) nodes.
    leaf_initializer: A Keras initializer for the leaves.
    split_regularizer: A Keras regularizer for the internal (splitting) nodes.
    leaf_regularizer: A Keras regularizer for the leaves.
  Input shape: A tensor of shape=[batch_size, input_dim].
  Output shape: A tensor of shape=[batch_size, output_logits_dim] if sum_outputs
    = True. Otherwise, a tensor of shape=[batch_size, trees_num *
    output_logits_dim].
  """

  def __init__(self,
               output_logits_dim,
               trees_num=1,
               depth=3,
               smooth_step_param=0.3,
               sum_outputs=True,
               parallelize_over_samples=False,
               split_initializer=RandomUniform(-0.01, 0.01),
               leaf_initializer=RandomUniform(-0.01, 0.01),
               split_regularizer=None,
               leaf_regularizer=None,
               **kwargs):
    """Initializes neural trees layer."""
    self.trees_num = trees_num
    self.depth = depth
    self.output_logits_dim = output_logits_dim
    self.smooth_step_param = smooth_step_param
    self.parallelize_over_samples = parallelize_over_samples
    self.sum_outputs = sum_outputs
    self.split_initializer = keras.initializers.get(split_initializer)
    self.leaf_initializer = keras.initializers.get(leaf_initializer)
    self.split_regularizer = keras.regularizers.get(split_regularizer)
    self.leaf_regularizer = keras.regularizers.get(leaf_regularizer)
    super(NeuralTrees, self).__init__(**kwargs)

  def build(self, input_shape):
    """Creates a keras layer."""
    num_leaves = 2**self.depth
    num_internal_nodes = num_leaves - 1
    input_dim = input_shape[1]
    # A list of node weights. Each element corresponds to one tree.
    self.node_weights = []
    # A list of leaf weights. Each element corresponds to one tree.
    self.leaf_weights = []
    # Create node/leaf weights for every tree.
    for _ in range(self.trees_num):
      self.node_weights.append(
          self.add_weight(
              name='node_weights',
              shape=[input_dim, num_internal_nodes],
              initializer=self.split_initializer,
              trainable=True,
              regularizer=self.split_regularizer))

      self.leaf_weights.append(
          self.add_weight(
              name='leaf_weights',
              shape=[self.output_logits_dim, num_leaves],
              initializer=self.leaf_initializer,
              trainable=True,
              regularizer=self.leaf_regularizer))

    super(NeuralTrees, self).build(input_shape)

  def call(self, inputs):
    """Predict op."""
    # A list of tree outputs. Each element corresponds to one tree.
    tree_logits = []
    for tree_index in range(self.trees_num):
      tree_logits.append(
          nt_compute_output_op(inputs, self.node_weights[tree_index],
                               self.leaf_weights[tree_index],
                               self.output_logits_dim, self.depth,
                               self.smooth_step_param,
                               self.parallelize_over_samples))
    if self.trees_num == 1:
      return tree_logits[0]
    elif self.sum_outputs:
      return tf.accumulate_n(tree_logits)
    else:
      return tf.concat(tree_logits, axis=1)

  def get_config(self):
    """Prepares a config with hyperparams."""
    config = ({
        'trees_num':
            self.trees_num,
        'depth':
            self.depth,
        'output_logits_dim':
            self.output_logits_dim,
        'smooth_step_param':
            self.smooth_step_param,
        'parallelize_over_samples':
            self.parallelize_over_samples,
        'sum_outputs':
            self.sum_outputs,
        'split_initializer':
            keras.initializers.serialize(self.split_initializer),
        'leaf_initializer':
            keras.initializers.serialize(self.leaf_initializer),
        'split_regularizer':
            keras.regularizers.serialize(self.split_regularizer),
        'leaf_regularizer':
            keras.regularizers.serialize(self.leaf_regularizer),
    })

    base_config = super(NeuralTrees, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def compute_output_shape(self, input_shape):
    """Defines an output shape."""
    if self.sum_outputs:
      return tf.TensorShape([input_shape[0], self.output_logits_dim])
    else:
      return tf.TensorShape(
          [input_shape[0], self.trees_num * self.output_logits_dim])
