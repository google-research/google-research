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

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for strip_pruning_vars."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow.compat.v1 as tf
from model_pruning.python import pruning
from model_pruning.python import strip_pruning_vars_lib
from model_pruning.python.layers import rnn_cells


def _get_number_pruning_vars(graph_def):
  number_vars = 0
  for node in graph_def.node:
    if re.match(
        r"^.*(mask$)|(threshold$)|(old_weight$)|(old_old_weight$)|(gradient$)",
        node.name):
      number_vars += 1
  return number_vars


def _get_node_names(tensor_names):
  return [
      strip_pruning_vars_lib._node_name(tensor_name)
      for tensor_name in tensor_names
  ]


class StripPruningVarsTest(tf.test.TestCase):

  def setUp(self):
    super(StripPruningVarsTest, self).setUp()
    param_list = [
        "pruning_frequency=1", "begin_pruning_step=1", "end_pruning_step=10",
        "nbins=2048", "threshold_decay=0.0", "prune_option=first_order_gradient"
    ]
    self.initial_graph = tf.Graph()
    self.initial_graph_def = None
    self.final_graph = tf.Graph()
    self.final_graph_def = None
    self.pruning_spec = ",".join(param_list)
    with self.initial_graph.as_default():
      self.sparsity = tf.Variable(0.5, name="sparsity")
      self.global_step = tf.train.get_or_create_global_step()
      self.increment_global_step = tf.assign_add(self.global_step, 1)
      self.mask_update_op = None

  def _build_convolutional_model(self, number_of_layers):
    # Create a graph with several conv2d layers
    base_depth = 64
    height, width = 7, 9
    input_tensor = tf.ones((8, height, width, base_depth))
    top_layer = input_tensor

    prev_depth = base_depth
    depth_step = 32

    with tf.variable_scope("conv_model"):
      for ix in range(number_of_layers):
        layer_name = "layer" + str(ix)
        with tf.variable_scope(layer_name) as scope:
          cur_depth = prev_depth + depth_step

          kernel = tf.Variable(
              tf.truncated_normal([3, 3, prev_depth, cur_depth],
                                  dtype=tf.float32),
              name="weights")
          top_layer = tf.nn.conv2d(
              top_layer,
              pruning.apply_mask(kernel, scope, "first_order_gradient"),
              [1, 1, 1, 1],
              padding="SAME")

          prev_depth = cur_depth

    return top_layer

  def _build_fully_connected_model(self, number_of_layers):
    base_depth = 128

    input_tensor = tf.ones((8, base_depth))
    top_layer = input_tensor

    prev_depth = base_depth
    depth_step = 128

    with tf.variable_scope("fc_model"):
      for ix in range(number_of_layers):
        layer_name = "layer" + str(ix)
        with tf.variable_scope(layer_name) as scope:
          cur_depth = prev_depth + depth_step

          kernel = tf.Variable(
              tf.truncated_normal([prev_depth, cur_depth], dtype=tf.float32),
              name="weights")
          bias = tf.Variable(
              tf.truncated_normal([cur_depth], dtype=tf.float32), name="biases")
          top_layer = tf.nn.relu_layer(
              top_layer,
              pruning.apply_mask(kernel, scope, "first_order_gradient"),
              bias,
              name=scope.name)

          prev_depth = cur_depth

    return top_layer

  def _build_lstm_model(self, number_of_layers):
    batch_size = 8
    dim = 10
    inputs = tf.Variable(tf.random_normal([batch_size, dim]))

    def lstm_cell():
      return rnn_cells.MaskedBasicLSTMCell(
          dim, forget_bias=0.0, state_is_tuple=True, reuse=False)

    cell = tf.nn.rnn_cell.MultiRNNCell(
        [lstm_cell() for _ in range(number_of_layers)], state_is_tuple=True)

    outputs = tf.nn.static_rnn(
        cell, [inputs], initial_state=cell.zero_state(batch_size, tf.float32))

    return outputs

  def _prune_model(self, session):
    pruning_hparams = pruning.get_pruning_hparams().parse(self.pruning_spec)
    p = pruning.Pruning(pruning_hparams, sparsity=self.sparsity)
    self.mask_update_op = p.conditional_mask_update_op()

    tf.global_variables_initializer().run()
    for _ in range(20):
      session.run(self.mask_update_op)
      session.run(self.increment_global_step)

  def _get_outputs(self, session, input_graph, tensors_list, graph_prefix=None):
    outputs = []

    for output_tensor in tensors_list:
      if graph_prefix:
        output_tensor = graph_prefix + "/" + output_tensor
      outputs.append(
          session.run(session.graph.get_tensor_by_name(output_tensor)))

    return outputs

  def _get_initial_outputs(self, output_tensor_names_list):
    with self.session(graph=self.initial_graph) as sess1:
      self._prune_model(sess1)
      reference_outputs = self._get_outputs(sess1, self.initial_graph,
                                            output_tensor_names_list)

      self.initial_graph_def = tf.graph_util.convert_variables_to_constants(
          sess1, sess1.graph.as_graph_def(),
          _get_node_names(output_tensor_names_list))
    return reference_outputs

  def _get_final_outputs(self, output_tensor_names_list):
    self.final_graph_def = strip_pruning_vars_lib.strip_pruning_vars_fn(
        self.initial_graph_def, _get_node_names(output_tensor_names_list))
    _ = tf.graph_util.import_graph_def(self.final_graph_def, name="final")

    with self.test_session(self.final_graph) as sess2:
      final_outputs = self._get_outputs(
          sess2,
          self.final_graph,
          output_tensor_names_list,
          graph_prefix="final")
    return final_outputs

  def _check_removal_of_pruning_vars(self, number_masked_layers):
    self.assertEqual(
        _get_number_pruning_vars(self.initial_graph_def), number_masked_layers)
    self.assertEqual(_get_number_pruning_vars(self.final_graph_def), 0)

  def _check_output_equivalence(self, initial_outputs, final_outputs):
    for initial_output, final_output in zip(initial_outputs, final_outputs):
      self.assertAllEqual(initial_output, final_output)

  def testConvolutionalModel(self):
    with self.initial_graph.as_default():
      number_masked_conv_layers = 5
      top_layer = self._build_convolutional_model(number_masked_conv_layers)
      output_tensor_names = [top_layer.name]
      initial_outputs = self._get_initial_outputs(output_tensor_names)

    # Remove pruning-related nodes.
    with self.final_graph.as_default():
      final_outputs = self._get_final_outputs(output_tensor_names)

    # Check that the final graph has no pruning-related vars
    self._check_removal_of_pruning_vars(number_masked_conv_layers)

    # Check that outputs remain the same after removal of pruning-related nodes
    self._check_output_equivalence(initial_outputs, final_outputs)

  def testFullyConnectedModel(self):
    with self.initial_graph.as_default():
      number_masked_fc_layers = 3
      top_layer = self._build_fully_connected_model(number_masked_fc_layers)
      output_tensor_names = [top_layer.name]
      initial_outputs = self._get_initial_outputs(output_tensor_names)

    # Remove pruning-related nodes.
    with self.final_graph.as_default():
      final_outputs = self._get_final_outputs(output_tensor_names)

    # Check that the final graph has no pruning-related vars
    self._check_removal_of_pruning_vars(number_masked_fc_layers)

    # Check that outputs remain the same after removal of pruning-related nodes
    self._check_output_equivalence(initial_outputs, final_outputs)

  def testLSTMModel(self):
    with self.initial_graph.as_default():
      number_masked_lstm_layers = 2
      outputs = self._build_lstm_model(number_masked_lstm_layers)
      output_tensor_names = [outputs[0][0].name]
      initial_outputs = self._get_initial_outputs(output_tensor_names)

    # Remove pruning-related nodes.
    with self.final_graph.as_default():
      final_outputs = self._get_final_outputs(output_tensor_names)

    # Check that the final graph has no pruning-related vars
    self._check_removal_of_pruning_vars(number_masked_lstm_layers)

    # Check that outputs remain the same after removal of pruning-related nodes
    self._check_output_equivalence(initial_outputs, final_outputs)


if __name__ == "__main__":
  tf.test.main()
