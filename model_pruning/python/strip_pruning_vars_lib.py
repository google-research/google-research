# coding=utf-8
# Copyright 2024 The Google Research Authors.
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
"""Utilities to remove pruning-related ops and variables from a GraphDef.
"""

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf


def _node_name(tensor_name):
  """Remove the trailing ':0' from the variable name."""
  if ':' not in tensor_name:
    return tensor_name

  return tensor_name.split(':')[0]


def _tensor_name(node_name):
  """Appends the :0 in the op name to get the canonical tensor name."""
  if ':' in node_name:
    return node_name

  return node_name + ':0'


def _get_masked_weights(input_graph_def):
  """Extracts masked_weights from the graph as a dict of {var_name:ndarray}."""
  input_graph = tf.Graph()
  with input_graph.as_default():
    tf.import_graph_def(input_graph_def, name='')

    with tf.Session(graph=input_graph) as sess:
      masked_weights_dict = {}
      for node in input_graph_def.node:
        if 'masked_weight' in node.name:
          masked_weight_val = sess.run(
              sess.graph.get_tensor_by_name(_tensor_name(node.name)))
          tf.logging.info(
              '%s has %d values, %1.2f%% zeros \n', node.name,
              np.size(masked_weight_val),
              100 - float(100 * np.count_nonzero(masked_weight_val)) /
              np.size(masked_weight_val))
          masked_weights_dict.update({node.name: masked_weight_val})
  return masked_weights_dict


def strip_pruning_vars_fn(input_graph_def, output_node_names):
  """Removes mask variable from the graph.

  Replaces the masked_weight tensor with element-wise multiplication of mask
  and the corresponding weight variable.

  Args:
    input_graph_def: A GraphDef in which the variables have been converted to
      constants. This is typically the output of
      tf.graph_util.convert_variables_to_constant()
    output_node_names: List of name strings for the result nodes of the graph

  Returns:
    A GraphDef in which pruning-related variables have been removed
  """
  masked_weights_dict = _get_masked_weights(input_graph_def)
  pruned_graph_def = tf.GraphDef()

  # Replace masked_weight with a const op containing the
  # result of tf.multiply(mask,weight)
  for node in input_graph_def.node:
    output_node = tf.NodeDef()
    if 'masked_weight' in node.name:
      output_node.op = 'Const'
      output_node.name = node.name
      dtype = node.attr['T']
      data = masked_weights_dict[node.name]
      output_node.attr['dtype'].CopyFrom(dtype)
      output_node.attr['value'].CopyFrom(
          tf.AttrValue(tensor=tf.make_tensor_proto(data)))

    else:
      output_node.CopyFrom(node)
    pruned_graph_def.node.extend([output_node])

  # Remove stranded nodes: mask and weights
  return tf.graph_util.extract_sub_graph(pruned_graph_def, output_node_names)


def graph_def_from_checkpoint(checkpoint_dir, output_node_names):
  """Converts checkpoint data to GraphDef.

  Reads the latest checkpoint data and produces a GraphDef in which the
  variables have been converted to constants.

  Args:
    checkpoint_dir: Path to the checkpoints.
    output_node_names: List of name strings for the result nodes of the graph.

  Returns:
    A GraphDef from the latest checkpoint

  Raises:
    ValueError: if no checkpoint is found
  """
  checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
  if checkpoint_path is None:
    raise ValueError('Could not find a checkpoint at: {0}.'
                     .format(checkpoint_dir))

  saver_for_restore = tf.train.import_meta_graph(
      checkpoint_path + '.meta', clear_devices=True)
  with tf.Session() as sess:
    saver_for_restore.restore(sess, checkpoint_path)
    graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph_def, output_node_names)

  return output_graph_def
