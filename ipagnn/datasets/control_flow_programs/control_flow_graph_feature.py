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

"""Control Flow Graph feature."""

from absl import logging  # pylint: disable=unused-import

import numpy as np
from python_graphs import cyclomatic_complexity as cyclomatic_complexity_lib
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from ipagnn.datasets.control_flow_programs import control_flow_programs_version
from ipagnn.datasets.control_flow_programs.program_generators import constants
from ipagnn.toolkit import shepherds as shepherds_lib


class ControlFlowGraphFeature(tfds.features.Text):
  """`FeatureConnector` for cfg, encoding to integers."""

  def __init__(self, include_back_edges, **kwargs):
    super(ControlFlowGraphFeature, self).__init__(**kwargs)
    self.include_back_edges = include_back_edges

  def get_tensor_info(self):
    """Gets the TensorInfos for the decoded Tensor features."""
    tensor_info = {
        # The index of the start node.
        'start_index': tfds.features.TensorInfo(shape=(), dtype=tf.int32),
        # The index of the exit node.
        'exit_index': tfds.features.TensorInfo(shape=(), dtype=tf.int32),
        # data: For each node, a sequence of integers representing the
        # code at that node.
        'data': tfds.features.TensorInfo(shape=(None, None), dtype=tf.float64),
        # For each node, the string of the code at that node.
        # 'strings': tfds.features.TensorInfo(shape=(None,), dtype=tf.string),
        # For each node, the number of elements in the data for that node.
        'lengths': tfds.features.TensorInfo(shape=(None,), dtype=tf.int64),
        # For each node, the line number of the program corresponding to it.
        'linenos': tfds.features.TensorInfo(shape=(None,), dtype=tf.int64),
        # The number of nodes in the graph.
        'count': tfds.features.TensorInfo(shape=(), dtype=tf.int32),
        # Identical to 'count', but represented with shape (1,) instead of ().
        'num_nodes': tfds.features.TensorInfo(shape=(1,), dtype=tf.int64),
        # The number of steps needed to cover recursively every block twice.
        'steps': tfds.features.TensorInfo(shape=(1,), dtype=tf.int64),
        'shape': tfds.features.TensorInfo(shape=(2,), dtype=tf.int32),
        'true_branch_nodes': tfds.features.TensorInfo(
            shape=(None,), dtype=tf.int64),
        'false_branch_nodes': tfds.features.TensorInfo(
            shape=(None,), dtype=tf.int64),
        'adjacency_matrix': tfds.features.TensorInfo(
            shape=(None, None), dtype=tf.float64),
        'adjacency_matrix_shape': tfds.features.TensorInfo(
            shape=(2,), dtype=tf.int32),
        'post_domination_matrix': tfds.features.TensorInfo(
            shape=(None, None), dtype=tf.float64),
        'post_domination_matrix_shape': tfds.features.TensorInfo(
            shape=(2,), dtype=tf.int32),
        'adjacency_list': tfds.features.TensorInfo(
            shape=(None, 2), dtype=tf.int64),
        'adjacency_list/source_indices': tfds.features.TensorInfo(
            shape=(None,), dtype=tf.int64),
        'adjacency_list/dest_indices': tfds.features.TensorInfo(
            shape=(None,), dtype=tf.int64),
        'adjacency_list/dense_shape': tfds.features.TensorInfo(
            shape=(2,), dtype=tf.int64),
        'adjacency_list/shape': tfds.features.TensorInfo(
            shape=(2,), dtype=tf.int64),
    }
    if control_flow_programs_version.at_least('0.0.44'):
      tensor_info.update({
          # The cyclomatic complexity of the program.
          'cyclomatic_complexity': tfds.features.TensorInfo(shape=(1,),
                                                            dtype=tf.int64),
          # The maximum level of indentation in the program.
          'max_indent': tfds.features.TensorInfo(shape=(1,), dtype=tf.float32),
      })
    if control_flow_programs_version.supports_edge_types():
      tensor_info.update({
          'edge_types': tfds.features.TensorInfo(shape=(None,), dtype=tf.int64),
      })
    return tensor_info

  def get_shepherd_info(self):
    mod_padding = 32
    shepherds_info = [
        shepherds_lib.NodeIndexShepherd('start_index',
                                        node_count_key='count',
                                        dtype=tf.int32),
        shepherds_lib.NodeIndexShepherd('exit_index',
                                        node_count_key='count',
                                        dtype=tf.int32),
        shepherds_lib.NodeIndicesShepherd('true_branch_nodes',
                                          node_count_key='count',
                                          dtype=tf.int64,
                                          shape=[None],
                                          mod_paddings=[mod_padding]),
        shepherds_lib.NodeIndicesShepherd('false_branch_nodes',
                                          node_count_key='count',
                                          dtype=tf.int64,
                                          shape=[None],
                                          mod_paddings=[mod_padding]),
        # shepherds_lib.DenseTensorShepherd('strings', dtype=tf.string),
        shepherds_lib.DenseTensorShepherd('num_nodes', dtype=tf.int64),
        shepherds_lib.DenseTensorShepherd('steps', dtype=tf.int64),
        shepherds_lib.DenseTensorShepherd('data', dtype=tf.float64,
                                          mod_paddings=[mod_padding, 1]),
        shepherds_lib.DenseTensorShepherd('lengths', dtype=tf.int64,
                                          mod_paddings=[mod_padding]),
        shepherds_lib.DenseTensorShepherd('linenos', dtype=tf.int64,
                                          mod_paddings=[mod_padding]),
        shepherds_lib.SparseTensorShepherd('adjacency_list'),
    ]
    if control_flow_programs_version.at_least('0.0.44'):
      shepherds_info.extend([
          shepherds_lib.DenseTensorShepherd('cyclomatic_complexity',
                                            dtype=tf.int64),
          shepherds_lib.DenseTensorShepherd('max_indent', dtype=tf.float32),
      ])
    if control_flow_programs_version.supports_edge_types():
      shepherds_info.extend([
          shepherds_lib.DenseTensorShepherd('edge_types', dtype=tf.int64,
                                            mod_paddings=[1]),
      ])
    return shepherds_info

  def get_serialized_info(self):
    """Return the tf-example features, as encoded for storage on disk."""
    info = self.get_tensor_info()
    # TODO(dbieber): See if you can switch to something general purpose like:
    # return {
    #     key: tfds.features.TensorInfo(shape=(None,), dtype=value.dtype)
    #     for key, value in info.items()
    # }
    info['start_index'] = tfds.features.TensorInfo(shape=(1,), dtype=tf.int32)
    info['exit_index'] = tfds.features.TensorInfo(shape=(1,), dtype=tf.int32)
    info['count'] = tfds.features.TensorInfo(shape=(1,), dtype=tf.int32)
    info['num_nodes'] = tfds.features.TensorInfo(shape=(1,), dtype=tf.int64)
    info['steps'] = tfds.features.TensorInfo(shape=(1,), dtype=tf.int64)
    if control_flow_programs_version.at_least('0.0.44'):
      info['max_indent'] = tfds.features.TensorInfo(
          shape=(1,), dtype=tf.float32)
      info['cyclomatic_complexity'] = tfds.features.TensorInfo(
          shape=(1,), dtype=tf.int64)
    if control_flow_programs_version.supports_edge_types():
      info['edge_types'] = tfds.features.TensorInfo(shape=(None,),
                                                    dtype=tf.int64)
    info['data'] = tfds.features.TensorInfo(shape=(None,), dtype=tf.float64)
    info['adjacency_matrix'] = tfds.features.TensorInfo(
        shape=(None,), dtype=tf.float64)
    info['post_domination_matrix'] = tfds.features.TensorInfo(
        shape=(None,), dtype=tf.float64)
    info['adjacency_list'] = tfds.features.TensorInfo(
        shape=(None,), dtype=tf.int64)
    return info

  def encode_example(self, cfg_and_python_source):
    cfg, python_source = cfg_and_python_source
    nodes = cfg.nodes
    lines = python_source.strip().split('\n')

    cyclomatic_complexity = cyclomatic_complexity_lib.cyclomatic_complexity2(
        cfg)

    # steps = 1  # Start with one step for reaching exit.
    # for line in lines:
    #   indent = (len(line) - len(line.lstrip())) / constants.INDENT_SPACES
    #   steps += 2 ** indent

    # if version < '0.0.38'
    # steps = 1  # Start with one step for reaching exit.
    # indents = []
    # for line in lines:
    #   indent = (len(line) - len(line.lstrip())) / constants.INDENT_SPACES
    #   while indents and indent <= indents[-1]:
    #     indents.pop()
    #   steps += 2 ** len(indents)
    #   if 'while' in line:
    #     indents.append(indent)

    max_indent = 0
    steps = 1  # Start with one step for reaching exit.
    indents = []
    for line in lines:
      indent = (len(line) - len(line.lstrip())) / constants.INDENT_SPACES
      max_indent = max(max_indent, indent)
      while indents and indent <= indents[-1]:
        indents.pop()
      steps += 2 ** len(indents)
      if 'while' in line:
        indents.append(indent)
        # We add steps at both levels of indentation for whiles.
        # Before for the initial condition check, after for subsequent condition
        # checks.
        steps += 2 ** len(indents)

    # cfg.nodes does not include an exit node, so we add 1.
    num_nodes = len(nodes) + 1
    exit_index = len(nodes)

    # Note that some of the nodes may have empty source.
    node_sources = [as_source(node, lines) for node in nodes]
    linenos = [node.instruction.node.lineno for node in nodes]
    # line_sources = python_source.strip().split('\n')

    if self.encoder:
      node_encodings = [
          self.encoder.encode(source)
          for source in node_sources
      ]
      # line_encodings = [
      #     self.encoder.encode(source)
      #     for source in line_sources
      # ]
    else:
      node_encodings = node_sources
      # line_encodings = line_sources
    node_encodings.append([])  # Finally add a blank line for the exit node.
    # line_encodings.append([])  # Finally add a blank line for the exit.
    linenos.append(len(lines) + 1)  # Add a lineno for the exit.

    # Pad encodings to all be the same length.
    padded_encodings = []
    encoding_lengths = []
    for encoding in node_encodings:
      encoding_lengths.append(len(encoding))
    max_len = max(encoding_lengths)
    for encoding in node_encodings:
      padded_encodings.append(
          np.pad(encoding, (0, max_len - len(encoding)), mode='constant'))
    padded_encodings = np.concatenate(padded_encodings)

    adjacency_matrix = get_adjacency_matrix(nodes, exit_index,
                                            self.include_back_edges)
    post_domination_matrix = get_post_domination_matrix(cfg)
    adjacency_list = get_adjacency_list(nodes, exit_index,
                                        self.include_back_edges)
    adjacency_list = np.array(adjacency_list, ndmin=2)
    adjacency_list.shape = (-1, 2)

    branch_list = np.array(get_branch_list(nodes, exit_index))
    true_branch_nodes = branch_list[:, 0]
    false_branch_nodes = branch_list[:, 1]

    encoded_example = {
        'start_index': [0],
        'exit_index': [exit_index],
        'data': padded_encodings,
        # 'strings': node_sources,
        'lengths': encoding_lengths,
        'linenos': linenos,
        'steps': [steps],
        'count': [num_nodes],
        'num_nodes': [num_nodes],
        'shape': [num_nodes, max_len],
        'true_branch_nodes': true_branch_nodes,
        'false_branch_nodes': false_branch_nodes,
        'adjacency_matrix': np.reshape(adjacency_matrix, (-1,)),
        'adjacency_matrix_shape': adjacency_matrix.shape,
        'post_domination_matrix': np.reshape(post_domination_matrix, (-1,)),
        'post_domination_matrix_shape': post_domination_matrix.shape,
        'adjacency_list': np.reshape(adjacency_list, (-1,)),
        'adjacency_list/source_indices': np.array(adjacency_list)[:, 1],
        'adjacency_list/dest_indices': np.array(adjacency_list)[:, 0],
        'adjacency_list/dense_shape': adjacency_matrix.shape,
        'adjacency_list/shape': [len(adjacency_list), 2],
    }
    if control_flow_programs_version.at_least('0.0.44'):
      encoded_example.update({
          'cyclomatic_complexity': [cyclomatic_complexity],
          'max_indent': [max_indent],
      })
    if control_flow_programs_version.supports_edge_types():
      encoded_example.update({
          'edge_types': get_edge_types(
              nodes, exit_index, self.include_back_edges)
      })
    return encoded_example

  def decode_example(self, tfexample_data):
    # Each node corresponds to a single statement and gets its own row in data.
    tfexample_data['start_index'] = tf.reshape(
        tfexample_data['start_index'], ())
    tfexample_data['exit_index'] = tf.reshape(tfexample_data['exit_index'], ())
    tfexample_data['count'] = tf.reshape(tfexample_data['count'], ())

    tfexample_data['data'] = tf.reshape(
        tfexample_data['data'], tfexample_data['shape'])
    tfexample_data['adjacency_matrix'] = tf.reshape(
        tfexample_data['adjacency_matrix'],
        tfexample_data['adjacency_matrix_shape'])
    tfexample_data['post_domination_matrix'] = tf.reshape(
        tfexample_data['post_domination_matrix'],
        tfexample_data['post_domination_matrix_shape'])
    tfexample_data['adjacency_list'] = tf.reshape(
        tfexample_data['adjacency_list'],
        tfexample_data['adjacency_list/shape'])
    return tfexample_data


def get_adjacency_matrix(nodes, exit_index, include_back_edges=False):
  """Computes the adjacency matrix for the control flow graph.

  Args:
    nodes: A list of control_flow.ControlFlowNodes.
    exit_index: The index of the exit node.
    include_back_edges: Whether to include back edges in the matrix.
  Returns:
    A numpy array representing the adjacency matrix for the control flow graph.
    adj[dest][src] == 1 indicates an edge from node index src to dest.
  """
  num_nodes = len(nodes) + 1  # Add one for the exit node.
  adj = np.zeros((num_nodes, num_nodes))
  def add_edge(dest, src):
    adj[dest, src] = 1
    if include_back_edges and src != dest:
      adj[src, dest] = 1

  branch_list = get_branch_list(nodes, exit_index)
  for index, branches in enumerate(branch_list):
    add_edge(branches[0], index)
    if branches[1] != branches[0]:
      add_edge(branches[1], index)
  return adj


def get_post_domination_matrix(cfg):
  """Computes the post-domination matrix for nodes in the control flow graph.

  A node i is post-dominated by another node j if every path from i to the exit
  includes j.

  Args:
    cfg: The control flow graph to compute the post domination matrix for.
  Returns:
    The 0/1 post-domination matrix. output[i, j] is 1 if i is post-dominated by
    j, 0 otherwise.
  """
  post_dominator_sets = get_post_dominator_sets(cfg)
  num_nodes = len(cfg.nodes) + 1  # Add one for the exit node.
  mat = np.zeros((num_nodes, num_nodes))

  # Create mapping from cfg_node to index in matrix.
  node_indexes = {
      cfg_node: i for i, cfg_node in enumerate(cfg.nodes)
  }
  node_indexes['<exit>'] = len(cfg.nodes)

  for node, post_dominators in post_dominator_sets.items():
    for post_dominator in post_dominators:
      mat[node_indexes[node], node_indexes[post_dominator]] = 1
  return mat


def get_post_dominator_sets(cfg):
  """Computes the set of post-dominating nodes for each node in the graph.

  A node i is post-dominated by another node j if every path from i to the exit
  includes j.

  Args:
    cfg: The control flow graph to compute the post domination matrix for.
  Returns:
    A dict with the post-domination sets for all control flow nodes. output[i]
    is the set of all control flow nodes j such that j post-dominates i. The
    exit node is represented by '<exit>'.
  """
  dominator_sets = {cfg_node: set(cfg.nodes) | {'<exit>'}
                    for cfg_node in cfg.nodes}
  dominator_sets['<exit>'] = {'<exit>'}

  def succ(cfg_node):
    """Returns the set of successors for a given control flow graph node."""
    cfg_node_is_end_of_block = cfg_node == cfg_node.block.control_flow_nodes[-1]
    if (cfg_node_is_end_of_block and
        any(block.label == '<exit>' for block in cfg_node.block.next)):
      # The node exits to the program exit node.
      return cfg_node.next | {'<exit>'}
    return cfg_node.next

  # Iterate the fixed-point equation until convergence to find post-dominators:
  # D(parent) = Union({parent}, Intersect(D(children)))
  # A node n is post-dominated by itself (trivially), and by any node which post
  # dominates all of n's immediate successors.
  change = True
  while change:
    change = False
    for cfg_node in reversed(cfg.nodes):
      old_value = dominator_sets[cfg_node].copy()
      for p in succ(cfg_node):
        dominator_sets[cfg_node] &= dominator_sets[p]
      dominator_sets[cfg_node] |= {cfg_node}
      if old_value != dominator_sets[cfg_node]:
        change = True
  return dominator_sets


def get_adjacency_list(nodes, exit_index, include_back_edges=False):
  """Computes the adjacency list for the control flow graph.

  Args:
    nodes: A list of control_flow.ControlFlowNodes.
    exit_index: The index of the exit node.
    include_back_edges: Whether to include back edges in the list.
  Returns:
    A Python list representing the adjacency list for the control flow graph.
    Each entry (dest, src) represents an edge.
  """
  edges = []
  def add_edge(dest, src):
    edges.append([dest, src])
    if include_back_edges and src != dest:
      edges.append([src, dest])

  branch_list = get_branch_list(nodes, exit_index)
  for index, branches in enumerate(branch_list):
    add_edge(branches[0], index)
    if branches[1] != branches[0]:
      add_edge(branches[1], index)
  return edges


def get_edge_types(nodes, exit_index, include_back_edges=False):
  """Computes the adjacency list for the control flow graph.

  Args:
    nodes: A list of control_flow.ControlFlowNodes.
    exit_index: The index of the exit node.
    include_back_edges: Whether to include back edges in the list.
  Returns:
    A list of edge types (ints) for each edge in the adjacency list of the
    control flow graph. Edge types are:
    0: No branch decision, Parent
    1: No branch decision, Child
    2: True branch, Parent
    3: True branch, Child
    4: False branch, Parent
    5: False branch, Child
  """
  edge_types = []
  def add_edge(dest, src, branch_kind):
    parent_edge_type = 2 * branch_kind  # Parent
    child_edge_type = 2 * branch_kind + 1  # Child
    edge_types.append(child_edge_type)
    if include_back_edges and src != dest:
      edge_types.append(parent_edge_type)

  branch_list = get_branch_list(nodes, exit_index)
  for index, branches in enumerate(branch_list):
    if branches[1] == branches[0]:
      add_edge(branches[0], index, 0)  # No branch decision
    else:
      add_edge(branches[0], index, 1)  # True
      add_edge(branches[1], index, 2)  # False
  return edge_types


def get_branch_list(nodes, exit_index):
  """Computes the branch list for the control flow graph.

  Args:
    nodes: A list of control_flow.ControlFlowNodes.
    exit_index: The index of the exit node.
  Returns:
    A Python list representing the branch options available from each node. Each
    entry in the list corresponds to a node in the control flow graph, with the
    final entry corresponding to the exit node (not present in the cfg). Each
    entry is a 2-tuple indicating the next node reached by the True and False
    branch respectively (these may be the same.) The exit node leads to itself
    along both branches.
  """
  indexes_by_id = {
      id(node): index for index, node in enumerate(nodes)
  }
  indexes_by_id[id(None)] = exit_index
  branches = []
  for node in nodes:
    node_branches = node.branches
    if node_branches:
      branches.append([indexes_by_id[id(node_branches[True])],
                       indexes_by_id[id(node_branches[False])]])
    else:
      try:
        next_node = next(iter(node.next))
        next_index = indexes_by_id[id(next_node)]
      except StopIteration:
        next_index = exit_index
      branches.append([next_index, next_index])

  # Finally we add branches from the exit node to itself.
  # Omit this if running on BasicBlocks rather than ControlFlowNodes, because
  # ControlFlowGraphs have an exit BasicBlock, but no exit ControlFlowNodes.
  branches.append([exit_index, exit_index])
  return branches


def as_source(node, lines):
  ast_node = node.instruction.node
  if ast_node:
    return lines[ast_node.lineno - 1]
  else:
    return '<exit>'
