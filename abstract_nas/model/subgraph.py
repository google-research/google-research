# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Utils for working with (sub)graph edit operations.

Includes both inserting / replacing a subgraph in a graph, as well as weight
inheritance.
"""

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from abstract_nas.model import Model
from abstract_nas.model.concrete import Graph
from abstract_nas.model.concrete import new_graph
from abstract_nas.model.concrete import Op
from abstract_nas.utils import canonicalize_tensor_name

Tensor = Any


class SubgraphNode(object):
  """SubgraphNode for Graph substitutions."""

  def __init__(self,
               op,
               output_names = None):
    # The op to be inserted
    self.op = op

    # Either empty (no rewiring).
    # Or, for every output in the op, a name corresponding to an input in the
    # graph into which the node is to be inserted.
    # If output_name does not exist or is None, then nothing happens. This is
    # used to support a situation where an op produces multiple outputs, only
    # a subset of which will be needed.
    output_names = [] if output_names is None else output_names
    self.output_names = [
        canonicalize_tensor_name(n) if n else n for n in output_names
    ]

SubgraphSpec = List[SubgraphNode]


# TODO(charlesjin) refactor for clarity
def replace_subgraph(graph,
                     subgraph):
  """Finds and replaces subgraph in graph.

  A subgraph is a list of ops in topological order.
  A subgraph is specified almost the same way as a graph, with one crucial
  difference: each op has an associated list of ``output_names''
  SubgraphNode(
    op=Op(name="layer1/gelu/1",
       type=Op.OpType.GELU,
       input_names=["layer1/conv"]),
    output_names=["layer1/relu"])
  )
  For each name in the output_names field, we update any ops in the old graph
  which consume the specified tensor with the corresponding output of the
  subgraph's op. In this case, any tensors which specified ``layer1/relu'' as an
  input value will instead receive ``layer1/gelu/1:0'' in that slot. Ops which
  produce multple outputs will specify additional tensors in the
  ``output_values'' (which correspond in order to the output tensors of the op).

  Our first step is to gather the mapping from inputs in the old graph to the
  corresponding new inputs in the subgraph. Any inputs which are not specified
  to be replaced retain their inputs in the old graph.

  We next insert all the subgraph nodes into the graph according to the first
  instance they are insertable (i.e., the first position in the topologically
  flattened graph at which all the required inputs would precede the node).

  Finally, we remove nodes with dangling outputs--those whose outputs are not
  backward reachable from the output tensor. For instance, in the example above,
  "layer1/relu" would be removed since we have updated any node which consumes
  its output to take "layer1/gelu/1" as input instead.

  Does NOT rigorously check to see if the replacement generates a legal graph
  i.e., the subgraph outputs may not have the same shapes as the original ops.
  This should be done by running a ShapeModel.

  Args:
    graph: the old graph
    subgraph: the subgraph to insert

  Returns:
    a config of the new graph
  """
  graph = copy.deepcopy(graph)
  subgraph = copy.deepcopy(list(subgraph))

  # Prepare all the new output rewiring.
  # Collect the outputs in the subgraph which are specified to replace inputs
  # in the original graph.
  node_mapping = {}
  for node in subgraph:
    if node.output_names:
      for idx, output_name in enumerate(node.output_names):
        if output_name is not None:
          node_mapping[output_name] = f"{node.op.name}:{idx}"
  # Update the output_names of the graph
  graph_outputs = []
  for output_name in graph.output_names:
    output_name = node_mapping.get(
        canonicalize_tensor_name(output_name), output_name)
    graph_outputs.append(output_name)
  graph.output_names = graph_outputs

  def is_insertable(new_op, ops, graph_inputs):
    """Returns whether the new_op is insertable in ops.

    The new_op is insertable if ops contains all the inputs consumed by new_op,
    i.e., does inserting new_op now preserve the topological order?

    Args:
      new_op: the op to insert.
      ops: the ops currently in the graph.
      graph_inputs: the inputs to the graph.
    """
    for input_name in new_op.input_names:

      # graph inputs are always available to consume
      if input_name in graph_inputs:
        continue

      # look for input_name in output produced by previous ops
      found = False
      for op in ops:
        for idx in range(op.num_outputs):
          if input_name == f"{op.name}:{idx}":
            found = True
            break
        if found: break
      if not found: return False
    return True

  # Insert subgraph into graph.ops while preserving topological order.
  # By the end, new_ops will contain old_ops + subgraph ops (in order).
  # Also rewire inputs in old_ops according to node_mapping.
  new_ops = []
  old_ops = list(graph.ops)
  graph_inputs = [
      canonicalize_tensor_name(input_name) for input_name in graph.input_names
  ]

  # Check that all outputs are produced
  missing_outputs = list(graph.output_names)

  def add_op(op):
    # Append the op
    new_ops.append(op)

    # Check if it produced a graph output
    if op.num_outputs == 1 and op.name in missing_outputs:
      missing_outputs.remove(op.name)
    for idx in range(op.num_outputs):
      output_name = f"{op.name}:{idx}"
      if output_name in missing_outputs:
        missing_outputs.remove(output_name)

  new_node = None
  while True:
    # Break early if we have produced all the required outputs.
    if not missing_outputs:
      break

    # subgraph is assumed to be topologically sorted, so we can insert the nodes
    # in subgraph into the new graph in that order.
    if new_node is None and subgraph:
      new_node = subgraph.pop(0)

    # Insert new_node into new_ops as soon as it is possible.
    # If we were successful in insertion, then we should try inserting the next
    # subgraph node immediately.
    # Otherwise, we need to start inserting ops from old_op to produce the
    # missing inputs.
    if new_node is not None and is_insertable(new_node.op, new_ops,
                                              graph_inputs):
      add_op(new_node.op)
      new_node = None
      continue

    if not old_ops:
      break
    old_op = old_ops.pop(0)

    # Updating the inputs to old_op
    new_inputs = []
    for inp in old_op.input_names:
      new_inp = node_mapping.get(inp, inp)
      new_inputs.append(new_inp)
    old_op.input_names = new_inputs

    # Insert an old_op into new_ops
    if not is_insertable(old_op, new_ops, graph_inputs):
      raise ValueError(f"Old op {old_op.name} is not insertable... the "
                       "original graph was not topological sorted.")
    add_op(old_op)

  # Check that the new graph produces the required output of the computation.
  if missing_outputs:
    raise ValueError(f"graph outputs {missing_outputs} have no producing node.")

  # Remove all dangling ops, i.e., those which do not affect the output.
  # Essentially traverses the graph in reverse topological order and removes the
  # ops which are not necessary at the point of traversal (i.e., do not produce
  # an input which is consumed but not produced by the graph thus far).
  missing_inputs = []
  reversed_ops = []

  # Checks for unique op names
  # This also guarantees that the tensor names are unique.
  op_names = []
  for op in new_ops[::-1]:
    needed = False

    if op.num_outputs == 1 and op.name in graph.output_names:
      # Node produces graph output
      needed = True
    for idx in range(op.num_outputs):
      output_name = f"{op.name}:{idx}"
      # Node produces graph output, or output is consumed by another op
      if output_name in graph.output_names or output_name in missing_inputs:
        needed = True

    # Op did not produce any needed inputs, so does not belong in the final
    # graph.
    if not needed:
      continue

    # We are going to insert op, so make sure the name is unique and does not
    # contain a colon (which is used to distinguish output tensors).
    if ":" in op.name:
      raise ValueError(f"Op name {op.name} contains ':'.")
    if op.name in op_names:
      raise ValueError(f"Op name {op.name} is not unique.")
    op_names.append(op.name)

    # Add node inputs to missing_inputs.
    for input_name in op.input_names:
      if input_name not in missing_inputs:
        missing_inputs.append(input_name)

    # Remove node outputs from missing_inputs.
    for idx in range(op.num_outputs):
      output_name = f"{op.name}:{idx}"
      if output_name in missing_inputs:
        missing_inputs.remove(output_name)

    # Insert op
    reversed_ops.append(op)

  # Remove graph inputs from missing_inputs
  for graph_inp in graph_inputs:
    if graph_inp in missing_inputs:
      missing_inputs.remove(graph_inp)

  # Make sure all inputs are provided
  if missing_inputs:
    raise ValueError(f"inputs {missing_inputs} have no producing node.")

  graph.ops = list(reversed(reversed_ops))
  return graph


class SubgraphModel():
  """A concrete subgraph.

  A concrete subgraph consists of:
  - The full graph, in which the subgraph is *already* embedded, i.e., you
  should call replace_subgraph BEFORE creating a ConcreteSubgraph!
  - An instantiation of the full graph, as specified by the state (i.e.,
  parameters). If None, the subgraph is treated as abstract.
  - An execution of the full graph, as specified by a set of inputs.
  - A specification of the subgraph, as defined by the list of subgraph nodes.
  If None, the subgraph is just the full graph.
  """

  def __init__(self,
               graph,
               constants,
               state,
               inputs,
               subgraph = None):
    self.graph = graph
    self.constants = constants
    self.state = state
    self.inputs = inputs
    self.subgraph: SubgraphSpec = subgraph if subgraph else []

    self.input_names = None
    self.output_names = None
    self.original_outputs = graph.output_names

    if subgraph:
      self._subgraph_to_names()

      # graph for graph inputs -> subg inputs
      self.subg_inputs_graph = copy.deepcopy(graph)
      self.subg_inputs_graph.output_names = self.input_names
      self.subg_inputs_model = Model(self.subg_inputs_graph, self.constants)
      self.subg_inputs = None

      # graph for graph inputs -> subg outputs
      self.subg_outputs_graph = copy.deepcopy(graph)
      self.subg_outputs_graph.output_names = self.output_names
      self.subg_outputs_model = Model(self.subg_outputs_graph, self.constants)
      self.subg_outputs = None

      # graph for subg inputs -> subg outputs
      subg_ops = [node.op for node in subgraph]
      self.subg_graph = new_graph(self.input_names, self.output_names, subg_ops)
      self.subg_model = Model(self.subg_graph, self.constants)
    else:
      self.input_names = [
          canonicalize_tensor_name(name) for name in graph.input_names
      ]
      self.output_names = [
          canonicalize_tensor_name(name) for name in graph.output_names
      ]

      # subg inputs = inputs to the graph
      self.subg_inputs_graph = None
      self.subg_inputs_model = None
      self.subg_inputs = inputs

      # graph for graph inputs -> subg outputs
      self.subg_outputs_graph = copy.deepcopy(graph)
      self.subg_outputs_model = Model(self.subg_outputs_graph, self.constants)
      self.subg_outputs = None

      # subg outputs = full graph outputs
      self.subg_graph = self.subg_outputs_graph
      self.subg_model = self.subg_outputs_model

  def _subgraph_to_names(self):
    """Populates the incoming and outgoing edges of the subgraph."""
    assert self.subgraph

    input_names = []
    output_names = []
    produced = []
    for node in self.subgraph:
      # check to see which inputs are incoming edges to the subgraph
      for input_name in node.op.input_names:
        if input_name not in produced and input_name not in input_names:
          input_names.append(input_name)

      # keep track of produced tensors (internal edges in the subgraph)
      for idx in range(node.op.num_outputs):
        produced.append(f"{node.op.name}:{idx}")

      # only the rewired outputs become externally visible to the graph
      for idx, output_name in enumerate(node.output_names):
        if output_name is not None:
          output_names.append(f"{node.op.name}:{idx}")

    self.input_names = input_names
    self.output_names = output_names

  def get_subg_inputs(
      self, graph_inputs,
      intermediates = False,
  ):
    """Returns the inputs to the subgraph given inputs to the full graph.

    Args:
      graph_inputs: The dictionary of input values to the full graph.
      intermediates: Whether to return all the inputs.

    Returns:
      The inputs to the subgraph.

    Raises:
      ValueError: If execution is necessary, but state is not provided.
    """

    # if no self.subg_inputs_model, then the subgraph is the full graph, so the
    # input to the subgraph is the same as the input to the full graph
    if not self.subg_inputs_model:
      return graph_inputs

    # execute the subg_inputs_model
    if not self.state:
      raise ValueError("Cannot execute subgraph without state.")
    if intermediates:
      old_output_names = self.subg_inputs_model.graph.output_names
      self.subg_inputs_model.graph.output_names = []
    subg_inputs = self.subg_inputs_model.apply(self.state, graph_inputs)
    if intermediates:
      self.subg_inputs_model.graph.output_names = old_output_names

    return subg_inputs

  def get_default_subg_inputs(self):
    """Returns the default inputs to the subgraph."""
    if self.subg_inputs is not None:
      return self.subg_inputs
    self.subg_inputs = self.get_subg_inputs(self.inputs)
    return self.subg_inputs

  def get_subg_outputs(
      self, graph_inputs
  ):
    """Returns the output from the subgraph given inputs to the full graph.

    Args:
      graph_inputs: The dictionary of input values to the full graph. If None,
        defaults to the stored input values.

    Returns:
      The outputs of the subgraph.

    Raises:
      ValueError: If execution is necessary, but state is not provided.
    """
    # execute the subg_outputs_model
    if not self.state:
      raise ValueError("Cannot execute subgraph without state.")
    return self.subg_outputs_model.apply(self.state, graph_inputs)

  def get_default_subg_outputs(self):
    """Returns the default outputs of the subgraph."""
    if self.subg_outputs is not None:
      return self.subg_outputs
    subg_inputs = self.get_default_subg_inputs()
    self.subg_outputs = self.execute_subg(subg_inputs)
    return self.subg_outputs

  def execute_subg(
      self, inputs
  ):
    """Returns the output from the subgraph given inputs to the subgraph.

    Args:
      inputs: The dictionary of input values to the subgraph.

    Returns:
      The outputs of the subgraph.

    Raises:
      ValueError: If state is not provided.
    """
    if not self.state:
      raise ValueError("Cannot execute subgraph without state.")
    return self.subg_model.apply(self.state, inputs)

  def update_subg_outputs(self, output_names):
    """Updates the outputs of the subgraph.

    Args:
      output_names: The list of new output_names.

    Raises:
      ValueError: If output_names are not produced in the subgraph.
    """

    for output_name in output_names:
      found = False
      for op in self.subg_graph.ops:
        for idx in range(op.num_outputs):
          if output_name == f"{op.name}:{idx}":
            found = True
            break
        if found: break
      if not found:
        raise ValueError(f"Requested output {output_name} not in subgraph.")

    self.output_names = output_names
    self.subg_graph.output_names = output_names
    self.subg_model.graph.output_names = output_names
    self.subg_outputs_graph.output_names = output_names
    self.subg_outputs_model.graph.output_names = output_names


def inherit_params(
    new_params,
    old_params):
  """Implements parameter inheritance.

  new_params inherits from old_params.
  This only does the top level params, matched by name
  e.g., layer0/conv is treated as a param, and not layer0/conv/kernel

  Args:
    new_params: the params doing the inheriting
    old_params: the params to be inherited

  Returns:
    inherited_params: old_params that were inherited
    trainable_params: new_params that were not inherited
  """
  inherited_params = {}
  trainable_params = {}

  for param in new_params.keys():
    if param in old_params:
      inherited_params[param] = old_params[param]
    else:
      trainable_params[param] = new_params[param]

  return inherited_params, trainable_params
