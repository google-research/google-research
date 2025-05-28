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

"""Class for progressive synthesizer."""

from __future__ import annotations

import copy
import functools
import random
import sys
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

from absl import logging

from abstract_nas.abstract.base import AbstractProperty
from abstract_nas.abstract.shape import ShapeProperty
from abstract_nas.model.concrete import new_graph
from abstract_nas.model.concrete import new_op
from abstract_nas.model.concrete import Op
from abstract_nas.model.concrete import OpType
from abstract_nas.model.subgraph import replace_subgraph
from abstract_nas.model.subgraph import SubgraphModel
from abstract_nas.model.subgraph import SubgraphNode
from abstract_nas.model.subgraph import SubgraphSpec
from abstract_nas.synthesis.base import AbstractSynthesizer as AS
from abstract_nas.synthesis.sequential import AbstractSequentialSynthesizer as SS

RESHAPE_OPS = frozenset([
    OpType.RESHAPE,
    OpType.FLATTEN,
    OpType.TRANSPOSE,
    # OpType.DENSE_GENERAL,  # not supported directly (reshape + dense)
    OpType.EINSUM,
])

BINARY_OPS = frozenset([
    OpType.EINSUM,
    OpType.ADD,
    OpType.MUL,
])

SynthesisSpec = Sequence[Tuple[SubgraphModel, Sequence[AbstractProperty]]]


def log_exc():
  exc_type, exc_value, exc_traceback = sys.exc_info()
  logging.info("%s", "".join(
      traceback.format_exception(exc_type, exc_value, exc_traceback)))


class SynthesisNode:
  """A node in the synthesis graph.

  The edges between SynthesisNodes are sequential graphs.
  """

  def __init__(
      self,
      generation,
      is_input,
      is_output,
      sequential_syn_ctr,
      op = None,
      parents = None,
      output_name = None,
      p = 0.0,
  ):
    self.generation = generation
    self.is_input = is_input
    self.is_output = is_output

    # A constructor for a sequential synthesizer.
    self.sequential_syn_ctr = sequential_syn_ctr
    self.p = p

    # The op and subgraph_node at this particular junction.
    self.op = op
    self.subgraph_node: SubgraphNode = None

    # The parent synthesis nodes, one for each op input.
    # The first element is the index of the parent output which is used as
    # input, and the second element is the actual SynthesisNode.
    self.parents: List[Tuple[int, SynthesisNode]] = parents if parents else []

    # The name of the output tensors of this op / subgraph_node
    self.output_names = None

    self.subgraph_models = []
    self.original_properties = []
    self.mutated_properties = []

    if is_input or is_output:
      assert output_name
      self.output_names = [output_name]
    else:
      assert op
      self.output_names = [f"{op.name}:{idx}" for idx in range(op.num_outputs)]

    if is_input:
      assert not parents
      assert output_name is not None
    else:
      assert parents
      if not is_output:
        assert len(parents) == len(op.input_names)  # pytype: disable=attribute-error

  def is_ready(self):
    """Returns whether the node is ready for synthesis.

    The node is ready when are all its parents are completed. A parent is
    complete when its subgraph_node has been synthesized.
    """
    if not self.parents:
      return True
    for _, parent in self.parents:
      if not parent.subgraph_node and not parent.is_input:
        return False
    return True

  def synthesize(
      self,
      subgraph_spec,
      subgraphs_and_props_per_parent,
  ):
    """Synthesizes the graph given the parents.

    Args:
      subgraph_spec: The SubgraphSpec thus far.
      subgraphs_and_props_per_parent: A sequence of SynthesisSpecs, one per
        parent.

    Returns:
      A sequence of topologically sorted new nodes which should be inserted into
      the synthesis graph (not including this one).

    Raises:
      ValueError: if parents are not ready.
      ValueError: if the graph does not fulfil the mutated properties.
    """
    if not self.is_ready():
      raise ValueError("Node is not ready for synthesis.")
    if self.is_input:
      return subgraph_spec

    original_output_names = list(self.output_names)

    # Create own op.
    if self.op:
      op = copy.deepcopy(self.op)
      op.name = f"gen{self.generation}/{op.name}"
      self.op = op

    # Synthesize subgraph from each parent to self
    parents_to_synthesize = list(self.parents)
    while parents_to_synthesize:
      parent = random.choice(parents_to_synthesize)
      if parent not in self.parents: continue
      parent_idx = self.parents.index(parent)
      subgraphs_and_props = subgraphs_and_props_per_parent[parent_idx]
      if subgraphs_and_props:
        subgraph_spec = self.synthesize_parent(
            parent_idx, subgraph_spec, subgraphs_and_props)
      parents_to_synthesize.remove(parent)

    # Update own op / SubgraphNode.
    if self.op:
      assert self.op.num_outputs == len(original_output_names)
      self.op.name += f"/{len(subgraph_spec)}"
      subg_node = SubgraphNode(self.op, output_names=original_output_names)
      subgraph_spec.append(subg_node)
      self.subgraph_node = subg_node

    return subgraph_spec

  def synthesize_parent(self, parent_idx, subgraph_spec,
                        subgraphs_and_props):
    """Synthesizes a subgraph from a single parent to self."""
    prefix = f"gen{self.generation}/"

    assert len(self.parents) <= 2  # Only support up to binary ops.
    assert self.op is not None
    assert self.is_output or self.op.num_outputs == 1

    parent_output_idx, parent_node = self.parents[parent_idx]
    input_name = parent_node.output_names[parent_output_idx]

    if self.is_output:
      output_name = self.output_names[0]
    else:
      output_name = self.op.input_names[parent_idx]

    synthesizer = self.sequential_syn_ctr(subgraphs_and_props)

    # Check if this edge is necessary.
    if (self.op and self.op.type in BINARY_OPS and
        self.op.type not in RESHAPE_OPS):

      if random.random() < self.p:
        op_name = f"{prefix}placeholder/{len(subgraph_spec)}"
        placeholder_op = new_op(
            op_name=op_name, op_type=OpType.NONE, input_names=[input_name])
        placeholder_node = SubgraphNode(
            placeholder_op, output_names=[output_name])
        subgraph_models = synthesizer.make_subgraph_models([placeholder_node])

        if synthesizer.verify(subgraph_models):
          # Remove this branch.
          other_input_name = self.op.input_names[1 - parent_idx]
          op_name = f"{prefix}id"
          self.op = new_op(op_name=op_name,
                           op_type=OpType.IDENTITY,
                           input_names=[other_input_name])
          self.output_names = [f"{op_name}:0"]

          del self.parents[parent_idx]
          return subgraph_spec

    new_spec = synthesizer.synthesize()[0].subgraph

    # Ensure all names are unique before adding to current spec
    for idx, node in enumerate(new_spec):
      if idx > 0:
        node.op.input_names = [f"{new_spec[idx - 1].op.name}:0"]
      node.op.name += f"/{len(subgraph_spec)}"
      subgraph_spec.append(node)

    if not self.is_output:
      subgraph_spec[-1].output_names = [None]
      input_names = list(self.op.input_names)
      input_names[parent_idx] = f"{subgraph_spec[-1].op.name}:0"
      self.op.input_names = input_names
    else:
      subgraph_spec[-1].output_names = [output_name]

    return subgraph_spec


class GraphSynthesizer(AS):
  """Synthesizer for arbitrary graphs."""

  def __init__(
      self,
      subgraphs_and_props,
      generation,
      sequential_ctr,
      abstract = True,
      p = 0.0):
    """Initializes a synthesizer.

    Args:
      subgraphs_and_props: A list of tuples providing subgraphs and the
        enclosing contexts into which the synthesized subgraph will be embedded,
        and the properties for each instantiation to satisfy. Note that each
        subgraph will have the same graph, but differing constants, state, and
        inputs.
      generation: The generation of the new subgraph, the value of which must be
        incremented at least once every time a graph is selected to have a
        subgraph in it replaced. This is primarily to ensure that the
        synthesizer is easily able to generate unique names for any newly
        inserted nodes.
      sequential_ctr: A constructor for a sequential synthesizer.
      abstract: Whether the properties of the synthesized subgraph need to
        satisfy the properties abstractly or concretely (i.e., using actual
        values).
      p: the probability of dropping an unneeded branch.
    """
    super().__init__(subgraphs_and_props, generation, abstract)
    self.sequential_ctr = sequential_ctr
    self.p = p

  def synthesize(self):
    """Synthesizes a subgraph satisfying all the properties."""

    # Extract the structure of the original graph.
    # We make a copy since we will be mutating the original graphs later via
    # replace_subgraphs.
    graph = self.subgraphs_and_props[0][0].subg_graph
    subg_spec = self.subgraphs_and_props[0][0].subgraph
    graph = new_graph(input_names=graph.input_names,
                      output_names=graph.output_names,
                      ops=[node.op for node in subg_spec])

    # parents is a map from a tensor name to the synthesis node that will
    # produce the sequential subgraph subsuming said output name; the index
    # corresponds to the index of the synthesis_node.op output that forms the
    # root of this sequential graph.
    # { input_name : ( output_idx, synthesis_node ) }
    parents = {}

    # Add input nodes.
    for input_name in graph.input_names:
      node = SynthesisNode(
          generation=self.generation,
          is_input=True,
          is_output=False,
          sequential_syn_ctr=self.sequential_ctr,
          output_name=input_name,
          p=self.p)
      parents[input_name] = (0, node)

    output_name_to_op: Dict[str, Op] = {}
    new_to_old_names: Dict[str, str] = {}
    old_to_new_names: Dict[str, str] = {}
    nodes_to_synthesize: List[SynthesisNode] = []

    def resolve_name(name, replaced_names):
      while name in replaced_names:
        name = replaced_names[name]
      return name

    def get_subg_spec(input_name, output_name):
      """Returns the sequential subgraph from input_name to output_name."""
      output_name = resolve_name(output_name, new_to_old_names)
      input_name = resolve_name(input_name, new_to_old_names)
      if input_name == output_name: return []

      # Construct subgraph in reverse.
      subg = []
      while output_name in output_name_to_op and output_name != input_name:
        output_op = copy.deepcopy(output_name_to_op[output_name])
        assert len(output_op.input_names) == 1
        output_name = output_op.input_names[0]
        output_op.input_names = [resolve_name(output_name, old_to_new_names)]
        subg.append(output_op)
        output_name = resolve_name(output_name, new_to_old_names)
      assert input_name == output_name
      assert subg
      subg.reverse()
      subg_spec = [SubgraphNode(op) for op in subg]
      subg_spec[-1].output_names = [f"{subg_spec[-1].op.name}:0"]
      return subg_spec

    # Add reshape / binary ops.
    for op in graph.ops:
      parent_nodes = [parents[input_name] for input_name in op.input_names]
      is_binary = op.type in BINARY_OPS
      is_reshape = op.type in RESHAPE_OPS
      if not is_binary:
        assert len(op.input_names) == 1

      if not is_binary and not is_reshape:
        # Since the op is not binary, it should only have one input.
        assert len(parent_nodes) == 1
        parent_idx, syn_node = parent_nodes[0]
        for idx in range(op.num_outputs):
          output_name_to_op[f"{op.name}:{idx}"] = op
          parents[f"{op.name}:{idx}"] = (parent_idx, syn_node)
      else:
        syn_node = SynthesisNode(
            generation=self.generation,
            is_input=False,
            is_output=False,
            sequential_syn_ctr=self.sequential_ctr,
            op=op,
            parents=parent_nodes,
            p=self.p)
        nodes_to_synthesize.append(syn_node)
        for idx in range(op.num_outputs):
          parents[f"{op.name}:{idx}"] = (idx, syn_node)

    # Add output nodes.
    for output_name in self.subgraphs_and_props[0][0].output_names:
      parent_node = parents[output_name]
      syn_node = SynthesisNode(
          generation=self.generation,
          is_input=False,
          is_output=True,
          sequential_syn_ctr=self.sequential_ctr,
          output_name=output_name,
          parents=[parent_node],
          p=self.p)
      nodes_to_synthesize.append(syn_node)

    # Main synthesis loop.
    subgraph_spec = []
    original_subgraphs = [m.graph for m, _ in self.subgraphs_and_props]
    while nodes_to_synthesize:
      new_subgraphs = [
          replace_subgraph(original_subgraph, subgraph_spec)
          for original_subgraph in original_subgraphs
      ]

      # Get the nodes which are ready to synthesize.
      ready = [node for node in nodes_to_synthesize if node.is_ready()]
      assert ready

      # Synthesize a random node.
      node = random.choice(ready)

      new_subgraphs_and_props_by_parent = []
      for parent_idx, parent in enumerate(node.parents):
        parent_output_idx, parent_node = parent
        start_name = parent_node.output_names[parent_output_idx]
        if node.is_output:
          assert len(node.parents) == 1
          assert len(node.output_names) == 1
          subg_spec = get_subg_spec(start_name, node.output_names[0])
        else:
          subg_spec = get_subg_spec(start_name, node.op.input_names[parent_idx])

        new_subgraphs_and_props = []
        if subg_spec:
          subgraph_models = self.make_subgraph_models(subg_spec, new_subgraphs)
          for subgraph_model, (_, props) in zip(subgraph_models,
                                                self.subgraphs_and_props):
            new_props = []
            for prop in props:
              new_prop = copy.deepcopy(prop)
              # Need to set input_values to None, otherwise will be using the
              # inputs to the original subgraph (whereas this is a subgraph of
              # the original subgraph).
              new_prop.input_values = None
              new_prop = new_prop.infer(subgraph_model)

              # As there are properties (e.g., LinearProperty) that subclass
              # ShapeProperty which *can* be safely mutated, we need to check
              # for the type explicitly.
              if type(new_prop) is not ShapeProperty or node.is_output:  # pylint: disable=unidiomatic-typecheck
                # TODO(charlesjin): safely mutate shape for non-outputs?
                new_prop = new_prop.mutate()
              new_props.append(new_prop)
            new_subgraphs_and_props.append((subgraph_model, new_props))
        new_subgraphs_and_props_by_parent.append(new_subgraphs_and_props)

      subgraph_spec = node.synthesize(subgraph_spec,
                                      new_subgraphs_and_props_by_parent)
      nodes_to_synthesize.remove(node)

      for subg_node in subgraph_spec:
        is_binary = subg_node.op.type in BINARY_OPS
        is_reshape = subg_node.op.type in RESHAPE_OPS
        if subg_node.output_names:
          for idx, output_name in enumerate(subg_node.output_names):
            if output_name:
              new_to_old_names[f"{subg_node.op.name}:{idx}"] = output_name
              old_to_new_names[output_name] = f"{subg_node.op.name}:{idx}"
        if is_binary or is_reshape:
          continue
        if subg_node.output_names:
          for output_name in subg_node.output_names:
            if output_name:
              output_name_to_op[output_name] = subg_node.op
        for idx in range(subg_node.op.num_outputs):
          output_name_to_op[f"{subg_node.op.name}:{idx}"] = subg_node.op

    subg_models = self.make_subgraph_models(subgraph_spec)
    return subg_models
