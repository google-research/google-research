# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Builds datasets for edge-supervised Python AST tasks.

This module is designed for tasks where we are given a dataset of ASTs, and a
set of ordered pairs of AST nodes representing the target directed edges. For
instance, the target edges may be syntactic traversals (where does this return
statement return from?) or dataflow analysis (where does this variable get
written to?).

Given a task of that form, this module handles encoding that graph into NDArrays
so that we can train an automaton layer to produce those directed edges as its
marginal distribution.
"""

from typing import List, Tuple, TypeVar, Optional

import dataclasses
import numpy as np

from gfsa import automaton_builder
from gfsa import graph_types
from gfsa import jax_util
from gfsa import sparse_operator

T = TypeVar("T")


@jax_util.register_dataclass_pytree
@dataclasses.dataclass
class GraphBundle:
  """A (possibly padded or batched) combination of automaton and regular graphs.

  May or may not be padded and/or batched. When padded, we require that
  everything is padded with zeros (for dense things, zero values; for sparse
  things, both zero indices and zero values).

  Attributes:
    automaton_graph: The graph as encoded for an automaton.
    graph_metadata: Metadata about the true size of this graph.
    node_types: <int[num_nodes]> specifying the node type of each node.
    edges: Sparse operator mapping from an array of per-edge-type values to a
      [num_nodes, num_nodes] adjacency matrix.
  """
  automaton_graph: automaton_builder.EncodedGraph
  graph_metadata: automaton_builder.EncodedGraphMetadata
  node_types: jax_util.NDArray
  edges: sparse_operator.SparseCoordOperator


def convert_graph_with_edges(
    graph,
    edges,
    builder,
):
  """Convert a graph with edges into an GraphBundle.

  The order of nodes in the returned example is guaranteed to match the
  order of the keys in `graph`.

  Args:
    graph: Graph to encode.
    edges: List of (source, dest, edge_type) pairs to add to the non-automaton
      graph representation (i.e. GNN edges or targets).
    builder: Builder to use to convert the graph.

  Returns:
    Encoded example.
  """
  # Encode the graph.
  encoded_graph, encoded_metadata = builder.encode_graph(graph, as_jax=False)

  # Look up node types.
  node_types = []
  for node_info in graph.values():
    node_types.append(builder.node_type_to_index[node_info.node_type])
  node_types = np.array(node_types, dtype=np.int32)

  # Build the indices for the edges.
  if edges:
    src_dest_pairs = []
    edge_types = []
    id_to_index_map = {node_id: i for i, node_id in enumerate(graph)}
    for src_id, dest_id, edge_type in edges:
      src_idx = id_to_index_map[src_id]
      dest_idx = id_to_index_map[dest_id]
      src_dest_pairs.append((src_idx, dest_idx))
      edge_types.append(edge_type)

    edge_operator = sparse_operator.SparseCoordOperator(
        input_indices=np.array(edge_types, dtype=np.int32).reshape([-1, 1]),
        output_indices=np.array(src_dest_pairs, dtype=np.int32),
        values=np.ones([len(edges)], dtype=np.int32),
    )
  else:
    # Handle case where there are no edges.
    edge_operator = sparse_operator.SparseCoordOperator(
        input_indices=np.empty([0, 1], dtype=np.int32),
        output_indices=np.empty([0, 2], dtype=np.int32),
        values=np.empty([0], dtype=np.int32),
    )

  return GraphBundle(
      automaton_graph=encoded_graph,
      graph_metadata=encoded_metadata,
      node_types=node_types,
      edges=edge_operator)


@jax_util.register_dataclass_pytree
@dataclasses.dataclass
class PaddingConfig:
  """Configuration specifying how examples get padded to a constant shape.

  Attributes:
    static_max_metadata: EncodedGraphMetadata for the padded graph size,
      specifying the maximum number of nodes (input tagged and untagged) that
      can appear. The transition matrix will be padded to these sizes in order
      to batch together multiple graph solves.
    max_initial_transitions: Maximum number of transitions from initial states
      to in-tagged states; equivalently, the maximum number of nonzero entries
      in initial_to_in_tagged.
    max_in_tagged_transitions: Maximum number of transitions from in-tagged
      states to in-tagged states; equivalently, the maximum number of nonzero
      entries in in_tagged_to_in_tagged.
    max_edges: Maximum number of edges in the graph.
  """
  static_max_metadata: automaton_builder.EncodedGraphMetadata
  max_initial_transitions: int
  max_in_tagged_transitions: int
  max_edges: int


def pad_example(example,
                config,
                allow_failure = False):
  """Pad an example so that it has a static shape determined by the config.

  The shapes of all NDArrays in the output will be fully determined by the
  config. Note that we do not pad the metadata or num_targets fields, since
  those are already of static shape; the values in those fields can be used
  to determine which elements of the other fields are padding and which elements
  are not.

  Args:
    example: The example to pad.
    config: Configuration specifying the desired padding size.
    allow_failure: If True, returns None instead of failing if example is too
      large.

  Returns:
    A padded example with static shape.

  Raises:
    ValueError: If the graph is too big to pad to this size.
  """
  # Check the size of the example.
  if example.graph_metadata.num_nodes > config.static_max_metadata.num_nodes:
    if allow_failure:
      return None
    raise ValueError("Example has too many nodes")

  if (example.graph_metadata.num_input_tagged_nodes >
      config.static_max_metadata.num_input_tagged_nodes):
    if allow_failure:
      return None
    raise ValueError("Example has too many input-tagged nodes")

  if (example.automaton_graph.initial_to_in_tagged.values.shape[0] >
      config.max_initial_transitions):
    if allow_failure:
      return None
    raise ValueError("Example has too many initial transitions")

  if (example.automaton_graph.in_tagged_to_in_tagged.values.shape[0] >
      config.max_in_tagged_transitions):
    if allow_failure:
      return None
    raise ValueError("Example has too many in-tagged transitions")

  if example.edges.values.shape[0] > config.max_edges:
    if allow_failure:
      return None
    raise ValueError("Example has too many edges")

  # Pad it out.
  return GraphBundle(
      automaton_graph=automaton_builder.EncodedGraph(
          initial_to_in_tagged=example.automaton_graph.initial_to_in_tagged
          .pad_nonzeros(config.max_initial_transitions),
          initial_to_special=jax_util.pad_to(
              example.automaton_graph.initial_to_special,
              config.static_max_metadata.num_nodes),
          in_tagged_to_in_tagged=(
              example.automaton_graph.in_tagged_to_in_tagged.pad_nonzeros(
                  config.max_in_tagged_transitions)),
          in_tagged_to_special=jax_util.pad_to(
              example.automaton_graph.in_tagged_to_special,
              config.static_max_metadata.num_input_tagged_nodes),
          in_tagged_node_indices=jax_util.pad_to(
              example.automaton_graph.in_tagged_node_indices,
              config.static_max_metadata.num_input_tagged_nodes),
      ),
      graph_metadata=example.graph_metadata,
      node_types=jax_util.pad_to(example.node_types,
                                 config.static_max_metadata.num_nodes),
      edges=example.edges.pad_nonzeros(config.max_edges),
  )


def zeros_like_padded_example(config):
  """Build an GraphBundle containing only zeros.

  This can be useful to initialize model parameters, or do tests.

  Args:
    config: Configuration specifying the desired padding size.

  Returns:
    An "example" filled with zeros of the given size.
  """
  return GraphBundle(
      automaton_graph=automaton_builder.EncodedGraph(
          initial_to_in_tagged=sparse_operator.SparseCoordOperator(
              input_indices=np.zeros(
                  shape=(config.max_initial_transitions, 1), dtype=np.int32),
              output_indices=np.zeros(
                  shape=(config.max_initial_transitions, 2), dtype=np.int32),
              values=np.zeros(
                  shape=(config.max_initial_transitions,), dtype=np.float32),
          ),
          initial_to_special=np.zeros(
              shape=(config.static_max_metadata.num_nodes,), dtype=np.int32),
          in_tagged_to_in_tagged=sparse_operator.SparseCoordOperator(
              input_indices=np.zeros(
                  shape=(config.max_in_tagged_transitions, 1), dtype=np.int32),
              output_indices=np.zeros(
                  shape=(config.max_in_tagged_transitions, 2), dtype=np.int32),
              values=np.zeros(
                  shape=(config.max_in_tagged_transitions,), dtype=np.float32),
          ),
          in_tagged_to_special=np.zeros(
              shape=(config.static_max_metadata.num_input_tagged_nodes,),
              dtype=np.int32),
          in_tagged_node_indices=np.zeros(
              shape=(config.static_max_metadata.num_input_tagged_nodes,),
              dtype=np.int32),
      ),
      graph_metadata=automaton_builder.EncodedGraphMetadata(
          num_nodes=0, num_input_tagged_nodes=0),
      node_types=np.zeros(
          shape=(config.static_max_metadata.num_nodes,), dtype=np.int32),
      edges=sparse_operator.SparseCoordOperator(
          input_indices=np.zeros(shape=(config.max_edges, 1), dtype=np.int32),
          output_indices=np.zeros(shape=(config.max_edges, 2), dtype=np.int32),
          values=np.zeros(shape=(config.max_edges,), dtype=np.int32),
      ),
  )
