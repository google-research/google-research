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

"""JAX/TF-serializable graph data struct that stores nodes, edges, features."""

import collections
from collections.abc import Iterable
import copy
import glob
import io
import json
from typing import Any, Callable, NamedTuple, Sequence, Optional

import networkx as nx
import numpy as np
import tqdm

import sparse_deferred as sd

open_file = open
glob_files = glob.glob



Tensor = sd.matrix.Tensor
Features = dict[str, Tensor]
FeatureSets = dict[str, Features]
Edge = tuple[tuple[Tensor, Ellipsis]|Tensor, Features]  # (endpoints, edge features)
Edges = dict[str, Edge]
Nodes = FeatureSets
Schema = dict[str, tuple[str, Ellipsis]]
_Schema = dict[str, tuple[dict[str, int], Ellipsis]]


def _assert_endpoints_within_node_range(
    graph,
    engine,
):
  """Checks if the adjacencies of edges are within the size of their node sets.

  Args:
    graph: Graph to validate.
    engine: ComputeEngine to use for efficient matrix operations.
  """
  for edge_name in graph.edges.keys():
    for i, node_name in enumerate(graph.schema[edge_name]):
      endpoints = graph.edges[edge_name][0][i]
      engine.assert_greater(
          graph.get_num_nodes(engine, node_name), engine.reduce_max(endpoints)
      )


class GraphStruct(NamedTuple):
  """{TF,JAX,etc}-Dataset-friendly struct for encoding Graph Structures.

  An instance is best-constructed using the `.new()` static method.
  """

  # Edge set name -> (edges [(src, tgt, ...), ...], {"feature": Tensor})
  edges: Edges
  # Node set name -> feature name -> Tensor.
  nodes: Nodes

  # For every key in `edges`, store names of node sets that `key` edge connects.
  schema_: _Schema

  @classmethod
  def new(
      cls,
      nodes = None,
      edges = None,
      schema = None,
      engine = None,
      validate = False,
  ):
    """Constructs a new instance.

    Args:
      nodes: Node set name -> feature name -> Tensor.
      edges: Edge set name -> (edges [(src, tgt, ...), ...], {"feature":
        Tensor})
      schema: Edge set name -> (source node set name, target node set name). If
        the schema is None and there is only one nodeset and one edgeset, a
        homogeneous schema is inferred.
      engine: ComputeEngine to use for efficient matrix operations used in
        validating graph structure.
      validate: If True, validate the input. If set, `engine` must be provided.
        If set, `schema` must be provided, unless inferred as an homogeneous
        schema.

    Returns:
      A new instance of `GraphStruct`.

    Raises:
      ValueError if `validate` is True and input is invalid.
    """
    edges = edges or {}
    nodes = nodes or {}
    schema = schema or {}

    if edges and validate:
      for k, v in edges.items():
        assert isinstance(k, str), 'Edges must be keyed by edge set name.'
        if not isinstance(v, tuple):
          err = (
              'Edges must be a tuple of (endpoints, features) where '
              '`endpoints` is a 2D array of (source, target) pairs with shape '
              '(2, num_edges) and `features` are a (potentially empty) '
              'dictionary of edge features.'
          )
          raise ValueError(err)

        edge_features = v[1]
        if not isinstance(edge_features, dict):
          raise ValueError(
              'Edges must define features as a (potentially empty) dict.'
          )

    if len(edges) == 1 and len(nodes) == 1 and not schema:
      edge_name = list(edges.keys())[0]
      node_name = list(nodes.keys())[0]
      schema = {edge_name: (node_name, node_name)}
    else:
      if validate:
        for edge_name in edges:
          if edge_name not in schema:
            raise ValueError(
                'Edge name %s is not in schema with keys (%s)'
                % (edge_name, ', '.join(schema.keys()))
            )

    graph = GraphStruct(
        nodes=nodes or {},
        edges=edges or {},
        schema_=GraphStruct.schema_names_as_dict_keys(schema or {}),
    )

    if validate and engine:
      _assert_endpoints_within_node_range(graph, engine)

    return graph

  @classmethod
  def from_nx(
      cls,
      graph,
      symmetrize = True,
      engine = None,
      validate = False,
      feature_name = 'embedding',
  ):
    """Constructs a new numpy GraphStruct from a NetworkX graph.

    Args:
      graph: NetworkX graph to convert to GraphStruct.
      symmetrize: If True, the graph will be symmetrized.
      engine: ComputeEngine to use for efficient matrix operations used in
        validating graph structure.
      validate: If True, validate the input. If set to True, `engine` must be
        provided.
      feature_name: Name of the dummy node feature.
    """
    # TODO(mgalkin): Copy all node features from the input nx graph.
    # list of dicts of node features
    features = list(dict(graph.nodes(data=True)).values())
    if features[0].get(feature_name, None) is not None:
      node_features = np.array(
          [d.get(feature_name) for d in features], dtype=np.float32
      )
      node_features = node_features.reshape(graph.number_of_nodes(), -1)
    else:
      node_features = np.ones((graph.number_of_nodes(), 1), dtype=np.float32)

    edge_list = np.array(list(graph.edges()), dtype=np.int32)  # (n_edges, 2)
    edge_list = edge_list.T
    senders = edge_list[0]
    receivers = edge_list[1]
    # Duplicate edges for directed representation
    full_senders = (
        np.concatenate([senders, receivers]) if symmetrize else senders
    )
    full_receivers = (
        np.concatenate([receivers, senders]) if symmetrize else receivers
    )
    return cls.new(
        nodes={'nodes': {feature_name: node_features}},
        edges={'edges': ((full_senders, full_receivers), {})},
        schema={'edges': ('nodes', 'nodes')},
        engine=engine,
        validate=validate,
    )

  def to_nx(self):
    """Converts a GraphStruct to a NetworkX graph.

    # TODO(mgalkin): preserve node and edge features.

    Returns:
      A NetworkX graph representation of the GraphStruct.
    """
    graph = nx.Graph()
    for edge_name in self.edges:
      for src, tgt in zip(
          self.edges[edge_name][0][0], self.edges[edge_name][0][1]
      ):
        graph.add_edge(src, tgt)
    return graph

  @property
  def schema(self):
    return GraphStruct.schema_names_as_strings(self.schema_)

  def update(
      self,
      *,
      nodes = None,
      edges = None,
      schema = None,
  ):
    """Returns a modified copy of this instance.

    Args:
      nodes: New node names and/or new feature names are accepted. If a feature
        name already exist, then value will be overwritten.
      edges: New edge names and/or new feature names are accepted. If an edge
        name is given, the adjacency list will be overwritten.
      schema: New edges must be amended here.
    """
    updated_nodes = _copy_features(self.nodes)
    updated_edges = {
        k: (endpoints, dict(feats))
        for k, (endpoints, feats) in self.edges.items()
    }
    updated_schema = dict(self.schema_)
    if schema is not None:
      updated_schema.update(GraphStruct.schema_names_as_dict_keys(schema))

    for new_edge_name, new_edge in (edges or {}).items():
      updated_edges[new_edge_name] = new_edge
    for new_node_name, new_node in (nodes or {}).items():
      if new_node_name not in updated_nodes:
        updated_nodes[new_node_name] = {}
      for new_feature_name, new_feature in new_node.items():
        updated_nodes[new_node_name][new_feature_name] = new_feature

    return GraphStruct(
        nodes=updated_nodes, edges=updated_edges, schema_=updated_schema
    )

  def add_pooling(
      self,
      engine,
      graph_features = None,
      stack_edges = False,
      excluded_node_sets = (),
      num_nodes_map = None,
  ):
    """Amends one virtual node 'g' (if not present) to connect to all nodes.

    Args:
      engine: Compute Engine to create edge endpoints. Node 'g' will be
        connected to every node from every node set.
      graph_features: Features to add at the graph level, specifically, to tie
        to node set 'g'.
      stack_edges: If False (default), will present pooled edges as a tuple of
        (source, target) Tensors. If True, will stack the source and target into
        a single 2D tensor with shape [[source], [target]]
      excluded_node_sets: A list of node sets to exclude from the pooling. For
        example, some tfgnn graph tensors have a '_readout' nodeset that doesn't
        make to add to the pooled node set 'g'.
      num_nodes_map: Optional number of nodes in each nodeset. If not specified,
        nodesets are required to have at least one feature (used to determine
        the number of nodes).

    Returns:
      new copy of the graph (does not modify existing).
    """
    if 'g' in self.nodes:
      raise ValueError('GraphStruct already has graph-level node-set "g"')
    for node_name in self.nodes.keys():
      edge_name = f'g_{node_name}'
      if edge_name in self.edges or edge_name in self.schema_:
        raise ValueError(f'GraphStruct already has edge "{edge_name}"')

    if graph_features is None:
      graph_features = {'id': engine.zeros([1, 1], 'int32')}
    else:
      for graph_feature in graph_features.values():
        size = engine.shape(graph_feature)[0]
        engine.assert_equal(size, 1)

    g_edges = {}
    g_edges_schema = {}
    for node_name, features in self.nodes.items():
      if node_name in excluded_node_sets:
        continue

      if num_nodes_map is not None:
        num_nodes = num_nodes_map[node_name]
      else:
        first_feature = list(features.values())[0]
        num_nodes = engine.shape(first_feature)[0]
      edge_name = f'g_{node_name}'

      if stack_edges:
        g_edges[edge_name] = (
            engine.reshape(
                engine.concat(
                    [
                        engine.zeros([num_nodes], 'int32'),
                        engine.range(num_nodes, 'int32'),
                    ],
                    axis=-1,
                ),
                shape=[2, num_nodes],
            ),
            {},
        )
      else:
        g_edges[edge_name] = (
            (
                engine.zeros([num_nodes], 'int32'),
                engine.range(num_nodes, 'int32'),
            ),
            {},
        )
      g_edges_schema[edge_name] = ('g', node_name)

    return self.update(
        nodes={'g': graph_features}, edges=g_edges, schema=g_edges_schema
    )

  def get_num_nodes(
      self, engine, node_name
  ):
    features = self.nodes.get(node_name, {})
    if not features:
      return 0
    first_feature = list(features.values())[0]
    if first_feature.shape[0] is None:
      return engine.shape(first_feature)[0]
    return first_feature.shape[0]

  def adj(
      self,
      engine,
      edge_name,
      values = None,
  ):
    """Adjacency over edge-set name.

    The multiplication `A @ X` (where `A` is the result of this function and `X`
    are some features) computes message passing source->target, i.e., each
    target computes features to be the weighted sum of rows of `X`.

    Args:
      engine: Compute engine that is used to construct `SparseMatrix`, i.e., for
        conducting the actual ops.
      edge_name: Name of edge. It must be in `.schema`.
      values: If given, must be a vector equal to number of edges that should
        assign weight to every edge.

    Returns:
      Multiplier with shape determined by `self.schema[edge_name]`.
      Specifically, with shape:
      `(num nodes for target node set, num nodes for source node set)`.
    """
    col_indices = self.edges[edge_name][0][0]  # One-at-a-time is TPU-friendly.
    row_indices = self.edges[edge_name][0][1]  # One-at-a-time is TPU-friendly.
    col_node_name, row_node_name = self.schema[edge_name]
    return sd.SparseMatrix(
        engine,
        indices=(row_indices, col_indices),
        dense_shape=(
            self.get_num_nodes(engine, row_node_name),
            self.get_num_nodes(engine, col_node_name),
        ),
        values=values,
    )

  def get_outgoing_neighbors(
      self, engine, edge_name, node_index
  ):
    """Returns neighbors of a given node and edge feature."""
    src_indices = self.edges[edge_name][0][0]
    tgt_indices = self.edges[edge_name][0][1]
    src_node_name = self.schema[edge_name][0]
    tgt_node_name = self.schema[edge_name][1]
    num_nodes = engine.maximum(
        self.get_num_nodes(engine, src_node_name),
        self.get_num_nodes(engine, tgt_node_name),
    )
    outdegree = engine.unsorted_segment_sum(
        engine.ones(engine.shape(src_indices)),
        src_indices,
        num_segments=num_nodes,
    )
    offsets = engine.concat([engine.zeros([1]), engine.cumsum(outdegree)], 0)
    argsorted_src_indices = engine.argsort(src_indices)
    sorted_tgts = engine.gather(tgt_indices, argsorted_src_indices)
    start_offset = offsets[node_index]
    end_offset = offsets[node_index + 1]
    neighbors = engine.gather(
        sorted_tgts,
        engine.cast(
            engine.range(end_offset - start_offset) + start_offset, 'int32'
        ),
    )
    return neighbors

  def incidence(
      self, engine, edge_name, endpoint_index = 0
  ):
    """Node to edge incidence matrix."""
    endpoint_ids = self.edges[edge_name][0][endpoint_index]
    endpoint_name = self.schema[edge_name][endpoint_index]
    num_edges = engine.shape(endpoint_ids)[0]
    num_nodes = self.get_num_nodes(engine, endpoint_name)
    return sd.SparseMatrix(
        engine,
        indices=(engine.range(num_edges, dtype='int32'), endpoint_ids),
        dense_shape=(num_edges, num_nodes),
        values=None,
    )

  @classmethod
  def schema_names_as_dict_keys(cls, schema):
    return {
        k: tuple([{endpoint: 0} for endpoint in v]) for k, v in schema.items()
    }

  @classmethod
  def schema_names_as_strings(cls, schema):
    # faster dict parsing as suggested by Gemini 2.5 Pro
    # works for binary edges
    # Prev: k: tuple([list(endpoint.keys())[0] for endpoint in v])
    return {k: (next(iter(v[0])), next(iter(v[1]))) for k, v in schema.items()}

  def serialize_as_db_npz(self):
    db = InMemoryDB()
    db.add(self)
    db.finalize()
    return db.get_npz_bytes()

  @classmethod
  def deserialize_db_npz(cls, graph_bytes):
    return InMemoryDB.from_bytes(graph_bytes).get_item(0)


def combine_graph_structs(
    engine,
    *graph_structs,
    stack_edges = False,
):
  """Combines multiple GraphStructs into one with multiple components.

  Args:
    engine: A sd.ComputeEngine instance
    *graph_structs: Input graph structs
    stack_edges: If False (default) combined edges will have format ((source,
      target), {}). If True will have format (T, {}) where T is a 2D shape like
      [[source], [target]].

  Returns:
    A single GraphStruct instance that combines all graphs.
  """
  # node set name -> feature name -> list of features.
  all_node_feats = collections.defaultdict(
      lambda: collections.defaultdict(list)
  )
  # edge name -> feature name -> list of features.
  all_edge_feats = collections.defaultdict(
      lambda: collections.defaultdict(list)
  )
  # edge name -> list of list of vectors.
  # Outer list contains endpoint types and inner list contain endpoint IDs, e.g.
  # `[[source_nodes_graph_1, source_graph2, ..], [target_nodes_graph 1, tgt 2]]`
  edge_endpoints = collections.defaultdict(list)
  node_offsets = collections.Counter()
  schema = {}

  for g in graph_structs:
    g_schema = g.schema
    schema.update(g_schema)
    # Add edges to batch.
    for edge_name, (endpoints, features) in g.edges.items():
      # Add endpoints.
      while len(edge_endpoints[edge_name]) < len(g_schema[edge_name]):
        edge_endpoints[edge_name].append([])
      for i, endpoint_node_name in enumerate(g_schema[edge_name]):
        endpoint = endpoints[i]
        edge_endpoints[edge_name][i].append(
            endpoint + node_offsets[endpoint_node_name]
        )
      # Add features.
      for feature_name, feature in features.items():
        all_edge_feats[edge_name][feature_name].append(feature)
    #
    # Add node features.
    for node_name, features in g.nodes.items():
      num_nodes = -1
      for feature_name, feature in features.items():
        all_node_feats[node_name][feature_name].append(feature)
        if isinstance(num_nodes, int) and num_nodes == -1:
          # sdtf engine in graph mode returns a symbolic value.
          num_nodes = engine.shape(feature)[0]
          # engine.assert_equal(num_nodes, engine.shape(feature)[0])
      #
      if isinstance(num_nodes, int) and num_nodes == -1:
        # sdtf engine in graph mode returns a symbolic value.
        num_nodes = 0
      node_offsets[node_name] += num_nodes

  edges = {}
  for edge_name in list(edge_endpoints.keys()):
    edge_features = {}
    for feature_name in list(all_edge_feats[edge_name].keys()):
      edge_features[feature_name] = engine.concat(
          all_edge_feats[edge_name][feature_name], axis=0
      )
    #
    endpoints = [engine.concat(el, axis=0) for el in edge_endpoints[edge_name]]

    if stack_edges:
      edges[edge_name] = (
          engine.reshape(engine.concat(endpoints, axis=-1), shape=[2, -1]),
          edge_features,
      )
    else:
      edges[edge_name] = (tuple(endpoints), edge_features)

  nodes = {}
  for node_name in list(all_node_feats.keys()):
    node_features = {}
    for feature_name in list(all_node_feats[node_name].keys()):
      node_features[feature_name] = engine.concat(
          all_node_feats[node_name][feature_name], axis=0
      )
    nodes[node_name] = node_features

  return GraphStruct.new(nodes=dict(nodes), edges=dict(edges), schema=schema)


def _copy_features(features):
  # 2-level deep copy.
  return {k: dict(v) for k, v in (features or {}).items()}


def are_graphs_exactly_equal(
    engine, g1, g2
):
  """Returns True if the graphs are exactly equal."""
  if g1.nodes.keys() != g2.nodes.keys():
    return False

  if g1.edges.keys() != g2.edges.keys():
    return False

  for node_name, feats1 in g1.nodes.items():
    feats2 = g2.nodes[node_name]
    if feats1.keys() != feats2.keys():
      return False

    for k, tensor1 in feats1.items():
      tensor2 = feats2[k]
      if engine.shape(tensor1) != engine.shape(tensor2):
        return False
      if engine.to_cpu(engine.reduce_any(tensor1 != tensor2)):
        return False

  for edge_name, edges1 in g1.edges.items():
    edges2 = g2.edges[edge_name]
    edges1 = edges1[0]  # Edges only
    edges2 = edges2[0]
    for endpoint1, endpoint2 in zip(edges1, edges2):
      if engine.shape(endpoint1) != engine.shape(endpoint2):
        return False
      if engine.to_cpu(engine.reduce_any(endpoint1 != endpoint2)):
        return False

  return True


class InsufficientPaddingError(ValueError):
  pass


class FixedSizePadder:
  """Adds padding to `GraphStruct` instances for fixed-sized tensors.

  Fixed-size tensors can be preferred when running on TPU accelerators.

  To use this class, you must first initialize it with statistics of your graphs
  then use it to pad graphs. The statistics can be initialized by invoking
  `calculate_pad_statistics`: this function records the *maximum* observerd size
  of every node and edge set, as well as the standard deviation (std) of sizes.

  Once initialized, the function: `pad_graph()` will add padding to the graph.
  Specifically, the node feature (tensors) will be padded with zeros. Similarly,
  edges will be inserted, among newly-added virtual nodes.

  Each node (or edge) size will become:

  `max observed [per calculate_pad_statistics] + slack*std + 1`

  NOTE: there will always be at least one more node or edge, even if the
  statistics show zero std. This is required for making virtual nodes.

  All sizes (for node-set features and edge-set features/adjacency lists) are
  affected.
  """

  def __init__(
      self,
      engine,
      slack = 1.0,
      stack_edges = True,
      fails_on_oversized_batch = False,
  ):
    """Initializes the FixedSizePadder.

    Args:
      engine: Execution engine used for tensor operations.
      slack: Padding margin, measured as a ratio of the standard deviation of
        observed sizes. The padding size is calculated as `max_size + slack *
        std_dev + 1`.
      stack_edges: If True, edge endpoints are stacked into a single `[2,
        num_edges]` tensor. If False, they are kept as a tuple of two
        `[num_edges]` tensors.
      fails_on_oversized_batch: If True, an `InsufficientPaddingError` is raised
        if a graph being padded exceeds the calculated fixed size for any node
        or edge set. If False, oversized parts of the graph will be truncated to
        fit the fixed size, which might result in missing edges, missing nodes,
        or fake edges pointing to real nodes.
    """
    # `('edge'|'node', NodeOrEdgeName) -> target size`
    # where `target size` is maximum observed size for node (or edge) set, plus
    # one, plus slack-times-std of observed sizes.
    self.sizes: dict[tuple[str, str], int] = {}
    self.slack = slack
    self._engine = engine
    self.stack_edges = stack_edges
    self.fails_on_oversized_batch = fails_on_oversized_batch

  def replace_engine(self, engine):
    """Creates a new equivalent padder with a different engine.

    Usage example:

    ```python
      p1 = FixedSizePadder(sdnp.engine)
      p1.calculate_pad_statistics(...)
      p2 = p1.replace_engine(sdjax.engine)
    ```

    Args:
      engine: Compute engine to use for efficient matrix operations.

    Returns:
      A new `FixedSizePadder`.
    """
    new_padder = FixedSizePadder(
        engine=engine, slack=self.slack, stack_edges=self.stack_edges
    )
    new_padder.sizes = copy.deepcopy(self.sizes)
    return new_padder

  def calculate_pad_statistics(
      self, examples, num_steps = 100
  ):
    """Measures the max and std of node & edge sizes of elements of `examples`.

    Calling this function is necessary before invoking `pad_graph`.

    Args:
      examples: iterable that yields `GraphStruct` examples.
      num_steps: If positive, considers this many samples of `examples`.
        Otherwise, iterates over all `examples`. Warning: this may run
        infinitely on infinite iterators (e.g., `dataset.repeat()`).
    """
    sizes: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for i, graph in enumerate(examples):
      # assert isinstance(graph, GraphStruct)
      if i > 0 and i >= num_steps:
        break
      for node_name, features in graph.nodes.items():
        value_list = sizes[('nodes', node_name)]
        if not features:
          value_list.append(0)
        else:
          value_list.append(list(features.values())[0].shape[0])

      for edge_name, edges_tuple in graph.edges.items():
        value_list = sizes[('edges', edge_name)]
        source_nodes = edges_tuple[0][0]
        # if len(value_list) and edge_set.sizes.shape != value_list[-1].shape:
        #   continue
        value_list.append(source_nodes.shape[0])

    self.sizes = {
        k: int(1 + max(v) + self.slack * np.std(v)) for k, v in sizes.items()
    }

  def pad_graph(self, graph):
    """Pads node-sets and edge-sets, with zeros, to max-seen during `calc..`.

    This function is useful for running on TPU hardware.

    Args:
      graph: contains any number of nodes and edges.

    Returns:
      graph with deterministic number of nodes and edges. See class docstring.

    Raises:
      ValueError: If `calculate_pad_statistics` has not been called.
      InsufficientPaddingError: If `fails_on_oversized_batch` is True and the
        input graph exceeds the fixed sizes determined during padding
        calculation.
    """
    if not self.sizes:
      raise ValueError(
          'No statistics have been initialized. '
          'Perhaps you forgot to invoke "calculate_pad_statistics"?'
      )
    # Edge set name -> (1D vectors containing endpoints**), {"feature": Tensor})
    edges: Edges = {}
    # ** tuple should have 2 entries for directed graphs

    nodes: Nodes = {}

    # For every key in `edges`, store names of node sets that `key` edge
    # connects.
    schema = graph.schema

    e = self._engine  # for short.
    for node_name, node_features in graph.nodes.items():
      padded_features = {}
      desired_size = self.sizes[('nodes', node_name)]

      for feature_name, feature in node_features.items():
        padded_feature = e.zeros(
            tuple([desired_size] + list(feature.shape[1:])), dtype=feature.dtype
        )
        if self.fails_on_oversized_batch and feature.shape[0] > desired_size:
          raise InsufficientPaddingError(
              f"Padding for node set '{node_name}' is insufficient. Required"
              f' at least {feature.shape[0]} nodes (including the padding'
              f' node), but the padder only defines {desired_size}.'
          )
        padded_feature = e.fill_padding(feature[:desired_size], padded_feature)
        padded_features[feature_name] = padded_feature

      nodes[node_name] = padded_features

    for edge_name, (edge_endpoints, features) in graph.edges.items():
      padded_features = {}
      desired_size = self.sizes[('edges', edge_name)]

      for feature_name, feature in features.items():
        feature = feature[:desired_size]  # if `is_oversized`.
        padded_feature = e.zeros(
            tuple([desired_size] + list(feature.shape[1:])), dtype=feature.dtype
        )
        if self.fails_on_oversized_batch and feature.shape[0] > desired_size:
          raise InsufficientPaddingError(
              f"Padding for edge set '{edge_name}' is insufficient. Required"
              f' at least {feature.shape[0]} nodes (including the padding'
              f' node), but the padder only defines {desired_size}.'
          )
        padded_feature = e.fill_padding(feature, padded_feature)
        padded_features[feature_name] = padded_feature

      # Note: This implementation creates ones array per endpoint and stack
      # them. This is necessary for JAX and TF Graph code. However, this is
      # slower than creating and modifying directly the array in numpy.
      # The benchmark cl/782798674 helps measuring this (30% slower in this
      # benchmark). If the NP speed becomes critical, separating the numpy from
      # the other implementation is an alternative solution.

      # For each endpoint (generally the source node and target node), compute
      # the padded array of edges.
      padded_endpoints = []
      for node_ids, node_name in zip(edge_endpoints, schema[edge_name]):
        # The node index used to pad the array correspond to the last node
        # (including the padding).
        padded_value = self.sizes[('nodes', node_name)] - 1
        # Create the padding array filled with padded value.
        # TODO(gbm): Implement using "fill(n)" instead of ones() * n.
        padded_endpoint = padded_value * e.ones(
            [desired_size], dtype=edge_endpoints[0].dtype
        )

        if self.fails_on_oversized_batch and node_ids.shape[0] > desired_size:
          raise InsufficientPaddingError(
              f"Padding for edge set '{edge_name}' is insufficient. Required at"
              f' least {node_ids.shape[0]} edges, but the padder only defines'
              f' {desired_size}.'
          )

        # Copy the real values.
        padded_endpoint = e.fill_padding(
            node_ids[:desired_size], padded_endpoint
        )
        padded_endpoints.append(padded_endpoint)

      # Merge the two end-points
      if self.stack_edges:
        edges[edge_name] = (
            e.stack(padded_endpoints, axis=0),
            padded_features,
        )
      else:
        edges[edge_name] = (tuple(padded_endpoints), padded_features)

    graph = GraphStruct.new(nodes=nodes, edges=edges, schema=schema)
    return graph

  def save_stats_to_json(self, path):
    """Saves the padding statistics to a JSON file.

    This function serializes the padding statistics to a JSON file. This is a
    simple serialization of the dictionary `self.sizes`. Since JSON does not
    support tuple as keys, the keys are converted to comma-separated strings.
    This is compatible with `load_stats_from_json()`.

    Args:
      path: Path to the JSON file.
    """
    with gfile.GFile(path, 'w') as f:
      # Convert tuple keys to comma-separated strings
      data_to_save = {','.join(k): int(v) for k, v in self.sizes.items()}
      json.dump(data_to_save, f)

  def load_stats_from_json(self, path):
    """Loads the padding statistics from a JSON file.

    This function loads the padding statistics from a JSON file. This is a
    simple deserialization of the dictionary `self.sizes`. Since JSON does not
    support tuple as keys, the keys are converted to comma-separated strings.
    This is compatible with `save_stats_to_json()`.

    Args:
      path: Path to the JSON file.
    """
    with gfile.GFile(path, 'r') as f:
      self.sizes = {tuple(k.split(',')): v for k, v in json.load(f).items()}


class InMemoryDB:
  """In-memory database of GraphStructs. It can be saved/loaded as numpy format.

  ```py
  db = InMemoryDB()
  for g in graphs:  # Many graphs with same schema.
    db.add(g)

  db.finalize()  # Activates `get_item`

  db.get_item(0)  # == graphs[0], etc.

  # tf.data.Dataset
  ds: tf.data.Dataset = sparse_deferred.tf.graph.db_to_tf_dataset(db)

  # NOTE: for the above to work, the features in `graphs` should be all
  # `tf.Tensor`s. If this is not the case, then you can convert all to tf.Tensor
  # by saving-then-loading
  db.save('temp_file.npz')
  tfdb = sparse_deferred.tf.graph.InMemoryDB.from_file('temp_file.npz')
  ds = tfdb.as_tf_dataset()
  ```
  """

  def __init__(self, flex_schema = False):
    self._sizes = collections.defaultdict(list)
    self._edges = {}
    self._features = collections.defaultdict(list)
    self.schema = None
    self._cumsum_sizes: dict[tuple[str, str], np.ndarray] | None = None
    self._flex_schema = flex_schema
    self._num_graphs = 0

  @property
  def features(self):
    """Internal representation: Concatenation of all graph features."""
    if not self._cumsum_sizes:
      raise ValueError('You have not yet called finalize().')
    return self._features

  def get_npz_bytes(
      self,
      save_fn = np.savez_compressed,
      start_id = 0,
      end_id = None,
  ):
    """Returns bytes of the npz file that can be loaded with `from_bytes()`."""
    if self._cumsum_sizes is None:
      raise ValueError('You have not yet called finalize().')
    if end_id is None:
      end_id = self.size
    np_kwargs = {}
    for (n_or_e, name, feat_name), v in self._features.items():
      idx_start = self._cumsum_size((n_or_e, name), start_id)
      idx_end = self._cumsum_size((n_or_e, name), end_id)
      np_kwargs[f'feat.{n_or_e}.{name}.{feat_name}'] = np.array(
          v[idx_start:idx_end]
      )
    for (n_or_e, name), v in self._sizes.items():
      np_kwargs[f'size.{n_or_e}.{name}'] = np.array(v[start_id:end_id])
      np_kwargs[f'csumsize.{n_or_e}.{name}'] = np.array(
          self._cumsum_size((n_or_e, name), range(start_id, end_id + 1))
      )
    for name, v in self._edges.items():
      idx_start = self._cumsum_size(('e', name), start_id)
      idx_end = self._cumsum_size(('e', name), end_id)
      np_kwargs[f'edge.{name}'] = np.array([ep[idx_start:idx_end] for ep in v])
    np_kwargs['schema'] = json.dumps(self.schema)
    bytes_io = io.BytesIO()
    save_fn(bytes_io, **np_kwargs)
    return bytes_io.getvalue()

  def _cumsum_size(self, key, index):
    """Returns `self._cumsum_sizes[key][index]` after validation."""
    if self._cumsum_sizes is None:
      raise ValueError('You have not yet called finalize().')
    if key not in self._cumsum_sizes:
      raise ValueError(
          f'Key {key} not found in {list(self._cumsum_sizes.keys())}'
      )
    if isinstance(index, int):
      if index < 0 or index >= len(self._cumsum_sizes[key]):
        raise ValueError(f'Index {index} out of bounds for {key}')
    else:  # must be Sequence.
      for idx in list(index):
        if idx < 0 or idx >= len(self._cumsum_sizes[key]):
          raise ValueError(f'Index {idx} out of bounds for {key}')

    return np.array(self._cumsum_sizes[key])[index]

  def save(
      self,
      filename,
      save_fn = np.savez_compressed,
      start_id = 0,
      end_id = None,
  ):
    """Saves DB to a numpy file that can be loaded with `from_file()`."""
    npz_bytes = self.get_npz_bytes(
        save_fn=save_fn, start_id=start_id, end_id=end_id
    )
    with gfile.GFile(filename, 'wb') as f:
      f.write(npz_bytes)

  def save_sharded(
      self,
      file_prefix,
      batch_size,
      save_fn = np.savez_compressed,
  ):
    """Saves DB onto multiple files, each containing <= `batch_size` graphs."""
    if self._cumsum_sizes is None:
      raise ValueError('You have not yet called finalize().')
    batch_indices = list(range(0, self.size, batch_size))
    batch_indices.append(self.size)
    for start_i, end_i in tqdm.tqdm(
        list(zip(batch_indices[:-1], batch_indices[1:]))
    ):
      end_i = min(end_i, self.size)
      filename = f'{file_prefix}-{start_i}-to-{end_i}'
      self.save(filename, save_fn=save_fn, start_id=start_i, end_id=end_i)

  def load_from_bytes(
      self, npz_bytes, to_device_fn = np.array
  ):
    """Restores DB instance serialized with `get_npz_bytes()`."""
    np_data = dict(np.load(io.BytesIO(npz_bytes), allow_pickle=True))
    self._cumsum_sizes = {}
    for k, np_arr in np_data.items():
      k_parts = k.split('.')
      if k_parts[0] == 'feat':
        self._features[tuple(k_parts[1:])] = to_device_fn(np_arr)
      elif k_parts[0] == 'size':
        self._sizes[tuple(k_parts[1:])] = to_device_fn(np_arr)
      elif k_parts[0] == 'csumsize':
        csizes = to_device_fn(np_arr)
        csizes -= csizes[0]
        self._cumsum_sizes[tuple(k_parts[1:])] = csizes
      elif k_parts[0] == 'edge':
        self._edges[k_parts[1]] = to_device_fn(np.array(np_arr, 'int32'))
      elif k == 'schema':
        self.schema = json.loads(str(np_arr))
        self.schema = {
            edge_set_name: tuple(endpoints)
            for edge_set_name, endpoints in self.schema.items()
        }

    self._np_features = self._features

  def load_from_file(
      self, filename, to_device_fn = np.array
  ):
    """Loads DB from a numpy file saved with `save()`."""
    self.load_from_bytes(gfile.GFile(filename, 'rb').read(), to_device_fn)

  @classmethod
  def from_sharded_files(
      cls, file_prefix, to_device_fn = np.array
  ):
    """Load from sharded files saved with `save_sharded()`."""
    filenames = glob_files(f'{file_prefix}-*')
    filenames.sort(key=lambda filename: int(filename.split('-')[-1]))
    entire_db = InMemoryDB()
    for filename in tqdm.tqdm(filenames):
      db = InMemoryDB.from_file(filename, to_device_fn)
      for i in range(db.size):
        entire_db.add(db.get_item(i))
    entire_db.finalize(to_device_fn)
    return entire_db

  @classmethod
  def from_file(
      cls, filename, to_device_fn = np.array
  ):
    """Recovers instance saved by save()."""
    db = InMemoryDB()
    db.load_from_file(filename, to_device_fn=to_device_fn)
    return db

  @classmethod
  def from_bytes(
      cls, npz_bytes, to_device_fn = np.array
  ):
    """Recovers instance saved by get_npz_bytes()."""
    db = InMemoryDB()
    db.load_from_bytes(npz_bytes, to_device_fn=to_device_fn)
    return db

  def verify_or_merge_schema(self, schema):
    """Verifies that the schema is compatible with the current one."""
    if self.schema is None:
      self.schema = dict(schema)
    elif not self._flex_schema:
      if self.schema != schema:
        raise ValueError(
            f'Schema mismatch: {self.schema} != {schema}. if this is intended, '
            'please set `flex_schema=True` when creating the InMemoryDB.'
        )
      assert self.schema == schema
    else:
      # Schema is set, but it is flexible (it can grow!).
      for edge_set_name, endpoints in schema.items():
        if edge_set_name not in self.schema:
          self.schema[edge_set_name] = endpoints
        else:
          assert self.schema[edge_set_name] == endpoints

  def add(self, g):
    """Adds a graph to the database."""
    if self._cumsum_sizes is not None:
      raise ValueError('You already called finalize().')
    self.verify_or_merge_schema(g.schema)
    added_features: set[tuple[str, str, str]] = set()
    added_artefacts: set[tuple[str, str]] = set()
    for ns_name, feats in g.nodes.items():
      if not self._sizes[('n', ns_name)]:
        self._sizes[('n', ns_name)] = [0] * self._num_graphs
      size = None
      for feat_name, feat_val in feats.items():
        self._add_feature(('n', ns_name, feat_name), feat_val)
        added_features.add(('n', ns_name, feat_name))
        if size is None:
          size = feat_val.shape[0]
          added_artefacts.add(('n', ns_name))
        else:
          if size != feat_val.shape[0]:
            raise ValueError(
                f'Features for node set {ns_name} have different '
                f'sizes: {size} != {feat_val.shape[0]}'
            )
      self._sizes[('n', ns_name)].append(size)

    # Add edges.
    zero_int_vec = np.zeros([0], dtype='int32')  # For empty edges.
    for es_name, (endpoints, feats) in g.edges.items():
      size = None
      assert (es_name in self._edges) == (('e', es_name) in self._sizes)
      if es_name not in self._edges:
        # When edge-set-name is seen for first time, prior-inserted graphs will
        # get empty edge-list (for that edge-set-name).
        self._edges[es_name] = [
            [zero_int_vec] * self._num_graphs for _ in range(len(endpoints))
        ]
        self._sizes[('e', es_name)] = [0] * self._num_graphs

      for i, endpoint in enumerate(endpoints):
        if size is None:
          size = endpoint.shape[0]
        else:
          if size != endpoint.shape[0]:
            raise ValueError(
                f'Endpoint lists for edge set {es_name} differ in '
                f'length: {size} != {endpoint.shape[0]}'
            )
        self._edges[es_name][i].append(np.array(endpoint, 'int32'))
      for feat_name, feat_val in feats.items():
        self._add_feature(('e', es_name, feat_name), feat_val)
        added_features.add(('e', es_name, feat_name))
        if size is None:
          size = feat_val.shape[0]
        else:
          if size != feat_val.shape[0]:
            raise ValueError(
                f'Features for edge set {es_name} have different '
                f'sizes: {size} != {feat_val.shape[0]}'
            )
      self._sizes[('e', es_name)].append(size)
      added_artefacts.add(('e', es_name))

    # Add zero size for all missing artefacts.
    for artefact_key in self._sizes:
      if artefact_key in added_artefacts:
        continue
      self._sizes[artefact_key].append(0)

    # Add zero features for all missing features.
    for feature_key, prev_features in self._features.items():
      if feature_key in added_features:
        continue
      num_entries = self._sizes[feature_key[:2]][-1]
      zero_feature = np.zeros(
          [num_entries] + list(prev_features[-1].shape[1:]),
          dtype=prev_features[-1].dtype,
      )
      self._features[feature_key].append(zero_feature)

    self._num_graphs += 1

  def finalize(self, to_device_fn = np.array):
    """Enables `get_item()` invocations, marking end of `add()` invocations."""
    if self._cumsum_sizes is not None:
      raise ValueError('You already called finalize().')
    self._cumsum_sizes = {}
    for k in list(self._sizes):
      self._sizes[k] = to_device_fn(self._sizes[k])
      self._cumsum_sizes[k] = to_device_fn(
          np.concatenate([[0], np.cumsum(self._sizes[k], axis=0)], axis=0)
      )
    for k in list(self._edges):
      for i in range(len(self._edges[k])):
        self._edges[k][i] = to_device_fn(
            np.concatenate(self._edges[k][i], axis=0)
        )
    for k in list(self._features):
      self._features[k] = to_device_fn(
          np.concatenate(self._features[k], axis=0)
      )

  def convert_to_device(self, to_device_fn):
    """Converts all features to device memory."""
    if self._cumsum_sizes is None:
      raise ValueError('You have not yet called finalize().')
    self._features = {k: to_device_fn(v) for k, v in self._features.items()}
    self._cumsum_sizes = {
        k: to_device_fn(v) for k, v in self._cumsum_sizes.items()
    }
    self._sizes = {k: to_device_fn(v) for k, v in self._sizes.items()}
    self._edges = {k: to_device_fn(v) for k, v in self._edges.items()}

  def get_item(self, i):
    """Returns a GraphStruct for the given index `0 <= i < self.size."""
    if self._cumsum_sizes is None:
      raise ValueError('You have not yet called finalize().')
    node_features = collections.defaultdict(dict)
    edge_features = collections.defaultdict(dict)
    for (n_or_e, name, feat_name), feat_val in self._features.items():
      start_idx = self._cumsum_sizes[(n_or_e, name)][i]
      end_idx = self._cumsum_sizes[(n_or_e, name)][i + 1]
      if n_or_e == 'n':
        node_features[name][feat_name] = feat_val[start_idx:end_idx]
      elif n_or_e == 'e':
        edge_features[name][feat_name] = feat_val[start_idx:end_idx]
    edges = {}
    for name, endpoints in self._edges.items():
      start_idx = self._cumsum_sizes[('e', name)][i]
      end_idx = self._cumsum_sizes[('e', name)][i + 1]
      el = tuple([ep[start_idx:end_idx] for ep in endpoints])
      edges[name] = (el, edge_features.get(name, {}))
    return GraphStruct.new(nodes=node_features, edges=edges, schema=self.schema)

  def get_item_with_engine(
      self, engine, i
  ):
    """Same as .get_item() but using ops of `engine`."""

    def range_slice(tensor, start_idx, end_idx):
      elements = engine.range(end_idx - start_idx, dtype='int32') + engine.cast(
          start_idx, dtype='int32'
      )
      return engine.gather(tensor, elements)

    if self._cumsum_sizes is None:
      raise ValueError('You have not yet called finalize().')
    node_features = collections.defaultdict(dict)
    edge_features = collections.defaultdict(dict)
    for (n_or_e, name, feat_name), feat_val in self._features.items():
      start_idx = self._cumsum_sizes[(n_or_e, name)][i]
      end_idx = self._cumsum_sizes[(n_or_e, name)][i + 1]
      if n_or_e == 'n':
        node_features[name][feat_name] = range_slice(
            feat_val, start_idx, end_idx
        )
      elif n_or_e == 'e':
        edge_features[name][feat_name] = range_slice(
            feat_val, start_idx, end_idx
        )
    edges = {}
    for name, endpoints in self._edges.items():
      start_idx = self._cumsum_sizes[('e', name)][i]
      end_idx = self._cumsum_sizes[('e', name)][i + 1]
      if isinstance(endpoints, list):
        num_endpoints = len(endpoints)
      else:
        num_endpoints = endpoints.shape[0]
      el = tuple([
          range_slice(endpoints[j], start_idx, end_idx)
          for j in range(num_endpoints)
      ])
      edges[name] = (el, edge_features.get(name, {}))
    return GraphStruct.new(nodes=node_features, edges=edges, schema=self.schema)

  @property
  def size(self):
    assert self._cumsum_sizes is not None
    if not self._cumsum_sizes:  # Empty.
      return 0
    return list(self._cumsum_sizes.values())[0].shape[0] - 1

  def _add_feature(self, key, feat_val):
    """Adds a feature to the database."""
    if key not in self._features:
      self._features[key] = []
      feature_outer_dims = list(feat_val.shape[1:])
      for i in range(self._num_graphs):
        subkey = (key[0], key[1])
        num_nodes = self._sizes[subkey][i]
        zero_feature = np.zeros(
            [num_nodes] + feature_outer_dims, dtype=feat_val.dtype
        )
        self._features[key].append(zero_feature)

    if (
        self._num_graphs > 0
        and feat_val.shape[1:] != self._features[key][-1].shape[1:]
    ):
      raise ValueError(
          f'Feature {key} differs in non-leading dimensions. '
          f'{feat_val.shape} VS {self._features[key][-1].shape}'
      )

    self._features[key].append(feat_val)
