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
import glob
import io
import json
from typing import Any, Callable, NamedTuple, Sequence

import numpy as np
import tqdm

import sparse_deferred as sd

open_file = open



Tensor = sd.matrix.Tensor
Features = dict[str, Tensor]
FeatureSets = dict[str, Features]
Edge = tuple[tuple[Tensor, Ellipsis], Features]  # (endpoints, edge features)
Edges = dict[str, Edge]
Nodes = FeatureSets
Schema = dict[str, tuple[str, Ellipsis]]
_Schema = dict[str, tuple[dict[str, int], Ellipsis]]


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
      nodes = None, edges = None,
      schema = None):
    """Constructs a new instance."""
    edges = edges or {}
    nodes = nodes or {}
    schema = schema or {}
    if len(edges) == 1 and len(nodes) == 1 and not schema:
      edge_name = list(edges.keys())[0]
      node_name = list(nodes.keys())[0]
      schema = {edge_name: (node_name, node_name)}
    else:
      for edge_name in edges:
        if edge_name not in schema:
          raise ValueError('Edge name %s is not in schema with keys (%s)' % (
              edge_name, ', '.join(schema.keys())))

    return GraphStruct(
        nodes=nodes or {}, edges=edges or {},
        schema_=GraphStruct.schema_names_as_dict_keys(schema or {}))

  @property
  def schema(self):
    return GraphStruct.schema_names_as_strings(self.schema_)

  def update(self,
             *,
             nodes = None, edges = None,
             schema = None):
    """Returns a modified copy of this instance.

    Args:
      nodes: New node names and/or new feature names are accepted. If a feature
        name already exist, then value will be overwritten.
      edges: New edge names and/or new feature names are accepted. If an edge
        name is given, the adjacency list will be overwritten.
      schema: New edges must be amended here.
    """
    updated_nodes = _copy_features(self.nodes)
    updated_edges = {k: (endpoints, dict(feats))
                     for k, (endpoints, feats) in self.edges.items()}
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
        nodes=updated_nodes, edges=updated_edges, schema_=updated_schema)

  def add_pooling(
      self,
      engine,
      graph_features = None):
    """Amends one virtual node 'g' (if not present) to connect to all nodes.

    Args:
      engine: Compute Engine to create edge endpoints. Node 'g' will be
        connected to every node from every node set.
      graph_features: Features to add at the graph level, specifically, to tie
        to node set 'g'.

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
      first_feature = list(features.values())[0]
      num_nodes = engine.shape(first_feature)[0]
      edge_name = f'g_{node_name}'
      g_edges[edge_name] = ((engine.zeros([num_nodes], 'int32'),
                             engine.range(num_nodes, 'int32')), {})
      g_edges_schema[edge_name] = ('g', node_name)

    return self.update(
        nodes={'g': graph_features}, edges=g_edges, schema=g_edges_schema)

  def get_num_nodes(
      self, engine, node_name):
    features = self.nodes.get(node_name, {})
    if not features:
      return 0
    first_feature = list(features.values())[0]
    if first_feature.shape[0] is None:
      return engine.shape(first_feature)[0]
    return first_feature.shape[0]

  def adj(self, engine, edge_name,
          values = None):
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
    col_indices, row_indices = self.edges[edge_name][0]
    col_node_name, row_node_name = self.schema[edge_name]
    return sd.SparseMatrix(
        engine, indices=(row_indices, col_indices),
        dense_shape=(self.get_num_nodes(engine, row_node_name),
                     self.get_num_nodes(engine, col_node_name)),
        values=values)

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

  def incidence(self, engine,
                edge_name,
                endpoint_index = 0):
    """Node to edge incidence matrix."""
    endpoint_ids = self.edges[edge_name][0][endpoint_index]
    endpoint_name = self.schema[edge_name][endpoint_index]
    num_edges = engine.shape(endpoint_ids)[0]
    num_nodes = self.get_num_nodes(engine, endpoint_name)
    return sd.SparseMatrix(
        engine, indices=(engine.range(num_edges, dtype='int32'), endpoint_ids),
        dense_shape=(num_edges, num_nodes),
        values=None)

  @classmethod
  def schema_names_as_dict_keys(cls, schema):
    return {k: tuple([{endpoint: 0} for endpoint in v])
            for k, v in schema.items()}

  @classmethod
  def schema_names_as_strings(cls, schema):
    return {k: tuple([list(endpoint.keys())[0] for endpoint in v])
            for k, v in schema.items()}


def combine_graph_structs(
    engine,
    *graph_structs):
  """Combines multiple GraphStructs into one with multiple components."""
  # node set name -> feature name -> list of features.
  all_node_feats = collections.defaultdict(
      lambda: collections.defaultdict(list))
  # edge name -> feature name -> list of features.
  all_edge_feats = collections.defaultdict(
      lambda: collections.defaultdict(list))
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
      while len(edge_endpoints[edge_name]) < len(endpoints):
        edge_endpoints[edge_name].append([])
      for i, (endpoint, endpoint_node_name) in enumerate(
          zip(endpoints, g_schema[edge_name])):
        edge_endpoints[edge_name][i].append(
            endpoint + node_offsets[endpoint_node_name])
      # Add features.
      for feature_name, feature in features.items():
        all_edge_feats[edge_name][feature_name].append(feature)
    #
    # Add node features.
    for node_name, features in g.nodes.items():
      num_nodes = -1
      for feature_name, feature in features.items():
        all_node_feats[node_name][feature_name].append(feature)
        if num_nodes == -1:
          num_nodes = engine.shape(feature)[0]
        else:
          engine.assert_equal(num_nodes, engine.shape(feature)[0])
      #
      if isinstance(num_nodes, int) and num_nodes == -1:
        num_nodes = 0
      node_offsets[node_name] += num_nodes

  edges = {}
  for edge_name in list(edge_endpoints.keys()):
    edge_features = {}
    for feature_name in list(all_edge_feats[edge_name].keys()):
      edge_features[feature_name] = engine.concat(
          all_edge_feats[edge_name][feature_name], axis=0)
    #
    endpoints = [engine.concat(el, axis=0) for el in edge_endpoints[edge_name]]
    edges[edge_name] = (tuple(endpoints), edge_features)

  nodes = {}
  for node_name in list(all_node_feats.keys()):
    node_features = {}
    for feature_name in list(all_node_feats[node_name].keys()):
      node_features[feature_name] = engine.concat(
          all_node_feats[node_name][feature_name], axis=0)
    nodes[node_name] = node_features

  return GraphStruct.new(nodes=dict(nodes), edges=dict(edges), schema=schema)


def _copy_features(features):
  # 2-level deep copy.
  return {k: dict(v) for k, v in (features or {}).items()}


def are_graphs_exactly_equal(
    engine, g1, g2):
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

  All sizes node-set (features) and edge-set (features and adjacency list)
  """

  def __init__(self, engine, slack = 1.0):
    # `('edge'|'node', NodeOrEdgeName) -> target size`
    # where `target size` is maximum observed size for node (or edge) set, plus
    # one, plus slack-times-std of observed sizes.
    self.sizes: dict[tuple[str, str], int] = {}
    self.slack = slack
    self._engine = engine

  def calculate_pad_statistics(
      self, examples, num_steps = 100):
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
      assert isinstance(graph, GraphStruct)
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

    self.sizes = {k: int(1 + max(v) + self.slack * np.std(v))
                  for k, v in sizes.items()}

  def pad_graph(self, graph):
    """Pads node-sets and edge-sets, with zeros, to max-seen during `calc..`.

    This function is useful for running on TPU hardware.

    Args:
      graph: contains any number of nodes and edges.

    Returns:
      graph with deterministic number of nodes and edges. See class docstring.
    """
    if not self.sizes:
      raise ValueError(
          'No statistics have been initialized. '
          'Perhaps you forgot to invoke "calculate_pad_statistics"?')
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
        feature = feature[:desired_size]  # if `is_oversized`.
        pad = self._engine.maximum(
            desired_size - self._engine.shape(feature)[0], 0
        )
        zeros = e.zeros(
            tuple([pad] + list(feature.shape[1:])), dtype=feature.dtype
        )
        padded_feature = e.concat([feature, zeros], axis=0)
        padded_feature = e.reshape(
            padded_feature, [desired_size] + list(padded_feature.shape[1:]))
        padded_features[feature_name] = padded_feature

      nodes[node_name] = padded_features

    for edge_name, (edge_endpoints, features) in graph.edges.items():
      padded_features = {}
      padded_endpoints = []
      desired_size = self.sizes[('edges', edge_name)]
      current_size = e.shape(edge_endpoints[0])[0]

      pad = e.maximum(desired_size - current_size, 0)
      e.assert_greater(pad, -1)

      for feature_name, feature in features.items():
        feature = feature[:desired_size]  # if `is_oversized`.
        zeros = e.zeros(
            tuple([pad] + list(feature.shape[1:])), dtype=feature.dtype
        )
        padded_feature = e.concat([feature, zeros], axis=0)
        padded_feature = e.reshape(
            padded_feature, [desired_size] + padded_feature.shape[1:]
        )
        padded_features[feature_name] = padded_feature

      edge_endpoints = [node_ids[:desired_size] for node_ids in edge_endpoints]
      # [[src1_is_valid, src2_is_valid, ...], [tgt1_is_valid, ...]]
      valid = e.cast(
          [
              ids < self.sizes[('nodes', node_name)]
              for ids, node_name in zip(edge_endpoints, schema[edge_name])
          ],
          dtype=bool,
      )
      valid = e.reduce_all(valid, axis=0)

      for node_ids, node_name in zip(edge_endpoints, schema[edge_name]):
        # Universe size (e.g., of source or target).
        max_endpoint = self.sizes[('nodes', node_name)] - 1
        node_ids = node_ids[:desired_size]
        node_ids = e.boolean_mask(node_ids, valid)
        pad = desired_size - e.shape(node_ids)[0]  # Need only to compute once.

        padded_ids = e.concat(
            [node_ids, e.ones((pad), dtype=node_ids.dtype) * max_endpoint],
            axis=0,
        )
        padded_ids = e.reshape(padded_ids, [desired_size])
        padded_endpoints.append(padded_ids)

      edges[edge_name] = (tuple(padded_endpoints), padded_features)

    graph = GraphStruct.new(nodes=nodes, edges=edges, schema=schema)
    return graph


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

  def __init__(self):
    self._sizes = collections.defaultdict(list)
    self._edges = {}
    self._features = collections.defaultdict(list)
    self.schema = None
    self._cumsum_sizes: dict[tuple[str, str], np.ndarray]|None = None

  @property
  def features(self):
    """Internal representation: Concatenation of all graph features."""
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
    """show the error."""
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

    return self._cumsum_sizes[key][index]

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
    with open_file(filename, 'wb') as f:
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
      self,
      npz_bytes,
      to_device_fn = np.array,
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
        self.schema = {edge_set_name: tuple(endpoints)
                       for edge_set_name, endpoints in self.schema.items()}

  def load_from_file(
      self, filename, to_device_fn = np.array
  ):
    """Loads DB from a numpy file saved with `save()`."""
    self.load_from_bytes(open_file(filename, 'rb').read(), to_device_fn)

  @classmethod
  def from_sharded_files(
      cls,
      file_prefix,
      to_device_fn = np.array,
  ):
    """Load from sharded files saved with `save_sharded()`."""
    filenames = glob.glob(f'{file_prefix}-*')
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
      cls,
      npz_bytes,
      to_device_fn = np.array,
  ):
    """Recovers instance saved by get_npz_bytes()."""
    db = InMemoryDB()
    db.load_from_bytes(npz_bytes, to_device_fn=to_device_fn)
    return db

  def add(self, g):
    """Adds a graph to the database."""
    if self._cumsum_sizes is not None:
      raise ValueError('You already called finalize().')
    if self.schema is None:
      self.schema = g.schema
    else:
      assert self.schema == g.schema
    for ns_name, feats in g.nodes.items():
      size = None
      for feat_name, feat_val in feats.items():
        self._features[('n', ns_name, feat_name)].append(feat_val)
        if size is None:
          size = feat_val.shape[0]
        else:
          if size != feat_val.shape[0]:
            raise ValueError(f'Features for node set {ns_name} have different '
                             f'sizes: {size} != {feat_val.shape[0]}')
      self._sizes[('n', ns_name)].append(size)
    for es_name, (endpoints, feats) in g.edges.items():
      size = None
      if es_name not in self._edges:
        self._edges[es_name] = [[] for _ in range(len(endpoints))]
      for i, endpoint in enumerate(endpoints):
        if size is None:
          size = endpoint.shape[0]
        else:
          if size != endpoint.shape[0]:
            raise ValueError(f'Endpoint lists for edge set {es_name} differ in '
                             f'length: {size} != {endpoint.shape[0]}')
        self._edges[es_name][i].append(np.array(endpoint, 'int32'))
      for feat_name, feat_val in feats.items():
        self._features[('e', es_name, feat_name)].append(feat_val)
        if size is None:
          size = feat_val.shape[0]
        else:
          if size != feat_val.shape[0]:
            raise ValueError(f'Features for edge set {es_name} have different '
                             f'sizes: {size} != {feat_val.shape[0]}')
      self._sizes[('e', es_name)].append(size)

  def finalize(self, to_device_fn = np.array):
    """Enables `get_item()` invocations, marking end of `add()` invocations."""
    if self._cumsum_sizes is not None:
      raise ValueError('You already called finalize().')
    self._cumsum_sizes = {}
    for k in list(self._sizes):
      self._sizes[k] = to_device_fn(self._sizes[k])
      self._cumsum_sizes[k] = to_device_fn(
          np.concatenate([[0], np.cumsum(self._sizes[k], axis=0)], axis=0))
    for k in list(self._edges):
      for i in range(len(self._edges[k])):
        self._edges[k][i] = to_device_fn(
            np.concatenate(self._edges[k][i], axis=0))
    for k in list(self._features):
      self._features[k] = to_device_fn(
          np.concatenate(self._features[k], axis=0))

  def get_item(self, i):
    """Returns a GraphStruct for the given index `0 <= i < self.size."""
    if self._cumsum_sizes is None:
      raise ValueError('You have not yet called finalize().')
    node_features = collections.defaultdict(dict)
    edge_features = collections.defaultdict(dict)
    for (n_or_e, name, feat_name), feat_val in self._features.items():
      start_idx = self._cumsum_sizes[(n_or_e, name)][i]
      end_idx = self._cumsum_sizes[(n_or_e, name)][i+1]
      if n_or_e == 'n':
        node_features[name][feat_name] = feat_val[start_idx:end_idx]
      elif n_or_e == 'e':
        edge_features[name][feat_name] = feat_val[start_idx:end_idx]
    edges = {}
    for name, endpoints in self._edges.items():
      start_idx = self._cumsum_sizes[('e', name)][i]
      end_idx = self._cumsum_sizes[('e', name)][i+1]
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
