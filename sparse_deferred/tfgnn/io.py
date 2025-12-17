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

"""Utility functions for importing from GraphTensor, without importing tfgnn."""

from typing import Any

import numpy as np

from sparse_deferred.implicit import matrix
from sparse_deferred.structs import graph_struct
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.core.framework import types_pb2  # pylint: disable=g-direct-tensorflow-import -- support liteweight JAX usecases

GraphStruct = graph_struct.GraphStruct
Tensor = Any


_SST_PREFIX_NODES = 'nodes/'
_SST_PREFIX_EDGES = 'edges/'
_SST_PREFIX_GRAPH = 'context/'


def graph_struct_from_tf_example(
    example,
    tfgnn_graph_schema,  # tensorflow_gnn.GraphSchema
    engine = None,
):
  """Returns `GraphStruct` constructed from tf.Example and `tfgnn.GraphSchema`.

  Args:
    example: tf.Example instances, which has features with names like "{nodes,
      context, edges}/feature_name", that are exported by tfgnn per:
      https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/data_prep.md
    tfgnn_graph_schema: instance of `GraphSchema` that describes the feature
      shapes, data-types, and identifies which endpoint-types each edge-set
      connects.
    engine: If not given, tensorflow ComputeEngine will be assumed. If you are
      running under a different compute library (e.g., JAX or numpy), provide
      the appropriate `ComputeEngine`.
  """
  if engine is None:
    engine = sdtf_engine()

  schema = {
      k: (es.source, es.target)
      for k, es in tfgnn_graph_schema.edge_sets.items()
  }

  # Mapping from node name to number of nodes
  num_nodes_map = {}
  graph_features = {}
  nodes = {}
  edges = {}
  for feature_name, feature_value in example.features.feature.items():
    if feature_name.startswith(_SST_PREFIX_GRAPH):
      graph_feat_name = feature_name[len(_SST_PREFIX_GRAPH) :]
      feature_schema = tfgnn_graph_schema.context.features[graph_feat_name]
      graph_features[graph_feat_name] = _read_feature(
          feature_schema,
          feature_value,
          engine,
      )
    elif feature_name.startswith(_SST_PREFIX_NODES):
      node_name, node_feat_name = feature_name[len(_SST_PREFIX_NODES) :].split(
          '.', 2
      )
      if node_name not in nodes:
        nodes[node_name] = {}
      if node_feat_name == '#size':
        num_nodes_map[node_name] = feature_value.int64_list.value[0]
        # Need to save num_nodes otherwise node sets without features will
        # be empty.
      else:
        feature_schema = tfgnn_graph_schema.node_sets[node_name].features[
            node_feat_name
        ]
        nodes[node_name][node_feat_name] = _read_feature(
            feature_schema,
            feature_value,
            engine,
            size=(
                example.features.feature[
                    f'nodes/{node_name}.#size'
                ].int64_list.value[0]
            ),
        )
    elif feature_name.startswith(_SST_PREFIX_EDGES):
      edge_name, edge_feat_name = feature_name[len(_SST_PREFIX_EDGES) :].split(
          '.', 2
      )
      if edge_name not in edges:
        source_ids = example.features.feature[
            f'{_SST_PREFIX_EDGES}{edge_name}.#source'
        ].int64_list.value
        target_ids = example.features.feature[
            f'{_SST_PREFIX_EDGES}{edge_name}.#target'
        ].int64_list.value
        source_ids = engine.cast(source_ids, dtype='int32')
        target_ids = engine.cast(target_ids, dtype='int32')
        edges[edge_name] = ((source_ids, target_ids), {})
      if edge_feat_name not in ('#source', '#target', '#size'):
        # Features #source and #target are already processed.
        # #size is not important.
        edges[edge_name][1][edge_feat_name] = _read_feature(
            tfgnn_graph_schema.edge_sets[edge_name].features[edge_feat_name],
            feature_value,
            engine,
        )

  empty_node_ids = engine.zeros(shape=[0], dtype='int32')
  for k, v in schema.items():
    if k not in edges:
      edges[k] = ((empty_node_ids, empty_node_ids), {})
      # if a node set is not in the graph, add an empty placeholder
      if v[0] not in nodes:
        nodes[v[0]] = _init_node_set(engine, tfgnn_graph_schema, v[0])
        if v[0] not in num_nodes_map:
          num_nodes_map[v[0]] = 0
      if v[1] not in nodes:
        nodes[v[1]] = _init_node_set(engine, tfgnn_graph_schema, v[1])
        if v[1] not in num_nodes_map:
          num_nodes_map[v[1]] = 0

  return GraphStruct.new(nodes=nodes, edges=edges, schema=schema).add_pooling(
      engine,
      graph_features,
      excluded_node_sets=['_readout'],
      num_nodes_map=num_nodes_map,
  )


def graph_struct_from_graph_tensor(
    graph_tensor,  # Any == GraphTensor.
    engine = None,
    stack_edges = False,
):
  """Converts `tfgnn.GraphTensor` to `GraphStruct`.

  Args:
    graph_tensor: TF GNN GraphTensor.
    engine: A sd.ComputeEngine instance
    stack_edges: If False (default) combined edges will have format ((source,
      target), {}). If True will have format (T, {}) where T is a 2D shape like
      [[source], [target]].

  Returns:
    A single GraphStruct instance that combines all graphs.
  """

  def _convert(tensor):
    """Convert input (e.g., tf.Tensor) to that of `engine` (e.g., JAX.Array)."""
    if engine is None or isinstance(engine, sdtf_engine().__class__):
      return tensor
    else:
      # TensorFlow -> CPU -> ComputeEngine Hardware (e.g., GPU, TPU, ...)
      assert isinstance(engine, matrix.ComputeEngine)
      return engine.concat([sdtf_engine().to_cpu(tensor)], axis=0)  # pytype: disable=attribute-error

  def _convert_dict(tensors):
    return {k: _convert(v) for k, v in tensors.items()}

  nodes = {
      node_set_name: _convert_dict(node_set.features)
      for node_set_name, node_set in graph_tensor.node_sets.items()
  }

  def _convert_adjacency(adjacency):
    if stack_edges:
      local_engine = engine or sdtf_engine()
      return local_engine.stack([adjacency.source, adjacency.target], axis=0)
    else:
      return (_convert(adjacency.source), _convert(adjacency.target))

  edges = {
      n: (
          _convert_adjacency(es.adjacency),
          _convert_dict(es.features),
      )
      for n, es in graph_tensor.edge_sets.items()
  }
  schema = {
      es_name: (es.adjacency.source_name, es.adjacency.target_name)
      for es_name, es in graph_tensor.edge_sets.items()
  }

  return GraphStruct.new(nodes=nodes, edges=edges, schema=schema).add_pooling(
      engine or sdtf_engine(), _convert_dict(graph_tensor.context.features)
  )


def _init_node_set(
    engine,
    graph_schema,  # Any == GraphSchema
    node_name,
):
  """Returns an empty node set with its features according to the graph schema."""
  features = {}
  for feature_name, feature_schema in graph_schema.node_sets[
      node_name
  ].features.items():
    if feature_schema.dtype == types_pb2.DT_INT32:
      features[feature_name] = engine.zeros(shape=[0], dtype='int32')
    elif feature_schema.dtype == types_pb2.DT_INT64:
      features[feature_name] = engine.zeros(shape=[0], dtype='int64')
    elif feature_schema.dtype == types_pb2.DT_FLOAT:
      features[feature_name] = engine.zeros(shape=[0], dtype='float32')
    elif feature_schema.dtype == types_pb2.DT_DOUBLE:
      features[feature_name] = engine.zeros(shape=[0], dtype='float32')
    elif feature_schema.dtype == types_pb2.DT_BOOL:
      features[feature_name] = engine.zeros(shape=[0], dtype='bool')
    elif feature_schema.dtype == types_pb2.DT_STRING:
      features[feature_name] = engine.zeros(shape=[0], dtype='str')
    else:
      raise ValueError(
          f'Unsupported dtype: {feature_schema.dtype} for feature'
          f' {feature_name}'
      )
  return features


def _read_feature(
    feature_schema,  # tensorflow_gnn graph_schema.Feature
    feature,
    engine,
    size = None,
):
  """Read feature per `feature_schema`, casting to Tensor via `engine`."""

  # For type of input format (e.g. int64_list),; check for a compatible
  # types_pb2 (e.g., int32, and bool are compatible with int64_list). If not
  # compatible output format is found, returns the input format directly (e.g.,
  # int64_list  is an array of int64).
  if feature.HasField('int64_list'):
    value = feature.int64_list.value
    if feature_schema.dtype == types_pb2.DT_INT32:
      tensor = engine.cast(value, dtype='int32')
    elif feature_schema.dtype == types_pb2.DT_BOOL:
      tensor = engine.cast(value, dtype='bool')
    else:
      tensor = engine.cast(value, dtype='int64')
  elif feature.HasField('float_list'):
    value = feature.float_list.value
    if feature_schema.dtype == types_pb2.DT_DOUBLE:
      tensor = engine.cast(value, dtype='float64')
    else:
      tensor = engine.cast(value, dtype='float32')
  elif feature.HasField('bytes_list'):
    value = feature.bytes_list.value
    tensor = engine.cast(value, dtype='string')
  else:
    # Missing values
    if feature_schema.dtype == types_pb2.DT_STRING:
      tensor = engine.zeros(shape=[], dtype='string')
    else:
      raise ValueError(
          f'No missing value for {feature!r} with schema {feature_schema!r} of'
          f' type {type(feature_schema)}.'
      )

  shape = [d.size for d in feature_schema.shape.dim]
  # some embedding features have shape [-1]
  if shape == [-1]:
    if size is None:
      raise ValueError("size is None, shape is [-1] - can't infer size.")
    return engine.reshape(tensor, [size, -1])
  else:
    return engine.reshape(tensor, [-1] + shape)


def graph_struct_to_tf_example(
    graph, engine = None
):
  """Returns `tf.Example` constructed from `GraphStruct`.

  Args:
    graph: `GraphStruct` to convert to `tf.Example`.
    engine: If not given, tensorflow ComputeEngine will be assumed. If you are
      running under a different compute library (e.g., JAX or numpy), provide
      the appropriate `ComputeEngine`.
  """
  example = example_pb2.Example()
  for node_set_name, features in graph.nodes.items():
    if node_set_name == 'g':
      prefix = 'context/'
    else:
      prefix = f'{_SST_PREFIX_NODES}{node_set_name}.'
      node_size = graph.get_num_nodes(engine, node_set_name)
      (
          example.features.feature[f'{prefix}#size'].int64_list.value.append(
              node_size
          )
      )
    _write_features(example, prefix, features)

  for edge_set_name, (endpoints, features) in graph.edges.items():
    if edge_set_name.startswith('g_'):
      # Ignore pooling edges.
      continue
    num_edges = endpoints[0].shape[0]
    # Add "edges/EdgeSet.#size" feature.
    (
        example.features.feature[
            f'{_SST_PREFIX_EDGES}{edge_set_name}.#size'
        ].int64_list.value.append(num_edges)
    )

    # Add "edges/EdgeSet.#source" and "...#target" features.
    (
        example.features.feature[
            f'{_SST_PREFIX_EDGES}{edge_set_name}.#source'
        ].int64_list.value.extend(np.array(endpoints[0]).flatten())
    )
    (
        example.features.feature[
            f'{_SST_PREFIX_EDGES}{edge_set_name}.#target'
        ].int64_list.value.extend(np.array(endpoints[1]).flatten())
    )

    # Features.
    _write_features(example, f'{_SST_PREFIX_EDGES}{edge_set_name}.', features)

  return example


def _write_features(
    example,
    prefix,
    features,
):
  """Writes `features` to `example` under `prefix`."""
  for feature_name, feature_value in features.items():
    if feature_name == '#size':
      # We already added this feature!
      continue
    feature = example.features.feature[f'{prefix}{feature_name}']
    feature_value = np.array(feature_value)
    if np.issubdtype(feature_value.dtype, np.integer):
      feature.int64_list.value.extend(feature_value.flatten())
    elif np.issubdtype(feature_value.dtype, np.floating):
      feature.float_list.value.extend(feature_value.flatten())
    elif np.issubdtype(feature_value.dtype, str):
      feature.bytes_list.value.extend(
          [f.encode() for f in feature_value.flatten()]
      )
    elif np.issubdtype(feature_value.dtype, bytes) or np.issubdtype(
        feature_value.dtype, np.object_
    ):
      feature.bytes_list.value.extend(feature_value.flatten())
    else:
      raise ValueError(f'Unsupported dtype: {feature_value.dtype}')


def sdtf_engine():
  """Imports and returns `sparse_deferred.tf.engine`."""
  from sparse_deferred import tf as sdtf  # pylint: disable=g-import-not-at-top

  return sdtf.engine


def sdnp_engine():
  """Imports and returns `sparse_deferred.numpy.engine`."""
  from sparse_deferred import np as sdnp  # pylint: disable=g-import-not-at-top

  return sdnp.engine


def graph_struct_from_tf_example_dict(
    example,
    tfgnn_graph_schema,
    engine = None,
):
  """Returns `GraphStruct` constructed from a parsed tf.Example.

  The difference between this and graph_struct_from_tf_example is that this
  function allows passing in an already parsed dictionary of features instead of
  a plain tf.Example.

  This is useful when paired with grain.fast_proto.parse_tf_example_experimental
  that parses a tf.Example into a dictionary of feature: numpy array.
  Here, we build a GraphStruct from this dictionary.

  Using numpy engine by default.

  Args:
    example: tf.Example instances, which has features with names like "{nodes,
      context, edges}/feature_name", that are exported by tfgnn per:
      https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/data_prep.md
    tfgnn_graph_schema: instance of `GraphSchema` that describes the feature
      shapes, data-types, and identifies which endpoint-types each edge-set
      connects.
    engine: If not given, numpy ComputeEngine will be assumed - since grain
      parses tf.Example into numpy arrays by default.
  """
  if engine is None:
    engine = sdnp_engine()

  schema = {
      k: (es.source, es.target)
      for k, es in tfgnn_graph_schema.edge_sets.items()
  }

  # Mapping from node name to number of nodes
  num_nodes_map = {}
  graph_features = {}
  nodes = {}
  edges = {}
  for feature_name, feature_value in example.items():
    if feature_name.startswith(_SST_PREFIX_GRAPH):
      graph_feat_name = feature_name[len(_SST_PREFIX_GRAPH) :]
      feature_schema = tfgnn_graph_schema.context.features[graph_feat_name]
      shape = [d.size for d in feature_schema.shape.dim]
      graph_features[graph_feat_name] = feature_value.reshape([-1] + shape)
    elif feature_name.startswith(_SST_PREFIX_NODES):
      node_name, node_feat_name = feature_name[len(_SST_PREFIX_NODES) :].split(
          '.', 2
      )
      if node_name not in nodes:
        nodes[node_name] = {}
      if node_feat_name == '#size':
        num_nodes_map[node_name] = feature_value[0]
        # Need to save num_nodes otherwise node sets without features will
        # be empty.
      else:
        feature_schema = tfgnn_graph_schema.node_sets[node_name].features[
            node_feat_name
        ]
        shape = [d.size for d in feature_schema.shape.dim]
        nodes[node_name][node_feat_name] = feature_value.reshape([-1] + shape)
    elif feature_name.startswith(_SST_PREFIX_EDGES):
      edge_name, edge_feat_name = feature_name[len(_SST_PREFIX_EDGES) :].split(
          '.', 2
      )
      if edge_name not in edges:
        source_ids = example[f'{_SST_PREFIX_EDGES}{edge_name}.#source']
        target_ids = example[f'{_SST_PREFIX_EDGES}{edge_name}.#target']
        edges[edge_name] = ((source_ids, target_ids), {})
      if edge_feat_name not in ('#source', '#target', '#size'):
        # Features #source and #target are already processed.
        # #size is not important.
        edges[edge_name][1][edge_feat_name] = feature_value

  empty_node_ids = engine.zeros(shape=[0], dtype='int32')
  for k, v in schema.items():
    if k not in edges:
      edges[k] = ((empty_node_ids, empty_node_ids), {})
      # if a node set is not in the graph, add an empty placeholder
      if v[0] not in nodes:
        nodes[v[0]] = _init_node_set(engine, tfgnn_graph_schema, v[0])
        if v[0] not in num_nodes_map:
          num_nodes_map[v[0]] = 0
      if v[1] not in nodes:
        nodes[v[1]] = _init_node_set(engine, tfgnn_graph_schema, v[1])
        if v[1] not in num_nodes_map:
          num_nodes_map[v[1]] = 0

  return GraphStruct.new(nodes=nodes, edges=edges, schema=schema).add_pooling(
      engine,
      graph_features,
      excluded_node_sets=['_readout'],
      num_nodes_map=num_nodes_map,
  )
