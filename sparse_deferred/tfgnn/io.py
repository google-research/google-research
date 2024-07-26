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

"""Utility functions for importing from GraphTensor, without importing tfgnn."""

from typing import Any

import tensorflow as tf
from sparse_deferred import tf as sdtf
from sparse_deferred.implicit import matrix
from sparse_deferred.structs import graph_struct


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
    example: tf.Example instances, which has features with names like
      "{nodes, context, edges}/feature_name", that are exported by tfgnn per:
      https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/data_prep.md
    tfgnn_graph_schema: instance of `GraphSchema` that describes the feature
      shapes, data-types, and identifies which endpoint-types each edge-set
      connects.
    engine: If not given, tensorflow ComputeEngine will be assumed. If you are
      running under a different compute library (e.g., JAX or numpy), provide
      the appropriate `ComputeEngine`.
  """
  if engine is None:
    engine = sdtf.engine

  schema = {k: (es.source, es.target)
            for k, es in tfgnn_graph_schema.edge_sets.items()}

  graph_features = {}
  nodes = {}
  edges = {}
  for feature_name, feature_value in example.features.feature.items():
    if feature_name.startswith(_SST_PREFIX_GRAPH):
      graph_features[feature_name[len(_SST_PREFIX_GRAPH):]] = _read_feature(
          tfgnn_graph_schema.context.features[feature_name],
          feature_value, engine)
    elif feature_name.startswith(_SST_PREFIX_NODES):
      node_name, node_feat_name = (
          feature_name[len(_SST_PREFIX_NODES):].split('.', 2))
      if node_name not in nodes:
        nodes[node_name] = {}
      nodes[node_name][node_feat_name] = _read_feature(
          tfgnn_graph_schema.node_sets[node_name].features[node_feat_name],
          feature_value, engine)
    elif feature_name.startswith(_SST_PREFIX_EDGES):
      edge_name, edge_feat_name = feature_name[
          len(_SST_PREFIX_EDGES):].split('.', 2)
      if edge_name not in edges:
        source_ids = (example.features.feature[
            f'{_SST_PREFIX_EDGES}{edge_name}.#source'].int64_list.value)
        target_ids = (example.features.feature[
            f'{_SST_PREFIX_EDGES}{edge_name}.#target'].int64_list.value)
        source_ids = tf.constant(source_ids, dtype=tf.int32)
        target_ids = tf.constant(target_ids, dtype=tf.int32)
        edges[edge_name] = ((source_ids, target_ids), {})
      if edge_feat_name in ('#source', '#target', '#size'):
        pass  # Already processed, above. #size is not important.
      else:
        edges[edge_name][1][edge_feat_name] = _read_feature(
            tfgnn_graph_schema.edge_sets[edge_name].features[edge_feat_name],
            feature_value, engine)
    else:
      raise ValueError(f'Unsupported feature: {feature_name}')

  return GraphStruct.new(
      nodes=nodes, edges=edges, schema=schema
  ).add_pooling(engine, graph_features)


def graph_struct_from_graph_tensor(
    graph_tensor,  # Any == GraphTensor.
    engine = None):
  """Converts `tfgnn.GraphTensor` to `GraphStruct`."""

  def _convert(tensor):
    """Convert input (e.g., tf.Tensor) to that of `engine` (e.g., JAX.Array)."""
    if engine is None and not isinstance(engine, sdtf.engine.__class__):
      return tensor
    else:
      # TensorFlow -> CPU -> ComputeEngine Hardware (e.g., GPU, TPU, ...)
      return engine.concat([sdtf.engine.to_cpu(tensor)], axis=0)

  def _convert_dict(tensors):
    return {k: _convert(v) for k, v in tensors.items()}

  nodes = {node_set_name: _convert_dict(node_set.features)
           for node_set_name, node_set in graph_tensor.node_sets.items()}
  edges = {n: ((_convert(es.adjacency.source), _convert(es.adjacency.target)),
               _convert_dict(es.features))
           for n, es in graph_tensor.edge_sets.items()}
  schema = {es_name: (es.adjacency.source_name, es.adjacency.target_name)
            for es_name, es in graph_tensor.edge_sets.items()}

  return GraphStruct.new(nodes=nodes, edges=edges, schema=schema).add_pooling(
      engine or sdtf.engine, _convert_dict(graph_tensor.context.features))


def _read_feature(feature_schema,  # tensorflow_gnn graph_schema.Feature
                  feature,
                  engine):
  """Read feature per `feature_schema`, casting to Tensor via `engine`."""
  if feature_schema.dtype == tf.int32.as_datatype_enum:
    tensor = engine.cast(feature.int64_list.value, dtype='int32')
  elif feature_schema.dtype == tf.int64.as_datatype_enum or feature.HasField(
      'int64_list'):
    tensor = engine.cast(feature.int64_list.value, dtype='int64')
  elif feature_schema.dtype == tf.float32.as_datatype_enum or feature.HasField(
      'float_list'):
    tensor = engine.cast(feature.float_list.value, dtype='float32')
  elif feature_schema.dtype == tf.string.as_datatype_enum or feature.HasField(
      'bytes_list'):
    tensor = engine.cast(feature.bytes_list.value, dtype='string')
  else:
    raise ValueError(f'Unsupported dtype: {feature_schema.dtype}')

  shape = [d.size for d in feature_schema.shape.dim]
  return engine.reshape(tensor, [-1] + shape)
