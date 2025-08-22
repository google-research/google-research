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

"""Offers functions to map features from nodes to edges and vice versa.

Example, to develop standard graph network layer, where:
 - Each edge feature is the concatenation of
   (its feature, source node features, target_node_features)
   --> fed into one FullyConnected layer.
 - Then each node sets its features to:
   FullyConnected(NodeFeatures, AvgSourceEdgeFeatures, AvgTargetEdgeFeatures)
   where "AvgSourceEdgeFeatures" is the average of all edge features from
   edges that have this node as a source.
   "AvgTargetEdgeFeatures" is the average of all edge features from
   edges that have this node as a target.

then, you can write the code:

```py
graph: graph_struct.GraphStruct = ...
engine = sdnp.engine   # (or of JAX, or of tf).

# Assume `graph` has node-set 'n' and edge-set 'e'.

# Edge features reads from nodes.
edge_features = map_nodes_to_incident_edges(
    engine, graph, 'e', edge_layer=concat_features)

# Map through fully-connected layer. (dense_layer_fn : Tensor -> Tensor).
edge_features = dense_layer_fn(edge_features)

# Node reads from edges.
node_features = combine_node_features(
    engine, graph, 'n', [('e', edge_features)], node_layer=concat_features)

# Map through another fully-connected layer.
node_features = dense_layer_fn_2(node_features)

# You can re-insert the node features into the graph.
graph = graph.update(nodes={'n': {'f': node_features}})
```
"""

from typing import Callable, NamedTuple, Optional, Sequence

import sparse_deferred as sd
from sparse_deferred.structs import graph_struct


class EdgeInput(NamedTuple):
  engine: sd.ComputeEngine
  edge_features: Optional[sd.Tensor]  # (numEdges, ...)
  # For "normal" graphs (where edge connects only 2 nodes), `endpoint_features`
  # will have `len == 2`. Each element has shape (numEdges, ...).
  # endpoint_features[0] defaults* to features of source nodes to the edge.
  # endpoint_features[1] defaults* to features of target nodes to the edge.
  # *if endpoint_ids is called with (0, 1).
  endpoint_features: list[sd.Tensor]


class NodeInput(NamedTuple):
  """Contains features of a node set and all its incident edges."""

  engine: sd.ComputeEngine
  node_features: sd.Tensor  # (numNodes, ...)
  incidence_matrices: list[sd.Matrix]  # Each has shape (numEdges, numNodes).
  preaggregated_edge_features: list[sd.Tensor]  # Each has shape (numEdges, ...)

  @property
  def edge_features(self):
    """Summarizes all edge features into nodes using average.

    In particular, each node gets the *average* of all edge features from edges
    where it is a source or a target.
    If you want to sum the features of edges, then manually invoke
    `aggregate_edge_features`, passing `normalization_fn=lambda adj: adj`.
    """
    return self.aggregate_edge_features()

  def aggregate_edge_features(
      self,
      normalization_fn = lambda adj: adj.normalize_right(),
  ):
    """Aggregates edge features to nodes."""

    aggregated_edge_features = []
    for mat, edge_features in zip(
        self.incidence_matrices, self.preaggregated_edge_features
    ):
      aggregated_edge_features.append(normalization_fn(mat.T) @ edge_features)

    return aggregated_edge_features


def all_features(inputs):
  if isinstance(inputs, NodeInput):
    return [inputs.node_features] + inputs.edge_features
  elif isinstance(inputs, EdgeInput):
    if inputs.edge_features is None:
      return inputs.endpoint_features
    else:
      return [inputs.edge_features] + inputs.endpoint_features
  else:
    raise ValueError(f'Unsupported inputs type: {type(inputs)}')


def concat_features(inputs):
  return inputs.engine.concat(all_features(inputs), axis=-1)


def sum_features(inputs):
  inputs.engine.add_n(all_features(inputs))


def map_nodes_to_incident_edges(
    engine,
    graph,
    edge_set_name,
    edge_feature_name = None,
    edge_layer = concat_features,
    endpoint_ids = (0, 1),
    node_feature_names = None,
):
  """Returns the edge function.

  Args:
    engine: The compute engine.
    graph: Graph containing edge set with name `edge_set_name`.
    edge_set_name: The features of endpoint nodes and the features of these
      edges will be extracted and fed into `edge_layer`.
    edge_feature_name: If None and `len(graph.edges[edge_set_name][1]) == 1`,
      then the only edge feature in `graph.edges[edge_set_name]` will be used.
      If `None` and `len(graph.edges[edge_set_name][1]) != 1`, then `None` will
      be the value of `edge_features` in `EdgeInput`. If given and not found,
      then `None` will be the value of `edge_features` in `EdgeInput`. If given
      and found in `graph.edges[edge_set_name]`, then the value of
      `edge_features` in `EdgeInput` will be
      `graph.edges[edge_set_name][1][edge_feature_name]`.
    edge_layer: Layer that can map `EdgeInput` into a Tensor.
    endpoint_ids: The order of node endpoint features to extract. If == (0, 1),
      then, `endpoint_features` will be array as `[src_feats, tgt_feats]`. You
      must set this manually if your graph is a hypergraph -- e.g. to (0, 1, 2).
    node_feature_names: Node feature names. If given, List must be same size as
  """
  node_set_names = graph.schema[edge_set_name]

  # Accumulate at every edge the features of every connected node.
  node_edge_features = []
  # M x N sparse matrices. Each is NumberOfEdges x NumberOfNodes.
  incident_matrices = [
      graph.incidence(engine, edge_set_name, i) for i in endpoint_ids
  ]
  for i, mat in zip(endpoint_ids, incident_matrices):
    node_set_name = node_set_names[i]
    if node_feature_names:
      node_feature_name = node_feature_names[i]
      if node_feature_name not in graph.nodes[node_set_name]:
        raise ValueError(
            f'Node set {node_set_name} does not have feature '
            f'{node_feature_name}.'
        )
    elif len(graph.nodes[node_set_name]) != 1:
      raise ValueError(
          f'Node set {node_set_name} has multiple features, but '
          'node_feature_names is not set.'
      )
    else:
      node_feature_name = list(graph.nodes[node_set_name].keys())[0]
    node_features = graph.nodes[node_set_name][node_feature_name]
    feats_at_edge = mat @ node_features
    node_edge_features.append(feats_at_edge)

  all_edge_features = graph.edges[edge_set_name][1]
  if edge_feature_name is None:
    if len(all_edge_features) != 1:
      edge_features = None
    else:
      edge_features = list(all_edge_features.values())[0]
  else:
    edge_features = all_edge_features.get(edge_feature_name, None)

  edge_input = EdgeInput(
      engine=engine,
      edge_features=edge_features,
      endpoint_features=node_edge_features,
  )

  return edge_layer(edge_input)


def combine_node_features(
    engine,
    graph,
    node_set_name,
    edge_names_and_features,
    node_feature_name = None,
    node_layer = sum_features,
):
  """Combines features of `node_set_name` with all its given edge features.

  Args:
    engine: Compute engine.
    graph: Graph containing `node_set_name`.
    node_set_name: The node set to combine features of.
    edge_names_and_features: List of `(edge_name, edge_features)`. `edge_name`s
      are only considered if they connect to `node_set_name`.
    node_feature_name: If None and `len(graph.nodes[node_set_name]) == 1`, then
      the only node feature in `graph.nodes[node_set_name]` will be used.
    node_layer: will be called on `NodeInput` with attribute `node_features` set
      to `graph.nodes[node_set_name][node_feature_name]`. and property
      `edge_features` returning a list of edge-features (gathered from
      `edge_names_and_features`) where `node_set_name` is an endpoint. The size
      of this list will be the total number of times that `node_set_name` is an
      endpoint in `edge_names_and_features`. Specifically, a homogeneous
      edge-set will insert two elements on this list: one when the node is a
      source and one when it is a target. If a node has no edges where it is
      source or target, then the corresponding row will be zero.

  Returns:
    `node_layer(NodeInput)`.
  """
  if not node_feature_name:
    if len(graph.nodes[node_set_name]) == 1:
      node_feature_name = list(graph.nodes[node_set_name].keys())[0]
    else:
      raise ValueError(
          f'Node set {node_set_name} has multiple features, but '
          'node_feature_name is not set.'
      )

  node_features = graph.nodes[node_set_name][node_feature_name]
  all_edge_features = []
  incidence_matrices = []
  for edge_name, edge_features in edge_names_and_features:
    for i, edge_node_set_name in enumerate(graph.schema[edge_name]):
      if edge_node_set_name == node_set_name:
        all_edge_features.append(edge_features)
        incidence_matrices.append(graph.incidence(engine, edge_name, i))

  node_input = NodeInput(
      engine=engine,
      node_features=node_features,
      incidence_matrices=incidence_matrices,
      preaggregated_edge_features=all_edge_features,
  )
  return node_layer(node_input)
