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

import numpy as np
import tensorflow as tf

from sparse_deferred import np as sdnp
from sparse_deferred import tf as sdtf
from sparse_deferred.nn import edges
from sparse_deferred.structs import graph_struct

#     1 <-- 0  -> 2
#      ^   |    /
#       \  |  /
#        \ v v
#          3
_GRAPH = graph_struct.GraphStruct.new(
    nodes={
        'n': {
            'f': np.expand_dims(np.arange(4, dtype='float32'), -1),
        }
    },
    edges={
        'e': (
            (np.array([0, 0, 0, 2, 3]), np.array([1, 2, 3, 3, 1])),
            {'f': 10 + np.expand_dims(np.arange(5, dtype='float32'), -1)},
        )
    },
)

_BI_DIR_GRAPH = _GRAPH.update(
    edges={
        'rev_e': (
            (_GRAPH.edges['e'][0][1], _GRAPH.edges['e'][0][0]),
            {'f': _GRAPH.edges['e'][1]['f'] * -1},
        )
    },
    schema={'rev_e': ('n', 'n')},
)


class EdgesTest(tf.test.TestCase):

  def test_map_nodes_to_incident_edges(self):
    edge_concat_features = edges.map_nodes_to_incident_edges(
        sdnp.engine, _GRAPH, 'e'
    )

    self.assertAllEqual(
        edge_concat_features.shape,
        # 5 edges. Each edge gets node features from 2 nodes and edge features.
        # Every feature is 1-dim (therefore total is 3).
        (5, 3),
    )

    self.assertAllEqual(  # Edge features come first.
        edge_concat_features[:, 0], 10 + np.arange(5, dtype='float32')
    )

    self.assertAllEqual(  # Source node features come next.
        edge_concat_features[:, 1], [0, 0, 0, 2, 3]
    )

    self.assertAllEqual(  # Target node features come last.
        edge_concat_features[:, 2], [1, 2, 3, 3, 1]
    )

  def test_combine_node_features(self):
    edge_concat_features = edges.map_nodes_to_incident_edges(
        sdnp.engine, _BI_DIR_GRAPH, 'e'
    )
    rev_edge_concat_features = edges.map_nodes_to_incident_edges(
        sdnp.engine, _BI_DIR_GRAPH, 'rev_e'
    )
    cache_node_input = {}

    def custom_node_layer(node_input):
      cache_node_input['call'] = node_input

    edges.combine_node_features(
        sdnp.engine,
        _BI_DIR_GRAPH,
        'n',
        [('e', edge_concat_features), ('rev_e', rev_edge_concat_features)],
        'f',
        node_layer=custom_node_layer,
    )

    self.assertLen(cache_node_input, 1)
    self.assertIn('call', cache_node_input)

    node_input = cache_node_input['call']

    self.assertAllEqual(
        node_input.node_features,
        np.expand_dims(np.arange(4, dtype='float32'), -1),
    )

    edge_features = node_input.edge_features
    self.assertLen(edge_features, 4)

    # pylint: disable=bad-whitespace
    self.assertAllEqual(
        edge_features[0],
        # from edge set 'e' onto source nodes
        np.array([
            [11.0, 0.0, 2.0],  # Node 0 averages edge features (10, 11, 12)
            # and its own features (0, 0, 0).
            # and its neighbors' features (1, 2, 3).
            [0.0, 0.0, 0.0],  # node is never a source.
            [13.0, 2.0, 3.0],
            [14.0, 3.0, 1.0],
        ]),
    )
    self.assertAllEqual(
        edge_features[1],
        # from edge set 'e' onto target nodes
        np.array([
            [0.0, 0.0, 0.0],  # Node 0 is never a target.
            [12.0, 1.5, 1.0],
            [11.0, 0.0, 2.0],
            [12.5, 1.0, 3.0],
        ]),
    )
    self.assertAllEqual(
        edge_features[2],
        np.array([
            [0.0, 0.0, 0.0],
            [-12.0, 1.0, 1.5],
            [-11.0, 2.0, 0.0],
            [-12.5, 3.0, 1.0],
        ]),
    )
    self.assertAllEqual(
        edge_features[3],
        np.array([
            [-11.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [-13.0, 3.0, 2.0],
            [-14.0, 1.0, 3.0],
        ]),
    )

  def test_smoke_integration(self):
    edge_features = edges.map_nodes_to_incident_edges(
        sdtf.engine, _GRAPH, 'e', edge_layer=edges.concat_features
    )

    dense_layer_fn = tf.keras.layers.Dense(10)
    dense_layer_fn_2 = tf.keras.layers.Dense(32)

    # Map through fully-connected layer. (dense_layer_fn : Tensor -> Tensor).
    edge_features = dense_layer_fn(edge_features)

    # Node reads from edges.
    node_features = edges.combine_node_features(
        sdtf.engine,
        _GRAPH,
        'n',
        [('e', edge_features)],
        node_layer=edges.concat_features,
    )

    # Map through another fully-connected layer.
    node_features = dense_layer_fn_2(node_features)

    # You can re-insert the node features into the graph.
    graph = _GRAPH.update(nodes={'n': {'f': node_features}})
    print(graph)


if __name__ == '__main__':
  tf.test.main()
