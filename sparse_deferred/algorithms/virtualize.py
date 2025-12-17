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

"""Provides mapping from a heterogeneous to a homogeneous graph."""

import sparse_deferred as sd
from sparse_deferred.structs import graph_struct


def virtualize(g,
               engine
               ):
  """Virtualize a graph."""
  # The output graph copies all node sets and features from the input graph.
  nodes = {node_set_name: dict(features)
           for node_set_name, features in g.nodes.items()}

  graph_level_features = nodes.pop('g', None)

  # Determine node-set order (by sort). Count nodes per node set, and find their
  # cumulative sum according to the order (sort). Use this cumulative sum to
  # set renumber all nodes into one list (virtual nodes) eand use this numbering
  # for new edge-set.
  node_set_names = list(sorted(nodes.keys()))
  node_set_index = {name: i for i, name in enumerate(node_set_names)}
  cards = []  # Cardinalities
  cumsum_cards = []  # Cumulative sum of cards
  for node_set_name in node_set_names:
    cards.append(g.get_num_nodes(engine, node_set_name))
    if not cumsum_cards:
      cumsum_cards.append(0)
    else:
      cumsum_cards.append(cards[-2] + cumsum_cards[-1])

  # Create new edge-set combining all edge sets. We give unique ID for every
  # node as [0, 1, .., sum of all node set cardinalities]. This numbering is
  # used in this new edge-set.
  all_edges = ([], [])  # to collect (virtual node -> virtual node) edges.
  for edge_name, (src_name, tgt_name) in sorted(g.schema.items()):
    if 'g' in (src_name, tgt_name):
      continue  # Skip pooling edges -- we will add them later. We need pooling
                # edges also to the virtual nodes.
    edges, features = g.edges[edge_name]
    if features:
      raise ValueError('Virtualization does not (yet) support edge features.')
    src_indices = edges[0]
    tgt_indices = edges[1]
    all_edges[0].append(src_indices + cumsum_cards[node_set_index[src_name]])
    all_edges[1].append(tgt_indices + cumsum_cards[node_set_index[tgt_name]])

  edges = {}
  edges['virtual'] = (
      engine.stack([engine.concat(all_edges[0], axis=0),
                    engine.concat(all_edges[1], axis=0)], axis=0),   # Edges,
      {})                                                            # Features.

  # Add all virtual nodes (one per real node) and all real to virtual edges.
  # Nodes:
  all_zeros_column_vector = engine.zeros((engine.add_n(cards), 1), 'float32')
  nodes['virtual'] = {'_empty': all_zeros_column_vector}
  # Edges:
  for i, node_set_name in enumerate(node_set_names):
    rang = engine.range(cards[i], 'int32')
    # Set to (edges, features):
    edges[node_set_name] = (
        engine.stack([rang, rang + cumsum_cards[i]], axis=0), {})

  new_g = graph_struct.GraphStruct.new(
      nodes=nodes,
      edges=edges,
      schema=(
          {node_set_name: (node_set_name, 'virtual') for node_set_name in nodes}
      ),
  )

  if graph_level_features:
    new_g = new_g.add_pooling(engine, graph_level_features)

  return new_g
