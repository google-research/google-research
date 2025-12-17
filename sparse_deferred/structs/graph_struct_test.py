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

import copy
import os

from absl.testing import parameterized
import networkx as nx
import numpy as np
import tensorflow as tf

from sparse_deferred import np as sdnp
from sparse_deferred import tf as sdtf
from sparse_deferred.structs import graph_struct


class GraphStructTest(tf.test.TestCase):

  def test_graph_struct_new_wo_schema(self):
    """Test the new method can construct properly."""
    gt = graph_struct.GraphStruct.new(
        nodes={'ns1': {'f1': tf.zeros([5, 2, 3]), 'f2': tf.ones([5, 1])}},
        edges={'e': ((tf.constant([0, 1, 2]), tf.constant([2, 3, 4])), {})},
    )
    self.assertEqual(gt.schema, {'e': ('ns1', 'ns1')})

  def test_graph_struct_new_w_schema(self):
    """Test the `new` method can construct properly."""
    gt = graph_struct.GraphStruct.new(
        nodes={
            'ns1': {'f1': tf.zeros([5, 2, 3]), 'f2': tf.ones([5, 1])},
            'ns2': {'f1': tf.zeros([6, 2, 3]), 'f2': tf.ones([6, 1])},
        },
        edges={'e': ((tf.constant([0, 1, 2]), tf.constant([2, 3, 4])), {})},
        schema={'e': ('ns1', 'ns2')},
    )

    self.assertEqual(gt.schema, {'e': ('ns1', 'ns2')})
    self.assertAllEqual(gt.nodes['ns1']['f1'], tf.zeros([5, 2, 3]))
    self.assertAllEqual(gt.nodes['ns2']['f2'], tf.ones([6, 1]))
    gt2 = gt.update(nodes={'ns2': {'f2': tf.zeros([6, 1])}})
    self.assertAllEqual(gt2.nodes['ns2']['f2'], tf.zeros([6, 1]))
    # `gt` should be unmutated.
    self.assertAllEqual(gt.nodes['ns2']['f2'], tf.ones([6, 1]))

  def test_add_pooling_with_test(self):
    g1 = graph_struct.GraphStruct.new(
        nodes={
            'ns1': {'f': np.array([[1.0, 2.0], [3.0, 4.0]])},
            'ns2': {'f': np.zeros([3]) + 5.0},
        },
        edges={'e': ((np.array([0, 1, 1]), np.array([0, 0, 1])), {})},
        schema={'e': ('ns1', 'ns2')},
    ).add_pooling(sdtf.engine, stack_edges=True)

    self.assertAllEqual(g1.edges['g_ns1'][0].shape, [2, 2])
    self.assertAllEqual(g1.edges['g_ns2'][0].shape, [2, 3])

  def test_add_pooling_without_features(self):
    g1 = graph_struct.GraphStruct.new(
        nodes={
            'ns1': {},
            'ns2': {},
        },
        edges={'e': ((np.array([0, 1, 1]), np.array([0, 0, 1])), {})},
        schema={'e': ('ns1', 'ns2')},
    ).add_pooling(
        sdtf.engine, stack_edges=True, num_nodes_map={'ns1': 2, 'ns2': 3}
    )

    self.assertAllEqual(g1.edges['g_ns1'][0].shape, [2, 2])
    self.assertAllEqual(g1.edges['g_ns2'][0].shape, [2, 3])

  def test_add_pooling(self):
    """Test the `add_pooling` method can construct properly."""
    g1 = graph_struct.GraphStruct.new(
        nodes={
            'ns1': {'f': np.array([1.0, 2.0])},
            'ns2': {'f': np.zeros([3]) + 5.0},
        },
        edges={'e': ((np.array([0, 1, 1]), np.array([0, 0, 1])), {})},
        schema={'e': ('ns1', 'ns2')},
    ).add_pooling(sdnp.engine)
    g2 = graph_struct.GraphStruct.new(
        nodes={
            'ns1': {'f': np.array([10.0, 20.0])},
            'ns2': {'f': np.zeros([2]) + 1000.0},
        },
        edges={'e': ((np.array([1, 0]), np.array([0, 1])), {})},
        schema={'e': ('ns1', 'ns2')},
    ).add_pooling(sdnp.engine, {'id': np.array([[5]])})

    # In absence of user-supplied features, node-set 'g' created with feature
    # 'id' = 0.
    self.assertSetEqual(set(g1.nodes['g'].keys()), {'id'})
    self.assertAllEqual(g1.nodes['g']['id'], np.array([[0]]))

    # User-supplied feature.
    self.assertSetEqual(set(g2.nodes['g'].keys()), {'id'})
    self.assertAllEqual(g2.nodes['g']['id'], np.array([[5]]))

    # Schema is updated to include pooling edges.
    self.assertEqual(g1.schema['g_ns1'], ('g', 'ns1'))
    self.assertEqual(g1.schema['g_ns2'], ('g', 'ns2'))
    self.assertEqual(g2.schema['g_ns1'], ('g', 'ns1'))
    self.assertEqual(g2.schema['g_ns2'], ('g', 'ns2'))

    # Singleton 'g' node has edges to all nodes:
    src, tgt = g1.edges['g_ns1'][0]
    self.assertAllEqual(src, np.zeros([g1.nodes['ns1']['f'].shape[0]]))
    self.assertAllEqual(tgt, np.arange(g1.nodes['ns1']['f'].shape[0]))
    src, tgt = g1.edges['g_ns2'][0]
    self.assertAllEqual(src, np.zeros([g1.nodes['ns2']['f'].shape[0]]))
    self.assertAllEqual(tgt, np.arange(g1.nodes['ns2']['f'].shape[0]))
    src, tgt = g2.edges['g_ns1'][0]
    self.assertAllEqual(src, np.zeros([g2.nodes['ns1']['f'].shape[0]]))
    self.assertAllEqual(tgt, np.arange(g2.nodes['ns1']['f'].shape[0]))
    src, tgt = g2.edges['g_ns2'][0]
    self.assertAllEqual(src, np.zeros([g2.nodes['ns2']['f'].shape[0]]))
    self.assertAllEqual(tgt, np.arange(g2.nodes['ns2']['f'].shape[0]))

    # Sanity check, we can "actually" use these edges to do pooling.
    # Lets consider the more complex case: after combining two graphs into one
    # (batch)
    g = graph_struct.combine_graph_structs(sdnp.engine, g2, g1)

    # Symbolic adjacency matrices for pooling and broadcasting.
    # Pool from node set "ns1" -> graph by mean-pooling:
    mean_pooling_adjacency = g.adj(sdnp.engine, 'g_ns1').T.normalize_right()
    # 2 graphs, 2 nodes each in 'ns1'
    self.assertAllEqual(mean_pooling_adjacency.shape, (2, 4))

    # Broadcasts from "g" -> "ns2"
    broadcast_adjacency = g.adj(sdnp.engine, 'g_ns2')
    # 2 graphs. ns2 has 3 and 2 nodes, in graph 1 and graph 2, resp.
    self.assertAllEqual(broadcast_adjacency.shape, (5, 2))

    y = mean_pooling_adjacency @ g.nodes['ns1']['f']  # FAILS
    # Average of ns1 in g2 is 15 and of g1 is 1.5
    self.assertAllEqual(y, np.array([15, 1.5]))
    # Broadcast these values to 'ns2'
    broadcasted = broadcast_adjacency @ y
    self.assertAllEqual(broadcasted, np.array([15, 15, 1.5, 1.5, 1.5]))

  def test_graph_struct_get_outgoing_neighbors(self):
    """Test that the graph can get neighbors."""
    gt = graph_struct.GraphStruct.new(
        nodes={'ns1': {'f1': tf.zeros([5, 2, 3]), 'f2': tf.ones([5, 1])}},
        edges={
            'e': ((tf.constant([2, 3, 4, 4]), tf.constant([0, 1, 0, 2])), {})
        },
    )
    no_neighbor = gt.get_outgoing_neighbors(sdtf.engine, 'e', 0)
    self.assertAllEqual(no_neighbor, tf.constant([]), str(no_neighbor))

    single_neighbor = gt.get_outgoing_neighbors(sdtf.engine, 'e', 2)
    self.assertAllEqual(single_neighbor, tf.constant([0]), str(single_neighbor))

    multiple_neighbors = gt.get_outgoing_neighbors(sdtf.engine, 'e', 4)
    self.assertAllEqual(
        multiple_neighbors, tf.constant([0, 2]), str(multiple_neighbors)
    )


class FixedSizePadderTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='stack_edges_false_np',
          stack_edges=False,
          engine=sdnp.engine,
      ),
      dict(
          testcase_name='stack_edges_true_np',
          stack_edges=True,
          engine=sdnp.engine,
      ),
      dict(
          testcase_name='stack_edges_false_tf',
          stack_edges=False,
          engine=sdtf.engine,
      ),
      dict(
          testcase_name='stack_edges_true_tf',
          stack_edges=True,
          engine=sdtf.engine,
      ),
  )
  def test_padding(self, stack_edges, engine):
    sizes = {
        ('node', 'author'): [5, 20, 3, 25, 40],
        ('node', 'paper'): [50, 30, 35, 40, 52],
        ('node', 'root'): [5, 5, 5, 5, 5],  # Suppose fixed size.
        ('edge', 'cites'): [10, 20, 10, 20, 10],
        ('edge', 'seed'): [5, 5, 5, 5, 5],  # Suppose fixed size.
        ('edge', 'writes'): [100, 60, 70, 80, 100],
    }
    toy_schema = {
        'cites': ('paper', 'paper'),
        'writes': ('author', 'paper'),
        'seed': ('paper', 'root'),
    }
    schema_with_pooling = {
        'g_author': ('g', 'author'),
        'g_paper': ('g', 'paper'),
        'g_root': ('g', 'root'),
    }
    schema_with_pooling.update(toy_schema)
    lengths = list(map(len, sizes.values()))
    assert min(lengths) == max(lengths)  # All lengths must be same.
    length = min(lengths)
    graph_structs: list[graph_struct.GraphStruct] = []

    def _graph_struct(i):
      num_authors = sizes[('node', 'author')][i]
      num_papers = sizes[('node', 'paper')][i]
      num_cites = sizes[('edge', 'cites')][i]
      num_seeds = sizes[('edge', 'seed')][i]
      num_writes = sizes[('edge', 'writes')][i]
      cites_src = np.array(
          np.random.uniform(low=0, high=num_papers - 1, size=[num_cites]),
          dtype='int32',
      )
      cites_tgt = np.array(
          np.random.uniform(low=0, high=num_papers - 1, size=[num_cites]),
          dtype='int32',
      )

      writes_src = np.array(
          np.random.uniform(low=0, high=num_authors - 1, size=[num_writes]),
          dtype='int32',
      )
      writes_tgt = np.array(
          np.random.uniform(low=0, high=num_papers - 1, size=[num_writes]),
          dtype='int32',
      )

      seed_src = np.array(
          np.random.uniform(low=0, high=num_papers - 1, size=[num_seeds]),
          dtype='int32',
      )
      seed_tgt = np.array(
          np.random.uniform(low=0, high=num_seeds - 1, size=[num_seeds]),
          dtype='int32',
      )
      return graph_struct.GraphStruct.new(
          nodes={
              'author': {
                  'embed': np.random.uniform(
                      low=-1.0, high=1.0, size=[num_authors, 64]
                  ),
              },
              'paper': {
                  'embed': np.random.uniform(
                      low=-1.0, high=1.0, size=[num_papers, 128]
                  ),
              },
              'root': {
                  'empty': np.zeros([num_seeds, 0]),
              },
          },
          edges={
              'cites': ((cites_src, cites_tgt), {}),
              'seed': ((seed_src, seed_tgt), {}),
              'writes': ((writes_src, writes_tgt), {}),
          },
          schema=toy_schema,
          engine=engine,
      ).add_pooling(engine, {})

    for i in range(length):
      graph_structs.append(_graph_struct(i))

    padder = graph_struct.FixedSizePadder(
        engine=engine,
        slack=1.5,
        stack_edges=stack_edges,
    )
    padder.calculate_pad_statistics(graph_structs)

    g = _graph_struct(0)
    padded_g = padder.pad_graph(g)

    # Schema should be like toy_schema but with addition of graph-pooling
    # edge-sets (as we invoked `.add_pooling()`).
    self.assertEqual(padded_g.schema, schema_with_pooling)

    self.assertSetEqual(set(padded_g.nodes.keys()), set(g.nodes.keys()))
    self.assertSetEqual(set(padded_g.edges.keys()), set(g.edges.keys()))

    # Verify node features are copied then padded correctly onto `padded_g`.
    for node_name, feats in g.nodes.items():
      padded_feats = padded_g.nodes[node_name]
      self.assertSetEqual(set(padded_feats.keys()), set(feats.keys()))
      for feat_name, value in feats.items():
        padded_value = padded_feats[feat_name]
        size = value.shape[0]  # original size.

        # Zero padding
        self.assertAllEqual(padded_value[:size], value)
        self.assertAllEqual(
            padded_value[size:], np.zeros_like(padded_value[size:])
        )

        batch_sizes = sizes[('node', node_name)]
        # Note: 1.5 is slack given to constructor above!
        expected_padded_size = int(
            1 + np.max(batch_sizes) + 1.5 * np.std(batch_sizes)
        )
        self.assertAllEqual(padded_value.shape[0], expected_padded_size)

    # Verify edges are copied then padded with connections among "virtual nodes"
    for edge_name, ((src, tgt), unused_features) in g.edges.items():
      if edge_name.startswith('g_'):
        continue
      if stack_edges:
        self.assertEqual(padded_g.edges[edge_name][0].shape[0], 2)
      padded_src, padded_tgt = padded_g.edges[edge_name][0]
      batch_sizes = sizes[('edge', edge_name)]
      expected_padded_size = int(
          1 + np.max(batch_sizes) + 1.5 * np.std(batch_sizes)
      )

      self.assertAllEqual(padded_src.shape, [expected_padded_size])
      self.assertAllEqual(padded_tgt.shape, [expected_padded_size])
      orig_size = src.shape[0]
      self.assertAllEqual(padded_src[:orig_size], src)
      self.assertAllEqual(padded_tgt[:orig_size], tgt)

      src_name, tgt_name = toy_schema[edge_name]
      # Virtual node should be the *last* one. Grab any feature tensor (e.g., 0)
      # and get its size.
      virtual_src_node = list(padded_g.nodes[src_name].values())[0].shape[0] - 1
      virtual_tgt_node = list(padded_g.nodes[tgt_name].values())[0].shape[0] - 1
      self.assertAllEqual(
          padded_src[orig_size:],
          np.ones_like(padded_src[orig_size:]) * virtual_src_node,
      )
      self.assertAllEqual(
          padded_tgt[orig_size:],
          np.ones_like(padded_tgt[orig_size:]) * virtual_tgt_node,
      )

  def test_replace_engine(self):
    padder = graph_struct.FixedSizePadder(
        engine=sdnp.engine,
        slack=1.5,
        stack_edges=True,
    )
    new_padder = padder.replace_engine(sdtf.engine)
    self.assertEqual(new_padder.sizes, padder.sizes)
    self.assertEqual(new_padder.slack, padder.slack)
    self.assertEqual(new_padder.stack_edges, padder.stack_edges)

  def test_graph_struct_new_from_nx(self):
    """Test the new_from_nx method can construct properly."""
    nx_graph = nx.Graph()
    nx_graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
    edge_list = np.array(list(nx_graph.edges()), dtype=np.int32)

    # Test basic conversion
    gs = graph_struct.GraphStruct.from_nx(nx_graph, symmetrize=False)
    self.assertEqual(gs.schema, {'edges': ('nodes', 'nodes')})
    sources, targets = gs.edges['edges'][0]
    self.assertAllEqual(sources, edge_list[:, 0])
    self.assertAllEqual(targets, edge_list[:, 1])

    # Test symmetrization
    gs_sym = graph_struct.GraphStruct.from_nx(nx_graph, symmetrize=True)
    sources_sym, targets_sym = gs_sym.edges['edges'][0]
    self.assertAllEqual(
        sources_sym, np.concatenate([edge_list[:, 0], edge_list[:, 1]], axis=0)
    )
    self.assertAllEqual(
        targets_sym, np.concatenate([edge_list[:, 1], edge_list[:, 0]], axis=0)
    )

    # Test dummy feature
    self.assertSetEqual(set(gs.nodes['nodes'].keys()), {'embedding'})
    self.assertAllEqual(
        gs.nodes['nodes']['embedding'], np.ones((3, 1), dtype=np.float32)
    )

    # Test with existing embedding feature
    nx_graph.nodes[0]['embedding'] = np.array([1.0, 2.0, 3.0])
    nx_graph.nodes[1]['embedding'] = np.array([4.0, 5.0, 6.0])
    nx_graph.nodes[2]['embedding'] = np.array([7.0, 8.0, 9.0])
    gs_with_features = graph_struct.GraphStruct.from_nx(
        nx_graph, symmetrize=True, feature_name='embedding'
    )
    self.assertSetEqual(
        set(gs_with_features.nodes['nodes'].keys()), {'embedding'}
    )
    self.assertAllEqual(
        gs_with_features.nodes['nodes']['embedding'],
        np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float32,
        ),
    )

    # test 1D features
    nx_graph.nodes[0]['embedding'] = 1.0
    nx_graph.nodes[1]['embedding'] = 2.0
    nx_graph.nodes[2]['embedding'] = 3.0
    gs_with_features = graph_struct.GraphStruct.from_nx(
        nx_graph, symmetrize=True, feature_name='embedding'
    )
    self.assertSetEqual(
        set(gs_with_features.nodes['nodes'].keys()), {'embedding'}
    )
    self.assertAllEqual(
        gs_with_features.nodes['nodes']['embedding'],
        np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
    )

  def test_graph_struct_to_nx(self):
    """Test the to_nx method can convert properly."""
    nx_graph = G1.to_nx()
    self.assertIsInstance(nx_graph, nx.Graph)
    self.assertEqual(nx_graph.number_of_nodes(), 3)
    self.assertEqual(nx_graph.number_of_edges(), 2)
    self.assertTrue(nx_graph.has_edge(0, 2))
    self.assertTrue(nx_graph.has_edge(2, 0))
    self.assertTrue(nx_graph.has_edge(1, 1))

  @parameterized.parameters(
      {
          'padding_node': 1,
          'padding_edge': 5,
          r'expected_msg': (
              r"Padding for node set 'n1' is insufficient. Required at least 2"
              r' nodes \(including the padding node\), but the padder only'
              r' defines 1.'
          ),
      },
      {
          'padding_node': 5,
          'padding_edge': 1,
          'expected_msg': (
              r"Padding for edge set 'e1' is insufficient. Required at least 2"
              r' edges, but the padder only defines 1.'
          ),
      },
      {
          'padding_node': 2,
          'padding_edge': 2,
          r'expected_msg': None,
      },
  )
  def test_oversized_batch_for_padding(
      self, padding_edge, padding_node, expected_msg
  ):
    padder = graph_struct.FixedSizePadder(
        engine=sdnp.engine, fails_on_oversized_batch=True
    )
    padder.sizes = {
        ('nodes', 'n1'): padding_node,
        ('edges', 'e1'): padding_edge,
    }
    # 2 nodes and 2 edges
    graph = graph_struct.GraphStruct.new(
        nodes={'n1': {'f': np.array([-1, -1])}},
        edges={'e1': (np.array([[0, 0], [0, 1]], dtype='int32'), {})},
    )
    if expected_msg is None:
      # Should not fail
      _ = padder.pad_graph(graph)
    else:
      with self.assertRaisesRegex(ValueError, expected_msg):
        _ = padder.pad_graph(graph)

  def test_save_and_load_stats_to_json(self):
    padder = graph_struct.FixedSizePadder(engine=sdnp.engine)
    padder.sizes = {
        ('nodes', 'n1'): 10,
        ('edges', 'e1'): 20,
        ('nodes', 'n2'): 15,
    }
    temp_dir = self.create_tempdir()
    filename = os.path.join(temp_dir.full_path, 'stats.json')

    padder.save_stats_to_json(filename)
    self.assertTrue(os.path.exists(filename))

    new_padder = graph_struct.FixedSizePadder(engine=sdnp.engine)
    new_padder.load_stats_from_json(filename)

    self.assertEqual(new_padder.sizes, padder.sizes)

  def test_load_stats_from_json_file_not_found(self):
    padder = graph_struct.FixedSizePadder(engine=sdnp.engine)
    with self.assertRaises(IOError):
      padder.load_stats_from_json('non_existent_file.json')

  def test_load_stats_from_json_ensure_backward_compatibility(self):
    """Tests that the JSON serialization is backward compatible.

    This ensures any future changes maintain compatibility with previously
    saved stats files.
    """
    sizes_write = """
    {
        "nodes,n1": 10,
        "edges,e1": 20,
        "nodes,n2": 15
    }
    """
    temp_dir = self.create_tempdir()
    filename = os.path.join(temp_dir.full_path, 'stats.json')
    with open(filename, 'w') as f:
      f.write(sizes_write)
    padder = graph_struct.FixedSizePadder(engine=sdnp.engine)
    padder.load_stats_from_json(filename)
    self.assertEqual(
        padder.sizes,
        {
            ('nodes', 'n1'): 10,
            ('edges', 'e1'): 20,
            ('nodes', 'n2'): 15,
        },
    )

G1 = graph_struct.GraphStruct.new(
    nodes={
        'person': {
            'names': np.array([
                'bryan',
                'sami',
                'mangpo',
            ]),
        },
    },
    edges={
        'friendship': ((np.array([0, 2, 1]), np.array([2, 0, 1])), {}),
    },
)
G2 = graph_struct.GraphStruct.new(
    nodes={
        'person': {
            'names': np.array([
                'mickey mouse',
                'don duck',
            ]),
        },
    },
    edges={
        'friendship': (
            (np.array([], dtype='int32'), np.array([], dtype='int32')),
            {},
        ),
    },
)
G3 = graph_struct.GraphStruct.new(
    nodes={
        'person': {
            'names': np.array([
                'cat',
                'panda',
                'bird',
            ]),
        },
    },
    edges={
        'friendship': ((np.array([0, 1]), np.array([1, 0])), {}),
    },
)


class CombineGraphStructsTest(tf.test.TestCase):

  def test_combine_three_graphs(self):

    combined_g = graph_struct.combine_graph_structs(sdnp.engine, G1, G2, G3)

    self.assertAllEqual(
        combined_g.nodes['person']['names'],
        np.array([
            'bryan',
            'sami',
            'mangpo',
            'mickey mouse',
            'don duck',
            'cat',
            'panda',
            'bird',
        ]),
    )
    source_ids, target_ids = combined_g.edges['friendship'][0]
    self.assertAllEqual(source_ids, np.array([0, 2, 1, 5, 6]))
    self.assertAllEqual(target_ids, np.array([2, 0, 1, 6, 5]))

  def test_combine_three_graphs_stacked_edges(self):
    combined_g = graph_struct.combine_graph_structs(
        sdtf.engine, G1, G2, G3, stack_edges=True
    )
    self.assertAllEqual(combined_g.edges['friendship'][0].shape, [2, 5])


class GraphsExactlyEqualTest(tf.test.TestCase):

  def test_are_graphs_exactly_equal(self):
    self.assertTrue(
        graph_struct.are_graphs_exactly_equal(
            sdnp.engine, G1, copy.deepcopy(G1)
        )
    )
    self.assertFalse(graph_struct.are_graphs_exactly_equal(sdnp.engine, G1, G2))
    self.assertFalse(graph_struct.are_graphs_exactly_equal(sdnp.engine, G1, G3))
    self.assertFalse(
        graph_struct.are_graphs_exactly_equal(
            sdnp.engine,
            G1,
            G1.update(nodes={'person': {'y': np.array([1, 2, 3])}}),
        )
    )


def _make_db_save_on_desk(save_path):
  db = graph_struct.InMemoryDB()
  db.add(G1)
  db.add(G2)
  db.add(G3)
  db.finalize()
  db.save(save_path)


class InMemoryDBTest(tf.test.TestCase, parameterized.TestCase):

  def test_save_and_load(self):
    save_path = os.path.join(self.get_temp_dir(), 'in_memory_db.npz')
    _make_db_save_on_desk(save_path)
    db = graph_struct.InMemoryDB.from_file(save_path)
    self.assertTrue(
        graph_struct.are_graphs_exactly_equal(sdnp.engine, G1, db.get_item(0))
    )
    self.assertTrue(
        graph_struct.are_graphs_exactly_equal(sdnp.engine, G2, db.get_item(1))
    )
    self.assertTrue(
        graph_struct.are_graphs_exactly_equal(sdnp.engine, G3, db.get_item(2))
    )

  def test_bytes_serialization_and_deserialization(self):
    db = graph_struct.InMemoryDB()
    for g in [G1, G2, G3]:
      db.add(g)
    db.finalize()
    npz_bytes = db.get_npz_bytes()
    self.assertIsInstance(npz_bytes, bytes)
    db2 = graph_struct.InMemoryDB.from_bytes(npz_bytes)
    self.assertEqual(3, db.size)
    self.assertTrue(
        graph_struct.are_graphs_exactly_equal(sdnp.engine, G1, db2.get_item(0))
    )
    self.assertTrue(
        graph_struct.are_graphs_exactly_equal(sdnp.engine, G2, db2.get_item(1))
    )
    self.assertTrue(
        graph_struct.are_graphs_exactly_equal(sdnp.engine, G3, db2.get_item(2))
    )

  def test_sharded_save_and_load(self):
    db = graph_struct.InMemoryDB()
    for g in [G1, G2, G3]:
      db.add(g)
    db.finalize()
    save_prefix = os.path.join(self.get_temp_dir(), 'in_memory_db.npz')
    db.save_sharded(save_prefix, batch_size=2)
    self.assertTrue(os.path.exists(save_prefix + '-0-to-2'))  # 2 graphs.
    self.assertTrue(os.path.exists(save_prefix + '-2-to-3'))  # 1 graph.

    db0 = graph_struct.InMemoryDB.from_file(save_prefix + '-0-to-2')
    db1 = graph_struct.InMemoryDB.from_file(save_prefix + '-2-to-3')
    self.assertEqual(2, db0.size)
    self.assertEqual(1, db1.size)
    self.assertTrue(
        graph_struct.are_graphs_exactly_equal(sdnp.engine, G1, db0.get_item(0))
    )
    self.assertTrue(
        graph_struct.are_graphs_exactly_equal(sdnp.engine, G2, db0.get_item(1))
    )
    self.assertTrue(
        graph_struct.are_graphs_exactly_equal(sdnp.engine, G3, db1.get_item(0))
    )

    db_all = graph_struct.InMemoryDB.from_sharded_files(save_prefix)
    self.assertEqual(3, db_all.size)
    self.assertTrue(
        graph_struct.are_graphs_exactly_equal(
            sdnp.engine, G1, db_all.get_item(0)
        )
    )
    self.assertTrue(
        graph_struct.are_graphs_exactly_equal(
            sdnp.engine, G2, db_all.get_item(1)
        )
    )
    self.assertTrue(
        graph_struct.are_graphs_exactly_equal(
            sdnp.engine, G3, db_all.get_item(2)
        )
    )

  def test_make_tf_dataset(self):
    save_path = os.path.join(self.get_temp_dir(), 'in_memory_db.npz')
    _make_db_save_on_desk(save_path)
    tfds = sdtf.graph.InMemoryDB.from_file(save_path).as_tf_dataset()

    graphs = list(tfds)
    self.assertLen(graphs, 3)
    self.assertTrue(
        graph_struct.are_graphs_exactly_equal(sdtf.engine, G1, graphs[0])
    )
    self.assertTrue(
        graph_struct.are_graphs_exactly_equal(sdnp.engine, G2, graphs[1])
    )
    self.assertTrue(
        graph_struct.are_graphs_exactly_equal(sdnp.engine, G3, graphs[2])
    )

  def test_validation_error(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'Edges must define features as a (potentially empty) dict.'
    ):
      # Attempt to construct a graph without specifying empty edge features.
      graph_struct.GraphStruct.new(
          nodes={'ns1': {'f1': tf.zeros([5, 2, 3]), 'f2': tf.ones([5, 1])}},
          edges={'e': (tf.constant([2, 3, 4, 4]), tf.constant([0, 1, 0, 2]))},
          validate=True,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='invalid_src',
          src=np.array([3], dtype='int32'),
          tgt=np.array([0], dtype='int32'),
      ),
      dict(
          testcase_name='invalid_tgt',
          src=np.array([0], dtype='int32'),
          tgt=np.array([3], dtype='int32'),
      ),
  )
  def test_validation_error_invalid_adjacency(self, src, tgt):
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError, 'Condition x > y did not hold.*'
    ):
      _ = graph_struct.GraphStruct.new(
          nodes={
              'ns1': {
                  'f1': np.array([1, 2], dtype='int32'),
                  'f2': np.array([2, 3], dtype='int32'),
              },
              'ns2': {
                  'f3': np.array([3, 4], dtype='int32'),
                  'f4': np.array([4, 5], dtype='int32'),
              },
          },
          edges={'e': ((src, tgt), {})},
          schema={'e': ('ns1', 'ns2')},
          validate=True,
          engine=sdtf.engine,
      )

  def test_flex_schema(self):
    g0 = graph_struct.GraphStruct.new(
        nodes={
            'a': {
                'f1': np.array([1, 2], dtype='int32'),
                'f2': np.array([2, 3], dtype='int32'),
            },
            'b': {
                'f3': np.array([3, 4], dtype='int32'),
            },
        },
        edges={
            'ab': ((np.array([0, 1]), np.array([1, 0])), {}),
        },
        schema={'ab': ('a', 'b')},
    )
    g1 = graph_struct.GraphStruct.new(
        nodes={
            'a': {
                'f2': np.array([20, 30], dtype='int32'),
            },
            'b': {
                'f3': np.array([30, 40], dtype='int32'),
                'f4': np.array([50, 60], dtype='int32'),
            },
        },
        edges={
            'ba': ((np.array([0, 1]), np.array([1, 0])), {}),
        },
        schema={'ba': ('b', 'a')},
    )
    g2 = graph_struct.GraphStruct.new(
        nodes={
            'b': {
                'f5': np.array([[300.0, 3.0], [400.0, 4.0]], dtype='float32'),
            },
        },
        edges={
            'bb': ((np.array([0, 1]), np.array([1, 0])), {}),
        },
        schema={'bb': ('b', 'b')},
    )

    db = graph_struct.InMemoryDB(flex_schema=True)
    db.add(g0)
    db.add(g1)
    db.add(g2)
    db.finalize()

    # The schema should be the union of all above.
    expected_schema = {
        'ab': ('a', 'b'),
        'ba': ('b', 'a'),
        'bb': ('b', 'b'),
    }
    self.assertDictEqual(db.schema, expected_schema)

    gg0 = db.get_item(0)
    gg1 = db.get_item(1)
    gg2 = db.get_item(2)
    self.assertDictEqual(gg0.schema, expected_schema)
    self.assertDictEqual(gg1.schema, expected_schema)
    self.assertDictEqual(gg2.schema, expected_schema)

    # node-set 'a' never saw new features.
    self.assertDictEqual(gg0.nodes['a'], g0.nodes['a'])
    self.assertAllEqual(gg0.edges['ab'][0], g0.edges['ab'][0])
    self.assertAllEqual(gg0.edges['ba'][0], [[], []])  # No 'ba' edges in g0.
    self.assertAllEqual(gg0.edges['bb'][0], [[], []])  # No 'bb' edges in g0.
    self.assertDictEqual(
        gg0.nodes['b'],
        {
            'f3': [3, 4],  # Came from g0.
            'f4': [0, 0],
            'f5': [[0, 0], [0, 0]],  # Not in g0, defaults to zero.
        },
    )
    self.assertAllEqual(gg2.edges['bb'][0], g2.edges['bb'][0])
    self.assertAllEqual(gg2.edges['ba'][0], [[], []])  # No 'ba' edges in g2.
    self.assertDictEqual(gg2.nodes['a'], {'f1': [], 'f2': []})  # Not in g2.
    self.assertDictEqual(
        gg2.nodes['b'],
        {
            'f3': [0, 0],
            'f4': [0, 0],
            'f5': g2.nodes['b']['f5'],
        },
    )


if __name__ == '__main__':
  tf.test.main()
