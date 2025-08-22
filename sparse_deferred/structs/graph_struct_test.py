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


class FixedSizePadderTest(tf.test.TestCase):

  def test_padding(self):
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
      ).add_pooling(sdnp.engine, {})

    for i in range(length):
      graph_structs.append(_graph_struct(i))

    padder = graph_struct.FixedSizePadder(
        engine=sdnp.engine, slack=1.5  # Use numpy (as features are numpy!)
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


class InMemoryDBTest(tf.test.TestCase):

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


if __name__ == '__main__':
  tf.test.main()
