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

import tensorflow as tf
from sparse_deferred.algorithms import virtualize
import sparse_deferred.tf as sdtf

GraphStruct = virtualize.graph_struct.GraphStruct


class VirtualizeTest(tf.test.TestCase):

  def test_virtualize(self):
    g = GraphStruct.new(
        nodes={
            'a': {'fa': tf.constant([10, 11, 12])},
            'b': {'fb': tf.constant([[20, 40], [21, 42]])},
        },
        # Edges are:
        #   ab:
        #     a0 -> b1
        #     a2 -> b0
        #   ba:
        #     b0 -> a1
        #   bb:
        #     b1 -> b0
        edges={
            'ab': (tf.constant([[0, 2], [1, 0]]), {}),
            'ba': (tf.constant([[0], [1]]), {}),
            'bb': (tf.constant([[1], [0]]), {}),
        },
        schema={
            'ab': ('a', 'b'),
            'ba': ('b', 'a'),
            'bb': ('b', 'b'),
        },
    ).add_pooling(sdtf.engine, {'id': tf.constant([7])})

    virtual_g = virtualize.virtualize(g, sdtf.engine)
    num_nonpooling_edges = [edge_name for edge_name in virtual_g.edges
                            if not edge_name.startswith('g_')]
    self.assertSetEqual(set(num_nonpooling_edges), {'a', 'b', 'virtual'})
    # g for graph-level pooling (since we added it in input `g`)
    self.assertSetEqual(set(virtual_g.nodes), {'a', 'b', 'virtual', 'g'})
    self.assertDictEqual(virtual_g.nodes['g'], g.nodes['g'])
    self.assertDictEqual(virtual_g.schema, {
        'a': ('a', 'virtual'),
        'b': ('b', 'virtual'),
        'virtual': ('virtual', 'virtual'),
        'g_a': ('g', 'a'),
        'g_b': ('g', 'b'),
        'g_virtual': ('g', 'virtual'),
    })
    # Self edges.
    self.assertAllEqual(virtual_g.edges['a'][0][0], tf.constant([0, 1, 2]))
    self.assertAllEqual(virtual_g.edges['a'][0][1], tf.constant([0, 1, 2]))
    self.assertAllEqual(virtual_g.edges['b'][0][0], tf.constant([0, 1]))
    self.assertAllEqual(virtual_g.edges['b'][0][1], tf.constant([3, 4]))

    # Virtual edges.
    self.assertAllEqual(
        virtual_g.edges['virtual'][0],
        tf.transpose([
            # a0 -> b1
            [0, 4],
            # a2 -> b0
            [2, 3],
            # b0 -> a1
            [3, 1],
            # b1 -> b0
            [4, 3]
        ]))

    # Features.
    self.assertAllEqual(virtual_g.nodes['a'], g.nodes['a'])
    self.assertAllEqual(virtual_g.nodes['b'], g.nodes['b'])


if __name__ == '__main__':
  tf.test.main()
