# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Tests for graph_io."""

import os
import tempfile
from absl.testing import absltest
from graph_sampler import graph_io
from graph_sampler import molecule_sampler
import igraph


class GraphIoTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    graphs = []

    graph = igraph.Graph([(0, 1), (0, 1), (1, 2)])
    graph.vs['element'] = ['O', 'O+', 'O-']
    graph['imporance'] = 1
    graph['weight'] = 0.5
    graphs.append(graph)

    graph = igraph.Graph([(0, 3), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6)])
    graph.vs['element'] = ['H', 'H', 'F', 'C', 'N', 'H', 'H']
    graph['imporance'] = 0.1
    graph['weight'] = 0.05
    graphs.append(graph)

    self.graphs = graphs

  def test_write_read_graphs(self):
    filename = tempfile.NamedTemporaryFile(delete=False).name

    # Write a file.
    stats = dict(summary='Some summary', quality=100.0)
    with open(filename, 'w') as f:
      for graph in self.graphs:
        graph_io.write_graph(graph, f)
      graph_io.write_stats(stats, f)

    # Check we can recover the data.
    recovered_stats = graph_io.get_stats(filename)
    recovered_graphs = []
    with open(filename, 'r') as f:
      for graph in graph_io.graph_reader(f):
        recovered_graphs.append(graph)

    self.assertEqual(stats, recovered_stats)
    self.assertEqual(len(self.graphs), len(recovered_graphs))
    for g1, g2 in zip(self.graphs, recovered_graphs):
      self.assertTrue(molecule_sampler.is_isomorphic(g1, g2))

    os.remove(filename)

  def test_invalid_graph_file(self):
    filename = tempfile.NamedTemporaryFile(delete=False).name
    with open(filename, 'w') as f:
      for graph in self.graphs:
        graph_io.write_graph(graph, f)

    with self.assertRaisesRegex(AssertionError, 'is not a stats graph'):
      graph_io.get_stats(filename)


if __name__ == '__main__':
  absltest.main()
