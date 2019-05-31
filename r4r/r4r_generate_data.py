# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Script to build R4R data from the original R2R data.

Link to the original R2R:
  https://niessner.github.io/Matterport/
"""

from __future__ import print_function

import argparse
import json
import os

import networkx as nx
import numpy as np


def main(args):
  """Generate R4R data from the original R2R data.

  Args:
    args: argparse containing paths to input and output files.
  """
  print('Generating R4R data with a {} meter cutoff.'.format(args.cutoff))

  def _connections_file_path(scan):
    return os.path.join(
        args.connections_dir, '{}_connectivity.json'.format(scan))

  def _house_file_path(scan):
    return os.path.join(
        args.scans_dir, '{}/house_segmentations/{}.house'.format(scan, scan))

  inputs = json.load(open(args.input_file_path))
  outputs = list()

  # Group by scan to save memory.
  scans = dict()
  for value in inputs:
    scan = value['scan']
    if scan not in scans:
      scans[scan] = []
    scans[scan].append(value)

  for scan, values in scans.items():
    print('Loading graph for scan {}.'.format(scan))

    # Load the scan graph.
    with open(_connections_file_path(scan)) as f:
      lines = json.load(f)
      nodes = np.array([x['image_id'] for x in lines])
      matrix = np.array([x['unobstructed'] for x in lines])
      mask = [x['included'] for x in lines]
      matrix = matrix[mask][:, mask]
      nodes = nodes[mask]

    with open(_house_file_path(scan)) as f:
      lines = f.readlines()
      tokens = [str.split(x) for x in lines if x.startswith('P')]
      pos = {x[1]: np.array(map(float, x[5:8])) for x in tokens}

    graph = nx.from_numpy_matrix(matrix)
    graph = nx.relabel.relabel_nodes(graph, dict(enumerate(nodes)))
    edge_attrs = {
        (u, v): {'weight': np.linalg.norm(pos[u] - pos[v])}
        for u, v in graph.edges
    }
    nx.set_edge_attributes(graph, edge_attrs)

    # Cache format: (node, (distance, path)) ((node obj, (dict, dict)))
    cache = dict(nx.all_pairs_dijkstra(graph, weight='weight'))
    shortest_distance = {k: v[0] for k, v in cache.items()}
    shortest_path = {k: v[1] for k, v in cache.items()}

    for first in values:
      for second in values:
        first_target = first['path'][-1]
        second_source = second['path'][0]

        if shortest_distance[first_target][second_source] < args.cutoff:
          value = dict()
          value['path'] = (
              first['path'][:-1]
              + shortest_path[first_target][second_source]
              + second['path'][1:])
          value['distance'] = (
              first['distance']
              + shortest_distance[first_target][second_source]
              + second['distance'])
          value['instructions'] = [
              x + y  # pylint: disable=g-complex-comprehension
              for x in first['instructions']
              for y in second['instructions']]
          value['heading'] = first['heading']
          value['path_id'] = len(outputs)
          value['scan'] = scan

          # Additional data.
          path_source = first['path'][0]
          path_target = second['path'][-1]
          value['shortest_path_distance'] = cache[path_source][0][path_target]
          value['shortest_path'] = cache[path_source][1][path_target]
          value['first_path_id'] = first['path_id']
          value['second_path_id'] = second['path_id']

          outputs.append(value)

  with open(args.output_file_path, 'w') as f:
    json.dump(outputs, f, indent=2, sort_keys=True, separators=(',', ': '))

  # Dataset summary metrics.
  avg_distance = np.mean([x['distance'] for x in outputs])
  avg_path_len = np.mean([len(x['path']) for x in outputs])
  avg_sp_distance = np.mean([x['shortest_path_distance'] for x in outputs])
  avg_sp_path_len = np.mean([len(x['shortest_path']) for x in outputs])

  print('******Final Results********')
  print('  Total paths generated: {}'.format(len(outputs)))
  print('  Average path distance (meters): {}'.format(avg_distance))
  print('  Average shortest path distance: {}'.format(avg_sp_distance))
  print('  Average path length (steps): {}'.format(avg_path_len))
  print('  Average shortest path length: {}'.format(avg_sp_path_len))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--scans_dir',
      dest='scans_dir',
      required=True,
      help='Path to the Matterport simulator scan data.')
  parser.add_argument(
      '--connections_dir',
      dest='scans_dir',
      required=True,
      help='Path to the Matterport simulator connection data.')
  parser.add_argument(
      '--input_file_path',
      dest='input_file_path',
      required=True,
      help='Path to read the R2R input data.')
  parser.add_argument(
      '--output_file_path',
      dest='output_file_path',
      required=True,
      help='Path to write the R4R output data.')
  parser.add_argument(
      '--cutoff',
      dest='cutoff',
      nargs='?',
      const=3.0,
      type=float,
      help='Maximum end-start distance for joining a pair of R2R paths.')
  main(parser.parse_args())
