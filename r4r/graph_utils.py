# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Utils for loading and drawing graphs of the houses."""

from __future__ import print_function

import json
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
from numpy.linalg import norm


def load(connections_file):
  """Loads a networkx graph for a given scan.

  Args:
    connections_file: A string with the path to the .json file with the
      connectivity information.
  Returns:
    A networkx graph.
  """
  with open(connections_file) as f:
    lines = json.load(f)
    nodes = np.array([x['image_id'] for x in lines])
    matrix = np.array([x['unobstructed'] for x in lines])
    mask = [x['included'] for x in lines]
    matrix = matrix[mask][:, mask]
    nodes = nodes[mask]
    pos2d = {x['image_id']: np.array(x['pose'])[[3, 7]] for x in lines}
    pos3d = {x['image_id']: np.array(x['pose'])[[3, 7, 11]] for x in lines}

  graph = nx.from_numpy_matrix(matrix)
  graph = nx.relabel.relabel_nodes(graph, dict(enumerate(nodes)))
  nx.set_node_attributes(graph, pos2d, 'pos2d')
  nx.set_node_attributes(graph, pos3d, 'pos3d')

  weight2d = {(u, v): norm(pos2d[u] - pos2d[v]) for u, v in graph.edges}
  weight3d = {(u, v): norm(pos3d[u] - pos3d[v]) for u, v in graph.edges}
  nx.set_edge_attributes(graph, weight2d, 'weight2d')
  nx.set_edge_attributes(graph, weight3d, 'weight3d')

  return graph


def draw(graph, predicted_path, reference_path, output_filename, **kwargs):
  """Generates a plot showing the graph, predicted and reference paths.

  Args:
    graph: A networkx graph.
    predicted_path: A list with the ids of the nodes in the predicted path.
    reference_path: A list with the ids of the nodes in the reference path.
    output_filename: A string with the path where to store the generated image.
    **kwargs: Key-word arguments for aesthetic control.
  """
  plt.figure(figsize=(10, 10))
  ax = plt.gca()
  pos = nx.get_node_attributes(graph, 'pos2d')

  # Zoom in.
  xs = [pos[node][0] for node in predicted_path + reference_path]
  ys = [pos[node][1] for node in predicted_path + reference_path]
  min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
  center_x, center_y = (min_x + max_x)/2, (min_y + max_y)/2
  zoom_margin = kwargs.get('zoom_margin', 1.3)
  max_range = zoom_margin * max(max_x - min_x, max_y - min_y)
  half_range = max_range / 2
  ax.set_xlim(center_x - half_range, center_x + half_range)
  ax.set_ylim(center_y - half_range, center_y + half_range)

  # Background graph.
  nx.draw(graph,
          pos,
          edge_color=kwargs.get('background_edge_color', 'lightgrey'),
          node_color=kwargs.get('background_node_color', 'lightgrey'),
          node_size=kwargs.get('background_node_size', 60),
          width=kwargs.get('background_edge_width', 0.5))

  # Prediction graph.
  predicted_path_graph = nx.DiGraph()
  predicted_path_graph.add_nodes_from(predicted_path)
  predicted_path_graph.add_edges_from(
      zip(predicted_path[:-1], predicted_path[1:]))
  nx.draw(predicted_path_graph,
          pos,
          arrowsize=kwargs.get('prediction_arrowsize', 15),
          edge_color=kwargs.get('prediction_edge_color', 'red'),
          node_color=kwargs.get('prediction_node_color', 'red'),
          node_size=kwargs.get('prediction_node_size', 130),
          width=kwargs.get('prediction_edge_width', 2.0))

  # Reference graph.
  reference_path_graph = nx.DiGraph()
  reference_path_graph.add_nodes_from(reference_path)
  reference_path_graph.add_edges_from(
      zip(reference_path[:-1], reference_path[1:]))
  nx.draw(reference_path_graph,
          pos,
          arrowsize=kwargs.get('reference_arrowsize', 15),
          edge_color=kwargs.get('reference_edge_color', 'dodgerblue'),
          node_color=kwargs.get('reference_node_color', 'dodgerblue'),
          node_size=kwargs.get('reference_node_size', 130),
          width=kwargs.get('reference_edge_width', 2.0))

  # Intersection graph.
  intersection_path_graph = nx.DiGraph()
  common_nodes = set(predicted_path_graph.nodes.keys()).intersection(
      set(reference_path_graph.nodes.keys()))
  intersection_path_graph.add_nodes_from(common_nodes)
  common_edges = set(predicted_path_graph.edges.keys()).intersection(
      set(reference_path_graph.edges.keys()))
  intersection_path_graph.add_edges_from(common_edges)
  nx.draw(intersection_path_graph,
          pos,
          arrowsize=kwargs.get('intersection_arrowsize', 15),
          edge_color=kwargs.get('intersection_edge_color', 'limegreen'),
          node_color=kwargs.get('intersection_node_color', 'limegreen'),
          node_size=kwargs.get('intersection_node_size', 130),
          width=kwargs.get('intersection_edge_width', 2.0))

  plt.savefig(output_filename)
  plt.close()

