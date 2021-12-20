# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Visualize the graph of nodes and their neighbors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import random

from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import tensorflow.compat.v1 as tf

from simpdom import constants


random.seed(42)
FLAGS = flags.FLAGS

flags.DEFINE_boolean("shuffle_neighbors", False,
                     "If to shuffle the neightbors during visualization.")
flags.DEFINE_integer("max_neighbor_size", 15,
                     "The maximum neighbors of each node to plot.")
flags.DEFINE_string("domtree_data_path", None,
                    "The path of domtree json files.")
flags.DEFINE_string("vertical", None,
                    "The specific vertical to generate graph for.")

VERTICAL_WEBSITES = constants.VERTICAL_WEBSITES
PAGE_INDEX = 0


def abbr(label):
  """Gets a better visualization with abbreviations of long attribute names."""

  if label == "fuel_economy":
    return "fuel"
  # publication_date and date_posted do not appear in the same vertical.
  elif label == "publication_date":
    return "date"
  elif label == "date_posted":
    return "date"
  elif label == "mpaa_rating":
    return "mpaa"
  elif label == "isbn_13":
    return "isbn13"
  else:
    return label


def construct_edge(filename):
  """Loads the data and constructs the edges, labels and colors to show.

  Nodes represent the text unit in web pages and edges will be added
  if two nodes are neighbors in the DOM tree. We only draw the graph
  for the first page of each website without loss of generality.

  Args:
    filename: the json file to load features of specific vertical and website.

  Returns:
    edge_dict: mapping from node xpath to its neighbors.
    label_dict: mapping from node xpath to its label.
    color_dict: mapping from node xpath to its color to plot.

  """

  # Prepare 5 potential colors for attribute nodes. Each page at most contains
  # 5 attributes.
  colors = ["pink", "yellow", "lightgreen", "orange", "plum"]
  color_count = 0
  with tf.gfile.Open(filename, "r") as f:
    data = json.load(f)
    # Use the first page (PAGE_INDEX: 0) of each website as a representative.
    page = data["features"][PAGE_INDEX]
    edge_dict = {}
    label_dict = {}
    color_dict = {}
    for instance in page:
      label = abbr(instance["label"])
      xpath = instance["xpath"]
      neighbors = instance["neighbors"]
      # Randomly select neighbors if there are too many.
      if FLAGS.shuffle_neighbors:
        random.shuffle(neighbors)
      neighbors = neighbors[:FLAGS.max_neighbor_size]
      # Neighbors contain a list of xpaths of the neighboring nodes.
      edge_dict[xpath] = neighbors
      label_dict[xpath] = label
      # Pick up a color for each node.
      if label not in color_dict:
        if label == "none":
          color_dict[label] = "lightblue"
        else:
          color_dict[label] = colors[color_count]
          color_count += 1

  return edge_dict, label_dict, color_dict


def construct_graph(edge_dict, label_dict, color_dict):
  """Adds nodes and edges to an undirected graph.

  Args:
    edge_dict: mapping from node xpath to its neighbors.
    label_dict: mapping from node xpath to its label.
    color_dict: mapping from node xpath to its color to plot.

  Returns:
    graph: an undirected graph.
    label_dict: mapping from node xpath to its label to plot.
    color_list: list of colors to paint on the graph's nodes.

  """
  color_list = []
  graph = nx.Graph()
  # Need to add nodes first to filter out the noises from the neighbor list.
  for node, _ in edge_dict.items():
    graph.add_node(node)
    color_list.append(color_dict[label_dict[node]])
    # Only plot the attribute labels of interest.
    if label_dict[node] == "none":
      label_dict[node] = ""

  for node, neighbors in edge_dict.items():
    for v in neighbors:
      # Only plot edges between variable nodes from the DOM tree.
      if v in graph.nodes:
        graph.add_edge(node, v)

  return graph, label_dict, color_list


def draw_graph(graph, filename, labels, \
               colors="lightblue", font_size=10, \
               node_size=60, edge_width=0.1, \
               edge_color="gray", font_color="tomato", \
               font_family="Courier New"):
  """Plots the graph with various style settings."""

  # Manually create a layout to generate each node's position.
  df = pd.DataFrame(index=graph.nodes(), columns=graph.nodes())
  for row, data in nx.shortest_path_length(graph):
    for col, dist in data.items():
      df.loc[row, col] = dist
  df = df.fillna(df.max().max())
  layout = nx.kamada_kawai_layout(graph, dist=df.to_dict())

  options = {
      "edge_color": edge_color,
      "node_color": colors,
      "node_size": node_size,
      "font_color": font_color,
      "font_family": font_family,
      "font_size": font_size,
      "width": edge_width,
  }
  # Plot the figures.
  nx.draw(graph, layout, labels=labels, **options)
  # Write the figures to files.
  with tf.gfile.GFile(filename, "w") as f:
    plt.savefig(f, format="pdf", bbox_inches="tight")
    plt.close()


def main(_):
  for vertical, sites in VERTICAL_WEBSITES.items():
    if FLAGS.vertical and vertical != FLAGS.vertical:
      continue
    for website in sites:
      logging.info("%s-%s", vertical, website)
      input_filename = os.path.join(FLAGS.domtree_data_path,
                                    "%s-%s.json" % (vertical, website))
      edge_dict, label_dict, color_dict = construct_edge(input_filename)
      graph, label_dict, color_map = construct_graph(edge_dict, label_dict,
                                                     color_dict)
      output_filename = os.path.join(FLAGS.domtree_data_path,
                                     "%s-%s-graph.pdf" % (vertical, website))
      draw_graph(graph, output_filename, label_dict, color_map)


if __name__ == "__main__":
  app.run(main)
