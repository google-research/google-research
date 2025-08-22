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

"""Utils for reading and writing necessary files."""


import glob
from typing import List, Tuple
import pandas as pd

from hst_clustering import dynamic_program

ROOT_ID = ""


def LoadFilesIntoDataFrame(
    glob_string, dimensions
):
  """Creates a data frame from a collection of csv files representing a HST.

  Args:
    glob_string: A regular expression encoding the files.
    dimensions: dimensions of the input vectors.

  Returns:
    A pandas dataframe with the data from the files.
  """
  files = glob.glob(glob_string)
  dfs = []
  columns = ["id", "right_child", "left_child", "diameter", "weight"]
  n_features = dimensions
  str_type_dict = {"id": str, "right_child": str, "left_child": str}
  feature_columns = [str(i) for i in range(n_features)]
  columns += feature_columns
  for file in files:
    with open(file, "r") as f:
      dfs.append(
          pd.read_csv(
              f, names=columns, keep_default_na=False, dtype=str_type_dict
          )
      )
  return pd.concat(dfs), feature_columns


def ReadRawData(file):
  """Reads raw data from a file into a pandas data frame.

  Expects data do be encoded as a csv with the first column an index column.

  Args:
    file: A path to a file.

  Returns:
    A pandas dataframe with the dataset.
  """
  data = pd.read_csv(open(file, "r"), index_col=0, sep=r"\s+", header=None)
  return data.values


def GetThreshold(
    threshold_constant, budget_split, epsilon
):
  return max(0, threshold_constant * budget_split / epsilon)


def DataFrameToTree(data):
  """Parses a HST dataset into an HST.

  Args:
    data: A pandas dataframe encoding the HST.

  Returns:
    An HST.
  """
  tree = dynamic_program.HST()
  nodes = {}
  negative_weights = 0
  for _, element in data.iterrows():
    if element.weight < 0:
      negative_weights += 1
      element.weight = 0
    node = dynamic_program.Node(element.weight, element.diameter, element.id)
    node.set_right_child(element.right_child)
    node.set_left_child(element.left_child)
    nodes[element.id] = node

  for key, node in nodes.items():
    if node.right_child not in nodes and node.left_child not in nodes:
      # Node is a leaf
      node.right_child = None
      node.left_child = None
      tree.add_node(node.node_id, node)
    # Node is not a leaf but doesn't have both children available. Complete the
    # children.
    elif node.right_child not in nodes:
      # Create an empty node with zero weight
      empty_node = dynamic_program.Node(
          0.0, 2.0 / 3 * node.diameter, node.node_id + "0"
      )
      node.right_child = empty_node
      tree.add_node(empty_node.node_id, empty_node)
      node.left_child = nodes[node.left_child]
      if key == ROOT_ID:
        tree.add_root(ROOT_ID, node)
      else:
        tree.add_node(node.node_id, node)
    elif node.left_child not in nodes:
      # Create an empty node with zero weight
      empty_node = dynamic_program.Node(
          0.0, 2.0 / 3 * node.diameter, node.node_id + "1"
      )
      tree.add_node(empty_node.node_id, empty_node)
      node.left_child = empty_node
      node.right_child = nodes[node.right_child]
      if key == ROOT_ID:
        tree.add_root(ROOT_ID, node)
      else:
        tree.add_node(node.node_id, node)
    else:
      # Internal node with both children.
      node.right_child = nodes[node.right_child]
      node.left_child = nodes[node.left_child]
      if key == ROOT_ID:
        tree.add_root(ROOT_ID, node)
      else:
        tree.add_node(node.node_id, node)
  return tree
