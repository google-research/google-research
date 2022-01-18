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

# Lint as: python3
"""Tree structure helper classes and functions."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import numpy as np
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS


class Tree(object):
  """Tree class for tree based colaborative filtering model.

  Module to define basic tree operations, as updating the tree sturcture,
  providing tree stats and more.
  Attributes:
    children: dict of dicts. For each level l (where -1 is the root level)
      children[l] holds as keys the indices of the used nodes in this level,
      and their values are are lists of their children node indices. Note that
      item indices are given a unique id by adding the number of total nodes
      to the item index as in tree_based/models/base
  """

  def __init__(self, tot_levels):
    self.children = {}
    for l in range(tot_levels+1):
      self.children[l-1] = {}

  def update(self, node_ind, parent, parent_level):
    """Adds a node to the tree, should be used bottom up."""
    if parent in self.children[parent_level]:
      self.children[parent_level][parent].append(node_ind)
    else:
      self.children[parent_level][parent] = [node_ind]

  def as_ragged(self, tot_nodes):
    """Convernt to a ragged tensor."""
    all_child = [[] for i in range(tot_nodes+1)]
    for level in self.children:
      for parent in self.children[level]:
        all_child[int(parent)] = [
            int(child) for child in self.children[level][parent]
        ]
    return tf.ragged.constant(all_child)

  def stats(self):
    """Calculates tree stats: num of used nodes, mean and std of node degrees."""
    used_nodes = {}
    mean_deg = {}
    std_deg = {}
    for level in self.children:
      used_nodes[level] = len(self.children[level].keys())
      degs = []
      for parent in self.children[level]:
        degs.append(len(self.children[level][parent]))
      mean_deg[level] = np.mean(degs)
      std_deg[level] = np.std(degs)
    return used_nodes, mean_deg, std_deg


def top_k_to_scores(top_k_rec, n_items):
  k = len(top_k_rec)
  scores = np.zeros(n_items)
  for i, rec in enumerate(top_k_rec):
    scores[int(rec)] = k-i
  return scores


def build_tree(closest_node_to_items, closest_node_to_nodes, nodes_per_level):
  """builds the item-nodes tree based on closest nodes.

  Builds the tree borrom up. Skips nodes that are not connected to any item.

  Args:
      closest_node_to_items: np.array of size (n_item, ) where
        closest_node_to_items[item_index] = closest node index one level up.
      closest_node_to_nodes: np.array of size (tot_n_nodes, ) where
        closest_node_to_nodes[node_index] = closest node index one level up.
      nodes_per_level: list of the number of nodes per level excluding the
        root and the leaves.

  Returns:
      tree: Tree class.
  """
  # root index is -1
  tot_levels = len(nodes_per_level)
  tree = Tree(tot_levels)
  # add leaves
  for leaf, node_parent in enumerate(closest_node_to_items):
    leaf = sum(nodes_per_level) + leaf  # unique leaf id
    tree.update(leaf, parent=node_parent, parent_level=tot_levels-1)
  # add internal nodes, bottom-up
  for level in range(tot_levels-1, -1, -1):
    first_node = sum(nodes_per_level[:level])
    last = sum(nodes_per_level[:level+1])
    for node in range(first_node, last):
      node_parent = closest_node_to_nodes[node]
      if node in tree.children[level]:
        tree.update(node, parent=node_parent, parent_level=level-1)
  return tree
