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

"""Program tree representation."""

# pylint: skip-file
import numpy as np
from aloe.rfill.utils.rfill_consts import RFILL_EDGE_TYPES, RFILL_NODE_TYPES


class ProgNode(object):
  """Token as node in program tree/graph."""

  def __init__(self, syntax, value=None, subtrees=None):
    """Initializer.

    Args:
      syntax: string representation of syntax
      value: string representation of actual value
      subtrees: list of tuple(edge_type, subtree nodes or single node)
    """
    self.syntax = syntax
    self.value = value

    self.children = []

    if subtrees is not None:
      for e_type, children in subtrees:
        if isinstance(children, list):
          for c in children:
            add_edge(parent_node=self, child_node=c, edge_type=e_type)
        else:
          add_edge(parent_node=self, child_node=children, edge_type=e_type)

  def get_name(self):
    if self.value is None:
      return self.syntax
    return self.syntax + '-' + str(self.value)

  def add_child(self, edge_type, child_node):
    self.children.append((edge_type, child_node))

  def pprint(self, tab_cnt=0):
    st = '  ' * tab_cnt + self.get_name()
    print(st)
    for _, c in self.children:
      c.pprint(tab_cnt=tab_cnt + 1)

  def __str__(self):
    st = '(' + self.get_name()
    for _, c in self.children:
      st += c.__str__()
    st += ')'
    return st

class AbsRFillNode(ProgNode):
  """abstract Subclass of RFillNode."""

  def pprint(self, tab_cnt=0):
    if self.syntax == 'RegexTok' or self.syntax == 'ConstTok':
      st = '  ' * tab_cnt + self.syntax + '('
      _, p1 = self.children[0]
      _, p2 = self.children[1]
      _, direct = self.children[2]
      name = p1.value
      st += '%s, %d, %s)' % (name, p2.value, direct.value)
      print(st)
      return
    st = '  ' * tab_cnt + self.get_name()
    print(st)
    for _, c in self.children:
      c.pprint(tab_cnt=tab_cnt + 1)



def filter_tree_nodes(root_node, key_set, out_list=None):
  if out_list is None:
    out_list = []
  if root_node.syntax in key_set:
    out_list.append(root_node)
  for _, c in root_node.children:
    filter_tree_nodes(c, key_set, out_list=out_list)
  return out_list


def add_edge(parent_node, child_node, edge_type):
  parent_node.add_child(edge_type, child_node)


class ProgGraph(object):
  """Program graph"""

  def __init__(self, tree_root, node_types=RFILL_NODE_TYPES, edge_types=RFILL_EDGE_TYPES, add_rev_edge=True):
    """Initializer.

    Args:
      tree_root: ProgNode type; the root of tree representation
      node_types: dict of nodetype to index
      edge_types: dict of edgetype to index
      add_rev_edge: whether add reversed edge
    """
    self.tree_root = tree_root
    self.add_rev_edge = add_rev_edge
    self.node_types = node_types
    self.edge_types = edge_types

    # list of tree nodes
    self.node_list = []

    # node feature index
    self.node_feats = []

    # list of (from_idx, to_idx, etype_int) tuples
    self.edge_list = []

    self.last_terminal = None  # used for linking terminals
    self.build_graph(self.tree_root)

    self.num_nodes = len(self.node_list)
    self.num_edges = len(self.edge_list)
    # unzipped version of edge list
    # self.from_list, self.to_list, self.edge_feats = \
    #     [np.array(x, dtype=np.int32) for x in zip(*self.edge_list)]

    self.node_feats = np.array(self.node_feats, dtype=np.int32)

    self.subexpr_ids = []
    for _, c in self.tree_root.children:
      self.subexpr_ids.append(c.index)

  def render(self, render_path):
    """Render the program graph to specified path."""
    import pygraphviz as pgv
    ag = pgv.AGraph(directed=True)
    e_idx2name = {}
    for key in self.edge_types:
      e_idx2name[self.edge_types[key]] = key

    for i, node in enumerate(self.node_list):
      ag.add_node(str(i) + '-' + node.get_name())

    for e in self.edge_list:
      x, y, et = e
      ename = e_idx2name[et]
      if ename.startswith('rev-'):
        continue
      x = str(x) + '-' + self.node_list[x].get_name()
      y = str(y) + '-' + self.node_list[y].get_name()
      ag.add_edge(x, y)
    ag.layout(prog='dot')
    ag.draw(render_path)

  def add_bidir_edge(self, from_idx, to_idx, etype_str):
    assert etype_str in self.edge_types
    self.edge_list.append((from_idx, to_idx, self.edge_types[etype_str]))
    if self.add_rev_edge:
      # add reversed edge
      rev_etype_str = 'rev-' + etype_str
      assert rev_etype_str in self.edge_types
      self.edge_list.append((to_idx, from_idx, self.edge_types[rev_etype_str]))

  def build_graph(self, cur_root):
    """recursively build program graph from program tree.

    Args:
      cur_root: current root of (sub)program

    Returns:
      index: index of this cur_root node
    """
    cur_root.index = len(self.node_list)
    self.node_list.append(cur_root)
    name = cur_root.get_name()
    if name not in self.node_types:
      raise NotImplementedError
    type_idx = self.node_types[name]
    cur_root.node_type = type_idx
    self.node_feats.append(type_idx)

    if len(cur_root.children):  # pylint: disable=g-explicit-length-test
      for e_type, c in cur_root.children:
        child_idx = self.build_graph(c)
        self.add_bidir_edge(cur_root.index, child_idx, e_type)
    else:  # add possible links between adjacent terminals
      if self.last_terminal is not None:
        self.add_bidir_edge(self.last_terminal.index, cur_root.index, 'succ')
      self.last_terminal = cur_root

    return cur_root.index

