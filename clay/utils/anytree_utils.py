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

"""Utility functions for anytree operations."""

import anytree


def delete_node(node):
  """Delete node in tree (root cannot be deleted).

  The descendants of the node are added to its parent. If you would like to
  remove all descendants too, you can use delete_branch or delete_descendants
  instead.

  Args:
    node: the node to be deleted.
  """
  if node.parent is None:
    return
  par = node.parent
  # Make sure to maintain the order in the parent children's list.
  par_children = list(par.children)
  idx = par_children.index(node)
  par.children = tuple(par_children[:idx]) + node.children + tuple(
      par_children[idx + 1:])

  for c in node.children:
    c.parent = par


def get_node_index(node):
  """Returns the index (position) of a given node."""
  if node.parent is None:
    return 0
  return node.parent.children.index(node)
