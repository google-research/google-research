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

"""Conversion library from tf.Example protos to VHNode tree objects."""

import dataclasses
from typing import List, Optional
import uuid

from absl import logging
import anytree
import attr
import tensorflow as tf

from clay.utils import example_utils
from clay.utils import view_hierarchy_fields
from clay.utils.image_utils import BBox

VH_FIELDS = view_hierarchy_fields.ViewHierarchyFields
N_FIELDS = view_hierarchy_fields.NodeFields


def get_all_attrs(obj):
  """Retrieve all attributes of an object."""
  return [f for f in vars(obj) if not f.startswith('__')]


# Note: 'eq=False' enables hashing by id, which is required when deleting nodes.
@attr.s(eq=False)
# The NodeMixin class inheritance transforms the node into an anytree.Node.
class VHNode(anytree.NodeMixin):
  """Class to hold tf.Example fields while infering labels from view hierarchy.

  This should mirror k/c/s/i/s/utils/view_hierarchy_fields.py.
  """
  # LINT.IfChange
  bbox: BBox = attr.ib(default=None)
  description: str = attr.ib(default='')
  text: str = attr.ib(default='')
  class_name: str = attr.ib(default='')
  resource_id: str = attr.ib(default='')
  superclass_name = attr.ib(default='')
  component_classification = attr.ib(default='')
  node_id: str = attr.ib(default='')

  text_size: int = attr.ib(default=0)
  font_family: str = attr.ib(default='')
  font_weight: int = attr.ib(default=0)
  opacity: float = attr.ib(default=1.)
  is_focusable: bool = attr.ib(default=False, converter=bool)
  is_focused: bool = attr.ib(default=False, converter=bool)
  is_selected: bool = attr.ib(default=False, converter=bool)
  is_checked: bool = attr.ib(default=False, converter=bool)
  is_clickable: bool = attr.ib(default=False, converter=bool)
  is_enabled: bool = attr.ib(default=True, converter=bool)
  visible_to_user: bool = attr.ib(default=True, converter=bool)
  visibility: str = attr.ib(default='visible')
  elevation: float = attr.ib(default=0.)
  hint_text: str = attr.ib(default='')
  tooltip_text: str = attr.ib(default='')
  is_checkable: bool = attr.ib(default=False, converter=bool)
  is_editable: bool = attr.ib(default=False, converter=bool)
  is_long_clickable: bool = attr.ib(default=False, converter=bool)
  is_scrollable: bool = attr.ib(default=False, converter=bool)
  is_accessibility_focused: bool = attr.ib(default=False, converter=bool)
  pkg_name: str = attr.ib(default='')
  a11y_node_hashcode: int = attr.ib(default=0)

  row_count: int = attr.ib(default=0)
  column_count: int = attr.ib(default=0)
  row_index: int = attr.ib(default=0)
  column_index: int = attr.ib(default=0)
  support_text_location: bool = attr.ib(default=True, converter=bool)
  drawing_order: int = attr.ib(default=0)
  accessibility_actions: int = attr.ib(default=0)

  # Tree node fields.
  idx: int = attr.ib(default=attr.Factory(lambda: uuid.uuid1().int))
  parent: 'VHNode' = attr.ib(default=None)

  def set_labels(self, cls, label):
    """Set class label and id in BBox."""
    annotated_bbox = dataclasses.replace(
        self.bbox, ui_class=cls, ui_label=label)
    self.bbox = annotated_bbox


def to_vh_node(example, idx, parent):
  """Convert a tf.Example entry to a VHNode."""
  field_names = get_all_attrs(N_FIELDS)
  kwargs = {'idx': idx, 'parent': parent}

  # Copy all fields in tf.Example (if they exist).
  for field_name in field_names:
    features = example_utils.get_feat_list(example,
                                           getattr(N_FIELDS, field_name))
    if len(features) > idx:
      kwargs[field_name] = features[idx]

  # Post-process visibility attribute.
  if 'visibility' not in kwargs:
    kwargs['visibility'] = 'visible'
  if 'visible_to_user' in kwargs:
    kwargs['visibility'] = 'visible' if kwargs['visible_to_user'] else ''

  bboxes = example_utils.get_bboxes(example)
  if len(bboxes) > idx:
    kwargs['bbox'] = bboxes[idx]
  return VHNode(**kwargs)


def _find_node_by_index(nodes, idx):
  """Returns node whose index is idx or None if it doesn't exist.

  Args:
    nodes: A list of VHNodes.
    idx: The index we are looking for.

  Returns:
    The VHNode whose idx attribute matches idx, or None if none is found.
  """
  for node in nodes:
    if node.idx == idx:
      return node


def tfexample_to_tree(example):
  """Convert a tf.Example to a tree of AnyNode and returns the root.

  Args:
    example: tf.Example to be converted.

  Returns:
    The root of the tree or None if the tf.Example does not contain any node.
  """
  parent_ids = example_utils.get_feat_list(example, VH_FIELDS.parent_id)
  num_nodes = len(parent_ids)
  if not num_nodes:
    return None

  nodes = []
  # Process nodes in the order of their parent id to make sure parents are
  # inserted before their children.
  for p_id, i in sorted(zip(parent_ids, range(num_nodes))):
    parent = _find_node_by_index(nodes, p_id)
    if parent is None and p_id != -1:
      logging.warning('Parent node %s not found, node discarded.', p_id)
    else:
      node = to_vh_node(example, i, parent)
      nodes.append(node)

  return nodes[0]


def _add_node_to_tfexample(node, example,
                           parent_idx):
  """Copy all fields to tf.Example (if they exist)."""
  for field_name in get_all_attrs(N_FIELDS):
    if hasattr(node, field_name):
      field = getattr(node, field_name)
      # Convert bool fields to int.
      if isinstance(field, bool):
        field = int(field)
      example_utils.add_feat(example, getattr(N_FIELDS, field_name), field)

  example_utils.add_feat(example, VH_FIELDS.parent_id, parent_idx)
  example_utils.add_feat(example, VH_FIELDS.is_leaf,
                         1 - int(bool(node.children)))


def _find_parent_idx(node, nodes):
  """Find parent index."""
  if node.parent is None:
    return -1
  try:
    return nodes.index(node.parent)
  except ValueError:
    logging.error('Parent not found in tree nodes.')
  return -1


def tree_to_tfexample(
    root,
    example = None):
  """Convert a tree of AnyNodes to a tf.Example.

  Args:
    root: AnyNode that represents the root of the tree.
    example: tf.Example to which to add the view hierarchy information.

  Returns:
    The tf.Example created.
  """
  if example is None:
    example = tf.train.Example()
  else:
    # Delete existing fields.
    for field_name in get_all_attrs(N_FIELDS):
      example_utils.del_feat(example, getattr(N_FIELDS, field_name))
    example_utils.delete_bboxes(example)

  if root is None:
    return example
  nodes = list(anytree.PreOrderIter(root))
  for node in nodes:
    _add_node_to_tfexample(node, example, _find_parent_idx(node, nodes))
    example_utils.add_bboxes(example, node.bbox)
  return example


def delete_hierarchy_nodes(example):
  """Delete hierarchy nodes from example in-place."""
  example_utils.delete_bboxes(example)
  features = example.features.feature

  # remove all image/view_hierarchy/* keys as they all relate to tree nodes.
  for key in list(features.keys()):
    if key.startswith('image/view_hierarchy'):
      example_utils.del_feat(example, key)
