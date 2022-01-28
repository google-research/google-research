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

"""Contains classes specifying naming conventions used for View hierarchy fields in tf.Example protos.

Specifies:
  ViewHierarchyFields: standard fields used for View hierarchy fields in
  tf.Example proto.
  NodeFields: standard fields used for nodes in the view hierarchy.
"""

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class Field:
  """Field class to store field related attributes."""
  name: str
  dtype: type  # pylint: disable=g-bare-generic
  default_value: Any = dataclasses.field(init=False)

  def __post_init__(self):
    # Set default_value using __setattr__ because the dataclass is frozen.
    if self.dtype in {int, float}:
      object.__setattr__(self, 'default_value', self.dtype(-1))
    elif self.dtype in {str, bool}:
      object.__setattr__(self, 'default_value', self.dtype())
    if not isinstance(self.default_value, self.dtype):
      raise TypeError(
          'The types of default value and dtype need to be the same, but found {} and {}'
          .format(type(self.default_value), self.dtype))


class ViewHierarchyFields:
  """Names for View hierarchy fields in tf.Example proto.

  Attributes:
    is_keyboard_deployed: whether the keyboard is deployed.
    id: id of the node.
    parent_id: parent id of the node.
    is_leaf: whether the node has children.
    node_id: node id in the view hierarchy.
    ui_type: ui node type.
    window_id: window id in the view hierarchy.
    pre_order_index: index of the pre-order traversal from root node.
    post_order_index: index of the post-order traversal from root node.
    node_depth: depth of the node in the view hierarchy tree.
  """
  is_keyboard_deployed = Field('image/view_hierarchy/is_keyboard_deployed',
                               bool)
  id = Field('image/view_hierarchy/id', int)
  parent_id = Field('image/view_hierarchy/parent_id', int)
  is_leaf = Field('image/view_hierarchy/is_leaf', bool)
  node_id = Field('image/view_hierarchy/node_id', int)
  ui_type = Field('image/view_hierarchy/type', str)
  window_id = Field('image/view_hierarchy/window_id', int)
  pre_order_index = Field('image/view_hierarchy/pre_order_index', int)
  post_order_index = Field('image/view_hierarchy/post_order_index', int)
  node_depth = Field('image/view_hierarchy/node_depth', int)


class NodeFields:
  """Names for UI element view hierarchy node fields in tf.Example proto.

  Attributes:
    description: content description.
    text: text associated with element (if it exists).
    hint_text: hint text associated with element (if it exists).
    tooltip_text: text displayed in a small popup window on hover or long press
      text associated with element (if it exists).
    class_name: Android class name.
    superclass_name: Closest Android superclass that is guaranteed to not be
      obfuscated (from Robo crawling).
    component_classification: Coarse classification of the element (from Robo).
    resource_id: resource id of the node.
    text_size: size of the text, or font size (if it exists).
    font_family: font family of the text (if it exists).
    font_weight: font weight of the text (if it exists).
    opacity: transparency of the node.
    is_focusable: whether the node is focusable.
    is_focused: whether the node is focused.
    is_clickable: whether the node is clickable.
    is_selected: whether the node is selected.
    is_checked: whether the node is checked.
    is_enabled: whether the node is enabled.
    is_checkable: whether the node is checkable.
    is_editable: whether the node is editable.
    is_long_clickable: whether the node is long clickable.
    is_scrollable: whether the node is scrollable.
    is_accessibility_focused: whether the node is accessibility focused.
    visibility: visibility of the node (string).
    visible_to_user: whether the node is visible (boolean).
    elevation: base z depth of the node.
    pkg_name: package name.
    a11y_node_hashcode: a11y node hashcode.
    row_count: Maximum number of rows created when automatically positioning
      children
    column_count: Maximum number of columns created when automatically
      positioning children.
    row_index: Row index which the element is located.
    column_index: Column index which the element is located.
    support_text_location: whether node support text location.
    drawing_order: node drawing order position.
    accessibility_actions: Integer id for the accessibility action.
    is_heading: whether the node is considered a heading.
    chrome_role: node chrome role.
    chrome_role_description: node chrome role description.
  """
  description = Field('image/view_hierarchy/description', str)
  text = Field('image/view_hierarchy/text', str)
  hint_text = Field('image/view_hierarchy/attributes/hint_text', str)
  tooltip_text = Field('image/view_hierarchy/attributes/tooltip_text', str)
  class_name = Field('image/view_hierarchy/class/name', str)
  resource_id = Field('image/view_hierarchy/attributes/id', str)
  # The following 2 fields are used in Robo crawling.
  superclass_name = Field('image/view_hierarchy/class/superclass_name', str)
  component_classification = Field(
      'image/view_hierarchy/attribute/component_classification', str)
  text_size = Field('image/view_hierarchy/attributes/text_size', int)
  font_family = Field('image/view_hierarchy/attributes/font_family', str)
  font_weight = Field('image/view_hierarchy/attributes/font_weight', int)
  opacity = Field('image/view_hierarchy/attributes/opacity', int)
  is_focusable = Field('image/view_hierarchy/attributes/focusable', bool)
  is_focused = Field('image/view_hierarchy/attributes/focused', bool)
  is_clickable = Field('image/view_hierarchy/attributes/clickable', bool)
  is_selected = Field('image/view_hierarchy/attributes/selected', bool)
  is_checked = Field('image/view_hierarchy/attributes/checked', bool)
  is_enabled = Field('image/view_hierarchy/attributes/enabled', bool)
  is_checkable = Field('image/view_hierarchy/attributes/is_checkable', bool)
  is_editable = Field('image/view_hierarchy/attributes/is_editable', bool)
  is_long_clickable = Field('image/view_hierarchy/attributes/is_long_clickable',
                            bool)
  is_scrollable = Field('image/view_hierarchy/attributes/is_scrollable', bool)
  is_accessibility_focused = Field(
      'image/view_hierarchy/attributes/is_accessibility_focused', bool)
  visibility = Field('image/view_hierarchy/attributes/visibility', str)
  visible_to_user = Field('image/view_hierarchy/attributes/visible_to_user',
                          bool)
  is_dismissable = Field('image/view_hierarchy/attributes/is_dismissable', bool)
  is_important_for_accessibility = Field(
      'image/view_hierarchy/attributes/is_important_for_accessibility', bool)

  elevation = Field('image/view_hierarchy/attributes/elevation', int)
  pkg_name = Field('image/view_hierarchy/attributes/pkg_name', str)
  a11y_node_hashcode = Field(
      'image/view_hierarchy/attributes/a11y_node_hashcode', int)
  row_count = Field('image/view_hierarchy/attributes/row_count', int)
  column_count = Field('image/view_hierarchy/attributes/column_count', int)
  row_index = Field('image/view_hierarchy/attributes/row_index', int)
  column_index = Field('image/view_hierarchy/attributes/column_index', int)
  support_text_location = Field(
      'image/view_hierarchy/attributes/support_text_location', bool)
  drawing_order = Field('image/view_hierarchy/attributes/drawing_order', int)
  accessibility_actions = Field(
      'image/view_hierarchy/attributes/accessibility_actions', int)
  is_heading = Field('image/view_hierarchy/attributes/is_heading', bool)
  chrome_role = Field('image/view_hierarchy/attributes/chrome_role', str)
  chrome_role_description = Field(
      'image/view_hierarchy/attributes/chrome_role_description', str)

  text_size_in_px = Field('image/view_hierarchy/attributes/text_size_in_px',
                          float)
  text_size_unit = Field('image/view_hierarchy/attributes/text_size_unit',
                         float)
  layout_size_width = Field('image/view_hierarchy/attributes/layout_size_width',
                            float)
  layout_size_height = Field(
      'image/view_hierarchy/attributes/layout_size_height', float)
  node_id = Field('image/view_hierarchy/node_id', str)
