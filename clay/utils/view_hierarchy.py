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

"""Classes representing view hierarchy data of Android emulator screenshot."""

import collections
from distutils.util import strtobool  # pylint: disable=g-importing-member
from enum import Enum  # pylint: disable=g-importing-member
import json
import math
import os
import re

from absl import flags
from absl import logging
import attr
from lxml import etree
import numpy as np

from clay.proto import observation_pb2

FLAGS = flags.FLAGS

ROOT_ID = '-1'
SCREEN_WIDTH = 540
SCREEN_HEIGHT = 960
ADJACENT_BOUNDING_BOX_THRESHOLD = 3


class UIObjectType(Enum):
  """Types of the different UI objects."""
  UNKNOWN = 0
  BUTTON = 1
  CHECKBOX = 2
  CHECKEDTEXTVIEW = 3
  EDITTEXT = 4
  IMAGEBUTTON = 5
  IMAGEVIEW = 6
  RADIOBUTTON = 7
  SLIDINGDRAWER = 8
  SPINNER = 9
  SWITCH = 10
  TABWIDGET = 11
  TEXTVIEW = 12
  TOGGLEBUTTON = 13
  VIDEOVIEW = 14

  # pyformat: disable
  APPWIDGETHOSTVIEW = 15          # android.appwidget.AppWidgetHostView
  VIEW = 16                       # android.view.View
  WEBVIEW = 17                    # android.webkit.WebView
  FRAMELAYOUT = 18                # android.widget.FrameLayout
  HORIZONTALSCROLLVIEW = 19       # android.widget.HorizontalScrollView
  LINEARLAYOUT = 20               # android.widget.LinearLayout
  LISTVIEW = 21                   # android.widget.ListView
  MULTIAUTOCOMPLETETEXTVIEW = 22  # android.widget.MultiAutoCompleteTextView
  PROGRESSBAR = 23                # android.widget.ProgressBar
  RELATIVELAYOUT = 24             # android.widget.RelativeLayout
  SCROLLVIEW = 25                 # android.widget.ScrollView
  TABHOST = 26                    # android.widget.TabHost
  VIEWSWITCHER = 27               # android.widget.ViewSwitcher
  SEEKBAR = 28                    # android.widget.SeekBar
  # pyformat: enable


class UIObjectGridLocation(Enum):
  """The on-screen grid location (3x3 grid) of an UI object."""
  TOP_LEFT = 0
  TOP_CENTER = 1
  TOP_RIGHT = 2
  LEFT = 3
  CENTER = 4
  RIGHT = 5
  BOTTOM_LEFT = 6
  BOTTOM_CENTER = 7
  BOTTOM_RIGHT = 8


@attr.s
class BoundingBox(object):
  """The bounding box with horizontal/vertical coordinates of an UI object."""
  x1 = attr.ib()
  y1 = attr.ib()
  x2 = attr.ib()
  y2 = attr.ib()


@attr.s
class UIObject(object):
  """Represents an UI object from the node in the view hierarchy."""
  obj_id = attr.ib()
  obj_index = attr.ib()
  parent_id = attr.ib()
  parent_index = attr.ib()
  obj_type = attr.ib()
  obj_name = attr.ib()
  word_sequence = attr.ib()
  text = attr.ib()
  resource_id = attr.ib()
  android_class = attr.ib()
  android_package = attr.ib()
  content_desc = attr.ib()
  clickable = attr.ib()
  visible = attr.ib()
  enabled = attr.ib()
  focusable = attr.ib()
  focused = attr.ib()
  scrollable = attr.ib()
  long_clickable = attr.ib()
  selected = attr.ib()
  bounding_box = attr.ib()
  grid_location = attr.ib()
  dom_location = attr.ib()
  is_leaf = attr.ib()
  checkable = attr.ib()
  checked = attr.ib()

  def is_actionable(self):
    return (self.enabled and
            (self.clickable or self.obj_type == UIObjectType.SEEKBAR) and
            self.visible)


def _build_word_sequence(text, content_desc, resource_id):
  """Returns a sequence of word tokens based on certain attributes.

  Args:
    text: `text` attribute of an element.
    content_desc: `content_desc` attribute of an element.
    resource_id: `resource_id` attribute of an element.

  Returns:
    A sequence of word tokens.
  """
  if text or content_desc:
    return re.findall(r"[\w']+|[?.!/,;:]", text if text else content_desc)
  else:
    name = resource_id.split('/')[-1]
    return list(filter(None, name.split('_')))


def _build_object_type(android_class,
                       type_num=15,
                       ancestors_class=None,
                       fix_type=False):
  """Returns the object type based on `class` attribute.

  Args:
    android_class: `class` attribute of an element (Android class).
    type_num: Seq2act use types from 0-14, set the default value for backward
      compatibility. -1 means all.
    ancestors_class: list of strs, if current class is UNKNOWN, consider the
      classes of ancestors. Now only json support this.
    fix_type: when type is not matched, try match with its surfix.

  Returns:
    The UIObjectType enum.
  """
  if type_num > 15 or type_num == -1:
    if android_class == 'android.appwidget.AppWidgetHostView':
      return UIObjectType.APPWIDGETHOSTVIEW
    if android_class == 'android.view.View':
      return UIObjectType.VIEW
    if android_class == 'android.webkit.WebView':
      return UIObjectType.WEBVIEW

  if android_class.startswith('android.widget'):
    widget_type = android_class.split('.')[2]
    for obj_type in UIObjectType:
      if type_num == -1 or obj_type.value < type_num:
        if obj_type.name == widget_type.upper():
          return obj_type

  # ancestors are sorted from subclass to base, so extract recursively:
  # "ancestors": [
  #   "android.support.v7.widget.AppCompatTextView",
  #   "android.widget.TextView",
  #   "android.view.View",
  #   "java.lang.Object"]
  if ancestors_class:
    return _build_object_type(ancestors_class[0], type_num, ancestors_class[1:])

  if fix_type:
    matched_type = None
    last_part = android_class.split('.')[-1]
    for one_type in UIObjectType:
      if last_part.lower().endswith(one_type.name.lower()):
        if not matched_type or len(one_type.name) > len(matched_type.name):
          matched_type = one_type
    if matched_type:
      # print('Fix type: %s -> %s' % (android_class, matched_type.name))
      return matched_type

  return UIObjectType.UNKNOWN


def _build_object_name(text, content_desc):
  """Returns the object name based on `text` or `content_desc` attribute.

  Args:
    text: The `text` attribute.
    content_desc: The `content_desc` attribute.

  Returns:
    The object name string.
  """
  return text if text else content_desc


def _build_bounding_box(bounds):
  """Returns the object bounding box based on `bounds` attribute.

  Args:
    bounds: The `bounds` attribute.

  Returns:
    The BoundingBox object.
  """
  match = re.compile(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]').match(bounds)
  assert match
  x1, y1, x2, y2 = match.groups()
  return BoundingBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2))


def _build_clickable(element, tree_child_as_clickable=True):
  """Returns whether the element is clickable or one of its ancestors is.

  Args:
    element: The etree.Element object.
    tree_child_as_clickable: treat all tree children as clickable

  Returns:
    A boolean to indicate whether the element is clickable or one of its
    ancestors is.
  """
  clickable = element.get('clickable')
  if clickable == 'false':
    for node in element.iterancestors():
      if node.get('clickable') == 'true':
        clickable = 'true'
        break

  # Below code is try to fix that: some target UI have 'clickable==False'
  # but it's clickable by human actually

  # Checkable elemnts should also be treated as clickable
  # Some menu items may have clickable==False but checkable==True
  if element.get('checkable') == 'true':
    clickable = 'true'

  # TODO(zhouxin) This plan is not fully decided so no UT covering for now
  if tree_child_as_clickable:
    p = element.getparent()
    while p:
      if p.get('class') == 'android.widget.ListView':
        clickable = 'true'
        break
      p = p.getparent()

  return strtobool(clickable)


def _pixel_distance(a_x1, a_x2, b_x1, b_x2):
  """Calculates the pixel distance between bounding box a and b.

  Args:
    a_x1: The x1 coordinate of box a.
    a_x2: The x2 coordinate of box a.
    b_x1: The x1 coordinate of box b.
    b_x2: The x2 coordinate of box b.

  Returns:
    The pixel distance between box a and b on the x axis. The distance
    on the y axis can be calculated in the same way. The distance can be
    positive number (b is right/bottom to a) and negative number
    (b is left or top to a).
  """
  # if a and b are close enough, then we set the their distance to be 1
  # because there are typically padding spaces inside an object's bounding box
  if b_x1 <= a_x2 and a_x2 - b_x1 <= ADJACENT_BOUNDING_BOX_THRESHOLD:
    return 1
  if a_x1 <= b_x2 and b_x2 - a_x1 <= ADJACENT_BOUNDING_BOX_THRESHOLD:
    return -1
  # overlap
  if (a_x1 <= b_x1 <= a_x2) or (a_x1 <= b_x2 <= a_x2) or (
      b_x1 <= a_x1 <= b_x2) or (b_x1 <= a_x2 <= b_x2):
    return 0
  elif b_x1 > a_x2:
    return b_x1 - a_x2
  else:
    return b_x2 - a_x1


def _grid_coordinate(x, width):
  """Calculates the 3x3 grid coordinate on the x axis.

  The grid coordinate on the y axis is calculated in the same way.

  Args:
    x: The x coordinate: [0, width).
    width: The screen width.

  Returns:
    The grid coordinate: [0, 2].
    Note that the screen is divided into 3x3 grid, so the grid coordinate
    uses the number from 0, 1, 2.
  """
  assert 0 <= x <= width
  grid_x_0 = width / 3
  grid_x_1 = 2 * grid_x_0
  if 0 <= x < grid_x_0:
    grid_coordinate_x = 0
  elif grid_x_0 <= x < grid_x_1:
    grid_coordinate_x = 1
  else:
    grid_coordinate_x = 2
  return grid_coordinate_x


def _grid_location(bbox, screen_width, screen_height):
  """Calculates the grid number of the UI object's bounding box.

  The screen can be divided into 3x3 grid:
  (0, 0) (0, 1) (0, 2)        0   1   2
  (1, 0) (1, 1) (1, 2)  --->  3   4   5
  (2, 0) (2, 1) (2, 2)        6   7   8

  Args:
    bbox: The bounding box of the UI object.
    screen_width: The width of the screen associated with the hierarchy.
    screen_height: The height of the screen associated with the hierarchy.

  Returns:
    The grid location number.
  """
  bbox_center_x = (bbox.x1 + bbox.x2) / 2
  bbox_center_y = (bbox.y1 + bbox.y2) / 2
  bbox_grid_x = _grid_coordinate(bbox_center_x, screen_width)
  bbox_grid_y = _grid_coordinate(bbox_center_y, screen_height)
  return UIObjectGridLocation(bbox_grid_y * 3 + bbox_grid_x)


def _build_etree_from_json(root, json_dict):
  """Builds the element tree from json_dict.

  Args:
    root: The current etree root node.
    json_dict: The current json_dict corresponding to the etree root node.
  """
  # set node attributes
  if root is None or json_dict is None:
    return
  # Set object id as traversal_id|pointer, used by different downstream apps.
  root.set(
      'obj_id', '{}|{}'.format(
          json_dict.get('_traversal_id'), json_dict.get('pointer', '')))
  x1, y1, x2, y2 = json_dict.get('bounds', [0, 0, 0, 0])
  root.set('bounds', '[%d,%d][%d,%d]' % (x1, y1, x2, y2))
  root.set('class', json_dict.get('class', ''))
  if 'ancestors' in json_dict:
    root.set('ancestors_class', ','.join(json_dict.get('ancestors', [])))
  # XML element cannot contain NULL bytes.
  root.set('text', json_dict.get('text', '').replace('\x00', ''))
  root.set('resource-id', json_dict.get('resource-id', ''))
  content_desc = json_dict.get('content-desc', [None])
  root.set(
      'content-desc',
      '' if content_desc[0] is None else content_desc[0].replace('\x00', ''))
  root.set('package', json_dict.get('package', ''))
  root.set(
      'visible',
      str(
          json_dict.get('visible-to-user', True) and
          json_dict.get('visibility', 'visible') == 'visible'))
  root.set('enabled', str(json_dict.get('enabled', False)))
  root.set('focusable', str(json_dict.get('focusable', False)))
  root.set('focused', str(json_dict.get('focused', False)))
  root.set(
      'scrollable',
      str(
          json_dict.get('scrollable-horizontal', False) or
          json_dict.get('scrollable-vertical', False)))
  root.set('clickable', str(json_dict.get('clickable', False)))
  root.set('long-clickable', str(json_dict.get('long-clickable', False)))
  root.set('selected', str(json_dict.get('selected', False)))
  if 'children' not in json_dict:  # leaf node
    return
  for child in json_dict['children']:
    # some json file has 'null' as one of the children.
    if child:
      child_node = etree.Element('node')
      root.append(child_node)
      _build_etree_from_json(child_node, child)


def generate_traversal_id_in_json(root):
  """Generate traversal id for objects in json view hierarchy.

  Traverse the view tree in a breadth-first manner. The object id is the list of
  child index that leads from the root to the object. For example, the root node
  has id `0`, the second child of the root has an id `0.1`.

  Args:
    root: the root node of the view.
  """
  if not root:
    return

  root['_traversal_id'] = '0'
  node_list = [root]

  for node in node_list:
    children = node.get('children', [])

    # Add children nodes to the list.
    for index, child in enumerate(children):
      if not child:
        # We don't remove None child beforehand as we want to keep the index
        # unchanged, so that we can use it to fetch a specific child in the list
        # directly.
        continue
      child['_traversal_id'] = '{}.{}'.format(node['_traversal_id'], index)
      node_list.append(child)


class Node(object):
  """Represents a node in the view hierarchy data from xml."""

  def __init__(self,
               element,
               dom_location=None,
               screen_width=SCREEN_WIDTH,
               screen_height=SCREEN_HEIGHT,
               idx=0,
               type_num=15,
               use_ancestors_class=False,
               fix_type=False):
    """Constructor.

    Args:
      element: The etree.Element object.
      dom_location: [depth, preorder-index, postorder-index] of element.
      screen_width: The width of the screen associated with the element.
      screen_height: The height of the screen associated with the element.
      idx: Its index in the generated node list.
      type_num: Seq2act use types from 0-14, set the default value for backward
        compatibility. -1 means all.
      use_ancestors_class: bool, current class is UNKNOWN, consider the classes
        of ancestors. Now only json support this.
      fix_type: when type is not matched, try match with its surfix.
    """
    is_leaf = not element.findall('.//node')
    self.element = element
    self._screen_width = screen_width
    self._screen_height = screen_height
    bbox = _build_bounding_box(element.get('bounds'))
    self.uiobject = UIObject(
        obj_id=element.get('obj_id', ''),
        obj_index=idx,
        parent_id=element.get('parent_id'),
        parent_index=None,  # need to be porpulated later
        obj_type=_build_object_type(
            element.get('class'),
            type_num,
            element.get('ancestors_class').split(',')
            if use_ancestors_class and element.get('ancestors_class') else None,
            fix_type=fix_type),
        obj_name=_build_object_name(
            element.get('text'), element.get('content_desc')),
        word_sequence=_build_word_sequence(
            element.get('text'), element.get('content-desc'),
            element.get('resource-id')),
        text=element.get('text'),
        resource_id=element.get('resource-id'),
        android_class=element.get('class'),
        android_package=element.get('package'),
        content_desc=element.get('content-desc'),
        clickable=_build_clickable(element),
        visible=strtobool(element.get('visible', default='true')),
        enabled=strtobool(element.get('enabled')),
        focusable=strtobool(element.get('focusable')),
        focused=strtobool(element.get('focused')),
        scrollable=strtobool(element.get('scrollable')),
        long_clickable=strtobool(element.get('long-clickable')),
        selected=strtobool(element.get('selected')),
        bounding_box=bbox,
        grid_location=_grid_location(bbox, self._screen_width,
                                     self._screen_height),
        dom_location=dom_location,
        is_leaf=is_leaf,
        checkable=strtobool(element.get('checkable', default='false')),
        checked=strtobool(element.get('checked', default='false')))

  def normalized_pixel_distance(self, other_node):
    """Calculates normalized pixel distance between this node and other node.

    Args:
      other_node: Another Node object.

    Returns:
      Normalized pixel distance on both horizontal and vertical direction.
    """
    h_distance = _pixel_distance(self.uiobject.bounding_box.x1,
                                 self.uiobject.bounding_box.x2,
                                 other_node.uiobject.bounding_box.x1,
                                 other_node.uiobject.bounding_box.x2)
    v_distance = _pixel_distance(self.uiobject.bounding_box.y1,
                                 self.uiobject.bounding_box.y2,
                                 other_node.uiobject.bounding_box.y1,
                                 other_node.uiobject.bounding_box.y2)

    return float(h_distance) / self._screen_width, float(
        v_distance) / self._screen_height

  def dom_distance(self, other_node):
    """Calculates dom distance between this node and other node.

    Args:
      other_node: Another Node object.

    Returns:
      The dom distance in between two leaf nodes: defined as the number of
      nodes on the path from one leaf node to the other on the tree.
    """
    intersection = [
        node for node in self.element.iterancestors()
        if node in other_node.element.iterancestors()
    ]
    assert intersection
    ancestor_list = list(self.element.iterancestors())
    other_ancestor_list = list(other_node.element.iterancestors())
    return ancestor_list.index(intersection[0]) + other_ancestor_list.index(
        intersection[0]) + 1


class DomLocationKey(Enum):
  """Keys of dom location info."""
  DEPTH = 0
  PREORDER_INDEX = 1
  POSTORDER_INDEX = 2


class ViewHierarchy(object):
  """Represents the view hierarchy data from UIAutomator dump."""

  def __init__(self, screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT):
    """Constructor.

    Args:
      screen_width: The pixel width of the screen for the view hierarchy.
      screen_height: The pixel height of the screen for the view hierarchy.
    """
    self._root = None
    self._root_element = None
    self._all_elements = []
    self._all_visible_leaves = []
    self._dom_location_dict = None
    self._screen_width = screen_width
    self._screen_height = screen_height

    self.max_children = 0
    self.pick_from_multiple_roots = False

  def load_json(self, json_content):
    """Builds the etree from json content.

    Args:
      json_content: The string containing json content.
    """
    json_dict = json.loads(json_content)
    if json_dict is None:
      raise ValueError('empty json file.')

    # Generate traversal ids.
    generate_traversal_id_in_json(json_dict['activity']['root'])

    self._root = etree.Element('hierarchy', rotation='0')
    self._root_element = etree.Element('node')
    self._root.append(self._root_element)
    self._root.set('obj_id', '-1')
    _build_etree_from_json(self._root_element, json_dict['activity']['root'])

    self._cache_elements()
    self._dom_location_dict = self._calculate_dom_location()

  def get_all_ui_objects(self,
                         scale=10,
                         use_ancestors_class=True,
                         include_invisible=False):
    """Returns all the UI objects.

    Args:
      scale: the times to scale the bitmap smaller. build a scaled_resolution
        int 2D array, each point is an int represents the pre-order index of
        node.
      use_ancestors_class: bool, current class is UNKNOWN, consider the classes
        of ancestors. Now only json support this.
      include_invisible: bool, whether in include invisible nodes.

    Returns:
      all_ui_objects, bitmap, missing_small_objects_in_bitmap, screen_state
    """
    # pylint: disable=g-complex-comprehension
    elements = (
        self._all_elements
        if include_invisible else self._all_valid_visible_elements)
    all_ui_objects = [
        Node(
            element,
            self._dom_location_dict[id(element)],
            self._screen_width,
            self._screen_height,
            idx,
            type_num=-1,
            use_ancestors_class=use_ancestors_class,
            fix_type=True).uiobject for idx, element in enumerate(elements)
    ]

    # Set parent index
    id_index_map = {o.obj_id: index for index, o in enumerate(all_ui_objects)}
    for o in all_ui_objects:
      if o.parent_id == '-1':
        o.parent_index = -1
      else:
        o.parent_index = id_index_map[o.parent_id]

    bitmap = None
    missing_small_objects_in_bitmap = []
    width, height = self._screen_width // scale, self._screen_height // scale
    bitmap = np.full(shape=[height, width], fill_value=-1, dtype=np.int32)

    for obj in all_ui_objects:
      box = obj.bounding_box

      # For x1 and y1, we need use ceil to make sure that after scale back to
      # original resolution the clicked point is still correct. Example:
      # [x1, x2] = [11, 35] and scale=10,
      #   [1, 3] is wrong, imagine the caller choose 1 and scale back to 10
      #   [2, 3] is correct
      x1 = math.ceil(box.x1 / scale)
      y1 = math.ceil(box.y1 / scale)
      x2 = math.floor(box.x2 / scale)
      y2 = math.floor(box.y2 / scale)
      for i in range(y1, y2):
        for j in range(x1, x2):
          bitmap[i][j] = obj.obj_index

      if x1 >= x2 and y1 >= y2:
        missing_small_objects_in_bitmap.append(obj)

    screen_state = self._uiobjects_to_screenstate_proto(all_ui_objects)

    return all_ui_objects, bitmap, missing_small_objects_in_bitmap, screen_state

  def _uiobjects_to_screenstate_proto(self, all_ui_objects, search_range=5):
    """Processes ui objects to observation_pb2.ScreenState."""
    screen_state = observation_pb2.ScreenState()
    label_used = set()

    def get_text(obj):
      # To get the 'name' of the a state element, we only consider 'text' or
      # 'content-desc' of the neighbours. 'resource-id' will introduce too many
      # noices, thus _build_word_sequence() is NOT used here.
      return obj.text.strip() or obj.content_desc.strip()

    need_update_value = {}  # {state_index: state_value}
    for obj in all_ui_objects:
      if obj.checkable:
        name = get_text(obj)
        value = 'true' if obj.checked else 'false'

        if not name or name.lower() in ['on', 'off']:
          # Search name for this element by looking forward and backward
          # Toggle buttons in Settings: there could be 2 texts associated w/ it
          # at left side, one has resource_id='android:id/title', and the other
          # has resource_id='android:id/summary', the 'title' one is preferred.
          start = max(obj.obj_index - search_range, 0)
          end = min(obj.obj_index + search_range + 1, len(all_ui_objects))
          max_prefix = ''
          title_found = False  # whether a 'title' neighbour found already

          neighbor_index_to_update = -1
          neighbor_name = None
          for neighbor_index in range(start, end):
            neighbor = all_ui_objects[neighbor_index]
            if neighbor.checkable or neighbor.obj_index in label_used:
              continue

            if get_text(neighbor):
              is_title = neighbor.resource_id == 'android:id/title'
              need_update = False
              if title_found and not is_title:
                need_update = False
              elif not title_found and is_title:
                need_update = True
              elif len(os.path.commonprefix([neighbor.obj_id, obj.obj_id
                                            ])) > len(max_prefix):
                need_update = True

              if need_update:
                neighbor_name = get_text(neighbor)
                neighbor_index_to_update = neighbor_index
                label_used.add(neighbor.obj_index)
                max_prefix = os.path.commonprefix([neighbor.obj_id, obj.obj_id])
                title_found = is_title

          if neighbor_name:
            need_update_value[neighbor_index_to_update] = value
            name = neighbor_name
          else:
            logging.error('Not found neighbor for checkable obj: %s', str(obj))

        screen_state.state_fields.add(name=name, value=value)
      else:  # if not obj.checkable:
        name = get_text(obj)
        screen_state.state_fields.add(name=name, value='')

    for neighbor_index_to_update, value in need_update_value.items():
      screen_state.state_fields[neighbor_index_to_update].value = value

    assert len(all_ui_objects) == len(screen_state.state_fields)
    return screen_state

  def get_all_text(self):
    """Returns a generator of all text in the view hierarchy."""
    for element in self._all_elements:
      all_text = []
      text = element.get('text', None)
      content = element.get('content_desc', [])
      if isinstance(content, str):
        all_text = [text, content]
      elif isinstance(content, list):
        all_text = content + [text]
      else:
        logging.warning('Can not extract text from node %s', element)

      for text in all_text:
        if text and text.strip():
          yield text

  def _cache_elements(self):
    """Catches all the elements in list."""
    # The view hierarchy has a <hierarchy> at top, which contains NO valuable
    # attributes, so when collecting we start from _root_element (_root[0])
    self._all_elements = list(self._root_element.iter('*'))

    self._all_visible_leaves = [
        element for element in self._all_elements if self._is_leaf(element) and
        strtobool(element.get('visible', default='true')) and
        self._is_within_screen_bound(element)
    ]

    # pylint: disable=g-complex-comprehension
    self._all_valid_visible_elements = [
        element for element in self._all_elements
        if strtobool(element.get('visible', default='true')) and
        self._is_within_screen_bound(element) and
        self._is_within_ancestors_bound(element)
    ]

  def _calculate_dom_location(self, generate_obj_id=False):
    """Calculate [depth, preorder-index, postorder-index] and obj_id of nodes.

    All elements will be filted and cached in self._all_visible_leaves.
    This is necessary because dom_location_dict use id(element) as keys, if
    call _root.iter('*') every time, the id(element) will not be a fixed value
    even for same element in XML.

    Args:
      generate_obj_id: whether generate obj_id by dom location. The generated id
        will be like "0.0.1.0.2"

    Returns:
      dom_location_dict, dict of
        {id(element): [depth, preorder-index, postorder-index]}
    """
    dom_location_dict = collections.defaultdict(lambda: [None, None, None])
    # Calculate the depth of all nodes.
    for element in self._all_elements:
      ancestors = list(element.iterancestors())
      dom_location_dict[id(element)][DomLocationKey.DEPTH.value] = len(
          ancestors)

    self._root.set('obj_id', ROOT_ID)
    if generate_obj_id:
      self._root_element.set('obj_id', '0')
    self._root_element.set('parent_id', ROOT_ID)
    # Calculate the pre/post index by calling pre/post iteration recursively.
    self._pre_order_iterate(
        self._root_element,
        dom_location_dict,
        next_index=1,
        generate_obj_id=generate_obj_id)
    self._post_order_iterate(
        self._root_element, dom_location_dict, next_index=0)
    return dom_location_dict

  def _pre_order_iterate(self, element, dom_location_dict, next_index,
                         generate_obj_id):
    """Preorder travel on hierarchy tree.

    Args:
      element: etree element which will be visited now.
      dom_location_dict: dict of
        {id(element): [depth, preorder-index, postorder-index]}
      next_index: next available index.
      generate_obj_id: whether generate obj_id by dom location. The generated id
        will be like "0.0.1.0.2"

    Returns:
      next_index: next available index.
    """
    dom_location_dict[id(element)][
        DomLocationKey.PREORDER_INDEX.value] = next_index
    next_index += 1

    child_index = 0
    num_children = 0
    for child in element:
      if child.getparent() == element:
        num_children += 1
        if generate_obj_id:
          child.set('obj_id', element.get('obj_id') + '.' + str(child_index))
        child.set('parent_id', element.get('obj_id'))
        next_index = self._pre_order_iterate(child, dom_location_dict,
                                             next_index, generate_obj_id)
        child_index += 1
    self.max_children = max(self.max_children, num_children)
    return next_index

  def _post_order_iterate(self, element, dom_location_dict, next_index):
    """Postorder travel on hierarchy tree.

    Args:
      element: etree element which will be visited now.
      dom_location_dict: dict of
        {id(element): [depth, preorder-index, postorder-index]}
      next_index: next available index.

    Returns:
      next_index: next available index.
    """
    for child in element:
      if child.getparent() == element:
        next_index = self._post_order_iterate(child, dom_location_dict,
                                              next_index)
    dom_location_dict[id(element)][
        DomLocationKey.POSTORDER_INDEX.value] = next_index
    return next_index + 1

  def _is_leaf(self, element):
    """Whether an etree element is leaf in hierarchy tree."""
    return not element.findall('.//node')

  def _is_within_screen_bound(self, element):
    """Whether an etree element's bounding box is within screen boundary."""
    if '-' in element.get('bounds'):  # nagative numbers
      return False
    bbox = _build_bounding_box(element.get('bounds'))
    in_x = (0 <= bbox.x1 <= self._screen_width) and (0 <= bbox.x2 <=
                                                     self._screen_width)
    in_y = (0 <= bbox.y1 <= self._screen_height) and (0 <= bbox.y2 <=
                                                      self._screen_height)
    x1_less_than_x2 = bbox.x1 < bbox.x2
    y1_less_than_y2 = bbox.y1 < bbox.y2
    return in_x and in_y and x1_less_than_x2 and y1_less_than_y2

  def _is_within_ancestors_bound(self, element):
    """Whether an etree element's bounding box within ancestors bboxes."""
    if '-' in element.get('bounds'):  # nagative numbers
      return False
    if not element.getparent().get('bounds'):  # root
      return True
    if not self._is_within_ancestors_bound(element.getparent()):
      return False

    bbox = _build_bounding_box(element.get('bounds'))
    parent_bbox = _build_bounding_box(element.getparent().get('bounds'))
    return (bbox.x1 >= parent_bbox.x1 and bbox.x2 <= parent_bbox.x2 and
            bbox.y1 >= parent_bbox.y1 and bbox.y2 <= parent_bbox.y2)
