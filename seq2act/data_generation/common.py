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

"""Functions shared among files under word2act/data_generation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import attr
from enum import Enum
import numpy as np
import tensorflow.compat.v1 as tf  # tf

from seq2act.data_generation import config
from seq2act.data_generation import view_hierarchy


gfile = tf.gfile


@attr.s
class MaxValues(object):
  """Represents max values for a task and UI."""

  # For instrction
  max_word_num = attr.ib(default=None)
  max_word_length = attr.ib(default=None)

  # For UI objects
  max_ui_object_num = attr.ib(default=None)
  max_ui_object_word_num = attr.ib(default=None)
  max_ui_object_word_length = attr.ib(default=None)

  def update(self, other):
    """Update max value from another MaxValues instance.

    This will be used when want to merge several MaxValues instances:

      max_values_list = ...
      result = MaxValues()
      for v in max_values_list:
        result.update(v)

    Then `result` contains merged max values in each field.

    Args:
      other: another MaxValues instance, contains updated data.
    """
    self.max_word_num = max(self.max_word_num, other.max_word_num)
    self.max_word_length = max(self.max_word_length, other.max_word_length)
    self.max_ui_object_num = max(self.max_ui_object_num,
                                 other.max_ui_object_num)
    self.max_ui_object_word_num = max(self.max_ui_object_word_num,
                                      other.max_ui_object_word_num)
    self.max_ui_object_word_length = max(self.max_ui_object_word_length,
                                         other.max_ui_object_word_length)


class ActionRules(Enum):
  """The rule_id to generate synthetic action."""
  SINGLE_OBJECT_RULE = 0
  GRID_CONTEXT_RULE = 1
  NEIGHBOR_CONTEXT_RULE = 2
  SWIPE_TO_OBJECT_RULE = 3
  SWIPE_TO_DIRECTION_RULE = 4
  REAL = 5  # The action is not generated, but a real user action.
  CROWD_COMPUTE = 6
  DIRECTION_VERB_RULE = 7  # For win, "click button under some tab/combobox
  CONSUMED_MULTI_STEP = 8  # For win, if the target verb is not direction_verb
  UNCONSUMED_MULTI_STEP = 9
  NO_VERB_RULE = 10


class ActionTypes(Enum):
  """The action types and ids of Android actions."""
  CLICK = 2
  INPUT = 3
  SWIPE = 4
  CHECK = 5
  UNCHECK = 6
  LONG_CLICK = 7
  OTHERS = 8
  GO_HOME = 9
  GO_BACK = 10


VERB_ID_MAP = {
    'check': ActionTypes.CHECK,
    'find': ActionTypes.SWIPE,
    'navigate': ActionTypes.SWIPE,
    'uncheck': ActionTypes.UNCHECK,
    'head to': ActionTypes.SWIPE,
    'enable': ActionTypes.CHECK,
    'turn on': ActionTypes.CHECK,
    'locate': ActionTypes.SWIPE,
    'disable': ActionTypes.UNCHECK,
    'tap and hold': ActionTypes.LONG_CLICK,
    'long press': ActionTypes.LONG_CLICK,
    'look': ActionTypes.SWIPE,
    'press and hold': ActionTypes.LONG_CLICK,
    'turn it on': ActionTypes.CHECK,
    'turn off': ActionTypes.UNCHECK,
    'switch on': ActionTypes.CHECK,
    'visit': ActionTypes.SWIPE,
    'hold': ActionTypes.LONG_CLICK,
    'switch off': ActionTypes.UNCHECK,
    'head': ActionTypes.SWIPE,
    'head over': ActionTypes.SWIPE,
    'long-press': ActionTypes.LONG_CLICK,
    'un-click': ActionTypes.UNCHECK,
    'tap': ActionTypes.CLICK,
    'check off': ActionTypes.UNCHECK,
    # 'power on': 21
}


class WinActionTypes(Enum):
  """The action types and ids of windows actions."""
  LEFT_CLICK = 2
  RIGHT_CLICK = 3
  DOUBLE_CLICK = 4
  INPUT = 5


@attr.s
class Action(object):
  """The class for a word2act action."""
  instruction_str = attr.ib(default=None)
  verb_str = attr.ib(default=None)
  obj_desc_str = attr.ib(default=None)
  input_content_str = attr.ib(default=None)
  action_type = attr.ib(default=None)
  action_rule = attr.ib(default=None)
  target_obj_idx = attr.ib(default=None)
  obj_str_pos = attr.ib(default=None)
  input_str_pos = attr.ib(default=None)
  verb_str_pos = attr.ib(default=None)
  # start/end position of one whole step
  step_str_pos = attr.ib(default=[0, 0])
  # Defalt action is 1-step consumed action
  is_consumed = attr.ib(default=True)

  def __eq__(self, other):
    if not isinstance(other, Action):
      return NotImplemented
    return self.instruction_str == other.instruction_str

  def is_valid(self):
    """Does valid check for action instance.

    Returns true when any component is None or obj_desc_str is all spaces.

    Returns:
      a boolean
    """
    invalid_obj_pos = (np.array(self.obj_str_pos) == 0).all()
    if (not self.instruction_str or invalid_obj_pos or
        not self.obj_desc_str.strip()):
      return False

    return True

  def has_valid_input(self):
    """Does valid check for input positions.

    Returns true when input_str_pos is not all default value.

    Returns:
      a boolean
    """
    return (self.input_str_pos != np.array([
        config.LABEL_DEFAULT_VALUE_INT, config.LABEL_DEFAULT_VALUE_INT
    ])).any()

  def regularize_strs(self):
    """Trims action instance's obj_desc_str, input_content_str, verb_str."""
    self.obj_desc_str = self.obj_desc_str.strip()
    self.input_content_str = self.input_content_str.strip()
    self.verb_str = self.verb_str.strip()

  def convert_to_lower_case(self):
    self.instruction_str = self.instruction_str.lower()
    self.obj_desc_str = self.obj_desc_str.lower()
    self.input_content_str = self.input_content_str.lower()
    self.verb_str = self.verb_str.lower()


@attr.s
class ActionEvent(object):
  """This class defines ActionEvent class.

  ActionEvent is high level event summarized from low level android event logs.
  This example shows the android event logs and the extracted ActionEvent
  object:

  Android Event Logs:
  [      42.407808] EV_ABS       ABS_MT_TRACKING_ID   00000000
  [      42.407808] EV_ABS       ABS_MT_TOUCH_MAJOR   00000004
  [      42.407808] EV_ABS       ABS_MT_PRESSURE      00000081
  [      42.407808] EV_ABS       ABS_MT_POSITION_X    00004289
  [      42.407808] EV_ABS       ABS_MT_POSITION_Y    00007758
  [      42.407808] EV_SYN       SYN_REPORT           00000000
  [      42.453256] EV_ABS       ABS_MT_PRESSURE      00000000
  [      42.453256] EV_ABS       ABS_MT_TRACKING_ID   ffffffff
  [      42.453256] EV_SYN       SYN_REPORT           00000000

  This log can be generated from this command during runing android emulator:
  adb shell getevent -lt /dev/input/event1

  If screen pixel size is [480,800], this is the extracted ActionEvent Object:
    ActionEvent(
      event_time = 42.407808
      action_type = ActionTypes.CLICK
      action_object_id = -1
      coordinates_x = [17033,]
      coordinates_y = [30552,]
      coordinates_x_pixel = [249,]
      coordinates_y_pixel = [747,]
      action_params = []
    )
  """

  event_time = attr.ib()
  action_type = attr.ib()
  coordinates_x = attr.ib()
  coordinates_y = attr.ib()
  action_params = attr.ib()
  # These fields will be generated by public method update_info_from_screen()
  coordinates_x_pixel = None
  coordinates_y_pixel = None
  object_id = config.LABEL_DEFAULT_INVALID_INT
  leaf_nodes = None  # If dedup, the nodes here will be less than XML
  debug_target_object_word_sequence = None

  def update_info_from_screen(self, screen_info, dedup=False):
    """Updates action event attributes from screen_info.

    Updates coordinates_x(y)_pixel and object_id from the screen_info proto.

    Args:
      screen_info: ScreenInfo protobuf
      dedup: whether dedup the UI objs with same text or content desc.
    Raises:
      ValueError when fail to find object id.
    """
    self.update_norm_coordinates((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    vh = view_hierarchy.ViewHierarchy()
    vh.load_xml(screen_info.view_hierarchy.xml.encode('utf-8'))
    if dedup:
      vh.dedup((self.coordinates_x_pixel[0], self.coordinates_y_pixel[0]))
    self.leaf_nodes = vh.get_leaf_nodes()
    ui_object_list = vh.get_ui_objects()
    self._update_object_id(ui_object_list)

  def _update_object_id(self, ui_object_list):
    """Updates ui object index from view_hierarchy.

    If point(X,Y) surrounded by multiple UI objects, select the one with
    smallest area.

    Args:
      ui_object_list: .
    Raises:
      ValueError when fail to find object id.
    """
    smallest_area = -1
    for index, ui_obj in enumerate(ui_object_list):
      box = ui_obj.bounding_box
      if (box.x1 <= self.coordinates_x_pixel[0] <= box.x2 and
          box.y1 <= self.coordinates_y_pixel[0] <= box.y2):
        area = (box.x2 - box.x1) * (box.y2 - box.y1)
        if smallest_area == -1 or area < smallest_area:
          self.object_id = index
          self.debug_target_object_word_sequence = ui_obj.word_sequence
          smallest_area = area

    if smallest_area == -1:
      raise ValueError(('Object id not found: x,y=%d,%d coordinates fail to '
                        'match every UI bounding box') %
                       (self.coordinates_x_pixel[0],
                        self.coordinates_y_pixel[0]))

  def update_norm_coordinates(self, screen_size):
    """Update coordinates_x(y)_norm according to screen_size.

    self.coordinate_x is scaled between [0, ANDROID_LOG_MAX_ABS_X]
    self.coordinate_y is scaled between [0, ANDROID_LOG_MAX_ABS_Y]
    This function recovers coordinate of android event logs back to coordinate
    in real screen's pixel level.

    coordinates_x_pixel = coordinates_x/ANDROID_LOG_MAX_ABS_X*horizontal_pixel
    coordinates_y_pixel = coordinates_y/ANDROID_LOG_MAX_ABS_Y*vertical_pixel

    For example,
    ANDROID_LOG_MAX_ABS_X = ANDROID_LOG_MAX_ABS_Y = 32676
    coordinate_x = [17033, ]
    object_cords_y = [30552, ]
    screen_size = (480, 800)
    Then the updated pixel coordinates are as follow:
      coordinates_x_pixel = [250, ]
      coordinates_y_pixel = [747, ]

    Args:
      screen_size: a tuple of screen pixel size.
    """
    (horizontal_pixel, vertical_pixel) = screen_size
    self.coordinates_x_pixel = [
        int(cord * horizontal_pixel / config.ANDROID_LOG_MAX_ABS_X)
        for cord in self.coordinates_x
    ]
    self.coordinates_y_pixel = [
        int(cord * vertical_pixel / config.ANDROID_LOG_MAX_ABS_Y)
        for cord in self.coordinates_y
    ]


# For Debug: Get distribution info for each cases
word_num_distribution_dict = collections.defaultdict(int)
word_length_distribution_dict = collections.defaultdict(int)


def get_word_statistics(file_path):
  """Calculates maximum word number/length from ui objects in one xml/json file.

  Args:
    file_path: The full path of a xml/json file.

  Returns:
    A tuple (max_word_num, max_word_length)
      ui_object_num: UI object num.
      max_word_num: The maximum number of words contained in all ui objects.
      max_word_length: The maximum length of words contained in all ui objects.
  """
  max_word_num = 0
  max_word_length = 0

  leaf_nodes = get_view_hierarchy_list(file_path)
  for view_hierarchy_object in leaf_nodes:
    word_sequence = view_hierarchy_object.uiobject.word_sequence
    max_word_num = max(max_word_num, len(word_sequence))
    word_num_distribution_dict[len(word_sequence)] += 1

    for word in word_sequence:
      max_word_length = max(max_word_length, len(word))
      word_length_distribution_dict[len(word)] += 1
  return len(leaf_nodes), max_word_num, max_word_length


def get_ui_max_values(file_paths):
  """Calculates max values from ui objects in multi xml/json files.

  Args:
    file_paths: The full paths of multi xml/json files.
  Returns:
    max_values: instrance of MaxValues.
  """
  max_values = MaxValues()
  for file_path in file_paths:
    (ui_object_num,
     max_ui_object_word_num,
     max_ui_object_word_length) = get_word_statistics(file_path)

    max_values.max_ui_object_num = max(
        max_values.max_ui_object_num, ui_object_num)
    max_values.max_ui_object_word_num = max(
        max_values.max_ui_object_word_num, max_ui_object_word_num)
    max_values.max_ui_object_word_length = max(
        max_values.max_ui_object_word_length, max_ui_object_word_length)
  return max_values


def get_ui_object_list(file_path):
  """Gets ui object list from view hierarchy leaf nodes.

  Args:
    file_path: file path of xml or json
  Returns:
    A list of ui objects according to view hierarchy leaf nodes.
  """

  vh = _get_view_hierachy(file_path)
  return vh.get_ui_objects()


def get_view_hierarchy_list(file_path):
  """Gets view hierarchy leaf node list.

  Args:
    file_path: file path of xml or json
  Returns:
    A list of view hierarchy leaf nodes.
  """
  vh = _get_view_hierachy(file_path)
  return vh.get_leaf_nodes()


def _get_view_hierachy(file_path):
  """Gets leaf nodes view hierarchy lists.

  Args:
    file_path: The full path of an input xml/json file.
  Returns:
    A ViewHierarchy object.
  Raises:
    ValueError: unsupported file format.
  """
  with gfile.GFile(file_path, 'r') as f:
    data = f.read()

  _, file_extension = os.path.splitext(file_path)
  if file_extension == '.xml':
    vh = view_hierarchy.ViewHierarchy(
        screen_width=config.SCREEN_WIDTH, screen_height=config.SCREEN_HEIGHT)
    vh.load_xml(data)
  elif file_extension == '.json':
    vh = view_hierarchy.ViewHierarchy(
        screen_width=config.RICO_SCREEN_WIDTH,
        screen_height=config.RICO_SCREEN_HEIGHT)
    vh.load_json(data)
  else:
    raise ValueError('unsupported file format %s' % file_extension)
  return vh
