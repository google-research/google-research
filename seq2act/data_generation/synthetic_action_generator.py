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

"""Generates synthetic actions from a list of ui objects on a screen.

The structure for generated actions follows the following rules:
  string_content* is for type action.
  grid_context: strings indicating 3x3 grid cell location, such as in TOP_RIGHT.
  neighbor_context: strings indicating two objects' spatial relationship,
                      such as 'on the top of'.
  swipe_context: string indicating swipe direction. such as UP/DOWN/TO_TOP/

1. Single object rule: If object A has valid name and valid type
  action = [verb, string_content*, object_name, object_type]
  target = A's index of the ui object list

  For example:
   Input google.com in the Search or type web address search box.
   |___| |________|        |________________________| |_________|
   verb   string           object_description_str    object_type_str
          content


2. Absolute location rule:
  Mobile screen is separated to 3*3 grids. If object A is the only object in
  one grid cell, and it has valid type:

  action = [verb, string_content*, object_type, grid_context]
  target = A's index of the ui object list

  For example:
   Type google.com in the search box at the top left corner.
   |__| |________|        |________| |____________________|
   verb   string           target A    grid_direction_str
          content         object type


3. Relative location rule: If object B is the closest neighbor of object A, and
  object A is clickable and object B has valid name and valid type:
  action = [verb, string_content*, A's object_type, neighbor_context, B's
  object_name, B's object_type]
  target = A's index of the ui object list

  For example:
   Type google.com in the search box on top of submit button
   |__| |________|        |________| |_______| |____| |_____|
   verb  string            target A   context    neighbor B
         content          object type           name    type


4. Swipe to object rule:
  action = [verb, swipe_object_str]
  target = -1
  params = [start_position, end_position]

  For example:
   Swipe to the submit button.
   |______| |_______________|
    verb     swipe_object_str


5. No verb rule:
  action = [object_str]
  target = target A
  verb_refs = [0, 0]
  verb_id = CLICK

  For example:

  Acknowledged & continue
  |_____________________|
       target A's
    object description



"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from enum import Enum
import numpy as np
import tensorflow.compat.v1 as tf

from seq2act.data_generation import common
from seq2act.data_generation import config
from seq2act.data_generation import proto_utils
from seq2act.data_generation import resources
from seq2act.data_generation import string_utils
from seq2act.data_generation import view_hierarchy

COMMA = ','


class Platform(Enum):
  """Andrdroid/Other Platform."""
  PLACE_HOLDER = 0
  ANDROID = 1


# Android
_CLICK_VERBS = [
    'click', 'tap', 'choose', 'press', 'select', 'launch', 'open', 'turn on',
    'turn off'
]
_TYPE_VERBS = ['type', 'enter', 'input', 'put', 'write']
_SWIPE_VERBS = ['swipe', 'flip', 'scroll', 'jump']

_CLICKABLE_OBJECT_DESC = {
    view_hierarchy.UIObjectType.BUTTON: ['button'],
    view_hierarchy.UIObjectType.CHECKEDTEXTVIEW: ['checkbox'],
    view_hierarchy.UIObjectType.CHECKBOX: ['checkbox'],
    view_hierarchy.UIObjectType.IMAGEBUTTON: ['icon'],
    view_hierarchy.UIObjectType.IMAGEVIEW: ['icon'],
    view_hierarchy.UIObjectType.RADIOBUTTON: ['radio button'],
    view_hierarchy.UIObjectType.TEXTVIEW: ['app'],
}
_TYPABLE_OBJECT_DESC = {view_hierarchy.UIObjectType.EDITTEXT: ['text box']}

_LOCATION_GRID_DICT = {
    'at the top left corner':
        view_hierarchy.UIObjectGridLocation.TOP_LEFT,
    'at the top':
        view_hierarchy.UIObjectGridLocation.TOP_CENTER,
    'at the top right corner':
        view_hierarchy.UIObjectGridLocation.TOP_RIGHT,
    'on the left':
        view_hierarchy.UIObjectGridLocation.LEFT,
    'at the center':
        view_hierarchy.UIObjectGridLocation.CENTER,
    'on the right':
        view_hierarchy.UIObjectGridLocation.RIGHT,
    'at the bottom left corner':
        view_hierarchy.UIObjectGridLocation.BOTTOM_LEFT,
    'at the bottom':
        view_hierarchy.UIObjectGridLocation.BOTTOM_CENTER,
    'at the bottom right corner':
        view_hierarchy.UIObjectGridLocation.BOTTOM_RIGHT,
}


class NeighborContextDesc(Enum):
  """The neighbor context relation's description."""
  TOP = ['above']
  BOTTOM = ['below']
  LEFT = ['to the left of', 'next to']
  RIGHT = ['to the right of', 'next to']


class SwipeContext(Enum):
  """The direction of swipe action.

  The 'desc' field is context description. The 'id' field is index of this
  context, which will be added to param as final label.
  """
  TO_TOP = {'desc': 'to the top', 'id': 0}
  TO_BOTTOM = {'desc': 'to the bottom', 'id': 1}
  UP = {'desc': 'up', 'id': 2}
  DOWN = {'desc': 'down', 'id': 3}
  LEFT = {'desc': 'to the left', 'id': 4}
  RIGHT = {'desc': 'to the right', 'id': 5}


def generate_all_actions(view_hierarchy_leaf_nodes, action_rules=('all')):
  """Generates actions and targets based on xml view hierarchy information.

  Args:
    view_hierarchy_leaf_nodes: list of view hierarchy leaf nodes.
    action_rules: used to specify which rules of actions to generate.

  Returns:
    A list of common.Action instances.
  """
  ui_object_list = [ele.uiobject for ele in view_hierarchy_leaf_nodes]
  vh_relation = proto_utils.get_view_hierarchy_leaf_relation(
      view_hierarchy_leaf_nodes)
  action_list = []
  if 'all' in action_rules or 'single' in action_rules:
    action_list.extend(
        _generate_single_object_actions(ui_object_list, Platform.ANDROID))
  if 'all' in action_rules or 'screen_loc' in action_rules:
    action_list.extend(_generate_absolute_location_action(ui_object_list))
  if 'all' in action_rules or 'neighbor_loc' in action_rules:
    action_list.extend(
        _generate_relative_location_action(ui_object_list,
                                           vh_relation['v_distance'],
                                           vh_relation['h_distance']))
  if 'all' in action_rules or 'swipe' in action_rules:
    action_list.extend(_generate_swipe_actions(ui_object_list))

  if 'all' in action_rules or 'no_verb' in action_rules:
    action_list.extend(_generate_no_verb_actions(ui_object_list))
  return action_list


def _generate_no_verb_actions(ui_object_list):
  """Generates action based on no verb rule, action_type should be CLICK.

  action = [object_description_str]

  Args:
    ui_object_list: list of ui objects

  Returns:
    A list of common.Action instances.
  """
  action_list = []
  action_rule = common.ActionRules.NO_VERB_RULE

  for object_index, ui_object in enumerate(ui_object_list):
    if _valid_clickable_object_with_name(ui_object):
      action_type = common.ActionTypes.CLICK
      verb_str = ''
      obj_desc_str = _truncate_name(ui_object.obj_name,
                                    config.MAX_OBJ_NAME_WORD_NUM)
      action = common.Action(
          verb_str=verb_str,
          obj_desc_str=obj_desc_str,
          input_content_str=config.LABEL_DEFAULT_VALUE_STRING,
          action_type=action_type,
          action_rule=action_rule,
          target_obj_idx=object_index)
      action_list.append(action)

  for action_element in action_list:
    _fill_action_info(action_element)
  return action_list


def _generate_swipe_actions(ui_object_list):
  """Generates swipe actions and targets.

  Rule:
    action = [verb, swipe_direction*, swipe_object_str*]
    target = ui_object's index in ui_object_list
    (* means optional)
  Args:
    ui_object_list:

  Returns:
    A list of common.Action instances.
  """
  action_list = []
  action_rule = common.ActionRules.SWIPE_TO_OBJECT_RULE

  for object_index, ui_object in enumerate(ui_object_list):
    if _valid_object_with_name(ui_object):
      (verb_str,
       action_type) = _get_verb_str_action_type('swipe', ui_object.obj_type)
      obj_desc_str = _get_obj_desc_str(action_rule, ui_object)
      action = common.Action(
          verb_str=verb_str,
          obj_desc_str=obj_desc_str,
          input_content_str=config.LABEL_DEFAULT_VALUE_STRING,
          action_type=action_type,
          action_rule=action_rule,
          target_obj_idx=object_index)
      action_list.append(action)

  for action_element in action_list:
    _fill_action_info(action_element)
  return action_list


def _generate_single_object_actions(ui_object_list, platform=Platform.ANDROID):
  """Generates single object actions and targets.

  Rule:
    action = [verb, string_content*, object_name, object_type]
    target = ui_object's index in ui_object_list

  Args:
    ui_object_list: a list of ui_objects
    platform: platform of UI objects.

  Returns:
    A list of common.Action instances.
  """
  action_list = []
  for object_index, ui_object in enumerate(ui_object_list):
    if hasattr(ui_object, 'bounding_box'):
      tf.logging.debug('ui_object.bounding_box: %s', [
          ui_object.bounding_box.x1, ui_object.bounding_box.y1,
          ui_object.bounding_box.x2, ui_object.bounding_box.y2
      ])
    action_list.extend(
        _generate_single_object_rule_action(ui_object, object_index, platform))
  return action_list


def _generate_absolute_location_action(ui_object_list):
  """Generates context grid actions and targets.

  Rule:
    action = [verb, string_context*, object_type, grid_context]
    target = ui_object's index in ui_object_list

  Args:
    ui_object_list: list of ui_objects

  Returns:
    A list of common.Action instances.
  """
  action_list = []
  for grid_direction_str, grid_num in _LOCATION_GRID_DICT.items():
    grid_objects_idx = [
        i for i in range(len(ui_object_list))
        if ui_object_list[i].grid_location == grid_num
    ]
    # If only one ui object locates in this grid, an action will be generated.
    if len(grid_objects_idx) == 1:
      object_in_grid = ui_object_list[grid_objects_idx[0]]
      action_list.extend(
          _generate_absolute_location_rule_action(object_in_grid,
                                                  grid_objects_idx[0],
                                                  grid_direction_str))
  return action_list


def _generate_relative_location_action(ui_object_list, ui_v_dist, ui_h_dist):
  """Generates context neighbor synthetic actions and targets.

  Rule:
  If Object B is closest neighbor of object A, and object A is clickable and
  object B has valid name and valid type.

    action = [verb, string_context*, A's object_type, neighbor_context, B's
    object_name, B's object_type]

    target = A's index in ui_object_list

  Args:
    ui_object_list: list of ui_objects
    ui_v_dist: ui objects' vertical distances. shape=[num_ui_obj, num_ui_obj]
    ui_h_dist: ui objects' horizontal distances. shape=[num_ui_obj, num_ui_obj]

  Returns:
    A list of common.Action instances.
  """
  action_list = []
  for object_idx, ui_object in enumerate(ui_object_list):
    if object_idx > ui_v_dist.shape[0]:
      assert False, ('ui_object_idx %d out of virtical distance bound %d' %
                     (object_idx, ui_v_dist.shape[0]))
    if object_idx > ui_h_dist.shape[0]:
      assert False, ('ui_object_idx %d out of horizontal distance bound %d' %
                     (object_idx, ui_h_dist.shape[0]))

    if _valid_clickable_object(ui_object) or _valid_typable_object(ui_object):
      neighbor_dict = _get_single_direction_neighbors(object_idx, ui_v_dist,
                                                      ui_h_dist)
      for neighbor_context, neighbor_index in neighbor_dict.items():
        neighbor_object = ui_object_list[neighbor_index]
        if _valid_object_with_name(neighbor_object):
          for neighbor_context_str in neighbor_context.value:
            action_list.extend(
                _generate_relative_location_rule_action(ui_object, object_idx,
                                                        neighbor_object,
                                                        neighbor_context_str))
  return action_list


def _get_single_direction_neighbors(object_idx, ui_v_dist, ui_h_dist):
  """Gets four 'single direction neighbors' for one target ui_object.

  If B is A's bottom/top 'single direction neighbor', it means B is the
  vertical closest neighbor among all object whose horizontal distance to A is
  smaller than margin threshold. Same with left/right direction neighbor.

  Args:
    object_idx: index number of target ui_object in ui_object_list
    ui_v_dist: ui objects' vertical distances. shape=[num_ui_obj, num_ui_obj]
    ui_h_dist: ui objects' horizontal distances. shape=[num_ui_obj, num_ui_obj]

  Returns:
    a dictionary, keys are NeighborContextDesc Instance, values are neighbor
    object index.
  """
  neighbor_dict = {}
  vertical_dist = ui_v_dist[object_idx]
  horizontal_dist = ui_h_dist[object_idx]
  bottom_neighbors = np.array([
      idx for idx in range(len(vertical_dist)) if vertical_dist[idx] > 0 and
      abs(horizontal_dist[idx]) < config.NORM_HORIZONTAL_NEIGHBOR_MARGIN
  ])
  top_neighbors = np.array([
      idx for idx in range(len(vertical_dist)) if vertical_dist[idx] < 0 and
      abs(horizontal_dist[idx]) < config.NORM_HORIZONTAL_NEIGHBOR_MARGIN
  ])
  right_neighbors = np.array([
      idx for idx in range(len(horizontal_dist)) if horizontal_dist[idx] > 0 and
      abs(vertical_dist[idx]) < config.NORM_VERTICAL_NEIGHBOR_MARGIN
  ])
  left_neighbors = np.array([
      idx for idx in range(len(horizontal_dist)) if horizontal_dist[idx] < 0 and
      abs(vertical_dist[idx]) < config.NORM_VERTICAL_NEIGHBOR_MARGIN
  ])

  if bottom_neighbors.size:
    neighbor_dict[NeighborContextDesc.TOP] = bottom_neighbors[np.argmin(
        vertical_dist[bottom_neighbors])]
  if top_neighbors.size:
    neighbor_dict[NeighborContextDesc.BOTTOM] = top_neighbors[np.argmax(
        vertical_dist[top_neighbors])]
  if right_neighbors.size:
    neighbor_dict[NeighborContextDesc.LEFT] = right_neighbors[np.argmin(
        horizontal_dist[right_neighbors])]
  if left_neighbors.size:
    neighbor_dict[NeighborContextDesc.RIGHT] = left_neighbors[np.argmax(
        horizontal_dist[left_neighbors])]

  return neighbor_dict


def _valid_clickable_object(ui_object):
  """Checks if a ui object has valid clickable type.

  For all objects, if they are clickable and they are not typable object, they
  will be considered valid.

  Args:
    ui_object: ui object.

  Returns:
     A boolean to indicate if a ui object has valid clickable type.
  """
  return not _valid_typable_object(ui_object) and ui_object.clickable


def _valid_clickable_object_with_name(ui_object, platform=Platform.ANDROID):
  """Checks if a ui object has a valid name and clickable type.

  For all objects, if they have a valid name and they are not typable object,
  they will be considered valid.

  Args:
    ui_object: ui object.
    platform: platform of UI objects.

  Returns:
     A boolean to indicate if a ui object has a valid name and clickable type.
  """
  return (not _valid_typable_object_with_name(ui_object, platform) and
          _valid_object_with_name(ui_object))


def _valid_typable_object(ui_object, platform=Platform.ANDROID):
  """Checks if a ui object has typable type.

  Args:
    ui_object: ui object.
    platform: platform of UI objects.

  Returns:
     A boolean to indicate if a ui object has typable type.
  """
  if platform == Platform.ANDROID:
    return ui_object.obj_type in _TYPABLE_OBJECT_DESC.keys()
  else:
    assert False, 'Wrong Platform'


def _valid_typable_object_with_name(ui_object, platform=Platform.ANDROID):
  """Checks if a ui object has a valid name and is typable.

  Args:
    ui_object: ui object.
    platform: platform of UI objects.

  Returns:
     A boolean to indicate whether the object is valid or not.
  """
  if platform == Platform.ANDROID:
    return (ui_object.obj_type in _TYPABLE_OBJECT_DESC.keys() and
            _valid_object_with_name(ui_object))
  else:
    assert False, 'Wrong Platform'


def _valid_object_with_name(ui_object):
  """Checks if a ui object has a valid name.

  Args:
    ui_object: ui object.

  Returns:
     A boolean to indicate if a ui object has a valid name.
  """
  return ui_object.obj_name


def _truncate_name(orig_str, word_num):
  """Gets a string of first word_num words from orig_str.

  If len(orig_str) > word_num, returns orig_str.

  Args:
    orig_str: Original string to be truncated.
    word_num: Number of words to be kept from orig_str.

  Returns:
    A truncated string.
  """
  if not orig_str:
    return orig_str
  tokens = string_utils.tokenizer(orig_str)
  if len(tokens) > word_num:
    orig_str = ' '.join(tokens[:word_num])
  return orig_str


def _get_obj_type_str(action_rule,
                      ui_object,
                      accept_empty=True,
                      platform=Platform.ANDROID):
  """Gets object type string.

  Args:
    action_rule: An ActionRules instance
    ui_object: UI object instance
    accept_empty: A flag to indicate if the type can be an empty string
    platform: Platform instance

  Returns:
    An ui object type string
  """

  obj_type_candidates = ['icon', 'item']
  if accept_empty:
    obj_type_candidates.append('')
  if action_rule == common.ActionRules.GRID_CONTEXT_RULE:
    obj_name = _truncate_name(ui_object.obj_name, config.MAX_OBJ_NAME_WORD_NUM)
    if obj_name:
      obj_type_candidates.append(obj_name)
  if platform == Platform.ANDROID:
    if ui_object.obj_type in _TYPABLE_OBJECT_DESC:
      obj_type_candidates.extend(_TYPABLE_OBJECT_DESC[ui_object.obj_type])
    elif ui_object.obj_type in _CLICKABLE_OBJECT_DESC:
      obj_type_candidates.extend(_CLICKABLE_OBJECT_DESC[ui_object.obj_type])
  else:
    assert False, 'wrong platform'
  return random.choice(obj_type_candidates)


def _get_obj_desc_str(action_rule,
                      ui_object,
                      context_direction_str=None,
                      target_ui_object=None,
                      platform=Platform.ANDROID):
  """Generates object description string.

  Args:
    action_rule: An ActionRules instance
    ui_object: UI object instance
    context_direction_str: A string to represent the context of ui object
    target_ui_object: target ui object. Only useful for neighbor context rule
    platform: Platform instance

  Returns:
    obj_desc: object description string
  """
  obj_desc = ''
  allow_empty_type = True
  if action_rule == common.ActionRules.GRID_CONTEXT_RULE:
    allow_empty_type = False
  obj_type = _get_obj_type_str(action_rule, ui_object, allow_empty_type,
                               platform)
  obj_name = _truncate_name(ui_object.obj_name, config.MAX_OBJ_NAME_WORD_NUM)
  obj_article = _get_obj_article_word()
  if action_rule == common.ActionRules.SINGLE_OBJECT_RULE or action_rule == common.ActionRules.SWIPE_TO_OBJECT_RULE:
    obj_desc = _concatenate_strs([obj_article, obj_name, obj_type])
  elif action_rule == common.ActionRules.GRID_CONTEXT_RULE:
    assert context_direction_str, 'Grid Direction String lost'
    obj_desc = _concatenate_strs([obj_article, obj_type, context_direction_str])
  elif action_rule == common.ActionRules.NEIGHBOR_CONTEXT_RULE:
    assert target_ui_object, 'Neighbor ui object lost'
    sub_obj_desc = _concatenate_strs([obj_article, obj_name, obj_type])
    target_obj_type = _get_obj_type_str(action_rule, ui_object, False, platform)
    target_obj_article = _get_obj_article_word()
    obj_desc = _concatenate_strs([
        target_obj_article, target_obj_type, context_direction_str, sub_obj_desc
    ])
  return obj_desc


def _get_verb_str_action_type(action_type, unused_obj_type,
                              platform=Platform.ANDROID):
  """Gets verb string and action type object.

  Args:
    action_type: A String to represent click / input / swipe
    unused_obj_type: UI object type
    platform: Platform instance

  Returns:
    action_verb_str: verb string
    action_type: ActionType instance
  """
  if action_type == 'click':
    if platform == Platform.ANDROID:
      verbs_and_actions = [(_CLICK_VERBS, common.ActionTypes.CLICK)]
    else:
      assert False, 'Wrong Platform'

  elif action_type == 'input':
    if platform == Platform.ANDROID:
      verbs_and_actions = [(_TYPE_VERBS, common.ActionTypes.INPUT)]
    else:
      assert False, 'Wrong Platform'
  else:
    assert action_type == 'swipe', 'Illegal action type received'
    verbs_and_actions = [(_SWIPE_VERBS, common.ActionTypes.SWIPE)]
  action_verbs, action_type = random.choice(verbs_and_actions)
  action_verb_str = random.choice(action_verbs)
  return (action_verb_str, action_type)


def _generate_single_object_rule_action(ui_object, target_object_id, platform):
  """Generates action based on single object rule.

  action = [verb, string_content*, object_description_str, object_type_str]

  Args:
    ui_object: ui object.
    target_object_id: target ui object id.
    platform: platform of UI objects.

  Returns:
    A list of common.Action instances.
  """
  input_content_list = []
  action_result_list = []
  action_rule = common.ActionRules.SINGLE_OBJECT_RULE

  if _valid_clickable_object_with_name(ui_object, platform):  # for CLICK
    (verb_str, action_type) = _get_verb_str_action_type('click',
                                                        ui_object.obj_type,
                                                        platform)
    input_content_list = [config.LABEL_DEFAULT_VALUE_STRING]
  elif _valid_typable_object_with_name(ui_object, platform):  # for INPUT
    (verb_str, action_type) = _get_verb_str_action_type('input',
                                                        ui_object.obj_type,
                                                        platform)
    input_content_list = [
        _generate_string_seq()
        for _ in range(config.INPUT_ACTION_UPSAMPLE_RATIO)
    ]
  else:
    return action_result_list
  obj_desc_str = _get_obj_desc_str(action_rule, ui_object, platform=platform)

  for input_content_str in input_content_list:
    action = common.Action(
        verb_str=verb_str,
        obj_desc_str=obj_desc_str,
        input_content_str=input_content_str,
        action_type=action_type,
        action_rule=action_rule,
        target_obj_idx=target_object_id)
    action_result_list.append(action)
  for action_element in action_result_list:
    _fill_action_info(action_element)
  return action_result_list


def _generate_absolute_location_rule_action(ui_object, target_object_id,
                                            grid_direction_str):
  """Generates action based on grid context rule.

  action = [verb, string_content*, object_type, grid_direction_str]

  Args:
    ui_object: ui object.
    target_object_id: target ui object id.
    grid_direction_str: a string that describs grid dicretion.

  Returns:
    A list of common.Action instances.
  """
  action_rule = common.ActionRules.GRID_CONTEXT_RULE
  action_result_list = []
  input_content_list = []
  if _valid_clickable_object(ui_object):
    (verb_str, action_type) = _get_verb_str_action_type('click',
                                                        ui_object.obj_type)
    input_content_list = [config.LABEL_DEFAULT_VALUE_STRING]
  elif _valid_typable_object_with_name(ui_object):
    (verb_str, action_type) = _get_verb_str_action_type('input',
                                                        ui_object.obj_type)
    input_content_list = [
        _generate_string_seq()
        for _ in range(config.INPUT_ACTION_UPSAMPLE_RATIO)
    ]
  obj_desc_str = _get_obj_desc_str(
      action_rule, ui_object, context_direction_str=grid_direction_str)
  for input_content_str in input_content_list:
    action = common.Action(
        verb_str=verb_str,
        obj_desc_str=obj_desc_str,
        input_content_str=input_content_str,
        action_type=action_type,
        action_rule=action_rule,
        target_obj_idx=target_object_id)
    action_result_list.append(action)
  for action_element in action_result_list:
    _fill_action_info(action_element)
  return action_result_list


def _generate_relative_location_rule_action(target_object, target_object_id,
                                            neighbor_object,
                                            context_direction_str):
  """Generates action based on neighbor context rule.

  action = [verb, string_content*, target_object_type, context_direction_str,
            neighbor_name_str, neighbor_type_str]
  A is target_object, B is neighbor_object.

  Args:
    target_object: ui object.
    target_object_id: target ui object id.
    neighbor_object: target ui object's neighbor object
    context_direction_str: a string that describs target object and neighbor
      object's context relation.

  Returns:
    A list of common.Action instances.
  """
  input_content_list = []
  action_result_list = []
  action_rule = common.ActionRules.NEIGHBOR_CONTEXT_RULE

  if _valid_clickable_object(target_object):
    (verb_str, action_type) = _get_verb_str_action_type('click',
                                                        target_object.obj_type)
    input_content_list = [config.LABEL_DEFAULT_VALUE_STRING]
  elif _valid_typable_object_with_name(target_object):
    (verb_str, action_type) = _get_verb_str_action_type('input',
                                                        target_object.obj_type)
    input_content_list = [
        _generate_string_seq()
        for _ in range(config.INPUT_ACTION_UPSAMPLE_RATIO)
    ]
  obj_desc_str = _get_obj_desc_str(
      action_rule,
      neighbor_object,
      context_direction_str=context_direction_str,
      target_ui_object=target_object)

  for input_content_str in input_content_list:
    action = common.Action(
        verb_str=verb_str,
        obj_desc_str=obj_desc_str,
        input_content_str=input_content_str,
        action_type=action_type,
        action_rule=action_rule,
        target_obj_idx=target_object_id)
    action_result_list.append(action)
  for action_element in action_result_list:
    _fill_action_info(action_element)
  return action_result_list


def _fill_action_info(action):
  """Fills components into action instance.

  Fills action.instruction_str, action.verb_str_pos, action.obj_str_pos,
  action.input_str_pos.

  Args:
    action: A common.Action instance.
  """
  def _is_ascii(s):
    return all(ord(c) < 128 for c in s)

  if not _is_ascii(action.obj_desc_str):
    tf.logging.info('Found an unconvertable unicode %s', action.obj_desc_str)
    return

  if not (isinstance(action.verb_str, str) and isinstance(
      action.obj_desc_str, str) and isinstance(action.input_content_str, str)):
    return
  action.regularize_strs()
  input_str_pos_padding = [
      config.LABEL_DEFAULT_VALUE_INT, config.LABEL_DEFAULT_VALUE_INT
  ]

  input_prep_word = _get_input_prep_word()
  swipe_prep_word = _get_swipe_prep_word()

  if action.action_rule == common.ActionRules.NO_VERB_RULE:
    action.instruction_str = action.obj_desc_str
    action.verb_str_pos = [0, 0]
    action.obj_str_pos = [0, _count_chars(action.obj_desc_str)]
    action.input_str_pos = input_str_pos_padding
    return

  if action.action_type in [common.ActionTypes.CLICK]:
    action.instruction_str = '%s %s' % (action.verb_str, action.obj_desc_str)
    action.verb_str_pos = [0, _count_chars(action.verb_str)]
    action.obj_str_pos = [
        _count_chars(action.verb_str) + 1,
        _count_chars(action.instruction_str)
    ]
    action.input_str_pos = input_str_pos_padding

  elif action.action_type in [common.ActionTypes.INPUT]:
    # There is no space between 4th and 5th string because the 2nd string,
    # article word, is optional.
    action.instruction_str = '%s %s %s %s' % (
        action.verb_str, action.input_content_str, input_prep_word,
        action.obj_desc_str)
    action.verb_str_pos = [0, _count_chars(action.verb_str)]
    action.input_str_pos = [
        _count_chars(action.verb_str) + 1,
        _count_chars('%s %s' % (action.verb_str, action.input_content_str))
    ]
    action.obj_str_pos = [
        _count_chars(
            '%s %s %s' %
            (action.verb_str, action.input_content_str, input_prep_word)) + 1,
        _count_chars(action.instruction_str)
    ]
  # All the rests are swipe actions
  else:
    action.instruction_str = '%s %s %s' % (action.verb_str, swipe_prep_word,
                                           action.obj_desc_str)
    action.verb_str_pos = [0, _count_chars(action.verb_str)]
    action.input_str_pos = input_str_pos_padding
    action.obj_str_pos = [
        _count_chars('%s %s' % (action.verb_str, swipe_prep_word)) + 1,
        _count_chars(action.instruction_str)
    ]


def _get_obj_article_word():
  object_article_words = ['', 'the']
  return random.choice(object_article_words)


def _get_input_prep_word():
  input_prep_words = ['into', 'in']
  return random.choice(input_prep_words)


def _get_swipe_prep_word():
  swipe_prep_words = ['to', 'until']
  return random.choice(swipe_prep_words)


def _count_chars(char_string):
  return len(char_string)


def _generate_string_seq():
  """Randomly generates a string of words.

  Some words will be splited by string_utils.tokenizer_with_punctuation
  afterwards, which means they may be original 5 words, but after the split,
  there are 6 or more words. So it checks the length before return the generated
  word sequences.

  Returns:
    A string containing multiple randomly generated words.
  """
  input_word_num = random.randint(1, config.MAX_INPUT_WORD_NUMBER)
  return ' '.join(resources.get_random_words(input_word_num))


def _concatenate_strs(string_list):
  return ' '.join([s for s in string_list if s])


def get_synthetic_feature_dict(synthetic_action_list,
                               max_word_num,
                               unused_max_word_length,
                               parse_consumed=False):
  """Get padded synthetic action feature dictionary.

  This dictionary contains all features related to the synthetic instructions.

  Args:
    synthetic_action_list: List of common.Action() instances
    max_word_num: max word number for padding
    unused_max_word_length: max word length for padding.
    parse_consumed: whether to parse consumed tag.

  Returns:
    a padded feature dictionary
  """
  feature = {
      'instruction_str': [],
      'instruction_rule_id': [],
      'instruction_word_id_seq': [],
      'verb_id_seq': [],
      'ui_target_id_seq': [],
      'verb_str_position_seq': [],
      'input_str_position_seq': [],
      'obj_desc_position_seq': [],
  }
  if parse_consumed:
    feature['consumed_tag'] = []
    feature['step_str_position_seq'] = []

  for action in synthetic_action_list:
    if not action.is_valid():
      continue
    action.convert_to_lower_case()
    word_id_seq, char_id_seq = string_utils.tokenize_to_ids(
        action.instruction_str)
    # skips the synthetic actions that have more than max_word_num tokens
    if len(word_id_seq) > max_word_num:
      tf.logging.info('[Dropped Long Synthetic Action]:%s',
                      action.instruction_str)
      continue
    feature['instruction_str'].append(action.instruction_str)
    feature['instruction_rule_id'].append(action.action_rule.value)
    feature['instruction_word_id_seq'].append(word_id_seq)
    # Enable this when using word token
    if 'instruction_char_id_seq' in feature:
      feature['instruction_char_id_seq'].append(char_id_seq)
    feature['verb_id_seq'].append(action.action_type.value)
    feature['ui_target_id_seq'].append(action.target_obj_idx)
    feature['verb_str_position_seq'].extend(
        string_utils.get_token_pos_from_char_pos(action.instruction_str,
                                                 action.verb_str_pos[0],
                                                 action.verb_str_pos[1]))
    feature['obj_desc_position_seq'].extend(
        string_utils.get_token_pos_from_char_pos(action.instruction_str,
                                                 action.obj_str_pos[0],
                                                 action.obj_str_pos[1]))
    if action.has_valid_input():
      feature['input_str_position_seq'].extend(
          string_utils.get_token_pos_from_char_pos(action.instruction_str,
                                                   action.input_str_pos[0],
                                                   action.input_str_pos[1]))
    else:
      feature['input_str_position_seq'].extend(action.input_str_pos)
    if parse_consumed:
      feature['consumed_tag'].append(int(action.is_consumed))
      step_token_pos = string_utils.get_token_pos_from_char_pos(
          action.instruction_str, action.step_str_pos[0],
          action.step_str_pos[1])
      feature['step_str_position_seq'].extend(step_token_pos)

  for key in feature:
    feature[key] = np.array(feature[key])

  phrase_count = feature['instruction_str'].shape[0]
  feature_padding_info = {
      'instruction_str': [phrase_count, np.string_, ''],
      'instruction_rule_id': [(phrase_count), np.int64, 0],
      'instruction_word_id_seq': [(phrase_count, max_word_num), np.int64, 0],
      'verb_id_seq': [(phrase_count), np.int64, 0],
      'ui_target_id_seq': [(phrase_count), np.int64, 0],
      'verb_str_position_seq': [(phrase_count * 2), np.int64, 0],
      'input_str_position_seq': [(phrase_count * 2), np.int64, 0],
      'obj_desc_position_seq': [(phrase_count * 2), np.int64, 0],
  }
  if parse_consumed:
    feature_padding_info['consumed_tag'] = [(phrase_count), np.int64, 0]
    feature_padding_info['step_str_position_seq'] = [(phrase_count * 2),
                                                     np.int64, 0]

  padding_shape, padding_type, padding_value = {}, {}, {}
  for key in feature_padding_info:
    shape, pad_type, value = feature_padding_info[key]
    padding_shape[key] = shape
    padding_type[key] = pad_type
    padding_value[key] = value

  padded_feature_dict = proto_utils.padding_dictionary(feature, padding_shape,
                                                       padding_type,
                                                       padding_value)
  return padded_feature_dict
