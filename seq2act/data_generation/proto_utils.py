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

"""Proto utility module containing helper functions.

The module handles tasks related to protobufs in word2act:
1. encodes word2act action and time_step into tf.train.Example proto2.
2. parses screeninfo protobuf into feature dictionary.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io

import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf

from seq2act.data_generation import string_utils
from seq2act.data_generation import view_hierarchy
from tensorflow.contrib import framework as contrib_framework
nest = contrib_framework.nest


def get_feature_dict(screen_info_proto, padding_shape=None, lower_case=False):
  """Gets screen feature dictionary from screen_info protobuf.

  Args:
    screen_info_proto: protobuf defined in word2act/proto/rehearsal.proto.
      Contains screenshot and xml
    padding_shape: The shape of padding size for final feature list. shape =
      (max_object_num, max_word_num, max_word_length) If the shape is not given,
      then returns the original list without padding.
    lower_case: lower case all the ui texts.

  Returns:
    A feature dictionary. If padding_shape is not None, all values of the
    dictionary are padded. The shape after padding is shown as 'shape = ...'.
    Otherwise, shapes of values are not a fixed value.
      screenshot: numpy array of screen_info_proto.screenshot
      'ui_obj_str_seq': uiobject's name/content_descriotion/resource_id, numpy
          array of strings.
      'ui_obj_word_id_seq': encoded word sequence, np int array, shape =
          (max_object_num, max_word_num)
      'ui_obj_char_id_seq': encoded char sequence, np int array, shape =
          (max_object_num, max_word_num, max_word_length)
      'ui_obj_type_seq': type sequence, np int array, shape = (max_object_num,)
      'ui_obj_clickable_seq': clickable sequence, np int array, shape =
          (max_object_num,)
      'ui_obj_cord_x_seq': x cordinate sequence, np int array, shape =
          (max_object_num*2,)
      'ui_obj_cord_y_seq': y cordinate sequence, np int array, shape =
          (max_object_num*2,)
      'ui_obj_v_distance': vertical relation matrix, np float array,
          shape = (max_object_num, max_object_num)
      'ui_obj_h_distance': horizontal relation matrix, np float array, shape =
          (max_object_num, max_object_num)
      'ui_obj_dom_distance': dom relation matrix, np int array, shape =
          (max_object_num, max_object_num)
      'ui_obj_dom_location_seq': dom index from tree traversal, np int array,
          shape = (max_object_num*3,)


  """
  screenshot = Image.open(io.BytesIO(screen_info_proto.screenshot.content))
  screenshot = np.asarray(screenshot, np.float32)
  vh = view_hierarchy.ViewHierarchy()
  vh.load_xml(screen_info_proto.view_hierarchy.xml.encode('utf-8'))
  view_hierarchy_leaf_nodes = vh.get_leaf_nodes()

  ui_object_features_dict = get_ui_objects_feature_dict(
      view_hierarchy_leaf_nodes, padding_shape, lower_case)
  ui_object_features_dict['screenshot'] = screenshot

  return ui_object_features_dict


def get_ui_objects_feature_dict(view_hierarchy_leaf_nodes,
                                padding_shape=None,
                                lower_case=False):
  """Gets ui object features dictionary from view hierarchy leaf nodes list.

  Args:
    view_hierarchy_leaf_nodes: A list of view hierarchy leaf node objects.
    padding_shape: The shape of padding size for final feature list. shape =
      (max_object_num, max_word_num, max_word_length) If the shape is not given,
      then returns the original list without padding.
    lower_case: lower case all the ui texts.

  Returns:
    A feature dictionary. If padding_shape is not None, all values of the
    dictionary are padded. The shape after padding is shown as 'shape = ...'.
    Otherwise, shapes of values are not a fixed value.
      'ui_obj_type_seq': type sequence, np int array, shape = (max_object_num,)
      'ui_obj_word_id_seq': encoded word sequence, np int array, shape =
          (max_object_num, max_word_num)
      'ui_obj_char_id_seq': encoded char sequence, np int array, shape =
          (max_object_num, max_word_num, max_word_length)
      'ui_obj_clickable_seq': clickable sequence, np int array, shape =
          (max_object_num,)
      'ui_obj_cord_x_seq': x cordinate sequence, np int array, shape =
          (max_object_num*2,)
      'ui_obj_cord_y_seq': y cordinate sequence, np int array, shape =
          (max_object_num*2,)
      'ui_obj_v_distance': vertical relation matrix, np float array, shape =
          (max_object_num, max_object_num)
      'ui_obj_h_distance': horizontal relation matrix, np float array, shape =
          (max_object_num, max_object_num)
      'ui_obj_dom_distance': dom relation matrix, np int array, shape =
          (max_object_num, max_object_num)
      'ui_obj_dom_location_seq': dom index from tree traversal, np int array,
          shape = (max_object_num*3,)
      'ui_obj_str_seq': uiobject's name/content_descriotion/resource_id,
          numpy array of strings.
  """
  ui_object_attributes = _get_ui_object_attributes(view_hierarchy_leaf_nodes,
                                                   lower_case)
  vh_relations = get_view_hierarchy_leaf_relation(view_hierarchy_leaf_nodes)
  if padding_shape is None:
    merged_features = {}
    for key in ui_object_attributes:
      if key == 'obj_str_seq':
        merged_features['ui_obj_str_seq'] = ui_object_attributes[key].copy()
      else:
        merged_features['ui_obj_' + key] = ui_object_attributes[key].copy()
    for key in vh_relations:
      merged_features['ui_obj_' + key] = vh_relations[key].copy()
    return merged_features
  else:
    if not isinstance(padding_shape, tuple):
      assert False, 'padding_shape %s is not a tuple.' % (str(padding_shape))
    if len(padding_shape) != 3:
      assert False, 'padding_shape %s contains not exactly 3 elements.' % (
          str(padding_shape))

  (max_object_num, max_word_num, _) = padding_shape
  obj_feature_dict = {
      'ui_obj_type_id_seq':
          padding_array(ui_object_attributes['type_id_seq'], (max_object_num,),
                        -1),
      'ui_obj_str_seq':
          padding_array(
              ui_object_attributes['obj_str_seq'], (max_object_num,),
              padding_type=np.string_,
              padding_value=''),
      'ui_obj_word_id_seq':
          padding_array(
              ui_object_attributes['word_id_seq'],
              (max_object_num, max_word_num),
              padding_value=0),
      'ui_obj_clickable_seq':
          padding_array(ui_object_attributes['clickable_seq'],
                        (max_object_num,)),
      'ui_obj_cord_x_seq':
          padding_array(ui_object_attributes['cord_x_seq'],
                        (max_object_num * 2,)),
      'ui_obj_cord_y_seq':
          padding_array(ui_object_attributes['cord_y_seq'],
                        (max_object_num * 2,)),
      'ui_obj_v_distance':
          padding_array(vh_relations['v_distance'],
                        (max_object_num, max_object_num), 0, np.float32),
      'ui_obj_h_distance':
          padding_array(vh_relations['h_distance'],
                        (max_object_num, max_object_num), 0, np.float32),
      'ui_obj_dom_distance':
          padding_array(vh_relations['dom_distance'],
                        (max_object_num, max_object_num)),
      'ui_obj_dom_location_seq':
          padding_array(ui_object_attributes['dom_location_seq'],
                        (max_object_num * 3,)),
  }
  return obj_feature_dict


def _get_ui_object_attributes(view_hierarchy_leaf_nodes, lower_case=False):
  """Parses ui object informationn from a view hierachy leaf node list.

  Args:
    view_hierarchy_leaf_nodes: a list of view hierachy leaf nodes.
    lower_case: lower case all the ui texts.

  Returns:
    An un-padded attribute dictionary as follow:
      'type_id_seq': numpy array of ui object types from view hierarchy.
      'word_id_seq': numpy array of encoding for words in ui object.
      'char_id_seq': numpy array of encoding for words in ui object.
      'clickable_seq': numpy array of ui object clickable status.
      'cord_x_seq': numpy array of ui object x coordination.
      'cord_y_seq': numpy array of ui object y coordination.
      'dom_location_seq': numpy array of ui object depth, pre-order-traversal
      index, post-order-traversal index.
      'word_str_sequence': numpy array of ui object name strings.
  """
  type_sequence = []
  word_id_sequence = []
  char_id_sequence = []
  clickable_sequence = []
  cord_x_sequence = []
  cord_y_sequence = []
  dom_location_sequence = []
  obj_str_sequence = []

  def _is_ascii(s):
    return all(ord(c) < 128 for c in s)

  for vh_node in view_hierarchy_leaf_nodes:
    ui_obj = vh_node.uiobject
    type_sequence.append(ui_obj.obj_type.value)
    cord_x_sequence.append(ui_obj.bounding_box.x1)
    cord_x_sequence.append(ui_obj.bounding_box.x2)
    cord_y_sequence.append(ui_obj.bounding_box.y1)
    cord_y_sequence.append(ui_obj.bounding_box.y2)
    clickable_sequence.append(ui_obj.clickable)
    dom_location_sequence.extend(ui_obj.dom_location)

    valid_words = [w for w in ui_obj.word_sequence if _is_ascii(w)]
    word_sequence = ' '.join(valid_words)

    if lower_case:
      word_sequence = word_sequence.lower()
    obj_str_sequence.append(word_sequence)

    word_ids, char_ids = string_utils.tokenize_to_ids(word_sequence)
    word_id_sequence.append(word_ids)
    char_id_sequence.append(char_ids)
  ui_feature = {
      'type_id_seq': np.array(type_sequence),
      'word_id_seq': np.array(word_id_sequence),
      'clickable_seq': np.array(clickable_sequence),
      'cord_x_seq': np.array(cord_x_sequence),
      'cord_y_seq': np.array(cord_y_sequence),
      'dom_location_seq': np.array(dom_location_sequence),
      'obj_str_seq': np.array(obj_str_sequence, dtype=np.str),
  }
  return ui_feature


def get_view_hierarchy_leaf_relation(view_hierarchy_leaf_nodes):
  """Calculates adjacency relation from list of view hierarchy leaf nodes.

  Args:
    view_hierarchy_leaf_nodes: a list of view hierachy leaf nodes.

  Returns:
    An un-padded feature dictionary as follow:
      'v_distance': 2d numpy array of ui object vertical adjacency relation.
      'h_distance': 2d numpy array of ui object horizontal adjacency relation.
      'dom_distance': 2d numpy array of ui object dom adjacency relation.
  """
  vh_node_num = len(view_hierarchy_leaf_nodes)
  vertical_adjacency = np.zeros((vh_node_num, vh_node_num), dtype=np.float32)
  horizontal_adjacency = np.zeros((vh_node_num, vh_node_num), dtype=np.float32)
  dom_adjacency = np.zeros((vh_node_num, vh_node_num), dtype=np.int64)
  for row in range(len(view_hierarchy_leaf_nodes)):
    for column in range(len(view_hierarchy_leaf_nodes)):
      if row == column:
        h_dist = v_dist = dom_dist = 0
      else:
        node1 = view_hierarchy_leaf_nodes[row]
        node2 = view_hierarchy_leaf_nodes[column]
        h_dist, v_dist = node1.normalized_pixel_distance(node2)
        dom_dist = node1.dom_distance(node2)
      vertical_adjacency[row][column] = v_dist
      horizontal_adjacency[row][column] = h_dist
      dom_adjacency[row][column] = dom_dist
  return {
      'v_distance': vertical_adjacency,
      'h_distance': horizontal_adjacency,
      'dom_distance': dom_adjacency
  }


def padding_dictionary(orig_dict, padding_shape_dict, padding_type_dict,
                       padding_value_dict):
  """Does padding for dictionary of array or numpy array.

  Args:
    orig_dict: Original dictionary.
    padding_shape_dict: Dictionary of padding shape, keys are field names,
      values are shape tuple
    padding_type_dict: Dictionary of padding shape, keys are field names, values
      are padded numpy type
    padding_value_dict: Dictionary of padding shape, keys are field names,
      values are shape tuple

  Returns:
    A padded dictionary.
  """
  # Asserting the keys of the four dictionaries are exactly same.
  assert (set(orig_dict.keys()) == set(padding_shape_dict.keys()) == set(
      padding_type_dict.keys()) == set(padding_value_dict.keys()))
  padded_dict = {}
  for key in orig_dict:
    if padding_shape_dict[key]:
      padded_dict[key] = padding_array(orig_dict[key], padding_shape_dict[key],
                                       padding_value_dict[key],
                                       padding_type_dict[key])
    else:
      padded_dict[key] = np.array(orig_dict[key], dtype=padding_type_dict[key])
  return padded_dict


def padding_array(orig_array,
                  padding_shape,
                  padding_value=0,
                  padding_type=np.int64):
  """Pads orig_array according to padding shape, number and type.

  The dimension of final result is the smaller dimension between
  orig_array.shape and padding_shape.

  For example:
    a = [[1,2],[3,4]]
    padding_array(a, (3,3), 0, np.int64) = [[1, 2, 0], [3, 4, 0], [0, 0, 0]]

    a = [[1,2,3,4],[5,6,7,8]]
    padding_array(a, (3,3), 0, np.int64) = [[1, 2, 3], [5, 6, 7], [0, 0, 0]]

  Args:
    orig_array: The original array before padding.
    padding_shape: The shape of padding.
    padding_value: The number to be padded into new array.
    padding_type: The data type to be padded into new array.

  Returns:
    A padded numpy array.
  """
  # When padding type is string, we need to initialize target_array with object
  # type first. And convert it back to np.string_ after _fill_array. Because
  # after initialized, numpy string array cannot hold longer string.
  # For example:
  #   >>> a = np.array([''], dtype = np.string_)
  #   >>> a
  #   array([''], dtype='|S1')
  #   >>> a[0] = 'foo'
  #   >>> a
  #   array(['f'], dtype='|S1')
  if padding_type == np.string_:
    used_pad_type = object
  else:
    used_pad_type = padding_type
  target_array = np.full(
      shape=padding_shape, fill_value=padding_value, dtype=used_pad_type)
  _fill_array(orig_array, target_array)
  if padding_type == np.string_:
    target_array = target_array.astype(np.string_)
  return target_array


def _fill_array(orig_array, target_array):
  """Fills elements from orig_array to target_array.

  If any dimension of orig_array is larger than target_array, only fills the
  array of their shared dimensions.

  Args:
    orig_array: original array that contains the filling numbers, could be numpy
      array or python list.
    target_array: target array that will be filled with original array numbers,
      numpy array

  Raises:
    TypeError: if the target_array is not a numpy array
  """
  if not isinstance(target_array, np.ndarray):
    raise TypeError('target array is not numpy array')
  if target_array.ndim == 1:
    try:
      orig_length = len(orig_array)
    except TypeError:
      tf.logging.exception(
          'orig_array %s and target_array %s dimension not fit',
          orig_array, target_array)
      orig_length = 0
    if len(target_array) < orig_length:
      target_array[:] = orig_array[:len(target_array)]
    else:
      target_array[:orig_length] = orig_array
      return
  else:
    for sub_orig, sub_target in zip(orig_array, target_array):
      _fill_array(sub_orig, sub_target)


def features_to_tf_example(features):
  """Converts feature dictionary into tf.Example protobuf.

  This function only supports to convert np.int and np.float array.

  Args:
    features: A feature dictionary. Keys are field names, values are np array.

  Returns:
    A tf.Example protobuf.

  Raises:
    ValueError: Feature dictionary's value field is not supported type.

  """
  new_features = {}
  for k, v in features.items():
    if not isinstance(v, np.ndarray):
      raise ValueError('Value field: %s is not numpy array' % str((k, v)))
    v = v.flatten()
    if np.issubdtype(v.dtype.type, np.string_):
      new_features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    elif np.issubdtype(v.dtype.type, np.integer):
      new_features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    elif np.issubdtype(v.dtype.type, np.floating):
      new_features[k] = tf.train.Feature(float_list=tf.train.FloatList(value=v))
    else:
      raise ValueError('Value for %s is not a recognized type; v: %s type: %s' %
                       (k, str(v[0]), str(type(v[0]))))
  return tf.train.Example(features=tf.train.Features(feature=new_features))
