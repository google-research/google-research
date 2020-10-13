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

"""Creates screen dataset with tfExample proto in TFRecord format.

For all the valid xml or json files in the input directory, it parses their
view hierarchy attributes, extracts the feature data into tf.train.Example proto
and saves the results with TFRecord format in the output
directory as multiple sharded files. A file containing the dimension data for
padding purpose is also created.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import operator
import os
import threading

import concurrent.futures
import numpy as np
import tensorflow.compat.v1 as tf  # tf

from seq2act.data_generation import common
from seq2act.data_generation import config
from seq2act.data_generation import proto_utils
from seq2act.data_generation import synthetic_action_generator
from seq2act.data_generation import view_hierarchy

gfile = tf.gfile
flags = tf.flags
FLAGS = flags.FLAGS


_INPUT_DIR = '/tmp/input'
_OUTPUT_DIR = '/tmp/'
_FILTER_FILE = '/tmp/'

_NUM_THREADS_DEFAULT = 10
_PADDING_DIMENSIONS_FILE_NAME = 'padding_dimensions.txt'
_TOKEN_TYPE = 'subtoken'
_NUM_SHARDS_DEFAULT = config.SHARD_NUM
_MAX_WORD_NUM_UPPER_DEFAULT = config.MAX_WORD_NUM_UPPER_BOUND
_MAX_WORD_LENGTH_UPPER_DEFAULT = config.MAX_WORD_LENGTH_UPPER_BOUND
_DATASET_TYPE_DEFAULT = 'rico'
_STATS_DIMENSIONS = False
_MAX_WORD_NUM = 30
_MAX_WORD_LENGTH = 23


_FREQ_OBJ_TYPE = [
    view_hierarchy.UIObjectType.UNKNOWN,
    view_hierarchy.UIObjectType.BUTTON,
    view_hierarchy.UIObjectType.IMAGEVIEW,
    view_hierarchy.UIObjectType.TEXTVIEW,
]

_INFREQUENT_OBJ_TYPE_MAX_RATIO = 0.9
_FILTER_ACTIONS_BY_NAME = True
_FILTER_ACTION_BY_TYPE = True

flags.DEFINE_string(
    'input_dir', _INPUT_DIR,
    'Full path to the directory containing the data files for a set of tasks.')
flags.DEFINE_string(
    'output_dir', _OUTPUT_DIR,
    'Full path to the directory for saving the tf record file.')
flags.DEFINE_string(
    'filter_file', _FILTER_FILE,
    'Full path to the directory for saving filter file or RICO.')
flags.DEFINE_integer(
    'num_threads', _NUM_THREADS_DEFAULT,
    'The number of threads to process the data files concurrently.')

flags.DEFINE_integer(
    'num_shards', _NUM_SHARDS_DEFAULT,
    'The number of sharded files to save the created dataset.')
flags.DEFINE_integer('max_word_num_upper', _MAX_WORD_NUM_UPPER_DEFAULT,
                     'The max number of words for building model features.')
flags.DEFINE_integer('max_word_length_upper', _MAX_WORD_LENGTH_UPPER_DEFAULT,
                     'The max length of words for building model features.')
flags.DEFINE_enum('dataset', _DATASET_TYPE_DEFAULT,
                  ['android_settings', 'rico'],
                  'The type of supported dataset.')
flags.DEFINE_enum('file_to_generate', 'tf_example',
                  ['tf_example', 'corpus'],
                  'Whether generate feature tfexample or corpus txt.')


debug_info_lock = threading.Lock()

longest_stats = collections.Counter()
distributions = collections.defaultdict(collections.Counter)


def _stat_distribution(name, value_list):
  with debug_info_lock:
    # longest_stats[name] = max(longest_stats[name], num)
    for value in value_list:
      distributions[name][value] += 1


sums = collections.defaultdict(int)


def _stat_sum(name, num):
  with debug_info_lock:
    sums[name] += num


def _get_data_dimensions(input_dir):
  """Processes the dimension data.

  The dimension data includes maximum word numbers and maximum word lengths
  from all the ui objects across all the .xml/.json files in input_dir.
  It will be used for padding purpose.

  The results are written in <output_dir>/padding_dimensions.txt.

  Args:
    input_dir: The directory that contains the input xml/json files.

  Returns:
    A tuple (max_word_num_all_files, max_word_length_all_files)
    max_word_num_all_files: The max number of words from all the ui objects
      across all the .xml/.json files.
    max_word_length_all_files: The max length of words from all the ui objects
      across all the .xml/.json files.
  """
  max_word_num_all_files = 0
  max_word_length_all_files = 0
  # We can use ThreadPool since these are IO-bound operations.
  with concurrent.futures.ThreadPoolExecutor(FLAGS.num_threads) as executor:
    futures = []
    for file_path in gfile.Glob(os.path.join(input_dir, '*.xml')) + gfile.Glob(
        os.path.join(input_dir, '*.json')):
      futures.append(executor.submit(common.get_word_statistics, file_path))

    for future in concurrent.futures.as_completed(futures):
      _, max_word_num_one_file, max_word_length_one_file = future.result()
      max_word_num_all_files = max(max_word_num_all_files,
                                   max_word_num_one_file)
      max_word_length_all_files = max(max_word_length_all_files,
                                      max_word_length_one_file)
  tf.logging.info('max_word_num_all_files=%d, max_word_length_all_files=%d',
                  max_word_num_all_files, max_word_length_all_files)

  return max_word_num_all_files, max_word_length_all_files


def _process_dimensions(input_dir, output_dir):
  """Processes the dimension data.

  The dimension data includes maximum word numbers and maximum word lengths
  from all the ui objects across all the .xml/.json files in input_dir.
  It will be used for padding purpose.

  The results are written in <output_dir>/padding_dimensions.txt.

  Args:
    input_dir: The directory that contains the input xml/json files.
    output_dir: The directory that saves output dimension data.

  Returns:
    A tuple (max_word_num, max_word_length)
    max_word_num: The max number of words for building model features.
    max_word_length: The max length of words for building model features.
  """
  tf.logging.info('Processing data dimensions...')
  max_word_num, max_word_length = _get_data_dimensions(input_dir)

  # Apply pre-configured upper bound to clip possibly rare outlier values.
  max_word_num = min(max_word_num, FLAGS.max_word_num_upper)
  max_word_length = min(max_word_length, FLAGS.max_word_length_upper)

  with gfile.GFile(
      os.path.join(output_dir, _PADDING_DIMENSIONS_FILE_NAME), 'w+') as f:
    f.write('max_word_num: %d\nmax_word_length: %d\n' %
            (max_word_num, max_word_length))
  return max_word_num, max_word_length


def _filter_synthetic_by_name_overlap(action_list,
                                      screen_feature,
                                      first_k_tokens=5,
                                      overlap_threshold=0.5):
  """Filter synthetic action by object name overlap.

  For each action, if any non-target object's name contains more than
  first_k_tokens*overlap_threshold same tokens from target object name, this
  action will be treated as invalid and will be filtered out.

  For example:
  If:
    target object = ['data_', 'usage_']
    non_target_object = ['no_', 'data_', 'limitation_', 'usage_', 'set_']
    first_k_tokens = 3
    overlap_threshold = 0.5
  non_target_object[0:first_k_tokens] = ['no_', 'data_', 'limitation_']
  Overlapped tokens are ['data_'], covered ratio is
  len(['data_']) / len(['data_', 'usage_']) = 0.5 >= overlap_threshold
  So this action is invalid and will be removed.

  Args:
    action_list: list of actions
    screen_feature: screen feature dictionary
    first_k_tokens: number of heading tokens selected to calculate token overlap
    overlap_threshold: Threshold of object name overlap ratio.

  Returns:
    valid action list
  """
  obj_tokens_id = screen_feature['ui_obj_word_id_seq']
  target_obj_idx = [action.target_obj_idx for action in action_list]
  target_objs_first_k_token = [
      screen_feature['ui_obj_word_id_seq'][idx][0:first_k_tokens]
      for idx in target_obj_idx
  ]
  filter_flag = []
  for obj_first_k_token in target_objs_first_k_token:
    trim_obj_first_k_token = np.trim_zeros(obj_first_k_token)
    target_token_appear_times = np.sum(
        np.array([
            np.isin(trim_obj_first_k_token, one_obj_tokens)
            for one_obj_tokens in obj_tokens_id
        ]),
        axis=1)
    filter_flag.append(
        np.sum(target_token_appear_times > overlap_threshold *
               trim_obj_first_k_token.shape[0]) <= 1)
  valid_actions = [
      action for action, valid in zip(action_list, filter_flag) if valid
  ]
  invalid_actions = [
      action for action, valid in zip(action_list, filter_flag) if not valid
  ]

  # Enable when debugging to see invalid synthetic
  # _stat_distribution('obj_name_invalid_synthetic',
  #                    [action.instruction_str for action in invalid_actions])
  _stat_sum('obj_name_invalid_synthetic', len(invalid_actions))

  return valid_actions


def _filter_synthetic_by_obj_type(ui_object_list,
                                  action_list,
                                  max_num_syn_per_screen=20):
  """Filters synthetic data by object type.

  For all synthetic actions, split them into frequent/non-frequent actions.
  First select non-frequent actions, non-frequent actions number should be no
  more than max_num_syn_per_screen*_INFREQUENT_OBJ_TYPE_MAX_RATIO.
  Then select frequent actions, total selected actions number should be no more
  than max_num_syn_per_screen.

  Args:
    ui_object_list: list of ui objects
    action_list: list of actions
    max_num_syn_per_screen: max number of synthetic sentence for each screen

  Returns:
    valid action list
  """
  max_infreq_num = int(max_num_syn_per_screen*_INFREQUENT_OBJ_TYPE_MAX_RATIO)
  freq_obj_actions = []
  infreq_obj_actions = []

  for action in action_list:
    if (ui_object_list[action.target_obj_idx].obj_type not in _FREQ_OBJ_TYPE or
        action.action_type == common.ActionTypes.INPUT):
      infreq_obj_actions.append(action)
    else:
      freq_obj_actions.append(action)
  if len(infreq_obj_actions) > max_infreq_num:
    valid_actions = np.random.choice(
        infreq_obj_actions, size=max_infreq_num, replace=False).tolist()
  else:
    valid_actions = infreq_obj_actions

  left_space = max_num_syn_per_screen - len(valid_actions)
  if len(freq_obj_actions) < left_space:
    valid_actions.extend(freq_obj_actions)
  else:
    valid_actions.extend(
        np.random.choice(freq_obj_actions, size=left_space,
                         replace=False).tolist())

  return valid_actions


def _get_full_feature_dict(dataset_type, file_path, max_word_num,
                           max_word_length):
  """Gets full padded feature dictionary from xml/json file_path.

  Args:
    dataset_type: The supported dataset type.
    file_path: The full path of xml/json file.
    max_word_num: The max number of words in each ui object.
    max_word_length: The max length of words in each ui object.

  Returns:
    padded_feature_dict: A dictionary that contains ui_object features, view
    hierarchy leaf node adjacency matrix features and synthetic action
    features. Each value of the feature dicionary is padded. The padding shape
    is as follow:
    feature_dict = {
      'instruction_str': synthetic action strings, np array of string,
          shape = (phrase_count,)
      'instruction_word_id_seq': encoded action words, np int array, shape =
          (phrase_count, max_word_num)
      'instruction_rule_id': representing action rule(single_object_rule/
          grid_context_rule/neighbor_context_rule), np int array,
          shape = (phrase_count,)

      'ui_obj_str_seq': string of ui object name/content_description/resourceid,
          np string array, shape = (ui_object_num)
      'ui_obj_word_id_seq': encoded word sequence, np int array, shape =
          (ui_object_num, max_word_num)
      'ui_obj_type_id_seq': object type ids, np int array, shape =
          (ui_object_num,)
      'ui_obj_clickable_seq': clickable sequence, np int array, shape =
          (ui_object_num,)
      'ui_obj_cord_x_seq': x coordinate sequence, np float array, shape =
          (ui_object_num*2,)
      'ui_obj_cord_y_seq': y coordinate sequence, np float array, shape =
          (ui_object_num*2,)
      'ui_obj_v_distance': vertical relation matrix, np float array, shape =
          (ui_object_num, ui_object_num)
      'ui_obj_h_distance': horizontal relation matrix, np float array, shape =
          (ui_object_num, ui_object_num)
      'ui_obj_dom_distance': dom relation matrix, np int array, shape =
          (ui_object_num, ui_object_num)
      'ui_obj_dom_location_seq': index of pre-order/in-order/post-order in view
          hierarchy tree. np int array, shape = (ui_object_num*3,)

      'verb_id_seq': representing action verb id(click/input/swipe), np int
          array, shape = (phrase_count,)
      'verb_str_position_seq': index of verb string, np int array,
          shape = (phrase_count*2,)
      'ui_target_id_seq': index of ui object target, np int array,
          shape = (phrase_count,)
      'input_str_position_seq': input words' start end position in instruction,
          np int array, shape = (phrase_count*2,)
      'obj_desc_position_seq': target object words' start end position,
          np int array, shape = (phrase_count*2,)
    }
  """
  view_hierarchy_leaf_nodes = common.get_view_hierarchy_list(file_path)
  ui_obj_list = [ele.uiobject for ele in view_hierarchy_leaf_nodes]

  ui_object_num = len(view_hierarchy_leaf_nodes)
  padded_obj_feature_dict = proto_utils.get_ui_objects_feature_dict(
      view_hierarchy_leaf_nodes,
      padding_shape=(ui_object_num, max_word_num, max_word_length),
      lower_case=True)

  actions = synthetic_action_generator.generate_all_actions(
      view_hierarchy_leaf_nodes,
      action_rules=('single', 'screen_loc', 'neighbor_loc', 'swipe'))

  if actions and _FILTER_ACTIONS_BY_NAME:
    actions = _filter_synthetic_by_name_overlap(
        actions,
        padded_obj_feature_dict,
        config.MAX_OBJ_NAME_WORD_NUM,
        overlap_threshold=0.5)
  if actions and _FILTER_ACTION_BY_TYPE:
    actions = _filter_synthetic_by_obj_type(
        ui_obj_list, actions, max_num_syn_per_screen=20)

  padded_syn_feature_dict = synthetic_action_generator.get_synthetic_feature_dict(
      actions, max_word_num, max_word_length)
  full_feature = {}
  full_feature.update(padded_obj_feature_dict)
  full_feature.update(padded_syn_feature_dict)

  # Normalize ui object's coordinates to be within [0,1].
  if dataset_type == 'android_settings':
    screen_width, screen_height = (config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
  else:
    screen_width, screen_height = (config.RICO_SCREEN_WIDTH,
                                   config.RICO_SCREEN_HEIGHT)
  full_feature['ui_obj_cord_x_seq'] = full_feature['ui_obj_cord_x_seq'] / float(
      screen_width)
  full_feature['ui_obj_cord_y_seq'] = full_feature['ui_obj_cord_y_seq'] / float(
      screen_height)

  return full_feature


def _assert_feature_value(feature):
  """Asserts feature value doesn't have -1, except for anchor features."""
  anchor_features = [
      'instruction_word_id_seq', 'ui_obj_type_id_seq', 'verb_id_seq'
  ]
  for key in feature:
    if key in anchor_features:
      continue
    if -1 in feature[key]:
      tf.logging.info('[FATAL]: Feature %d contains -1', key)
      return False
  return True


def _assert_feature_shape(feature, expected_shape):
  """Asserts feature shape is legal, same as expected_shape."""
  assert set(feature.keys()) == set(expected_shape.keys(
  )), '[FATAL] feature keys %s different from expected %s' % (
      sorted(feature.keys()), sorted(expected_shape.keys()))
  for key in feature:
    if feature[key].shape != expected_shape[key]:
      tf.logging.info('[FATAL] feature %s shape is different from expected',
                      key)
      return False
  return True


def _produce_corpus(corpus_writer, writer_lock, file_path):
  """Writes all UI names."""
  view_hierarchy_leaf_nodes = common.get_view_hierarchy_list(file_path)
  ui_obj_list = [ele.uiobject for ele in view_hierarchy_leaf_nodes]
  ui_names = [ui.obj_name.lower().strip() for ui in ui_obj_list if ui.obj_name]
  corpus = '\n'.join(ui_names)
  with writer_lock:
    corpus_writer.write(corpus + '\n')


def _process_features(tf_record_writer, writer_lock,
                      dataset_type, file_path, max_word_num, max_word_length):
  """Processes features from one xml/json file.

  Args:
    tf_record_writer: TFRecordWriter object
    writer_lock: The lock to protect the writer's write from multi threads.
    dataset_type: The supported dataset type.
    file_path: The full path of xml/json file.
    max_word_num: The max number of words in each ui object.
    max_word_length: The max length of words in each ui object. synthetic input
      actions.
  """
  feature_dict = _get_full_feature_dict(
      dataset_type,
      file_path,
      max_word_num,
      max_word_length,
  )
  phrase_count = feature_dict['instruction_str'].shape[0]
  ui_object_num = feature_dict['ui_obj_str_seq'].shape[0]

  expected_feature_shape = {
      'instruction_str': (phrase_count,),
      'instruction_word_id_seq': (phrase_count, max_word_num),
      'instruction_rule_id': (phrase_count,),
      'ui_obj_str_seq': (ui_object_num,),
      'ui_obj_word_id_seq': (ui_object_num, max_word_num),
      'ui_obj_type_id_seq': (ui_object_num,),
      'ui_obj_clickable_seq': (ui_object_num,),
      'ui_obj_cord_x_seq': (ui_object_num*2,),
      'ui_obj_cord_y_seq': (ui_object_num*2,),
      'ui_obj_v_distance': (ui_object_num, ui_object_num),
      'ui_obj_h_distance': (ui_object_num, ui_object_num),
      'ui_obj_dom_distance': (ui_object_num, ui_object_num),
      'ui_obj_dom_location_seq': (ui_object_num*3,),
      'verb_id_seq': (phrase_count,),
      'ui_target_id_seq': (phrase_count,),
      'verb_str_position_seq': (phrase_count*2,),
      'input_str_position_seq': (phrase_count*2,),
      'obj_desc_position_seq': (phrase_count*2,),
  }

  _stat_distribution('ui_obj_type_id_seq', feature_dict['ui_obj_type_id_seq'])
  _stat_distribution('verb_id_seq', feature_dict['verb_id_seq'])
  _stat_distribution('instruction_rule_id', feature_dict['instruction_rule_id'])
  target_objs = feature_dict['ui_target_id_seq']
  _stat_distribution('target_obj_type',
                     feature_dict['ui_obj_type_id_seq'][target_objs])

  # When feature_dict['verb_id_seq'] is not always padded value, generate
  # tfexample
  if (_assert_feature_shape(feature_dict, expected_feature_shape) and
      _assert_feature_value(feature_dict) and
      not np.array(feature_dict['verb_id_seq'] ==
                   config.LABEL_DEFAULT_INVALID_INT).all()):
    tf_proto = proto_utils.features_to_tf_example(feature_dict)
    with writer_lock:
      tf_record_writer.write(tf_proto.SerializeToString())


def _write_dataset(dataset_type, input_dir, output_dir, max_word_num,
                   max_word_length):
  """Processes features from xml/json files and writes results to dataset files.

  Args:
    dataset_type: The supported dataset type.
    input_dir: The directory that contains the input xml/json files.
    output_dir: The directory that saves output dimension data.
    max_word_num: The max number of words in each ui object.
    max_word_length: The max length of words in each ui object. synthetic input
      actions.
  """
  tf.logging.info('Processing data features...')
  tf_record_writers = []
  writer_locks = []
  for shard in range(FLAGS.num_shards):
    tf_record_writers.append(
        tf.python_io.TFRecordWriter(
            os.path.join(
                output_dir, 'one_shot_%s_%s_%d.tfrecord' %
                (dataset_type, _TOKEN_TYPE, shard))))
    writer_locks.append(threading.Lock())

  num_processed_files = 0
  # We can use ThreadPool since these are IO-bound operations.
  with concurrent.futures.ThreadPoolExecutor(FLAGS.num_threads) as executor:
    futures = []
    all_file_path = gfile.Glob(os.path.join(input_dir, '*.xml')) + gfile.Glob(
        os.path.join(input_dir, '*.json'))

    all_file_path = filter_file_by_name(all_file_path)
    assert len(all_file_path) == 24598

    for file_path in sorted(all_file_path):
      shard = num_processed_files % FLAGS.num_shards
      futures.append(
          executor.submit(_process_features, tf_record_writers[shard],
                          writer_locks[shard], dataset_type, file_path,
                          max_word_num, max_word_length))
      num_processed_files += 1
    concurrent.futures.wait(futures)

  for shard in range(FLAGS.num_shards):
    tf_record_writers[shard].close()


def filter_file_by_name(file_path):
  """Filters input file by name."""
  filter_filepath = FLAGS.filter_file
  valid_data_set = set()
  with gfile.Open(filter_filepath, 'r') as f:
    for line in f.read().split('\n'):
      valid_data_set.add(line)
  return [fp for fp in file_path if fp.split('/')[-1] in valid_data_set]


def create_dataset(dataset_type, input_dir, output_dir):
  """Converts xml/json files in input_dir to tfExample files in output_dir.

  tfExample file contains multiple tf.Example proto and the feature dictionary
  of tf.Example proto defined by _get_full_feature_dict()

  Args:
    dataset_type: The supported dataset type.
    input_dir: The directory that contains input xml/json file.
    output_dir: The directory that saves output files.
  """
  if _STATS_DIMENSIONS:
    max_word_num, max_word_length = _process_dimensions(input_dir, output_dir)
  else:
    max_word_num, max_word_length = _MAX_WORD_NUM, _MAX_WORD_LENGTH
  _write_dataset(dataset_type, input_dir, output_dir, max_word_num,
                 max_word_length)


def main(_):
  create_dataset(FLAGS.dataset, FLAGS.input_dir, FLAGS.output_dir)

  tf.logging.info('\n\n%s\n\n', longest_stats)
  if FLAGS.file_to_generate == 'tf_example':
    with open(os.path.join(FLAGS.output_dir, 'stats.txt'), 'w+') as writer:

      for key, distribution in distributions.items():
        writer.write(
            '%s: %s\n' %
            (key, sorted(distribution.items(), key=operator.itemgetter(0))))

      for key, distribution in sums.items():
        writer.write('%s: %s\n' %
                     (key, sorted(sums.items(), key=operator.itemgetter(0))))


if __name__ == '__main__':
  FLAGS.set_default('logtostderr', True)
  tf.app.run(main)
