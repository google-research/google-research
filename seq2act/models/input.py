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

"""The input function of seq2act models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum
from tensor2tensor.layers import common_layers
import tensorflow.compat.v1 as tf  # tf

NUM_TOKENS_PER_OBJ = 30
NUM_TOKENS_PER_SYN = 30


class DataSource(Enum):
  """The class that represents word2act data source."""
  RICO_SCA = 'rico_sca'
  ANDROID_HOWTO = 'android_howto'
  PIXEL_HELP = 'pixel_help'

  @staticmethod
  def from_str(label):
    if label == 'rico_sca':
      return DataSource.RICO_SCA
    elif label == 'android_howto':
      return DataSource.ANDROID_HOWTO
    elif label == 'pixel_help':
      return DataSource.PIXEL_HELP
    else:
      raise ValueError('Unrecognized source %s' % label)


MAX_UI_OBJECT_NUM = {
    DataSource.PIXEL_HELP: 93,
}

MAX_TOKEN_NUM = {
    DataSource.ANDROID_HOWTO: 30,
    DataSource.RICO_SCA: 30,
    DataSource.PIXEL_HELP: 153,
}

# ['connect_str',  token_id(connector_str)]
# token id based on all_source_lower_case_vocab_59429
PADDED_CONCATENATORS = [
    [5, 0, 0],
    [115, 0, 0],
    [8, 32, 0],
    [115, 8, 32],
    [12, 0, 0],
]

CONCATENATORS_STR = [
    ', ',
    ' , ',
    ' and then ',
    ' , and then ',
    '. '
]


def _construct_padding_info(data_source, load_dom_dist, load_extra):
  """Constructs the padding info tuple."""
  token_num = MAX_TOKEN_NUM[data_source]
  # Model uses this anchor padding value to mask out the padded features.
  anchor_padding_value_int = tf.cast(-1, tf.int32)
  padding_value_int = tf.cast(0, tf.int32)
  padding_value_str = tf.cast('', tf.string)
  # Tuple of (feature name, padded_shape, padded_value)
  padding_info = [
      ('task', [None], padding_value_int),
      ('rule', [], padding_value_int),
      ('verbs', [None], padding_value_int),
      ('input_refs', [None, 2], padding_value_int),
      ('obj_refs', [None, 2], padding_value_int),
      ('verb_refs', [None, 2], padding_value_int),
      ('objects', [None], padding_value_int),
      ('obj_text', [None, None, token_num], padding_value_int),
      ('obj_type', [None, None], anchor_padding_value_int),
      ('obj_clickable', [None, None], padding_value_int),
      ('obj_screen_pos', [None, None, 4], tf.cast(0, tf.int32)),
      ('obj_dom_pos', [None, None, 3], padding_value_int),
      ('agreement_count', [], padding_value_int),
      ('data_source', [], padding_value_int),
  ]
  if load_dom_dist:
    padding_info.append(('obj_dom_dist', [None, None, None], padding_value_int))
  if load_extra:
    padding_info.append(('task_id', [], padding_value_str))
    padding_info.append(('raw_task', [], padding_value_str))
    padding_info.append(('obj_raw_text', [None, None], padding_value_str))

  padded_shapes = {}
  padded_values = {}
  for (key, padding_shape, padding_value) in padding_info:
    padded_shapes[key] = padding_shape
    padded_values[key] = padding_value
  return padded_shapes, padded_values


def input_fn(data_files,
             batch_size,
             repeat=-1,
             data_source=DataSource.RICO_SCA,
             required_agreement=2,
             max_range=1000,
             max_dom_pos=2000,
             max_pixel_pos=100,
             load_dom_dist=False,
             load_extra=False,
             buffer_size=8 * 1024,
             shuffle_size=8 * 1024,
             required_rule_id_list=None,
             shuffle_repeat=True,
             mean_synthetic_length=1.0,
             stddev_synthetic_length=0.0,
             load_screen=True,
             shuffle_files=True):
  """Retrieves batches of data for training.

  Adds padding to ensure all dimension in one batch are always same.

  Args:
    data_files: A list of file names to initialize the TFRecordDataset
    batch_size:  Number for the size of the batch.
    repeat: the number of times to repeat the input data.
    data_source: A DataSource instance.
    required_agreement: the minimum agreement required.
    max_range: the max range.
    max_dom_pos: the max dom pos.
    max_pixel_pos: the max screen pixels.
    load_dom_dist: whether to load the dom distance feature.
    load_extra: whether to load the raw text data.
    buffer_size: the buffer size for prefetching.
    shuffle_size: the shuffle size.
    required_rule_id_list: the list of required rule ids.
    shuffle_repeat: whether to shuffle and repeat.
    mean_synthetic_length: the mean length for synthetic sequence.
    stddev_synthetic_length: the stddev length for synthetic sequence.
    load_screen: whether to load screen features.
    shuffle_files: shuffling file names.
  Returns:
    a tf.dataset.Dateset object.
  Raises:
    ValueError: The data_format is neither 'recordio' nor 'tfrecord'.
  """
  if not isinstance(data_source, DataSource):
    assert False, 'data_source %s unsupported' % str(data_source)
  padded_shapes, padded_values = _construct_padding_info(
      data_source, load_dom_dist, load_extra)
  if not isinstance(data_files, (list,)):
    data_files = [data_files]
  all_files = tf.concat(
      values=[tf.matching_files(f) for f in data_files], axis=0)
  if repeat == -1 and shuffle_files:
    all_files = tf.random.shuffle(all_files)
  if data_files[0].endswith('.recordio'):
    dataset = tf.data.RecordIODataset(all_files)
  elif data_files[0].endswith('.tfrecord'):
    dataset = tf.data.TFRecordDataset(
        all_files, num_parallel_reads=10 if repeat == -1 else None)
  else:
    assert False, 'Data_format %s is not supported.' % data_files[0]

  def _map_fn(x):
    return parse_tf_example(x, data_source, max_range, max_dom_pos,
                            max_pixel_pos, load_dom_dist=load_dom_dist,
                            load_extra=load_extra,
                            append_eos=(data_source != DataSource.RICO_SCA or
                                        mean_synthetic_length == 1.0),
                            load_screen=load_screen)
  dataset = dataset.map(_map_fn)
  def _is_enough_agreement(example):
    return tf.greater_equal(example['agreement_count'], required_agreement)
  dataset = dataset.filter(_is_enough_agreement)

  def _length_filter(example):
    return tf.less(tf.shape(example['obj_refs'])[0], 20)
  dataset = dataset.filter(_length_filter)

  def _filter_data_by_rule(example, rule_id_list):
    return tf.reduce_any(
        [tf.equal(example['rule'], rule_id) for rule_id in rule_id_list])
  if data_source == DataSource.RICO_SCA and required_rule_id_list is not None:
    dataset = dataset.filter(
        lambda x: _filter_data_by_rule(x, required_rule_id_list))

  # (TODO: liyang) tf.data.experimental.bucket_by_sequence_length
  if shuffle_repeat:
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(
        shuffle_size, count=repeat))
  dataset = dataset.padded_batch(
      batch_size, padded_shapes=padded_shapes, padding_values=padded_values)
  if data_source == DataSource.RICO_SCA and mean_synthetic_length > 1.0:
    def _stitch_fn(x):
      return _batch_stitch(x, mean_length=mean_synthetic_length,
                           stddev=stddev_synthetic_length)
    dataset = dataset.map(_stitch_fn)
  dataset = dataset.prefetch(buffer_size=buffer_size)
  return dataset


def hybrid_input_fn(data_files_list,
                    data_source_list,
                    batch_size_list,
                    max_range=1000,
                    max_dom_pos=2000,
                    max_pixel_pos=100,
                    load_dom_dist=False,
                    load_extra=False,
                    buffer_size=8 * 1024,
                    mean_synthetic_length=1.0,
                    stddev_synthetic_length=0.0,
                    hybrid_batch_size=128,
                    boost_input=False,
                    load_screen=True,
                    shuffle_size=1024):
  """Combines multiple datasouces."""
  mixed_dataset = None
  for data_files, data_source, batch_size in zip(
      data_files_list, data_source_list, batch_size_list):
    dataset = input_fn(data_files, batch_size, repeat=-1,
                       data_source=data_source,
                       required_agreement=-1,
                       max_range=max_range, max_dom_pos=max_dom_pos,
                       max_pixel_pos=max_pixel_pos,
                       load_dom_dist=load_dom_dist,
                       load_extra=load_extra,
                       buffer_size=0,
                       mean_synthetic_length=mean_synthetic_length,
                       stddev_synthetic_length=stddev_synthetic_length,
                       shuffle_repeat=False,
                       load_screen=load_screen)
    if mixed_dataset is None:
      mixed_dataset = dataset
    else:
      mixed_dataset = dataset.concatenate(mixed_dataset)

  mixed_dataset = mixed_dataset.unbatch()
  # Boost input examples
  if boost_input:
    def _input_booster(example):
      with tf.control_dependencies([tf.rank(example['input_refs']), 2]):
        has_input = tf.reduce_any(
            tf.greater(example['input_refs'][:, 1],
                       example['input_refs'][:, 0]))
        return tf.logical_or(has_input, tf.less(tf.random_uniform([]), 0.1))
    dataset = dataset.filter(_input_booster)
  # Remix single examples
  mixed_dataset = mixed_dataset.shuffle(hybrid_batch_size * shuffle_size)
  # Batch again
  padded_shapes, padded_values = _construct_padding_info(
      data_source_list[0], load_dom_dist, load_extra)
  mixed_dataset = mixed_dataset.padded_batch(
      hybrid_batch_size, padded_shapes=padded_shapes,
      padding_values=padded_values)
  mixed_dataset = mixed_dataset.repeat()
  mixed_dataset = mixed_dataset.prefetch(buffer_size=buffer_size)
  return mixed_dataset


def parse_tf_example(example_proto,
                     data_source,
                     max_range=100,
                     max_dom_pos=2000,
                     max_pixel_pos=100,
                     load_dom_dist=False,
                     load_extra=False,
                     append_eos=True,
                     load_screen=True):
  """Parses an example TFRecord proto into dictionary of tensors.

  Args:
    example_proto: TFRecord format proto that contains screen information.
    data_source: A DataSource instance.
    max_range: the max range.
    max_dom_pos: the maximum dom positoin.
    max_pixel_pos: the max dom position.
    load_dom_dist: whether to load the feature.
    load_extra: whether to load the extra data for debugging.
    append_eos: whether to append eos.
    load_screen: whether to load screen features.
  Returns:
    feature: The parsed tensor dictionary with the input feature data
    label: The parsed label tensor with the input label for the feature
  """
  feature_spec = {
      'instruction_word_id_seq':
          tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
      'input_str_position_seq':
          tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
      'obj_desc_position_seq':
          tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
      'verb_str_position_seq':
          tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
      'agreement_count':
          tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
      'instruction_rule_id':
          tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
  }
  if load_screen:
    feature_spec['verb_id_seq'] = tf.FixedLenSequenceFeature(
        [], tf.int64, allow_missing=True)
    feature_spec['ui_target_id_seq'] = tf.FixedLenSequenceFeature(
        [], tf.int64, allow_missing=True)
    feature_spec['ui_obj_word_id_seq'] = tf.FixedLenSequenceFeature(
        [], tf.int64, allow_missing=True)
    feature_spec['ui_obj_type_id_seq'] = tf.FixedLenSequenceFeature(
        [], tf.int64, allow_missing=True)
    feature_spec['ui_obj_clickable_seq'] = tf.FixedLenSequenceFeature(
        [], tf.int64, allow_missing=True)
    feature_spec['ui_obj_cord_x_seq'] = tf.FixedLenSequenceFeature(
        [], tf.float32, allow_missing=True)
    feature_spec['ui_obj_cord_y_seq'] = tf.FixedLenSequenceFeature(
        [], tf.float32, allow_missing=True)
    feature_spec['ui_obj_dom_location_seq'] = tf.FixedLenSequenceFeature(
        [], tf.int64, allow_missing=True)

  if load_dom_dist:
    feature_spec['ui_obj_dom_distance'] = tf.FixedLenSequenceFeature(
        [], tf.int64, allow_missing=True)
  if load_extra:
    feature_spec['instruction_str'] = tf.FixedLenSequenceFeature(
        [], tf.string, allow_missing=True)
    feature_spec['task_id'] = tf.FixedLenSequenceFeature(
        [], tf.string, allow_missing=True)
    feature_spec['ui_obj_str_seq'] = tf.FixedLenSequenceFeature(
        [], tf.string, allow_missing=True)

  feature_dict = tf.parse_single_example(example_proto, feature_spec)

  for key in feature_dict:
    if feature_dict[key].dtype == tf.int64:
      feature_dict[key] = tf.cast(feature_dict[key], tf.int32)
  if data_source == DataSource.ANDROID_HOWTO:
    tf.logging.info('Parsing android_howto dataset')
    feature = _process_android_howto(feature_dict, max_range=max_range,
                                     load_dom_dist=load_dom_dist,
                                     load_extra=load_extra)
  elif data_source == DataSource.RICO_SCA:
    tf.logging.info('Parsing synthetic dataset')
    feature = _process_rico_sca(
        feature_dict, max_range=max_range, max_dom_pos=max_dom_pos,
        load_dom_dist=load_dom_dist,
        load_extra=load_extra,
        load_screen=load_screen)
  elif data_source == DataSource.PIXEL_HELP:
    tf.logging.info('Parsing test dataset')
    feature = _process_pixel_help(feature_dict, data_source,
                                  load_dom_dist=load_dom_dist,
                                  load_extra=load_extra)
  else:
    raise ValueError('Unsupported datasource %s' % str(data_source))
  # Remove padding from "task"
  feature['task'] = tf.boolean_mask(feature['task'],
                                    tf.not_equal(feature['task'], 0))
  feature['obj_screen_pos'] = tf.to_int32(
      feature['obj_screen_pos'] * (max_pixel_pos - 1))
  # Appending EOS and padding to match the appended length
  if append_eos:
    feature['input_refs'] = tf.pad(feature['input_refs'], [[0, 1], [0, 0]])
    feature['obj_refs'] = tf.pad(feature['obj_refs'], [[0, 1], [0, 0]])
    step_num = tf.size(feature['task'])
    feature['verb_refs'] = tf.concat(
        [feature['verb_refs'], [[step_num, step_num + 1]]], axis=0)
    feature['task'] = tf.pad(feature['task'], [[0, 1]], constant_values=1)
    feature['obj_text'] = tf.pad(feature['obj_text'], [[0, 1], [0, 0], [0, 0]])
    feature['obj_clickable'] = tf.pad(feature['obj_clickable'],
                                      [[0, 1], [0, 0]])
    feature['obj_type'] = tf.pad(
        feature['obj_type'], [[0, 1], [0, 0]], constant_values=-1)
    feature['obj_screen_pos'] = tf.pad(feature['obj_screen_pos'],
                                       [[0, 1], [0, 0], [0, 0]])
    feature['obj_dom_pos'] = tf.pad(feature['obj_dom_pos'],
                                    [[0, 1], [0, 0], [0, 0]])
    if load_dom_dist:
      feature['obj_dom_dist'] = tf.pad(feature['obj_dom_dist'],
                                       [[0, 1], [0, 0], [0, 0]])
    feature['objects'] = tf.pad(feature['objects'], [[0, 1]])
    feature['verbs'] = tf.pad(feature['verbs'], [[0, 1]])
  return feature


def _bound_refs(feature, max_range):
  """Makes sure the refs are in the allowed range."""
  for key in feature:
    if not key.endswith('_refs'):
      continue
    feature[key] = tf.where(
        tf.greater(feature[key][:, 1] - feature[key][:, 0], max_range),
        tf.stack([feature[key][:, 0], feature[key][:, 0] + max_range], axis=1),
        feature[key])


def _process_android_howto(feature_dict, max_range, load_dom_dist=False,
                           load_extra=False):
  """Processes webanswer feature dictionary.

  Args:
    feature_dict: feature dictionary
    max_range: the max range.
    load_dom_dist: whether to load the dom distance feature.
    load_extra: whether to load the extra data for debugging.
  Returns:
    A processed feature dictionary.
  """

  feature = {
      'task': tf.reshape(feature_dict['instruction_word_id_seq'], [-1]),
      'input_refs': tf.reshape(feature_dict['input_str_position_seq'], [-1, 2]),
      'obj_refs': tf.reshape(feature_dict['obj_desc_position_seq'], [-1, 2]),
      'verb_refs': tf.reshape(feature_dict['verb_str_position_seq'], [-1, 2]),
      'agreement_count': tf.reshape(feature_dict['agreement_count'], [])
  }
  if load_extra:
    feature['task_id'] = tf.constant('empty_task_id', dtype=tf.string)
    feature['raw_task'] = tf.reshape(feature_dict['instruction_str'], [])
  _bound_refs(feature, max_range)
  _load_fake_screen(feature, load_extra, load_dom_dist)
  return feature


def _load_fake_screen(feature, load_extra, load_dom_dist):
  """Loads a fake screen."""
  # Fills in fake ui object features into feature dictionary.
  step_num = tf.shape(feature['verb_refs'])[0]
  obj_num = 1
  if load_extra:
    feature['obj_raw_text'] = tf.fill([step_num, obj_num], '')
  feature['data_source'] = tf.constant(1, dtype=tf.int32)
  feature['obj_text'] = tf.zeros([step_num, obj_num, NUM_TOKENS_PER_OBJ],
                                 tf.int32)
  feature['obj_type'] = tf.cast(tf.fill([step_num, obj_num], -1), tf.int32)
  feature['obj_clickable'] = tf.zeros([step_num, obj_num], tf.int32)
  feature['obj_screen_pos'] = tf.zeros([step_num, obj_num, 4], tf.float32)
  feature['obj_dom_pos'] = tf.zeros([step_num, obj_num, 3], tf.int32)
  if load_dom_dist:
    feature['obj_dom_dist'] = tf.zeros([step_num, obj_num, obj_num], tf.int32)
  feature['objects'] = tf.zeros([step_num], tf.int32)
  feature['verbs'] = tf.zeros([step_num], tf.int32)
  feature['rule'] = tf.constant(5, dtype=tf.int32)


def _batch_stitch(features, mean_length=4.0, stddev=2.0):
  """Stitches a batch of single-step data to a batch of multi-step data."""
  batch_size = common_layers.shape_list(features['task'])[0]
  num_sequences = tf.maximum(
      tf.to_int32(tf.to_float(batch_size) / mean_length), 1)
  lengths = tf.random.truncated_normal(shape=[num_sequences],
                                       mean=mean_length, stddev=stddev)
  max_length = tf.reduce_max(lengths) * (
      tf.to_float(batch_size) / tf.reduce_sum(lengths))
  max_length = tf.to_int32(tf.ceil(max_length))
  total_items = max_length * num_sequences
  num_paddings = total_items - batch_size
  indices = tf.random.shuffle(tf.range(total_items))
  for key in features:
    shape_list = common_layers.shape_list(features[key])
    assert len(shape_list) >= 1
    with tf.control_dependencies([
        tf.assert_greater_equal(num_paddings, 0,
                                name='num_paddings_positive')]):
      paddings = [[0, num_paddings]] + [[0, 0]] * (len(shape_list) - 1)
    features[key] = tf.pad(features[key], paddings,
                           constant_values=-1 if key == 'obj_type' else 0)
    features[key] = tf.gather(features[key], indices)
    shape = [num_sequences, max_length]
    if len(shape_list) >= 2:
      shape += shape_list[1:]
    features[key] = tf.reshape(features[key], shape)
  # Remove all-padding seqs
  step_mask = tf.reduce_any(tf.greater(features['task'], 1), axis=-1)
  mask = tf.reduce_any(step_mask, axis=-1)
  step_mask = tf.boolean_mask(step_mask, mask)
  for key in features:
    features[key] = tf.boolean_mask(features[key], mask=mask)
  num_sequences = tf.shape(features['task'])[0]
  # Sort steps within each seq
  _, step_indices = tf.math.top_k(tf.to_int32(step_mask), k=max_length)
  step_indices = step_indices + tf.expand_dims(
      tf.range(num_sequences) * max_length, 1)
  step_indices = tf.reshape(step_indices, [-1])
  for key in features:
    shape_list = common_layers.shape_list(features[key])
    features[key] = tf.gather(tf.reshape(features[key], [-1] + shape_list[2:]),
                              step_indices)
    features[key] = tf.reshape(features[key], shape_list)
  features = _stitch(features)
  return features


def _stitch(features):
  """Stitch features on the first dimension."""
  full_mask = tf.greater(features['task'], 1)
  step_mask = tf.reduce_any(full_mask, axis=-1)
  step_mask_exclude_last = tf.pad(step_mask,
                                  [[0, 0], [0, 1]],
                                  constant_values=False)[:, 1:]
  num_sequences = common_layers.shape_list(features['task'])[0]
  num_steps = common_layers.shape_list(features['task'])[1]
  connectors = tf.constant(PADDED_CONCATENATORS)
  # Select connectors
  connector_indices = tf.random.uniform(
      [num_sequences * num_steps], minval=0,
      maxval=len(PADDED_CONCATENATORS), dtype=tf.int32)
  selected_connectors = tf.reshape(
      tf.gather(connectors, connector_indices),
      [num_sequences, num_steps, len(PADDED_CONCATENATORS[0])])
  selected_connectors = tf.multiply(
      selected_connectors,
      tf.expand_dims(tf.to_int32(step_mask_exclude_last), 2),
      name='connector_mask')
  features['task'] = tf.concat([features['task'], selected_connectors], axis=-1)
  ref_offsets = tf.expand_dims(
      tf.cumsum(tf.reduce_sum(tf.to_int32(tf.greater(features['task'], 1)), -1),
                exclusive=True, axis=-1), 2)
  features['task'] = tf.reshape(features['task'], [num_sequences, -1])
  full_mask = tf.greater(features['task'], 1)
  full_mask_int = tf.to_int32(full_mask)
  indices = tf.where(tf.sequence_mask(lengths=tf.reduce_sum(full_mask_int, -1)))
  values = tf.boolean_mask(tf.reshape(features['task'], [-1]),
                           tf.reshape(full_mask, [-1]))
  sparse_task = tf.sparse.SparseTensor(
      indices=indices, values=values,
      dense_shape=tf.to_int64(tf.shape(features['task'])))
  # Stitch task and raw_task
  stitched_features = {}
  stitched_features['task'] = tf.sparse_tensor_to_dense(sparse_task)
  max_len = tf.reduce_max(
      tf.reduce_sum(tf.to_int32(tf.greater(stitched_features['task'], 1)), -1))
  stitched_features['task'] = stitched_features['task'][:, :max_len]
  if 'raw_task' in features:
    connector_strs = tf.reshape(
        tf.gather(tf.constant(CONCATENATORS_STR), connector_indices),
        [num_sequences, num_steps])
    masked_connector_strs = tf.where(
        step_mask_exclude_last,
        connector_strs, tf.fill(tf.shape(connector_strs), ''))
    stitched_features['raw_task'] = tf.strings.reduce_join(
        tf.strings.reduce_join(tf.concat([
            tf.expand_dims(features['raw_task'], 2),
            tf.expand_dims(masked_connector_strs, 2)], axis=2), axis=-1), -1)
  # Stitch screen sequences
  action_lengths = tf.reduce_sum(tf.to_int32(
      tf.greater(features['verb_refs'][:, :, 0, 1],
                 features['verb_refs'][:, :, 0, 0])), -1)
  max_action_length = tf.reduce_max(action_lengths)
  def _pad(tensor, padding_value=0):
    shape_list = common_layers.shape_list(tensor)
    assert len(shape_list) >= 2
    padding_list = [[0, 0], [0, 1]] + [[0, 0]] * (len(shape_list) - 2)
    return tf.pad(tensor[:, :max_action_length],
                  padding_list, constant_values=padding_value)
  for key in features.keys():
    if key.endswith('_refs'):
      features[key] = tf.squeeze(features[key], 2)
      ref_mask = tf.expand_dims(tf.to_int32(
          tf.not_equal(features[key][:, :, 0],
                       features[key][:, :, 1])), 2)
      stitched_features[key] = tf.multiply(
          (features[key] + ref_offsets), ref_mask, name='ref_mask')
      stitched_features[key] = _pad(stitched_features[key])
    elif key in ['verbs', 'objects', 'consumed', 'obj_dom_pos',
                 'obj_text', 'obj_type', 'obj_clickable', 'obj_screen_pos',
                 'verb_refs', 'obj_refs', 'input_refs', 'obj_dom_dist']:
      features[key] = tf.squeeze(features[key], 2)
      stitched_features[key] = features[key]
      stitched_features[key] = _pad(
          stitched_features[key],
          padding_value=-1 if key == 'obj_type' else 0)
    elif key not in ['task', 'raw_task']:
      stitched_features[key] = features[key][:, 0]
  # Append eos to 'task'
  stitched_features['task'] = tf.pad(stitched_features['task'],
                                     [[0, 0], [0, 1]])
  task_mask = tf.to_int32(tf.greater(stitched_features['task'], 1))
  task_eos_mask = tf.pad(task_mask, [[0, 0], [1, 0]], constant_values=1)[:, :-1]
  stitched_features['task'] = stitched_features['task'] + (
      task_eos_mask - task_mask)
  # Append eos
  verb_mask = tf.to_int32(tf.greater(stitched_features['verbs'], 1))
  verb_eos_mask = tf.pad(verb_mask, [[0, 0], [1, 0]], constant_values=1)[:, :-1]
  verb_eos = verb_eos_mask - verb_mask
  stitched_features['verbs'] = stitched_features['verbs'] + verb_eos
  # Append last step refs to 'verb_refs'
  task_lengths = tf.where(tf.equal(stitched_features['task'], 1))[:, 1]
  eos_pos = tf.to_int32(tf.stack([task_lengths, task_lengths + 1], axis=1))
  action_mask = tf.to_int32(
      tf.sequence_mask(action_lengths, max_action_length + 1))
  action_and_eos_mask = tf.pad(action_mask, [[0, 0], [1, 0]],
                               constant_values=1)[:, :-1]
  verb_ref_eos = action_and_eos_mask - action_mask
  eos_refs = tf.multiply(
      tf.tile(tf.expand_dims(eos_pos, 1), [1, max_action_length + 1, 1]),
      tf.expand_dims(verb_ref_eos, 2), name='verb_ref_eos')
  stitched_features['verb_refs'] += eos_refs
  return stitched_features


def _process_rico_sca(feature_dict, max_range, max_dom_pos,
                      load_dom_dist=False, load_extra=False, load_screen=True):
  """Processes one_shot feature dictionary.

  Args:
    feature_dict: feature dictionary
    max_range: the max range.
    max_dom_pos: the max dom pos.
    load_dom_dist: whether to load the dom distance feature.
    load_extra: whether to load the extra data for debugging.
    load_screen: whether to load the screen features.
  Returns:
    A processed feature dictionary.
  """
  phrase_count = tf.size(feature_dict['obj_desc_position_seq']) // 2
  feature = {
      'task':
          tf.reshape(feature_dict['instruction_word_id_seq'],
                     [phrase_count, NUM_TOKENS_PER_SYN]),
      'input_refs':
          tf.reshape(feature_dict['input_str_position_seq'],
                     [phrase_count, 1, 2]),
      'obj_refs':
          tf.reshape(feature_dict['obj_desc_position_seq'],
                     [phrase_count, 1, 2]),
      'verb_refs':
          tf.reshape(feature_dict['verb_str_position_seq'],
                     [phrase_count, 1, 2]),
      'rule':
          tf.reshape(feature_dict['instruction_rule_id'], [phrase_count]),
  }
  selected_synthetic_action_idx = tf.random_uniform(
      shape=(), minval=0, maxval=phrase_count, dtype=tf.int32)
  for key in feature:
    feature[key] = feature[key][selected_synthetic_action_idx]
  if load_extra:
    feature['raw_task'] = tf.reshape(
        feature_dict['instruction_str'],
        [phrase_count])[selected_synthetic_action_idx]
    feature['task_id'] = tf.constant('empty_task_id', dtype=tf.string)
  if load_screen:
    feature['verbs'] = tf.reshape(
        feature_dict['verb_id_seq'],
        [phrase_count, 1])[selected_synthetic_action_idx]
    feature['objects'] = tf.reshape(
        feature_dict['ui_target_id_seq'],
        [phrase_count, 1])[selected_synthetic_action_idx]
    feature['obj_text'] = tf.reshape(feature_dict['ui_obj_word_id_seq'],
                                     [1, -1, NUM_TOKENS_PER_OBJ])
    feature['obj_type'] = tf.reshape(
        feature_dict['ui_obj_type_id_seq'], [1, -1])
    feature['obj_clickable'] = tf.reshape(feature_dict['ui_obj_clickable_seq'],
                                          [1, -1])
    def _make_obj_screen_pos():
      return tf.concat([
          tf.reshape(feature_dict['ui_obj_cord_x_seq'], [1, -1, 2]),
          tf.reshape(feature_dict['ui_obj_cord_y_seq'], [1, -1, 2])
      ], 2)

    feature['obj_screen_pos'] = tf.cond(
        tf.equal(
            tf.size(feature_dict['ui_obj_cord_x_seq']),
            0), lambda: tf.fill([1, tf.shape(feature['obj_type'])[1], 4], 0.),
        _make_obj_screen_pos)
    feature['obj_dom_pos'] = tf.reshape(feature_dict['ui_obj_dom_location_seq'],
                                        [1, -1, 3])
    feature['obj_dom_pos'] = tf.minimum(feature['obj_dom_pos'], max_dom_pos - 1)
    if load_dom_dist:
      num_ui_obj = tf.to_int32(
          tf.sqrt(tf.to_float(tf.size(feature_dict['ui_obj_dom_distance']))))
      feature['obj_dom_dist'] = tf.reshape(feature_dict['ui_obj_dom_distance'],
                                           [1, num_ui_obj, num_ui_obj])
    if load_extra:
      feature['obj_raw_text'] = tf.reshape(feature_dict['ui_obj_str_seq'],
                                           [1, -1])
  else:
    _load_fake_screen(feature, load_extra, load_dom_dist)
  _bound_refs(feature, max_range)
  feature['data_source'] = tf.constant(0, dtype=tf.int32)
  feature['agreement_count'] = tf.constant(100, dtype=tf.int32)

  return feature


def _process_pixel_help(feature_dict, data_source, load_dom_dist=False,
                        load_extra=False):
  """Processes testing data feature dictionary.

  Args:
    feature_dict: feature dictionary
    data_source: TEST_PIXEL_HELP
    load_dom_dist: whether to load the dom distance feature.
    load_extra: whether to load the extra data for debugging.
  Returns:
    A processed feature dictionary.
  """
  step_num = tf.size(feature_dict['verb_id_seq'])
  feature = {
      'task':
          tf.reshape(feature_dict['instruction_word_id_seq'], [-1]),
      'obj_text':
          tf.reshape(feature_dict['ui_obj_word_id_seq'], [
              step_num, MAX_UI_OBJECT_NUM[data_source],
              MAX_TOKEN_NUM[data_source]
          ]),
      'obj_type':
          tf.reshape(feature_dict['ui_obj_type_id_seq'],
                     [step_num, MAX_UI_OBJECT_NUM[data_source]]),
      'obj_clickable':
          tf.reshape(feature_dict['ui_obj_clickable_seq'],
                     [step_num, MAX_UI_OBJECT_NUM[data_source]]),
      # pylint: disable=g-long-ternary
      'obj_screen_pos': (
          tf.reshape(tf.concat([
              tf.reshape(feature_dict['ui_obj_cord_x_seq'], [step_num, -1, 2]),
              tf.reshape(feature_dict['ui_obj_cord_y_seq'], [step_num, -1, 2])
          ], axis=2), [step_num, MAX_UI_OBJECT_NUM[data_source], 4])),
      'obj_dom_pos':
          tf.reshape(feature_dict['ui_obj_dom_location_seq'],
                     [step_num, MAX_UI_OBJECT_NUM[data_source], 3]),
      'verbs':
          tf.reshape(feature_dict['verb_id_seq'], [step_num]),
      'objects':
          tf.reshape(feature_dict['ui_target_id_seq'], [step_num]),
      'input_refs':
          tf.reshape(feature_dict['input_str_position_seq'], [step_num, 2]),
      'obj_refs':
          tf.reshape(feature_dict['obj_desc_position_seq'], [step_num, 2]),
      'verb_refs':  # No data for Pixel on the field
          tf.zeros([step_num, 2], tf.int32),
      'agreement_count':
          tf.constant(100, dtype=tf.int32),
  }
  if load_dom_dist:
    feature['obj_dom_dist'] = tf.reshape(
        feature_dict['ui_obj_dom_distance'],
        [step_num, MAX_UI_OBJECT_NUM[data_source],
         MAX_UI_OBJECT_NUM[data_source]])
  feature['rule'] = tf.constant(5, dtype=tf.int32)
  if load_extra:
    feature['task_id'] = tf.reshape(feature_dict['task_id'], [])
    feature['raw_task'] = tf.reshape(feature_dict['instruction_str'], [])
    feature['obj_raw_text'] = tf.reshape(
        feature_dict['ui_obj_str_seq'],
        [step_num, MAX_UI_OBJECT_NUM[data_source]])
  feature['data_source'] = tf.constant(2, dtype=tf.int32)
  return feature
