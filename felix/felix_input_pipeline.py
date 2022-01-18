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

"""Input function utils."""

from typing import Sequence

import tensorflow as tf


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.io.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.cast(t, tf.int32)
    example[name] = t

  return example


def _get_file_reader():
  return tf.data.TFRecordDataset


def create_felix_dataset(input_patterns,
                         seq_length,
                         max_predictions_per_seq = 20,
                         batch_size = 512,
                         is_training = True,
                         use_insertion = True,
                         use_pointing = False,
                         use_weighted_labels = True):
  """Creates input dataset from (tf)records files for Felix model.

  Args:
    input_patterns: List of regex pattern for specifying input files.
    seq_length: Maximum length of sequence.
    max_predictions_per_seq: Only used for insertion, maximum number of
                             insertions.
    batch_size: Size of batch.
    is_training: Whether dataset is used for training.
    use_insertion: Whether the dataset is intended for the insertion model.
    use_pointing: Whether pointing is used within dataset. This flag is ignored
                  if use_insertion is true. Currently only True is supported.
    use_weighted_labels: Whether different labels were given different weights.
                         Primarly used to increase the importance of rare tags.


  Returns:
    tensorflow dataset.
  """
  insertion_name_to_features = {
      'input_ids':
          tf.io.FixedLenFeature([seq_length], tf.int64),
      'input_mask':
          tf.io.FixedLenFeature([seq_length], tf.int64),
      'segment_ids':
          tf.io.FixedLenFeature([seq_length], tf.int64),
      'masked_lm_positions':
          tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
      'masked_lm_ids':
          tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
      'masked_lm_weights':
          tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32)
  }
  tagging_name_to_features = {
      'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
      'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
      'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
      'labels': tf.io.FixedLenFeature([seq_length], tf.int64),
  }
  if use_weighted_labels:
    tagging_name_to_features['labels_mask'] = tf.io.FixedLenFeature(
        [seq_length], tf.float32)
  else:
    tagging_name_to_features['labels_mask'] = tf.io.FixedLenFeature(
        [seq_length], tf.int64)
  if use_pointing:
    tagging_name_to_features['point_indexes'] = tf.io.FixedLenFeature(
        [seq_length], tf.int64)
  if use_insertion:
    name_to_features = insertion_name_to_features
  else:
    name_to_features = tagging_name_to_features

  for input_pattern in input_patterns:
    if not tf.io.gfile.glob(input_pattern):
      raise ValueError('%s does not match any files.' % input_pattern)

  dataset = tf.data.Dataset.list_files(input_patterns, shuffle=is_training)
  options = tf.data.Options()
  options.experimental_distribute.auto_shard_policy = (
      tf.data.experimental.AutoShardPolicy.DATA)
  options.experimental_deterministic = not is_training
  dataset = dataset.with_options(options)
  if is_training:
    dataset = dataset.repeat()

    # We set shuffle buffer to exactly match total number of
    # training files to ensure that training data is well shuffled.
    input_files = []
    for input_pattern in input_patterns:
      input_files.extend(tf.io.gfile.glob(input_pattern))
    dataset = dataset.shuffle(len(input_files))

  # In parallel, create tf record dataset for each training file.
  # cycle_length = 8 means that up to 8 files will be read and deserialized in
  # parallel. You may want to increase this number if you have a large number of
  # CPU cores.
  dataset = dataset.interleave(
      _get_file_reader(),
      cycle_length=None,  # Let tf.data runtime decide based on available CPU.
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  decode_fn = lambda record: _decode_record(record, name_to_features)
  dataset = dataset.map(
      decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def _select_data_from_record(record):
    """Filter out features to use for pretraining."""
    if use_insertion:
      x = {
          'input_word_ids': record['input_ids'],
          'input_mask': record['input_mask'],
          'input_type_ids': record['segment_ids'],
          'masked_lm_positions': record['masked_lm_positions'],
          'masked_lm_ids': record['masked_lm_ids'],
          'masked_lm_weights': record['masked_lm_weights'],
      }

      y = record['masked_lm_weights']

    else:
      x = {
          'input_word_ids': record['input_ids'],
          'input_mask': record['input_mask'],
          'input_type_ids': record['segment_ids'],
          'edit_tags': record['labels'],
      }
      if use_weighted_labels:
        x['labels_mask'] = record['labels_mask']
      else:
        x['labels_mask'] = record['input_mask']
      if use_pointing:
        x['pointers'] = record['point_indexes']

      y = record['input_ids']

    return (x, y)

  dataset = dataset.map(
      _select_data_from_record,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if is_training:
    dataset = dataset.shuffle(50000)

  dataset = dataset.batch(batch_size, drop_remainder=is_training)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset
