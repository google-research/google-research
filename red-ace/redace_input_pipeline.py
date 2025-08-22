# coding=utf-8
# Copyright 2025 The Google Research Authors.
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


def create_redace_dataset(
    input_file,
    seq_length,
    batch_size=512,
    is_training=True,
    use_weighted_labels=True,
):
  """Creates input dataset from tfrecords files for RED-ACE model.

  Args:
    input_file: Input file path.
    seq_length: Maximum length of sequence.
    batch_size: Size of batch.
    is_training: Whether dataset is used for training.
    use_weighted_labels: Whether different labels were given different weights.
      Primarly used to increase the importance of rare tags.

  Returns:
    tensorflow dataset.
  """
  tagging_name_to_features = {
      'input_ids':
          tf.io.FixedLenFeature([seq_length], tf.int64),
      'input_mask':
          tf.io.FixedLenFeature([seq_length], tf.int64),
      'segment_ids':
          tf.io.FixedLenFeature([seq_length], tf.int64),
      'bucketed_confidence_scores':
          tf.io.FixedLenFeature([seq_length], tf.int64),
      'labels':
          tf.io.FixedLenFeature([seq_length], tf.int64),
  }

  tagging_name_to_features['labels_mask'] = tf.io.FixedLenFeature([seq_length],
                                                                  tf.float32)

  name_to_features = tagging_name_to_features

  d = tf.data.Dataset.from_tensor_slices(tf.constant([input_file]))
  dataset = d.interleave(
      tf.data.TFRecordDataset,
      cycle_length=1,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  ).repeat()

  if is_training:
    dataset = dataset.shuffle(buffer_size=min(1, 100))
  options = tf.data.Options()
  options.experimental_distribute.auto_shard_policy = (
      tf.data.experimental.AutoShardPolicy.DATA)
  options.experimental_deterministic = not is_training
  dataset = dataset.with_options(options)
  decode_fn = lambda record: _decode_record(record, name_to_features)
  dataset = dataset.map(
      decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def _select_data_from_record(record):
    """Filter out features to use for pretraining."""
    x = {
        'input_word_ids': record['input_ids'],
        'input_mask': record['input_mask'],
        'input_type_ids': record['segment_ids'],
        'input_confidence_scores': record['bucketed_confidence_scores'],
        'edit_tags': record['labels'],
    }
    if use_weighted_labels and 'labels_mask' in record:
      x['labels_mask'] = record['labels_mask']
    else:
      x['labels_mask'] = record['input_mask']

    y = record['input_ids']

    return (x, y)

  dataset = dataset.map(
      _select_data_from_record,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if is_training:
    dataset = dataset.shuffle(50000)

  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset
