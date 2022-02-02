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

"""Functions to read and preprocess data for Contrack models."""

import logging
from typing import Any, Dict, Text, Tuple

import tensorflow as tf

from contrack import env
from contrack.encoding import Encodings

Manifest = Dict[str, Any]


def build_manifests(encodings):
  """Returns the feature descriptions for Contrack SequenceExamples."""
  input_vec_length = encodings.enref_encoding_length
  prediction_vec_length = encodings.prediction_encoding_length

  # Feature manifests
  context_feature_manifest = {
      'state_seq_length': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
      'token_seq_length': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
      'scenario_id': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
      'sender': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
      'participants': tf.io.VarLenFeature(dtype=tf.string),
  }

  sequence_feature_manifest = {
      'state_seq':
          tf.io.FixedLenSequenceFeature(
              shape=[input_vec_length], dtype=tf.float32),
      'token_seq':
          tf.io.FixedLenSequenceFeature(
              shape=[input_vec_length], dtype=tf.float32),
      'word_seq':
          tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.string),
      'annotation_seq':
          tf.io.FixedLenSequenceFeature(
              shape=[prediction_vec_length], dtype=tf.float32),
  }

  return context_feature_manifest, sequence_feature_manifest


def _parse_features(serialized_seqex, context_manifest,
                    sequence_manifest):
  """Parse SequenceExample proto and return Dict of the features therein."""
  context, feature_lists, _ = tf.io.parse_sequence_example(
      serialized_seqex,
      context_features=context_manifest,
      sequence_features=sequence_manifest)

  features = dict(context)
  features.update((k, feature_lists[k]) for k in sequence_manifest)

  return features


def _rotate_ids_tensor(tensor, offset):
  """Rotate ids to start from start_index instead of zero."""
  # Input shape is [num_batch, num_positions, num_features]
  shape = tf.shape(tensor)
  num_batch = shape[0]
  num_positions = shape[1]
  num_features = shape[2]

  # Create an updates tensor for tf.scatter_nd. This tensor contains indices
  # into seq so that calling tf.scatter_nd(ids_t, updates, ...) just returns seq
  # again. However, if one adds a constant n to the last column of updates,
  # tf.scatter_nd(ids_t, updates, ...) rotates the features by n positions.
  u1 = tf.tile(tf.expand_dims(tf.range(num_batch), -1), [1, num_features])
  u1 = tf.pad(
      tf.reshape(u1, [num_batch, num_features, 1]), [[0, 0], [0, 0], [0, 1]])
  u2 = tf.pad(tf.expand_dims(tf.range(num_features), -1), [[0, 0], [1, 0]])
  u2 = tf.tile([u2], [num_batch, 1, 1])
  updates = u1 + u2

  # This tensor can be added to updates so that tf.scatter_nd rotates features
  # as specified by b.
  rotation = tf.expand_dims(
      tf.expand_dims(tf.dtypes.cast(offset, tf.int32), -1), -1)
  rotation = tf.pad(rotation, [[0, 0], [0, 0], [1, 0]])
  rotation = tf.tile(rotation, [1, num_features, 1])

  # Transpose seq, then rotate individual examples, then transpose back.
  ids_t = tf.transpose(tensor, tf.constant([0, 2, 1], tf.int32))
  new_shape = tf.concat([[num_batch], [2 * num_features], [num_positions]], 0)
  rotated_t = tf.scatter_nd(updates + rotation, ids_t, new_shape)
  rotated = tf.transpose(rotated_t, tf.constant([0, 2, 1]))
  rotated = rotated[:, :, :num_features] + rotated[:, :, num_features:]

  return rotated


def rotate_enref_ids(encodings,
                     features):
  """Adds a semi-random offset to all enref ids in the batch."""
  scenario_id = features['scenario_id']
  offset = tf.strings.to_hash_bucket_fast(scenario_id, 7)

  # Rotate enref ids in enref sequence
  enrefs = features['state_seq']
  enref_id = encodings.as_enref_encoding(enrefs).enref_id
  rotated_id = _rotate_ids_tensor(enref_id.slice(), offset)
  enrefs = enref_id.replace(rotated_id)

  group_ids = encodings.as_enref_encoding(enrefs).enref_membership
  rotated_id = _rotate_ids_tensor(group_ids.slice(), offset)
  enrefs = group_ids.replace(rotated_id)

  features['state_seq'] = enrefs

  # Rotate enref ids in predictions sequence
  predictions = features['annotation_seq']
  enref_id = encodings.as_prediction_encoding(predictions).enref_id
  rotated_id = _rotate_ids_tensor(enref_id.slice(), offset)
  predictions = enref_id.replace(rotated_id)

  group_ids = encodings.as_prediction_encoding(predictions).enref_membership
  rotated_id = _rotate_ids_tensor(group_ids.slice(), offset)
  predictions = group_ids.replace(rotated_id)

  features['annotation_seq'] = predictions

  return features


def read_training_data(data_glob, config,
                       encodings):
  """Read training data from files specified by data_glob."""
  context_manifest, sequence_manifest = build_manifests(encodings)

  with tf.name_scope('read_training_data'):
    filenames = tf.io.gfile.glob(data_glob)
    if not filenames:
      raise ValueError(f'No files found for training glob {data_glob}')
    logging.info('Reading from files %s', str(filenames))
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.shuffle(config.shuffle_buffer_size)
    dataset = dataset.batch(config.batch_size).repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(
        lambda x: _parse_features(x, context_manifest, sequence_manifest))

    if config.rotate_enref_ids:
      dataset = dataset.map(lambda x: rotate_enref_ids(encodings, x))

    return dataset


def read_eval_data(data_glob, config,
                   encodings):
  """Read evaluation data from files specified by data_glob."""
  context_manifest, sequence_manifest = build_manifests(encodings)

  with tf.name_scope('read_eval_data'):
    filenames = tf.io.gfile.glob(data_glob)
    if not filenames:
      raise ValueError(f'No files found for eval glob {data_glob}')
    logging.info('Reading from files %s', str(filenames))
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        lambda x: _parse_features(x, context_manifest, sequence_manifest))

    return dataset
