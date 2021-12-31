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

"""Construct a tf.data.Dataset of protein training data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import tensorflow.compat.v1 as tf
from tensorflow.contrib import lookup as contrib_lookup
import utils

DATA_ROOT_DIR = 'data/'
TEST_FOLD = 'test'
DEV_FOLD = 'dev'
TRAIN_FOLD = 'train'
ALL_FOLD = '*'
DATA_FOLD_VALUES = [TRAIN_FOLD, DEV_FOLD, TEST_FOLD, ALL_FOLD]

SEQUENCE_KEY = 'sequence'
SEQUENCE_LENGTH_KEY = 'sequence_length'
SEQUENCE_ID_KEY = 'id'
LABEL_KEY = 'label'

DATASET_FEATURES = {
    SEQUENCE_KEY: tf.FixedLenFeature([], tf.string),
    LABEL_KEY: tf.io.RaggedFeature(dtype=tf.string, row_splits_dtype=tf.int64),
    SEQUENCE_ID_KEY: tf.FixedLenFeature([], tf.string)
}


def _map_sequence_to_ints(example, amino_acid_table):
  """Take amino acids in features as strings and replaces them with ints.

  Args:
    example: dictionary from string to tensor, containing key
      SEQUENCE_KEY.
    amino_acid_table: tf.contrib.lookup.index_table_from_tensor.

  Returns:
    dict from string to tensor, where the value at SEQUENCE_KEY is
    converted from a np.array of string labels to a np.array of ints.
  """
  seq = example[SEQUENCE_KEY]
  seq_char_by_char_sparse = tf.string_split([seq], delimiter='')
  seq_char_by_char = seq_char_by_char_sparse.values
  seq_indices = amino_acid_table.lookup(seq_char_by_char)
  example[SEQUENCE_KEY] = seq_indices
  return example


def _map_labels_to_ints(example, protein_class_table):
  """Take labels in features as strings and replaces them with ints.

  Args:
    example: dictionary from string to tensor, containing key LABEL_KEY.
    protein_class_table: tf.contrib.lookup.index_table_from_tensor.

  Returns:
    dict from string to tensor, where the value at LABEL_KEY is converted
    from a np.array of string labels to a np.array of ints.
  """
  label_indices = protein_class_table.lookup(example[LABEL_KEY])
  # In a multilabel task there are multiple labels in the label field.
  # Any labels not in the vocab are mapped to -1 and we then remove them
  # with the below:
  label_mask = tf.not_equal(label_indices, -1)
  label_indices = tf.boolean_mask(label_indices, label_mask)
  example[LABEL_KEY] = label_indices
  return example


def _to_one_hot_sequence(indexed_sequence_tensors):
  """Convert ints in sequence to one-hots.

  Turns indices (in the sequence) into one-hot vectors.

  Args:
    indexed_sequence_tensors: dict containing SEQUENCE_KEY field.
        For example: {
          'sequence': '[1, 3, 3, 4, 12, 6]'  # This is the amino acid sequence.
            ... }

  Returns:
    indexed_sequence_tensors with the same overall structure as the input,
    except that SEQUENCE_KEY field has been transformed to a one-hot
    encoding.
    For example:
    {
      # The first index in sequence is from letter C, which
      # is at index 1 in the amino acid vocabulary, and the second is from
      # E, which is at index 4.
      SEQUENCE_KEY: [[0, 1, 0, ...], [0, 0, 0, 1, 0, ...]...]
      ...
    }
  """
  indexed_sequence_tensors[SEQUENCE_KEY] = tf.one_hot(
      indices=indexed_sequence_tensors[SEQUENCE_KEY],
      depth=len(utils.AMINO_ACID_VOCABULARY))
  return indexed_sequence_tensors


def _add_sequence_length(example):
  example[SEQUENCE_LENGTH_KEY] = tf.strings.length(example[SEQUENCE_KEY])
  return example


def non_batched_dataset(train_dev_or_test,
                        label_vocab,
                        data_root_dir=DATA_ROOT_DIR):
  """Constructs a dataset of examples.

  Args:
    train_dev_or_test: one of _DEV_FOLD_VALUES. The source examples to load into
      a dataset.
    label_vocab: list of string.
    data_root_dir: path to tfrecord examples.

  Returns:
    tf.data.Dataset, where each example is of form
    {
        SEQUENCE_KEY: one-hot of amino acid characters
        SEQUENCE_LENGTH_KEY: length of sequence
        SEQUENCE_ID_KEY: unique identifier for protein
        LABEL_KEY: rank-1 tensor of integer labels from label_vocab,
    }
  """
  if train_dev_or_test not in DATA_FOLD_VALUES:
    raise ValueError(('Only train, dev, test and * are supported datasets.'
                      ' Received {}.').format(train_dev_or_test))
  dataset_files = [
      os.path.join(data_root_dir, f)
      for f in tf.gfile.ListDirectory(data_root_dir)
      if train_dev_or_test in f
  ]

  tfrecord_dataset = tf.data.TFRecordDataset(dataset_files)

  dataset = tfrecord_dataset.map(lambda record: tf.io.parse_single_example(  # pylint: disable=g-long-lambda
      record, DATASET_FEATURES))
  dataset = dataset.map(_add_sequence_length)

  amino_acid_table = contrib_lookup.index_table_from_tensor(
      utils.AMINO_ACID_VOCABULARY,
      default_value=len(utils.AMINO_ACID_VOCABULARY))
  protein_class_table = contrib_lookup.index_table_from_tensor(
      mapping=label_vocab)

  dataset = dataset.map(lambda ex: _map_sequence_to_ints(ex, amino_acid_table))
  dataset = dataset.map(lambda ex: _map_labels_to_ints(ex, protein_class_table))
  dataset = dataset.map(_to_one_hot_sequence)

  if train_dev_or_test == TRAIN_FOLD:
    dataset = dataset.repeat()

  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  return dataset


def batched_dataset(input_dataset, batch_size):
  """Batches and pads input_dataset.

  Args:
    input_dataset: tf.data.Dataset. output of _non_padded_dataset.
    batch_size: int.

  Returns:
    tf.data.Dataset. Because sequences are non-uniform length, and because
    the number of labels for a sequence is variable, the sequence and label
    features are padded with 0 and -1, respectively.
  """

  padding_values = {
      SEQUENCE_KEY: tf.constant(0, tf.float32),
      LABEL_KEY: tf.constant(-1, tf.int64),

      # Padding value is unused since this is always provided upstream and is
      # of deterministic shape.
      SEQUENCE_LENGTH_KEY: tf.constant(0, tf.int32),

      # Padding value is unused since this is always provided upstream and is
      # of deterministic shape.
      SEQUENCE_ID_KEY: ''
  }

  dataset = input_dataset.batch(batch_size, padding_values)
  return dataset


def make_input_fn(batch_size, data_file_pattern, train_dev_or_test,
                  label_vocab):
  """Makes an input_fn, according to the `Estimator` `input_fn` interface.

  Args:
    batch_size: int.
    data_file_pattern: A file path pattern that has your examples.
    train_dev_or_test: one of _DEV_FOLD_VALUES. The source examples to load into
      a dataset.
    label_vocab: list of string.

  Returns:
    input_fn to be used by Estimator.
  """

  def _input_fn():
    """`Estimator`-compatible input_fn."""

    dataset = non_batched_dataset(
        train_dev_or_test=train_dev_or_test,
        data_root_dir=data_file_pattern,
        label_vocab=label_vocab)
    dataset = batched_dataset(dataset, batch_size)
    itr = dataset.make_initializable_iterator()

    data_ops = itr.get_next()
    features = {
        SEQUENCE_KEY: data_ops[SEQUENCE_KEY],
        SEQUENCE_LENGTH_KEY: data_ops[SEQUENCE_LENGTH_KEY],
    }
    labels = {
        LABEL_KEY: data_ops[LABEL_KEY],
        SEQUENCE_ID_KEY: data_ops[SEQUENCE_ID_KEY]
    }

    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, itr.initializer)

    return features, labels

  return _input_fn
