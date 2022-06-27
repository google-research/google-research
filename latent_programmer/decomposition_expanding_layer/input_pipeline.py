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

"""Input pipeline for Robust-fill dataset."""

import tensorflow.compat.v2 as tf

from latent_programmer.tasks.robust_fill import dsl

gfile = tf.io.gfile


def partition_ragged_tensor(x, num_partitions=1):
  """Partition rows of x into num_splits as evenly as possible."""

  step = tf.cast(x.nrows() // num_partitions, tf.int32)
  remainder = tf.cast(x.nrows() % num_partitions, tf.int32)
  if step >= 1:
    row_splits = x.row_splits
    first_part = tf.ones(remainder, dtype=tf.int32) * (step + 1)
    second_part = tf.ones(num_partitions - remainder, dtype=tf.int32) * step
    partial_lengths = tf.concat([[0], first_part, second_part], axis=0)
    partial_row_splits = tf.gather(row_splits, tf.cumsum(partial_lengths))
    x = tf.RaggedTensor.from_row_splits(
        values=x.values,
        row_splits=partial_row_splits)
  return x


def create_dataset_from_tf_record(file_pattern, token_id_table, char_id_table,
                                  num_partial_programs=1):
  """Returns an instance of tf.data.Dataset."""

  filenames = gfile.glob(file_pattern)
  raw_dataset = tf.data.TFRecordDataset(filenames)

  char_table = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(
          # Add padding.
          [''] + list(char_id_table.keys()),
          [0] + list(char_id_table.values()),
          key_dtype=tf.string,
          value_dtype=tf.int64),
      len(char_id_table) + 1)
  eos_token = token_id_table[dsl.EOS]

  def _parse_fn(record):
    """Parses a record into a feature_dict."""
    feature_values = tf.io.parse_single_example(
        serialized=record,
        features={
            'i/o':
                tf.io.FixedLenFeature([], tf.string, default_value=''),
            'program_encoding':
                tf.io.FixedLenFeature([], tf.string, default_value=''),
        })

    ios = tf.strings.split(
        tf.strings.split(feature_values['i/o'], sep='>'), sep='<')

    inputs, outputs = ios.merge_dims(0, 1)[::2], ios.merge_dims(0, 1)[1::2]
    # Step 1. Parse inputs into tokens.
    inputs = tf.strings.unicode_split(inputs, 'UTF-8').to_tensor()
    inputs = char_table.lookup(inputs)  # Map characters to tokens.

    # Step 2. Parse outputs into tokens.
    split_outputs = tf.strings.unicode_split(
        tf.strings.split(outputs, sep='|'), 'UTF-8')
    outputs = split_outputs.merge_dims(1, 2).to_tensor()
    outputs = char_table.lookup(outputs)
    # Partition output into substrings by partial program.
    split_outputs = tf.map_fn(
        lambda x: partition_ragged_tensor(x, num_partial_programs),
        split_outputs).to_tensor()
    split_outputs = char_table.lookup(split_outputs)

    # Step 3. Parse program into tokens.
    program_encoding = tf.strings.split(
        tf.strings.split(
            feature_values['program_encoding'],
            sep='|'),
        sep=' ')
    program_encoding = tf.strings.to_number(
        program_encoding, out_type=tf.int32)

    # Partition the rows of program into partial programs.
    program_encoding = partition_ragged_tensor(
        program_encoding, num_partial_programs)

    # Add EOS token to each partial program.
    program_encoding = tf.map_fn(
        lambda x: tf.concat([x, [eos_token]], axis=-1),
        program_encoding).to_tensor()

    n_rows = tf.shape(program_encoding)[0]
    if n_rows < num_partial_programs:
      n_cols = tf.shape(program_encoding)[1]
      pad_sequence = tf.one_hot(0, n_cols, on_value=eos_token, dtype=tf.int32)
      pad_block = tf.repeat([pad_sequence], [num_partial_programs - n_rows],
                            axis=0)
      program_encoding = tf.concat([program_encoding, pad_block], axis=0)

    return inputs, outputs, program_encoding, split_outputs

  dataset = raw_dataset.map(_parse_fn)
  return dataset
