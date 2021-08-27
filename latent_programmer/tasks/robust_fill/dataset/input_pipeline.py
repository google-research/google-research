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

# Lint as: python3
"""Input pipeline for Robust-fill dataset."""

import tensorflow.compat.v2 as tf

from latent_programmer.tasks.robust_fill import dsl

gfile = tf.io.gfile


def create_dataset_from_tf_record(file_pattern, token_id_table, char_id_table):
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
    ios = tf.strings.unicode_split(ios, 'UTF-8')
    ios = ios.to_tensor()
    ios = char_table.lookup(ios)  # Map characters to integer tokens.

    program_encoding = tf.strings.to_number(
        tf.strings.split(feature_values['program_encoding'], sep=' '),
        out_type=tf.int32)
    # Add EOS token.
    eos_token = token_id_table[dsl.EOS]
    program_encoding = tf.concat([program_encoding, [eos_token]], axis=-1)
    return ios[:, 0], ios[:, 1], program_encoding

  dataset = raw_dataset.map(_parse_fn)
  return dataset
