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

"""Input pipeline for Robust-fill dataset."""

import tensorflow as tf

from latent_programmer.tasks.deepcoder import deepcoder_dsl
from latent_programmer.tasks.robust_fill import dsl as robust_fill_dsl

gfile = tf.io.gfile

SEPARATOR_TOKEN = '|'


def create_robust_fill_dataset(
    file_pattern, spec_token_id_table, num_examples, entire_programs,
    renaming_dict):
  """Loads a RobustFill step-by-step dataset.

  Args:
    file_pattern: A file pattern for the TFRecord files to read.
    spec_token_id_table: Mapping from characters (tokens) to token IDs for the
      I/O specification vocabulary.
    num_examples: The number of examples in an I/O specification.
    entire_programs: Whether the dataset contains decomposition data (False) or
      entire programs (True).
    renaming_dict: A dict mapping from the new name of fields in this dataset to
      the old name as in the original TFRecord files.

  Returns:
    A tf.data.Dataset containing dictionaries where the keys are the same as in
    `renaming_dict`.
  """
  filenames = gfile.glob(file_pattern)
  raw_dataset = tf.data.TFRecordDataset(filenames)

  spec_vocab_table = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(
          # Add padding.
          [''] + list(spec_token_id_table.keys()),
          [0] + list(spec_token_id_table.values()),
          key_dtype=tf.string,
          value_dtype=tf.int64),
      len(spec_token_id_table) + 1)
  eos_id = spec_token_id_table[robust_fill_dsl.EOS]

  def _parse_fn(record):
    """Parses a record into a feature_dict."""
    empty_default = [''] * num_examples
    if entire_programs:
      feature_values = tf.io.parse_single_example(
          serialized=record,
          features={
              'inputs':
                  tf.io.FixedLenFeature([num_examples], tf.string,
                                        default_value=empty_default),
              'outputs':
                  tf.io.FixedLenFeature([num_examples], tf.string,
                                        default_value=empty_default),
              'program':
                  tf.io.FixedLenFeature([], tf.string, default_value=''),
          })
    else:
      feature_values = tf.io.parse_single_example(
          serialized=record,
          features={
              'inputs':
                  tf.io.FixedLenFeature([num_examples], tf.string,
                                        default_value=empty_default),
              'outputs':
                  tf.io.FixedLenFeature([num_examples], tf.string,
                                        default_value=empty_default),
              'next_part':
                  tf.io.FixedLenFeature([num_examples], tf.string,
                                        default_value=empty_default),
              'corrupted_next_part':
                  tf.io.FixedLenFeature([num_examples], tf.string,
                                        default_value=empty_default),
              'program_part':
                  tf.io.FixedLenFeature([], tf.string, default_value=''),
          })

    # Map characters to tokens.
    inputs = tf.strings.unicode_split(
        feature_values['inputs'], 'UTF-8').to_tensor()
    inputs = spec_vocab_table.lookup(inputs)

    outputs = tf.strings.unicode_split(
        feature_values['outputs'], 'UTF-8').to_tensor()
    outputs = spec_vocab_table.lookup(outputs)

    if entire_programs:
      program = tf.strings.split(
          tf.strings.split(feature_values['program'], sep='|'), sep=' ')
      program = program.merge_dims(0, -1)
      program = tf.strings.to_number(program, out_type=tf.int32)
      program = tf.concat([program, [eos_id]], axis=-1)
      all_data_dict = {
          'inputs': inputs,
          'outputs': outputs,
          'program': program,
      }

    else:
      next_part = tf.strings.unicode_split(
          feature_values['next_part'], 'UTF-8').to_tensor()
      next_part = spec_vocab_table.lookup(next_part)

      corrupted_next_part = tf.strings.unicode_split(
          feature_values['corrupted_next_part'], 'UTF-8').to_tensor()
      corrupted_next_part = spec_vocab_table.lookup(corrupted_next_part)

      joined_next_part = tf.strings.reduce_join(feature_values['next_part'],
                                                separator=SEPARATOR_TOKEN)
      joined_next_part = tf.strings.unicode_split(joined_next_part, 'UTF-8')
      joined_next_part = spec_vocab_table.lookup(joined_next_part)
      joined_next_part = tf.concat([joined_next_part, [eos_id]], axis=-1)

      program_part = tf.strings.split(feature_values['program_part'], sep=' ')
      program_part = tf.strings.to_number(program_part, out_type=tf.int32)
      program_part = tf.concat([program_part, [eos_id]], axis=-1)

      # inputs: [num_strings, max_length_of_input]
      # outputs: [num_strings, max_length_of_output]
      # next_part: [num_strings, max_length_of_output_part]
      # corrupted_next_part: [num_strings, max_length_of_output_part]
      # joined_next_part: [num_strings * (max_length_of_part + 1)]
      # program_part: [max_length_of_program_part]
      all_data_dict = {
          'inputs': inputs,
          'outputs': outputs,
          'next_part': next_part,
          'corrupted_next_part': corrupted_next_part,
          'joined_next_part': joined_next_part,
          'program_part': program_part,
      }

    return {
        new_name: all_data_dict[old_name]
        for new_name, old_name in renaming_dict.items()
    }

  dataset = raw_dataset.map(_parse_fn)
  return dataset


def create_deepcoder_dataset(
    file_pattern, token_to_id, num_examples, entire_programs, renaming_dict):
  """Loads a DeepCoder step-by-step dataset.

  Args:
    file_pattern: A file pattern for the TFRecord files to read.
    token_to_id: Mapping from tokens to token IDs for the DeepCoder vocabulary.
    num_examples: The number of examples in an I/O specification.
    entire_programs: Whether the dataset contains decomposition data (False) or
      entire programs (True).
    renaming_dict: A dict mapping from the new name of fields in this dataset to
      the old name as in the original TFRecord files.

  Returns:
    A tf.data.Dataset containing dictionaries where the keys are the same as in
    `renaming_dict`.
  """
  filenames = gfile.glob(file_pattern)
  raw_dataset = tf.data.TFRecordDataset(filenames)

  vocab_table = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(
          list(token_to_id.keys()),
          list(token_to_id.values()),
          key_dtype=tf.string,
          value_dtype=tf.int64),
      len(token_to_id))
  eos_id = deepcoder_dsl.EOS_ID

  def _parse_fn(record):
    """Parses a record into a feature_dict."""
    empty_default = [''] * num_examples
    if entire_programs:
      feature_values = tf.io.parse_single_example(
          serialized=record,
          features={
              'inputs':
                  tf.io.FixedLenFeature([num_examples], tf.string,
                                        default_value=empty_default),
              'outputs':
                  tf.io.FixedLenFeature([num_examples], tf.string,
                                        default_value=empty_default),
              'program':
                  tf.io.FixedLenFeature([], tf.string, default_value=''),
          })
    else:
      feature_values = tf.io.parse_single_example(
          serialized=record,
          features={
              'inputs':
                  tf.io.FixedLenFeature([num_examples], tf.string,
                                        default_value=empty_default),
              'outputs':
                  tf.io.FixedLenFeature([num_examples], tf.string,
                                        default_value=empty_default),
              'next_part':
                  tf.io.FixedLenFeature([num_examples], tf.string,
                                        default_value=empty_default),
              'corrupted_next_part':
                  tf.io.FixedLenFeature([num_examples], tf.string,
                                        default_value=empty_default),
              'program_part':
                  tf.io.FixedLenFeature([], tf.string, default_value=''),
          })

    # Map tokens to ids.
    inputs = tf.strings.split(feature_values['inputs'], sep=' ').to_tensor()
    inputs = vocab_table.lookup(inputs)

    outputs = tf.strings.split(feature_values['outputs'], sep=' ').to_tensor()
    outputs = vocab_table.lookup(outputs)

    if entire_programs:
      program = tf.strings.split(feature_values['program'], sep=' ')
      program = vocab_table.lookup(program)
      program = tf.concat([program, [eos_id]], axis=-1)
      all_data_dict = {
          'inputs': inputs,
          'outputs': outputs,
          'program': program,
      }

    else:
      next_part = tf.strings.split(
          feature_values['next_part'], sep=' ').to_tensor()
      next_part = vocab_table.lookup(next_part)

      corrupted_next_part = tf.strings.split(
          feature_values['corrupted_next_part'], sep=' ').to_tensor()
      corrupted_next_part = vocab_table.lookup(corrupted_next_part)

      joined_next_part = tf.strings.reduce_join(feature_values['next_part'],
                                                separator=' | ')
      joined_next_part = tf.strings.split(joined_next_part, sep=' ')
      joined_next_part = vocab_table.lookup(joined_next_part)
      joined_next_part = tf.concat([joined_next_part, [eos_id]], axis=-1)

      program_part = tf.strings.split(feature_values['program_part'], sep=' ')
      program_part = vocab_table.lookup(program_part)
      program_part = tf.concat([program_part, [eos_id]], axis=-1)

      # Remove the part like 'x0 =' and keep only the RHS of the assignment.
      program_part_rhs = program_part[2:]

      # inputs: [num_examples, max_length_of_input]
      # outputs: [num_examples, max_length_of_output]
      # next_part: [num_examples, max_length_of_output_part]
      # corrupted_next_part: [num_examples, max_length_of_output_part]
      # joined_next_part: [num_examples * (max_length_of_part + 1)]
      # program_part: [max_length_of_program_part + 1]
      # program_part_rhs: [max_length_of_program_part - 1]
      all_data_dict = {
          'inputs': inputs,
          'outputs': outputs,
          'next_part': next_part,
          'corrupted_next_part': corrupted_next_part,
          'joined_next_part': joined_next_part,
          'program_part': program_part,
          'program_part_rhs': program_part_rhs,
      }

    return {
        new_name: all_data_dict[old_name]
        for new_name, old_name in renaming_dict.items()
    }

  dataset = raw_dataset.map(_parse_fn)
  return dataset
