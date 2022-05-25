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

from latent_programmer.tasks.robust_fill import dsl as robust_fill_dsl
from latent_programmer.tasks.scan import scan_vocab

gfile = tf.io.gfile


def create_robust_fill_dataset_from_tf_record(
    file_pattern, program_token_id_table, spec_token_id_table,
    max_target_length):
  """Returns an instance of tf.data.Dataset."""
  del program_token_id_table  # TODO(kshi): Unused until we process the program.

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
  separator_id = spec_token_id_table['|']

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

    # Parse inputs into tokens.
    inputs = tf.strings.unicode_split(inputs, 'UTF-8').to_tensor()
    inputs = spec_vocab_table.lookup(inputs)  # Map characters to tokens.

    # Parse outputs into tokens.
    outputs_with_separators = (
        tf.strings.unicode_split(outputs, 'UTF-8').to_tensor())
    outputs_with_separators = spec_vocab_table.lookup(outputs_with_separators)
    split_outputs = tf.strings.unicode_split(
        tf.strings.split(outputs, sep='|'), 'UTF-8')
    outputs = split_outputs.merge_dims(1, 2).to_tensor()
    outputs = spec_vocab_table.lookup(outputs)

    # Compute indices for the start of each part of the spec, w.r.t. the
    # original spec.
    separator_indices = tf.where(tf.equal(outputs_with_separators,
                                          separator_id))[:, 1]
    separator_indices = tf.reshape(separator_indices,
                                   (tf.shape(outputs_with_separators)[0], -1))
    start_indices = separator_indices - tf.expand_dims(
        tf.range(tf.shape(separator_indices)[1], dtype=tf.int64), 0)
    start_indices = tf.concat((tf.zeros((tf.shape(start_indices)[0], 1),
                                        dtype=tf.int64),
                               start_indices), axis=1)

    num_examples = tf.shape(start_indices)[0]
    num_parts = tf.shape(start_indices)[1]

    # Construct the shifted spec suffixes.
    flat_start_indices = tf.reshape(start_indices, (-1,))
    prefix_mask = (1 - tf.sequence_mask(
        flat_start_indices, maxlen=tf.shape(outputs)[-1], dtype=tf.int64))
    masked_outputs = tf.repeat(outputs, num_parts, axis=0) * prefix_mask
    output_suffixes = tf.vectorized_map(
        fn=lambda x: tf.roll(x[0], x[1], axis=0),
        elems=(masked_outputs, -flat_start_indices))

    # Compute indices for the start/end of spec parts, w.r.t. the shifted spec
    # suffixes.
    ground_truth_start_indices = tf.zeros((num_examples * num_parts,),
                                          dtype=tf.int64)
    cumulative_end_indices = tf.concat(
        (start_indices, tf.math.count_nonzero(outputs, axis=-1, keepdims=True)),
        axis=1)
    ground_truth_end_indices = tf.reshape(
        cumulative_end_indices[:, 1:] - cumulative_end_indices[:, :-1], (-1,))

    # Construct the actual spec parts to predict.
    range_indices = tf.expand_dims(
        tf.range(tf.shape(output_suffixes)[-1], dtype=tf.int64), axis=0)
    part_mask = tf.where(tf.logical_and(
        range_indices >= tf.expand_dims(ground_truth_start_indices, axis=1),
        range_indices < tf.expand_dims(ground_truth_end_indices, axis=1)), 1, 0)
    output_parts = output_suffixes * tf.cast(part_mask, tf.int64)
    output_parts = tf.pad(output_parts, [[0, 0], [0, 1]])  # Make room for sep.
    # TODO(kshi): roll output_parts leftward by start_indices for SCAN.
    first_zero_index = tf.math.count_nonzero(output_parts, axis=-1)
    output_parts += tf.one_hot(first_zero_index,
                               depth=tf.shape(output_parts)[-1],
                               dtype=tf.int64) * separator_id

    # Reshape everything so that different spec suffixes become different
    # dataset elements.
    output_suffixes_reshaped = tf.transpose(
        tf.reshape(output_suffixes, (num_examples, num_parts, -1)), (1, 0, 2))
    output_parts_reshaped = tf.transpose(
        tf.reshape(output_parts, (num_examples, num_parts, -1)), (1, 0, 2))
    inputs_reshaped = tf.reshape(
        tf.tile(inputs, (num_parts, 1)), (num_parts, num_examples, -1))
    ground_truth_start_indices_reshaped = tf.transpose(
        tf.reshape(ground_truth_start_indices, (num_examples, num_parts)))
    ground_truth_end_indices_reshaped = tf.transpose(
        tf.reshape(ground_truth_end_indices, (num_examples, num_parts)))

    # Combine spec parts from all examples into one sequence with separator
    # tokens between examples and ending in EOS.
    shifts = tf.cumsum(tf.concat(
        (tf.zeros((num_parts, 1), dtype=tf.int64),
         ground_truth_end_indices_reshaped[:, :-1] + 1), 1), axis=-1)
    flat_shifts = tf.reshape(shifts, (-1,))
    output_len = tf.shape(output_parts_reshaped)[-1]
    flat_spec_parts = tf.reshape(output_parts_reshaped, (-1, output_len))
    flat_spec_parts = tf.pad(flat_spec_parts,
                             [[0, 0], [0, max_target_length - output_len]])
    combined_spec_parts = tf.vectorized_map(
        fn=lambda x: tf.roll(x[0], x[1], axis=0),
        elems=(flat_spec_parts, flat_shifts))
    combined_spec_parts = tf.reshape(combined_spec_parts,
                                     (num_parts, num_examples, -1))
    combined_spec_parts = tf.reduce_sum(combined_spec_parts, axis=1)
    first_zero_index = tf.math.count_nonzero(combined_spec_parts, axis=-1)
    combined_spec_parts += tf.one_hot(first_zero_index,
                                      depth=tf.shape(combined_spec_parts)[-1],
                                      dtype=tf.int64) * eos_id

    # Create a dataset containing data for all spec suffixes.
    dataset = tf.data.Dataset.from_tensor_slices({
        'inputs': inputs_reshaped,
        'outputs': output_suffixes_reshaped,
        'spec_parts': combined_spec_parts,
        'start_index': ground_truth_start_indices_reshaped,
        'end_index': ground_truth_end_indices_reshaped})
    return dataset

  # Create one big flat dataset containing all spec suffixes for all problems.
  dataset = raw_dataset.flat_map(_parse_fn)
  return dataset


def create_scan_dataset_from_tf_record(
    file_pattern, token_id_table, char_id_table, use_bos_separators):
  """Returns an instance of tf.data.Dataset."""
  del char_id_table
  if 1 + 1 == 2:  # Avoid an unreachable code lint error.
    raise NotImplementedError()  # TODO(kshi): Implement.

  filenames = gfile.glob(file_pattern)
  raw_dataset = tf.data.TFRecordDataset(filenames)

  bos_token = token_id_table[scan_vocab.BOS]
  eos_token = token_id_table[scan_vocab.EOS]

  def _parse_fn(record):
    """Parses a record into a feature_dict."""
    feature_values = tf.io.parse_single_example(
        serialized=record,
        features={
            'input': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'output': tf.io.FixedLenFeature([], tf.string, default_value=''),
        })

    # Step 1. Parse input tokens.
    input_str = feature_values['input']
    input_split = tf.strings.split(input_str, sep=' ')
    input_tokens = tf.strings.to_number(input_split, out_type=tf.int32)
    input_tokens = tf.expand_dims(input_tokens, axis=0)

    # Step 2. Create dummy "output" (analogous to RobustFill's output).
    dummy = input_tokens

    # Step 3. Parse output program into tokens.
    program = feature_values['output']
    # Add BOS between every part, then add BOS followed by EOS. `program` has a
    # | between parts (not at the beginning or end of the sequence, and no
    # spaces around |).
    if use_bos_separators:
      program = tf.strings.join([
          tf.strings.regex_replace(program, r'\|', ' {} '.format(bos_token)),
          ' {} {}'.format(bos_token, eos_token),
      ])
    else:
      program = tf.strings.join([
          tf.strings.regex_replace(program, r'\|', ' '),
          ' {}'.format(eos_token),
      ])
    # Parse numbers.
    program = tf.strings.split(program, sep=' ')
    program = tf.strings.to_number(program, out_type=tf.int32)

    return input_tokens, dummy, program

  dataset = raw_dataset.map(_parse_fn)
  return dataset
