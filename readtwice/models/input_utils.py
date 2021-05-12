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

"""Utilities for creating or transforming model inputs."""

from typing import Dict, List, Optional, Text, Tuple, Union

import tensorflow.compat.v1 as tf


def make_block_pos_features(block_ids):
  """Creates feature with block relative positions in the original document."""
  block_ids_expanded_0 = tf.expand_dims(block_ids, 0)
  x = tf.cast(
      tf.logical_and(
          tf.equal(tf.expand_dims(block_ids, 1), block_ids_expanded_0),
          tf.not_equal(block_ids_expanded_0, 0)), tf.int32)

  # pylint: disable=line-too-long
  # `tf.linalg.band_part(x, -1, 0)` sets to lower triangual part of matrix to 0.
  # See https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/linalg/band_part
  # for more details.
  # pylint: enable=line-too-long
  return tf.reduce_sum(tf.linalg.band_part(x, -1, 0), 1)


def mask_same_entity_mentions(
    token_ids, annotation_begins,
    annotation_ends, annotation_labels,
    masked_lm_positions, masked_lm_weights,
    mask_token_id, apply_across_whole_batch):
  """Mask mentions from the same entity."""
  batch_size = tf.shape(token_ids)[0]
  block_length = tf.shape(token_ids)[1]
  max_num_annotations = tf.shape(annotation_labels)[1]
  mask_begin = tf.sequence_mask(annotation_begins, block_length, dtype=tf.int32)
  mask_end = tf.sequence_mask(annotation_ends + 1, block_length, dtype=tf.int32)

  # Ignore annotations with 0 ([PAD]) and 1 ([UNK]) labels
  is_annotation_not_pad_or_unk = tf.logical_and(
      tf.not_equal(annotation_labels, 0), tf.not_equal(annotation_labels, 1))

  # [batch_size, max_num_annotations]
  is_annotation_masked = tf.reduce_max(
      tf.cast(
          tf.logical_and(
              tf.less_equal(
                  tf.expand_dims(annotation_begins, -1),
                  tf.expand_dims(masked_lm_positions, 1)),
              tf.greater_equal(
                  tf.expand_dims(annotation_ends, -1),
                  tf.expand_dims(masked_lm_positions, 1))),
          dtype=tf.int32) *
      tf.expand_dims(tf.cast(masked_lm_weights, dtype=tf.int32), 1), -1)

  if apply_across_whole_batch:
    # [batch_size * max_num_annotations, batch_size * max_num_annotations]
    are_annotations_for_the_same_entity = tf.cast(
        tf.logical_and(
            tf.equal(
                tf.reshape(annotation_labels,
                           [batch_size * max_num_annotations, 1]),
                tf.reshape(annotation_labels,
                           [1, batch_size * max_num_annotations])),
            tf.reshape(is_annotation_not_pad_or_unk,
                       [batch_size * max_num_annotations, 1])),
        dtype=tf.int32)

    # [batch_size * max_num_annotations]
    should_annotation_be_masked = tf.einsum(
        'i,ij->j',
        tf.reshape(is_annotation_masked, [batch_size * max_num_annotations]),
        are_annotations_for_the_same_entity)

    # [batch_size, max_num_annotations]
    should_annotation_be_masked = tf.reshape(should_annotation_be_masked,
                                             [batch_size, max_num_annotations])
  else:
    # [batch_size, max_num_annotations, max_num_annotations]
    are_annotations_for_the_same_entity = tf.cast(
        tf.logical_and(
            tf.equal(
                tf.expand_dims(annotation_labels, -1),
                tf.expand_dims(annotation_labels, 1)),
            tf.expand_dims(is_annotation_not_pad_or_unk, -1)),
        dtype=tf.int32)

    # [batch_size, max_num_annotations]
    should_annotation_be_masked = tf.einsum(
        'bi,bij->bj', is_annotation_masked, are_annotations_for_the_same_entity)

  should_annotation_be_masked = tf.minimum(should_annotation_be_masked, 1)

  should_token_be_masked = (
      tf.reduce_max((mask_end - mask_begin) *
                    tf.expand_dims(should_annotation_be_masked, -1), 1))

  # [batch_size, block_length]
  return (token_ids * (1 - should_token_be_masked) +
          mask_token_id * should_token_be_masked)


def make_is_span_maskable_features(num_blocks_per_example,
                                   block_length, max_num_annotations,
                                   annotation_begins,
                                   annotation_ends,
                                   annotation_labels):
  """Prepares is-token-belongs-to-an-annotation mask."""
  annotation_begins = tf.reshape(annotation_begins,
                                 [num_blocks_per_example, max_num_annotations])
  annotation_ends = tf.reshape(annotation_ends,
                               [num_blocks_per_example, max_num_annotations])
  annotation_labels = tf.reshape(annotation_labels,
                                 [num_blocks_per_example, max_num_annotations])
  annotation_mask = tf.expand_dims(
      tf.cast(tf.not_equal(annotation_labels, 0), tf.int32), -1)

  mask_begin = tf.sequence_mask(annotation_begins, block_length, dtype=tf.int32)
  mask_begin_plus_one = tf.sequence_mask(
      annotation_begins + 1, block_length, dtype=tf.int32)
  mask_end = tf.sequence_mask(annotation_ends + 1, block_length, dtype=tf.int32)

  def make_mask(x):
    x = x * annotation_mask
    x = tf.reduce_sum(x, 1)
    x = tf.minimum(x, 1)
    x = tf.reshape(x, [num_blocks_per_example * block_length])
    return x

  return (make_mask(mask_end - mask_begin),
          make_mask(mask_end - mask_begin_plus_one))


def dynamic_padding_1d(tensor, length,
                       padding_token_id):
  """Padds or truncates 1D tensor to a specified length."""
  length_to_pad = length - tf.shape(tensor)[0]
  paddings = tf.expand_dims(
      tf.concat(
          [tf.constant([0]),
           tf.expand_dims(tf.maximum(length_to_pad, 0), 0)],
          axis=0), 0)

  def pad():
    return tf.pad(
        tensor, paddings, 'CONSTANT', constant_values=padding_token_id)

  padded_tensor = tf.cond(
      length_to_pad > 0,
      true_fn=pad,
      false_fn=lambda: tensor[:length],
      strict=True)
  padded_tensor.set_shape(length)
  return padded_tensor


def get_num_examples_in_tf_records(paths):
  if isinstance(paths, str):
    paths = [paths]
  num_examples = 0
  for path in paths:
    num_examples += sum(1 for _ in tf.python_io.tf_record_iterator(path))
  return num_examples


def get_block_params_from_input_file(input_file):
  """Extract the `num_blocks_per_example` and `block_length` from the record."""
  first_record = next(tf.python_io.tf_record_iterator(input_file))
  first_example = tf.train.Example.FromString(first_record)
  num_blocks_per_example = len(
      first_example.features.feature['block_ids'].int64_list.value)
  max_seq_len = len(
      first_example.features.feature['token_ids'].int64_list.value)
  if max_seq_len % num_blocks_per_example != 0:
    raise ValueError('Record contain inconsistent input: '
                     'num_blocks_per_example={}, max_seq_len={}'.format(
                         num_blocks_per_example, max_seq_len))
  block_length = max_seq_len // num_blocks_per_example
  return num_blocks_per_example, block_length


def get_num_annotations_from_input_file(input_file):
  """Extract the `max_num_annotations` (per block) from the record."""
  first_record = next(tf.python_io.tf_record_iterator(input_file))
  first_example = tf.train.Example.FromString(first_record)

  num_annotations_per_example = None
  # For historical reasons, the data could have either
  # `annotation_*` features or `answer_annotation_*` features.
  # We allow both options for backward compatibility.
  if 'annotation_labels' in first_example.features.feature:
    num_annotations_per_example = len(
        first_example.features.feature['annotation_labels'].int64_list.value)
  if 'answer_annotation_labels' in first_example.features.feature:
    assert num_annotations_per_example is None
    num_annotations_per_example = len(
        first_example.features.feature['answer_annotation_labels'].int64_list
        .value)
  # Currently, we force the number of entity and answer annotations to
  # be the same. That could be changed in the future rather easily.
  if 'entity_annotation_labels' in first_example.features.feature:
    assert num_annotations_per_example is not None
    assert num_annotations_per_example == len(
        first_example.features.feature['entity_annotation_labels'].int64_list
        .value)
  num_blocks_per_example = len(
      first_example.features.feature['block_ids'].int64_list.value)
  if num_annotations_per_example % num_blocks_per_example != 0:
    raise ValueError(
        'Record contain inconsistent input: '
        'num_blocks_per_example={}, num_annotations_per_example={}'.format(
            num_blocks_per_example, num_annotations_per_example))
  return num_annotations_per_example // num_blocks_per_example


def get_span_prediction_example_decode_fn(
    num_blocks_per_example,
    block_length,
    max_num_answer_annotations = None,
    max_num_entity_annotations = None,
    extra_int_features_shapes = None):
  """Returns a decode function to parse a single example into Tensors."""
  max_seq_len = num_blocks_per_example * block_length
  name_to_features = {
      'token_ids': tf.FixedLenFeature([max_seq_len], tf.int64),
      'block_ids': tf.FixedLenFeature([num_blocks_per_example], tf.int64),
      'prefix_length': tf.FixedLenFeature([num_blocks_per_example], tf.int64),
  }
  if not extra_int_features_shapes:
    extra_int_features_shapes = dict()
  for feature_name, feature_shape in extra_int_features_shapes.items():
    total_length = 1
    for x in feature_shape:
      total_length *= x
    name_to_features[feature_name] = tf.FixedLenFeature([total_length],
                                                        tf.int64)

  if max_num_answer_annotations is not None:
    max_num_annotations_total = (
        num_blocks_per_example * max_num_answer_annotations)
    name_to_features.update({
        'answer_annotation_begins':
            tf.FixedLenFeature([max_num_annotations_total], tf.int64),
        'answer_annotation_ends':
            tf.FixedLenFeature([max_num_annotations_total], tf.int64),
        'answer_annotation_labels':
            tf.FixedLenFeature([max_num_annotations_total], tf.int64),
    })
  if max_num_entity_annotations is not None:
    max_num_annotations_total = (
        num_blocks_per_example * max_num_entity_annotations)
    name_to_features.update({
        'entity_annotation_begins':
            tf.FixedLenFeature([max_num_annotations_total], tf.int64),
        'entity_annotation_ends':
            tf.FixedLenFeature([max_num_annotations_total], tf.int64),
        'entity_annotation_labels':
            tf.FixedLenFeature([max_num_annotations_total], tf.int64),
    })

  reshape_features = {
      'token_ids': [num_blocks_per_example, block_length],
      'answer_annotation_begins': [
          num_blocks_per_example, max_num_answer_annotations
      ],
      'answer_annotation_ends': [
          num_blocks_per_example, max_num_answer_annotations
      ],
      'answer_annotation_labels': [
          num_blocks_per_example, max_num_answer_annotations
      ],
      'entity_annotation_begins': [
          num_blocks_per_example, max_num_entity_annotations
      ],
      'entity_annotation_ends': [
          num_blocks_per_example, max_num_entity_annotations
      ],
      'entity_annotation_labels': [
          num_blocks_per_example, max_num_entity_annotations
      ],
  }
  reshape_features.update(extra_int_features_shapes)

  def _decode_fn(record):
    """Decodes a serialized tf.train.Example to a dictionary of Tensors.

    Arguments:
      record: A scalar string Tensor containing a serialized tf.train.Example.

    Returns:
      A dictionary of the decoded (and derived) Tensors.
    """
    example = tf.io.parse_single_example(record, name_to_features)

    for name in example.keys():
      t = example[name]
      if name in reshape_features:
        t = tf.reshape(t, reshape_features[name])
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    example['block_pos'] = make_block_pos_features(example['block_ids'])
    return example

  return _decode_fn
