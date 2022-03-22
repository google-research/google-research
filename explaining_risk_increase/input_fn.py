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

"""Input function to observation sequence model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
from tensorflow.contrib import data as contrib_data

CONTEXT_KEY_PREFIX = 'c-'
SEQUENCE_KEY_PREFIX = 's-'


def _example_index_to_sparse_index(example_indices, batch_size):
  """Creates a 2D sparse index tensor from a list of 1D example indices.

  For example, this would do the transformation:
  [0, 0, 0, 1, 3, 3] -> [[0,0], [0,1], [0,2], [1,0], [3,0], [3,1]]

  The second column of the output tensor is the running count of the occurrences
  of that example index.

  Args:
    example_indices: A sorted 1D Tensor with example indices.
    batch_size: The batch_size. Could be larger than max(example_indices) if the
      last examples of the batch do not have the feature present.

  Returns:
    The sparse index tensor.
    The maxmium length of a row in this tensor.
  """
  binned_counts = tf.bincount(example_indices, minlength=batch_size)
  max_len = tf.to_int64(tf.reduce_max(binned_counts))
  return tf.where(tf.sequence_mask(binned_counts)), max_len


def _remove_empty_timesteps(sp_tensor):
  """Creates a 3D SparseTensor skipping empty time steps.

  Args:
    sp_tensor: A SparseTensor with at least 2 dimensions (subsequent ones will
      be ignored and simply flattened into the 2nd dimension).

  Returns:
    A 3D SparseTensor with index 0 for dimension 3 and a series from [0,..k]
    for dimension 1 for each batch entry.
  """

  batch_size = tf.to_int32(sp_tensor.dense_shape[0])
  indices, max_len = _example_index_to_sparse_index(
      tf.to_int32(sp_tensor.indices[:, 0]), batch_size)
  indices = tf.concat([indices, tf.zeros_like(indices[:, 0:1])], axis=1)
  return tf.SparseTensor(
      indices=indices,
      values=sp_tensor.values,
      dense_shape=[batch_size, max_len, 1])


def _extend_with_dummy(extend_with, to_extend, dummy_value='n/a'):
  """Extends one SparseTensor with dummy_values at positions of other."""
  dense_shape = tf.to_int64(
      tf.concat([[tf.shape(extend_with)[0]],
                 [tf.maximum(tf.shape(extend_with)[1],
                             tf.shape(to_extend)[1])],
                 [tf.maximum(tf.shape(extend_with)[2],
                             tf.shape(to_extend)[2])]],
                axis=0))
  additional_indices = tf.sets.set_difference(
      tf.SparseTensor(
          indices=extend_with.indices,
          values=tf.zeros_like(extend_with.values, dtype=tf.int32),
          dense_shape=dense_shape),
      tf.SparseTensor(
          indices=to_extend.indices,
          values=tf.zeros([tf.shape(to_extend.indices)[0]], dtype=tf.int32),
          dense_shape=dense_shape)).indices
  # Supply defaults for all other indices.
  default = tf.tile(
      tf.constant([dummy_value]), multiples=[tf.shape(additional_indices)[0]])

  string_value = (
      tf.as_string(to_extend.values)
      if to_extend.values.dtype != tf.string else to_extend.values)
  return tf.sparse_reorder(
      tf.SparseTensor(
          indices=tf.concat([to_extend.indices, additional_indices], axis=0),
          values=tf.concat([string_value, default], axis=0),
          dense_shape=dense_shape))


def _sparse_intersect_indices(sp_tensor, required_sp_tensor):
  """Filters timestamps in sp_tensor to those present in required_sp_tensor."""
  # We extend both sp_tensor and required_sp_tensor with each others indices
  # so that they have the same indices.
  # E.g. their dense representation of one batch entry could be:
  # [dummy, dummy, 1 ]
  dummy_value = 'n/a'
  dummy_required_sp_tensor = _extend_with_dummy(
      sp_tensor, required_sp_tensor, dummy_value)
  dummy_sp_tensor = _extend_with_dummy(required_sp_tensor, sp_tensor,
                                       dummy_value)
  # We get rid to dummy values both for indices in the required_sp_tensor and
  # the sp_tensor.
  # First get rid of indices with dummy values in dummy_required_sp_tensor.
  in_required = tf.sparse_retain(
      dummy_sp_tensor,
      tf.logical_not(tf.equal(dummy_required_sp_tensor.values, dummy_value)))
  # Remove empty timesteps so that the timesteps align with the original
  # required_sp_tensor.
  # Then remove the indices with dummy values.
  in_required = tf.sparse_retain(
      _remove_empty_timesteps(in_required),
      tf.logical_not(tf.equal(in_required.values, dummy_value)))
  if sp_tensor.values.dtype != tf.string:
    in_required = tf.SparseTensor(
        indices=in_required.indices, dense_shape=in_required.dense_shape,
        values=tf.strings.to_number(
            in_required.values, out_type=sp_tensor.values.dtype))
  return in_required


def _dense_intersect_indices(tensor, required_sp_tensor):
  required_2d_indices = required_sp_tensor.indices[:, 0:2]
  values = tf.gather_nd(tensor, required_2d_indices)
  indices, max_len = _example_index_to_sparse_index(
      tf.to_int32(required_sp_tensor.indices[:, 0]),
      tf.to_int32(required_sp_tensor.dense_shape[0]))
  return tf.expand_dims(
      tf.sparse_to_dense(
          indices, tf.stack([required_sp_tensor.dense_shape[0], max_len]),
          values),
      axis=2)


def _intersect_indices(sequence_feature, required_sp_tensor):
  if isinstance(sequence_feature, tf.SparseTensor):
    return _sparse_intersect_indices(sequence_feature, required_sp_tensor)
  else:
    return _dense_intersect_indices(sequence_feature, required_sp_tensor)


def _make_parsing_fn(mode, label_name, sequence_features,
                     dense_sequence_feature):
  """Creates an input function to an estimator.

  Args:
    mode: The execution mode, as defined in tf.estimator.ModeKeys.
    label_name: Name of the label present as context feature in the
      SequenceExamples.
    sequence_features: List of sequence features (strings) that are valid keys
      in the tf.SequenceExample.
    dense_sequence_feature: Name of the float sequence feature.

  Returns:
    Two dictionaries with the parsing config for the context features and
    sequence features.
  """
  sequence_features_config = dict()
  for feature in sequence_features:
    dtype = tf.string
    if feature == dense_sequence_feature:
      dtype = tf.float32
    sequence_features_config[feature] = tf.io.VarLenFeature(dtype)

  sequence_features_config['eventId'] = tf.io.FixedLenSequenceFeature(
      [], tf.int64, allow_missing=False)

  context_features_config = dict()
  context_features_config['timestamp'] = tf.io.FixedLenFeature(
      [], tf.int64, default_value=-1)
  context_features_config['sequenceLength'] = tf.io.FixedLenFeature(
      [], tf.int64, default_value=-1)

  if mode != tf_estimator.ModeKeys.PREDICT:
    context_features_config[label_name] = tf.io.VarLenFeature(tf.string)

  def _parse_fn(serialized_examples):
    """Parses tf.(Sparse)Tensors from the serialized tf.SequenceExamples.

    Requires TF versions >= 1.12 but is faster than _parse_fn_old.
    Args:
      serialized_examples: A batch of serialized tf.SequenceExamples.

    Returns:
      A dictionary from name to (Sparse)Tensors of the context and sequence
      features.
    """
    context, sequence, _ = tf.io.parse_sequence_example(
        serialized_examples,
        context_features=context_features_config,
        sequence_features=sequence_features_config,
        name='parse_sequence_example')
    feature_map = dict()
    for k, v in context.items():
      feature_map[CONTEXT_KEY_PREFIX + k] = v
    for k, v in sequence.items():
      feature_map[SEQUENCE_KEY_PREFIX + k] = v
    return feature_map

  return _parse_fn


def _make_feature_engineering_fn(required_sp_tensor_name, label_name):
  """Creates an input function to an estimator.

  Args:
    required_sp_tensor_name: Name of the SparseTensor that is required. Other
      sequence features will be reduced to times at which this SparseTensor is
      also present.
    label_name: Name of label.

  Returns:
    Two dictionaries with the parsing config for the context features and
    sequence features.
  """

  def _process(examples):
    """Supplies input to our model.

    This function supplies input to our model after parsing.
    Args:
      examples: The dictionary from key to (Sparse)Tensors with context and
        sequence features.

    Returns:
      A tuple consisting of 1) a dictionary of tensors whose keys are
      the feature names, and 2) a tensor of target labels if the mode
      is not INFER (and None, otherwise).
    """
    # Combine into a single dictionary.
    feature_map = {}
    # Flatten sparse tensor to compute event age. This dense tensor also
    # contains padded values. These will not be used when gathering elements
    # from the dense tensor since each sparse feature won't have a value
    # defined for the padding.
    padded_event_age = (
        # Broadcast current time along sequence dimension.
        tf.expand_dims(examples.pop(CONTEXT_KEY_PREFIX + 'timestamp'), 1)
        # Subtract time of events.
        - examples.pop(SEQUENCE_KEY_PREFIX + 'eventId'))
    examples[SEQUENCE_KEY_PREFIX + 'deltaTime'] = padded_event_age

    if CONTEXT_KEY_PREFIX + label_name in examples:
      label = examples.pop(CONTEXT_KEY_PREFIX + label_name)
      label = tf.sparse.to_dense(tf.SparseTensor(
          indices=label.indices, dense_shape=[label.dense_shape[0], 1],
          values=tf.ones_like(label.values, dtype=tf.float32)))
      feature_map[CONTEXT_KEY_PREFIX + label_name] = label

    for k, v in examples.items():
      if k.startswith(CONTEXT_KEY_PREFIX):
        feature_map[k] = v
      else:
        feature_map[k] = _intersect_indices(
            v, examples[SEQUENCE_KEY_PREFIX + required_sp_tensor_name])
    sequence_length = tf.reduce_sum(
        _intersect_indices(
            tf.ones_like(examples[SEQUENCE_KEY_PREFIX + 'deltaTime']),
            examples[SEQUENCE_KEY_PREFIX + required_sp_tensor_name]),
        axis=1)
    feature_map[CONTEXT_KEY_PREFIX + 'sequenceLength'] = sequence_length
    return feature_map

  return _process


def get_input_fn(mode,
                 input_files,
                 label_name,
                 sequence_features,
                 dense_sequence_feature,
                 required_sequence_feature,
                 batch_size,
                 shuffle=True):
  """Creates an input function to an estimator.

  Args:
    mode: The execution mode, as defined in tf.estimator.ModeKeys.
    input_files: List of input files in TFRecord format containing
      tf.SequenceExamples.
    label_name: Name of the label present as context feature in the
      SequenceExamples.
    sequence_features: List of sequence features (strings) that are valid keys
      in the tf.SequenceExample.
    dense_sequence_feature: Name of float sequence feature.
    required_sequence_feature: Name of SparseTensor sequence feature that
      determines which events will be kept.
    batch_size: The size of the batch when reading in data.
    shuffle: Whether to shuffle the examples.

  Returns:
    A function that returns a dictionary of features and the target labels.
  """

  def input_fn():
    """Supplies input to our model.

    This function supplies input to our model, where this input is a
    function of the mode. For example, we supply different data if
    we're performing training versus evaluation.
    Returns:
      A tuple consisting of 1) a dictionary of tensors whose keys are
      the feature names, and 2) a tensor of target labels if the mode
      is not INFER (and None, otherwise).
    """
    is_training = mode == tf_estimator.ModeKeys.TRAIN
    num_epochs = None if is_training else 1

    with tf.name_scope('read_batch'):
      file_names = input_files
      files = tf.data.Dataset.list_files(file_names)
      if shuffle:
        files = files.shuffle(buffer_size=len(file_names))
      dataset = (
          files.apply(
              contrib_data.parallel_interleave(
                  tf.data.TFRecordDataset, cycle_length=10)).repeat(num_epochs))
      if shuffle:
        dataset = dataset.shuffle(buffer_size=100)
      parse_fn = _make_parsing_fn(mode, label_name, sequence_features,
                                  dense_sequence_feature)
      feature_engineering_fn = _make_feature_engineering_fn(
          required_sequence_feature, label_name)
      feature_map = (
          dataset.batch(batch_size)
          # Parallelize the input processing and put it behind a
          # queue to increase performance by removing it from the
          # critical path of per-step-computation.
          .map(parse_fn, num_parallel_calls=8)
          .map(feature_engineering_fn, num_parallel_calls=8)
          .prefetch(buffer_size=1).make_one_shot_iterator().get_next())
      label = None
      if mode != tf_estimator.ModeKeys.PREDICT:
        label = feature_map.pop(CONTEXT_KEY_PREFIX + label_name)
      return feature_map, {label_name: label}

  return input_fn
