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

"""Get data."""
# TODO(joelshor): The current cropping scheme biases data towards the beginning
# of samples. Consider random cropping the longer utterances, instead of taking
# the first N entries.

from typing import Callable, List, Optional, Union
from absl import logging
import numpy as np
import tensorflow as tf


AUTO_ = tf.data.experimental.AUTOTUNE


def get_data(file_patterns,
             reader,
             samples_key,
             target_key,
             batch_size,
             loop_forever,
             shuffle,
             max_samples_length = None,
             shuffle_buffer_size = 10000,
             output_dimension = 1024,
             label_key = None,
             speaker_id_key = None,
             min_samples_length = 4000,
             samples_are_float = True):
  """Gets data for TRILL distillation from a teacher or precomputed values.

  NOTE: Float audio *must* already be normalized to [-1, 1].

  Args:
    file_patterns: Single or list of globs for input data.
    reader: Class used to parse data on disk.
    samples_key: Name of audio samples in tf.Examples.
    target_key: Location of the target embeddings.
    batch_size: Batch size of data in returned tf.data.Dataset.
    loop_forever: Python bool. Whether to loop forever.
    shuffle: Python bool. Whether to shuffle data.
    max_samples_length: For memory reasons, we optionally cap the duration of
      samples. Samples longer than this will be randomly cropped to some
      subslice.
    shuffle_buffer_size: Size of shuffle buffer.
    output_dimension: Feature dimension of teacher output.
    label_key: Optional key to pass through for label. Useful for downstream
      eval.
    speaker_id_key: Optional key to pass through for label. Useful for
      downstream eval.
    min_samples_length: Minimum required length of samples, for a 16kHz
      sample. Samples less than this will be dropped.
    samples_are_float: If False, read samples as int64 and normalize to [-1, 1]
      floats.
  Returns:
    A tf.data.Dataset of (audio samples, regression targets).
  """
  if isinstance(file_patterns, str):
    file_patterns = [file_patterns]
  files_list = []
  for file_pattern in file_patterns:
    file_list = tf.io.gfile.glob(file_pattern)
    if not file_list:
      raise ValueError(f'Files not found: {file_pattern}')
    files_list.extend(file_list)

  # Use precomputed targets. We trust the data generation process to create
  # inputs of the right size, so use fixed-length input for samples.
  features = {
      samples_key:
          tf.io.VarLenFeature(tf.float32 if samples_are_float else tf.int64),
      target_key:
          tf.io.FixedLenFeature([output_dimension], tf.float32),
  }
  # If required, pass through the label and speaker ID..
  if label_key:
    features[label_key] = tf.io.FixedLenFeature(shape=(), dtype=tf.string)
  if speaker_id_key:
    if not label_key:
      raise ValueError('Must use label_key if using speaker_id_key')
    features[speaker_id_key] = tf.io.FixedLenFeature(shape=(), dtype=tf.string)

  # Load data into a batch that truncates to the shortest element, the process.
  ds = tf.data.experimental.make_batched_features_dataset(
      file_pattern=files_list,
      batch_size=batch_size,
      num_epochs=None if loop_forever else 1,
      reader_num_threads=AUTO_,
      parser_num_threads=AUTO_,
      features=features,
      reader=reader,
      shuffle=shuffle,
      shuffle_buffer_size=shuffle_buffer_size,
      prefetch_buffer_size=AUTO_,
      sloppy_ordering=True)

  def _int_to_float(kv):
    assert kv[samples_key].dtype != tf.float32, kv[samples_key].dtype
    kv[samples_key] = (
        tf.cast(kv[samples_key], tf.float32) / np.iinfo(np.int16).max)
    return kv

  if not samples_are_float:
    ds = ds.map(_int_to_float, num_parallel_calls=AUTO_)

  def _make_tuple(kv):
    # TODO(joelshor): Consider using ragged Tensors here.
    out = [kv[samples_key], kv[target_key]]
    if label_key:
      out += [kv[label_key]]
    if speaker_id_key:
      out += [kv[speaker_id_key]]
    return tuple(out)

  ds = (
      ds.map(_make_tuple, num_parallel_calls=AUTO_).map(
          _sparse_to_cropped_dense, num_parallel_calls=AUTO_)
      # Drop batches that are way to short.
      .filter(lambda *args: tf.shape(args[0])[1] > min_samples_length))

  def _crop(*args):
    samples = args[0]
    samples.shape.assert_has_rank(2)
    # If audio is shorter, this is a noop.
    if tf.shape(samples)[1] <= max_samples_length:
      logging.info('[get_data.crop] No cropping needed.')
      return args
    logging.info('[get_data.crop] Cropping to %i...', max_samples_length)
    samples = tf.expand_dims(tf.expand_dims(samples, axis=-1), axis=-1)
    # Pretend input is batches images of shape (..., height, width, channels).
    cropped_samples = tf.keras.layers.RandomCrop(
        height=max_samples_length, width=1)(
            samples)
    cropped_samples = tf.squeeze(cropped_samples, axis=[2, 3])
    return (cropped_samples,) + args[1:]

  if max_samples_length:
    ds = ds.map(_crop, num_parallel_calls=AUTO_)

  if speaker_id_key:
    expected_elems = 4
  elif label_key:
    expected_elems = 3
  else:
    expected_elems = 2
  assert len(ds.element_spec) == expected_elems, ds.element_spec
  ds.element_spec[0].shape.assert_is_compatible_with([None,
                                                      None])  # audio samples
  ds.element_spec[1].shape.assert_is_compatible_with(
      [None, output_dimension])  # teacher embeddings

  return ds


def _sparse_to_cropped_dense(*args):
  """Converts the sparse samples to a dense, with no empty zeros."""
  sparse_samples = args[0]

  # Convert sparse tensor to dense, with implausible values for empty.
  dv = 99
  dense_samples = tf.sparse.to_dense(sparse_samples, default_value=dv)
  # Assuming valid entries are to the left of some index for each row, find
  # the pivot per-row and truncate there.
  endpoint_per_row = tf.reduce_sum(
      tf.cast(dense_samples < dv, tf.int32), axis=1)
  min_endpoint = tf.reduce_min(endpoint_per_row)
  # Slice to the minimum.
  # TODO(joelshor): This biases data towards the beginning. Consider random
  # cropping the longer utterances, instead of taking the first N entries.
  dense = dense_samples[:, :min_endpoint]

  # Check that there's no padding.
  tf.debugging.assert_equal(tf.reduce_sum(tf.cast(dense == dv, tf.int32)), 0)

  # Pass through the other information.
  return (dense,) + args[1:]
