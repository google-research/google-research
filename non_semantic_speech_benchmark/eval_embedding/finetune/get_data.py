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

import functools
import tensorflow as tf


def get_data(
    file_pattern,
    reader,
    samples_key,
    min_length,
    label_key,
    label_list,
    batch_size,
    loop_forever,
    shuffle,
    shuffle_buffer_size=10000,
    label_type=tf.int64):
  """Gets the data for TRILL finetuning.

  This function is *always* stochastic.

  Args:
    file_pattern: Glob for input data.
    reader: Class used to parse data on disk.
    samples_key: Name of audio samples in tf.Examples.
    min_length: The minimum audio length. Should take sample rate into account.
      Examples smaller than this are dropped. Examples longer than this are
      randomly cropped to this size.
    label_key: Name of label key in tf.Examples.
    label_list: Python list of all possible label values.
    batch_size: Batch size of data in returned tf.data.Dataset.
    loop_forever: Python bool. Whether to loop forever.
    shuffle: Python bool. Whether to shuffle data.
    shuffle_buffer_size: Size of shuffle buffer.
    label_type: Type of label field. Usually `tf.string` or `tf.int64`.

  Returns:
    A tf.data.Dataset of (samples, onehot labels).
  """
  assert tf.io.gfile.glob(file_pattern), file_pattern

  # Audio samples are variable length.
  features = {
      samples_key: tf.io.VarLenFeature(dtype=tf.float32),
      label_key: tf.io.FixedLenFeature(
          shape=(), dtype=label_type, default_value=None),
  }

  # Load data into a dataset of batch size 1. Then preprocess.
  ds = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=1,
      num_epochs=None if loop_forever else 1,
      reader_num_threads=tf.data.experimental.AUTOTUNE,
      parser_num_threads=tf.data.experimental.AUTOTUNE,
      features=features,
      reader=reader,
      shuffle=shuffle,
      shuffle_buffer_size=shuffle_buffer_size,
      prefetch_buffer_size=tf.data.experimental.AUTOTUNE,
      sloppy_ordering=True)

  ds = tf_data_pipeline(
      ds, samples_key, label_key, label_list, min_length, batch_size)

  return ds


def tf_data_pipeline(ds, samples_key, label_key, label_list, min_length,
                     batch_size):
  """Create tf.data pipeline for reading data."""
  # Filter examples that are too short, crop, map labels to integers, and batch.
  @tf.function
  def _filter_fn(kv):
    d_shape = kv[samples_key].dense_shape
    one = tf.cast(1, d_shape.dtype)
    tf.debugging.assert_equal(d_shape[0], one)  # We expect batch size to be 1.
    return d_shape[1] > min_length
  def _crop(kv):
    samples, label = kv[samples_key], kv[label_key]
    samples = tf.sparse.to_dense(samples)
    samples.shape.assert_has_rank(2)
    samples = tf.squeeze(samples, axis=0)
    samples = tf.image.random_crop(samples, [min_length], seed=123, name='crop')
    samples.set_shape([min_length])
    return (samples, label)
  ds = (ds
        .filter(_filter_fn)
        .map(_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .map(functools.partial(_y_to_onehot, label_list=label_list),
             num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(batch_size)
        .prefetch(2))

  return ds


def _y_to_onehot(samples, labels, label_list):
  """Map label to int."""
  samples.shape.assert_has_rank(1)
  labels.shape.assert_has_rank(1)

  # Let's do some y remapping trickery.
  labels = tf.squeeze(labels, axis=0)
  if labels.dtype != tf.string:
    labels = tf.strings.as_string(labels)
  y_out = tf.where(tf.math.equal(label_list, labels))
  y_out.shape.assert_has_rank(2)
  y_out = tf.one_hot(y_out[0, 0], len(label_list))

  return samples, y_out
