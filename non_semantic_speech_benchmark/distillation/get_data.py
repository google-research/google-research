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
"""Get data."""

import tensorflow as tf
from non_semantic_speech_benchmark import file_utils


def get_data(
    file_pattern,
    teacher_fn,
    output_dimension,
    reader,
    samples_key,
    min_length,
    batch_size,
    loop_forever,
    shuffle,
    shuffle_buffer_size=10000):
  """Gets the data for TRILL distillation.

  This function is *always* stochastic.

  Args:
    file_pattern: Glob for input data.
    teacher_fn: A function that takes 1 argument and returns label embeddings.
    output_dimension: Feature dimension of teacher output.
    reader: Class used to parse data on disk.
    samples_key: Name of audio samples in tf.Examples
    min_length: The minimum audio length. Should take sample rate into account.
      Examples smaller than this are dropped. Examples longer than this are
      randomly cropped to this size.
    batch_size: Batch size of data in returned tf.data.Dataset.
    loop_forever: Python bool. Whether to loop forever.
    shuffle: Python bool. Whether to shuffle data.
    shuffle_buffer_size: Size of shuffle buffer

  Returns:
    A tf.data.Dataset of (audio samples, regression targets).
  """
  assert file_utils.Glob(file_pattern), file_pattern
  assert callable(teacher_fn)

  # Audio samples are variable length.
  features = {
      samples_key: tf.io.VarLenFeature(dtype=tf.float32),
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

  ds = tf_data_pipeline(ds, teacher_fn, samples_key, min_length, batch_size,
                        output_dimension)

  return ds


def get_precomputed_data(file_pattern, output_dimension, frontend_key,
                         target_key, batch_size, num_epochs,
                         shuffle_buffer_size):
  """Gets precomputed frontend features and targets for TRILL distillation.

  This function is *always* stochastic.
  Args:
    file_pattern: Glob for input data.
    output_dimension: Feature dimension of teacher output.
    frontend_key: Name of frontend features in tf.Examples.
    target_key: Name of targets (i.e. teacher model outputs) in tf.Examples.
    batch_size: Batch size of data in returned tf.data.Dataset.
    num_epochs: Number of training epochs.
    shuffle_buffer_size: Size of dataset shuffle buffer.

  Returns:
    A tf.data.Dataset of (frontend features, regression targets).
  """

  def _parse_examples(record):
    feature_description = {
        frontend_key: tf.io.FixedLenFeature([int(96 * 64)], tf.float32),
        target_key: tf.io.FixedLenFeature([output_dimension], tf.float32),
    }
    example = tf.io.parse_example(record, feature_description)
    return tf.reshape(example['audio'], (96, 64)), example['label']

  options = tf.data.Options()
  options.experimental_deterministic = False
  files_ds = tf.data.Dataset.list_files(file_pattern).with_options(options)
  return tf.data.TFRecordDataset(files_ds)\
    .map(_parse_examples)\
    .batch(batch_size, drop_remainder=True)\
    .shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)\
    .repeat(num_epochs)\
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


def tf_data_pipeline(ds, teacher_fn, samples_key, min_length, batch_size,
                     output_dimension):
  """Create tf.data pipeline for reading data."""
  # Filter examples that are too short, crop, map labels to integers, and batch.
  @tf.function
  def _filter_fn(kv):
    d_shape = kv[samples_key].dense_shape
    one = tf.cast(1, d_shape.dtype)
    tf.debugging.assert_equal(d_shape[0], one)  # We expect batch size to be 1.
    return d_shape[1] > min_length
  def _crop(kv):
    samples = kv[samples_key]
    samples = tf.sparse.to_dense(samples)
    samples.shape.assert_has_rank(2)
    samples = tf.squeeze(samples, axis=0)
    samples = tf.image.random_crop(samples, [min_length], seed=123, name='crop')
    samples.set_shape([min_length])
    return samples
  def _audio_to_embeddings(samples):
    teacher_embeddings = teacher_fn(samples)
    teacher_embeddings.shape.assert_has_rank(2)
    teacher_embeddings.set_shape([None, output_dimension])
    return (samples, teacher_embeddings)
  autotune_ = tf.data.experimental.AUTOTUNE
  ds = (ds
        .filter(_filter_fn)
        .map(_crop, num_parallel_calls=autotune_)
        .batch(batch_size)
        .map(_audio_to_embeddings, num_parallel_calls=autotune_)
        .prefetch(2))

  return ds


def savedmodel_to_func(saved_model, output_key, sample_rate=16000):
  def _saved_model_fn(audio):
    out = saved_model(audio, sample_rate)[output_key]
    if out.shape.rank != 2:
      batch_dim = tf.shape(audio)[0]
      out = tf.reshape(out, [batch_dim, -1])
    return out
  return _saved_model_fn
