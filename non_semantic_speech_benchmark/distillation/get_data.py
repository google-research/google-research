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


SAMPLES_ = 'samples_'
TARGETS_ = 'targets_'


def get_data(file_pattern,
             output_dimension,
             reader,
             samples_key,
             min_length,
             batch_size,
             loop_forever,
             shuffle,
             teacher_fn=None,
             target_key=None,
             shuffle_buffer_size=10000):
  """Gets data for TRILL distillation from a teacher or precomputed values.

  Args:
    file_pattern: Glob for input data.
    output_dimension: Feature dimension of teacher output.
    reader: Class used to parse data on disk.
    samples_key: Name of audio samples in tf.Examples.
    min_length: The minimum audio length. Should take sample rate into account.
      Examples smaller than this are dropped. Examples longer than this are
      randomly cropped to this size. If we are using precomputed targets,
      drop examples that aren't equal to this.
    batch_size: Batch size of data in returned tf.data.Dataset.
    loop_forever: Python bool. Whether to loop forever.
    shuffle: Python bool. Whether to shuffle data.
    teacher_fn: Optional. A function that takes 1 argument and returns label
      embeddings. If `None`, get precomputed data from disk. If present, run
      teacher function as part of the input pipeline. If `teacher_fn` is `None`,
      `target_key` must be not None.
    target_key: Required if reading precomputed features. Location of the target
      embeddings.
    shuffle_buffer_size: Size of shuffle buffer.
  Returns:
    A tf.data.Dataset of (audio samples, regression targets).
  """
  assert file_utils.Glob(file_pattern), file_pattern

  if teacher_fn is None:
    assert target_key
    # Use precomputed targets. We trust the data generation process to create
    # inputs of the right size, so use fixed-length input for samples.
    features = {
        samples_key: tf.io.FixedLenFeature([min_length], tf.float32),
        target_key: tf.io.FixedLenFeature([output_dimension], tf.float32),
    }
    cur_batch_size = batch_size
    def _rename_dict(kv):
      return {SAMPLES_: kv[samples_key], TARGETS_: kv[target_key]}
  else:
    assert target_key is None
    features = {
        samples_key: tf.io.VarLenFeature(dtype=tf.float32),
    }
    cur_batch_size = 1
    def _rename_dict(kv):
      return {SAMPLES_: kv[samples_key]}

  # Load data into a dataset of batch size 1, then preprocess if necessary.
  ds = (
      tf.data.experimental.make_batched_features_dataset(
          file_pattern=file_pattern,
          batch_size=cur_batch_size,
          num_epochs=None if loop_forever else 1,
          reader_num_threads=tf.data.experimental.AUTOTUNE,
          parser_num_threads=tf.data.experimental.AUTOTUNE,
          features=features,
          reader=reader,
          shuffle=shuffle,
          shuffle_buffer_size=shuffle_buffer_size,
          prefetch_buffer_size=tf.data.experimental.AUTOTUNE,
          sloppy_ordering=True)
      .map(_rename_dict, num_parallel_calls=tf.data.experimental.AUTOTUNE))

  if teacher_fn is not None:
    # Create target embeddings from `teacher_fn`.
    assert callable(teacher_fn)
    @tf.function
    def _audio_to_embeddings(samples):
      return _audio_to_embeddings_fn(samples, teacher_fn, output_dimension)
    ds = (ds
          .filter(lambda kv: _filter_fn(kv, min_length))
          .map(lambda kv: _crop_fn(kv, min_length),
               num_parallel_calls=tf.data.experimental.AUTOTUNE)
          .batch(batch_size)
          .map(_audio_to_embeddings,
               num_parallel_calls=tf.data.experimental.AUTOTUNE))

  # Convert results to tuple.
  ds = (ds
        .map(lambda kv: (kv[SAMPLES_], kv[TARGETS_]),
             num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .prefetch(2))

  assert len(ds.element_spec) == 2, ds.element_spec
  ds.element_spec[0].shape.assert_has_rank(2)  # audio samples
  ds.element_spec[1].shape.assert_has_rank(2)  # teacher embeddings
  assert tuple(ds.element_spec[0].shape) == (None, min_length)
  assert tuple(ds.element_spec[1].shape) == (None, output_dimension)

  return ds


def _filter_fn(kv, min_length):
  """Filter examples that are too short."""
  d_shape = kv[SAMPLES_].dense_shape
  one = tf.cast(1, d_shape.dtype)
  tf.debugging.assert_equal(d_shape[0], one)  # We expect batch size to be 1.
  return d_shape[1] > min_length


def _crop_fn(kv, min_length):
  """Crop audio to expected length."""
  samples = kv[SAMPLES_]
  samples = tf.sparse.to_dense(samples)
  samples.shape.assert_has_rank(2)
  samples = tf.squeeze(samples, axis=0)
  samples = tf.image.random_crop(samples, [min_length], seed=123, name='crop')
  samples.set_shape([min_length])
  return {SAMPLES_: samples}


def _audio_to_embeddings_fn(kv, teacher_fn, output_dimension):
  """Map audio to teacher labels."""
  samples = kv[SAMPLES_]
  teacher_embeddings = teacher_fn(samples)
  teacher_embeddings.shape.assert_has_rank(2)
  teacher_embeddings.set_shape([None, output_dimension])
  return {SAMPLES_: samples, TARGETS_: teacher_embeddings}


def savedmodel_to_func(saved_model, output_key, sample_rate=16000):
  def _saved_model_fn(audio):
    out = saved_model(audio, sample_rate)[output_key]
    if out.shape.rank != 2:
      batch_dim = tf.shape(audio)[0]
      out = tf.reshape(out, [batch_dim, -1])
    return out
  return _saved_model_fn
