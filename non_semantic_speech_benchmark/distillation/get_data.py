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

from typing import Callable, Dict, Optional, List, Union

import numpy as np
import tensorflow as tf

SAMPLES_ = 'samples_'
TARGETS_ = 'targets_'
LABELS_ = 'labels_'
SPEAKER_IDS_ = 'speaker_ids_'

AUTO_ = tf.data.experimental.AUTOTUNE


def get_data(file_patterns,
             output_dimension,
             reader,
             samples_key,
             min_length,
             batch_size,
             loop_forever,
             shuffle,
             teacher_fn = None,
             target_key = None,
             label_key = None,
             speaker_id_key = None,
             shuffle_buffer_size = 10000,
             normalize_to_pm_one = False):
  """Gets data for TRILL distillation from a teacher or precomputed values.

  Args:
    file_patterns: Single or list of globs for input data.
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
    label_key: Optional name of label key in tf.Examples.
    speaker_id_key: Location of speaker ID.
    shuffle_buffer_size: Size of shuffle buffer.
    normalize_to_pm_one: Whether to normalize to plus/minus 1.
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

  if teacher_fn is None:
    assert target_key
    # Use precomputed targets. We trust the data generation process to create
    # inputs of the right size, so use fixed-length input for samples.
    features = {
        samples_key: tf.io.FixedLenFeature([min_length], tf.float32),
        target_key: tf.io.FixedLenFeature([output_dimension], tf.float32),
    }
    cur_batch_size = batch_size
    rename_dict = {SAMPLES_: samples_key, TARGETS_: target_key}
  else:
    assert target_key is None
    features = {
        samples_key: tf.io.VarLenFeature(dtype=tf.float32),
    }
    cur_batch_size = 1
    rename_dict = {SAMPLES_: samples_key}

  # Read the label if required.
  if label_key:
    features[label_key] = tf.io.FixedLenFeature(shape=(), dtype=tf.string)
    rename_dict[LABELS_] = label_key

  # Read the speaker ID if required.
  if speaker_id_key:
    if not label_key:
      raise ValueError('Must use label_key if using speaker_id_key')
    features[speaker_id_key] = tf.io.FixedLenFeature(shape=(), dtype=tf.string)
    rename_dict[SPEAKER_IDS_] = speaker_id_key

  def _rename_dict(kv):
    return {k: kv[v] for k, v in rename_dict.items()}

  # Load data into a dataset of batch size 1, then preprocess if necessary.
  ds = (
      tf.data.experimental.make_batched_features_dataset(
          file_pattern=files_list,
          batch_size=cur_batch_size,
          num_epochs=None if loop_forever else 1,
          reader_num_threads=AUTO_,
          parser_num_threads=AUTO_,
          features=features,
          reader=reader,
          shuffle=shuffle,
          shuffle_buffer_size=shuffle_buffer_size,
          prefetch_buffer_size=AUTO_,
          sloppy_ordering=True).map(_rename_dict, num_parallel_calls=AUTO_))

  if teacher_fn is not None:
    # Create target embeddings from `teacher_fn`.
    assert callable(teacher_fn)
    @tf.function
    def _audio_to_embeddings(samples):
      return _audio_to_embeddings_fn(samples, teacher_fn, output_dimension)

    ds = (
        ds.filter(lambda kv: _filter_fn(kv, min_length)).map(
            lambda kv: _crop_fn(kv, min_length),
            num_parallel_calls=AUTO_).batch(batch_size).map(
                _audio_to_embeddings, num_parallel_calls=AUTO_))
    if label_key:
      def _sqz_lbls(kv):
        lbls_wout_bd = tf.squeeze(kv[LABELS_], axis=1)
        # Copy.
        ret = {k: v for k, v in kv.items()}
        ret[LABELS_] = lbls_wout_bd
        return ret
      ds = ds.map(_sqz_lbls, num_parallel_calls=AUTO_)

  # Possibly normalize samples.
  def _normalize(kv):
    assert SAMPLES_ in kv
    unnormalized_samples = kv[SAMPLES_]

    # Normalize.
    normalized_samples = unnormalized_samples / np.iinfo(np.int16).max
    kv[SAMPLES_] = normalized_samples

    return kv

  if normalize_to_pm_one:
    ds = ds.map(_normalize)

  # Convert results to tuple.
  def _to_tup(kv):
    if speaker_id_key:
      assert label_key
      return (kv[SAMPLES_], kv[TARGETS_], kv[LABELS_], kv[SPEAKER_IDS_])
    elif label_key:
      return (kv[SAMPLES_], kv[TARGETS_], kv[LABELS_])
    else:
      return (kv[SAMPLES_], kv[TARGETS_])

  ds = ds.map(_to_tup, num_parallel_calls=AUTO_).prefetch(2)
  if speaker_id_key:
    expected_len = 4
  elif label_key:
    expected_len = 3
  else:
    expected_len = 2
  assert len(ds.element_spec) == expected_len, ds.element_spec
  ds.element_spec[0].shape.assert_is_compatible_with(
      [None, min_length])  # audio samples
  ds.element_spec[1].shape.assert_is_compatible_with(
      [None, output_dimension])  # teacher embeddings
  if label_key:
    ds.element_spec[2].shape.assert_is_compatible_with([None])  # labels
  if speaker_id_key:
    ds.element_spec[3].shape.assert_is_compatible_with([None])  # speaker_id

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

  # Deepcopy outputs.
  ret = {k: v for k, v in kv.items()}
  ret[SAMPLES_] = samples
  return ret


def _audio_to_embeddings_fn(kv,
                            teacher_fn,
                            output_dimension):
  """Map audio to teacher labels."""
  samples = kv[SAMPLES_]
  teacher_embeddings = teacher_fn(samples)
  teacher_embeddings.shape.assert_has_rank(2)
  teacher_embeddings.set_shape([None, output_dimension])

  # Deepcopy outputs.
  ret = {k: v for k, v in kv.items()}
  ret[TARGETS_] = teacher_embeddings
  return ret


def _lbls_to_onehot(kv,
                    label_list):
  """Map label to int."""
  lbls = kv[LABELS_]
  lbls.shape.assert_has_rank(1)

  # Let's do some y remapping trickery.
  y_in = tf.expand_dims(lbls, axis=1)
  y_out = tf.where(tf.math.equal(label_list, y_in))[:, 1]
  y_out = tf.one_hot(y_out, len(label_list))

  # Deepcopy outputs.
  ret = {k: v for k, v in kv.items()}
  ret[LABELS_] = y_out
  return ret


def savedmodel_to_func(
    saved_model,
    output_key,
    sample_rate = 16000):
  """Makes a savedmodel fn."""

  def _saved_model_fn(audio):
    out = saved_model(audio, sample_rate)[output_key]
    if out.shape.rank != 2:
      batch_dim = tf.shape(audio)[0]
      out = tf.reshape(out, [batch_dim, -1])
    return out
  return _saved_model_fn
