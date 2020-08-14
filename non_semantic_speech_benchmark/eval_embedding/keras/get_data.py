# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

import functools
import tensorflow.compat.v2 as tf
from non_semantic_speech_benchmark import file_utils


def get_data(
    file_pattern,
    reader,
    embedding_name,
    embedding_dim,
    label_name,
    label_list,
    bucket_boundaries,
    bucket_batch_sizes,
    loop_forever,
    shuffle,
    shuffle_buffer_size=10000):
  """Gets the data for keras training.

  Args:
    file_pattern: Glob for input data.
    reader: Class used to parse data on disk.
    embedding_name: Name of embedding in tf.Examples.
    embedding_dim: Fixed size of embedding.
    label_name: Name of label key in tf.Examples.
    label_list: Python list of all possible label values.
    bucket_boundaries: Boundaries for bucketing.
    bucket_batch_sizes: Batch size for bucket size.
    loop_forever: Python bool. Whether to loop forever.
    shuffle: Python bool. Whether to shuffle data.
    shuffle_buffer_size: Size of shuffle buffer.

  Returns:
    A tf.data.Dataset of (embeddings, onehot labels).
  """
  assert file_utils.Glob(file_pattern), file_pattern
  assert isinstance(bucket_boundaries, (tuple, list))
  if isinstance(bucket_boundaries[0], str):
    bucket_boundaries = [int(x) for x in bucket_boundaries]

  emb_key = f'embedding/{embedding_name}'
  label_key = label_name

  features = {
      emb_key: tf.io.VarLenFeature(dtype=tf.float32),
      label_key: tf.io.FixedLenFeature(
          shape=(), dtype=tf.string, default_value=None),
  }

  # Load data into a dataset.
  ds = (tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=1,
      num_epochs=None if loop_forever else 1,
      reader_num_threads=tf.data.experimental.AUTOTUNE,
      parser_num_threads=2,
      features=features,
      reader=reader,
      shuffle=shuffle,
      shuffle_buffer_size=shuffle_buffer_size,
      prefetch_buffer_size=tf.data.experimental.AUTOTUNE,
      sloppy_ordering=True)
        .map(lambda kv: (kv[emb_key], kv[label_key]),
             num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .map(functools.partial(_reshape_full, embedding_dim=embedding_dim),
             num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .map(functools.partial(_y_to_onehot, label_list=label_list),
             num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .map(_remove_batchdim_one,
             num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .apply(
            tf.data.experimental.bucket_by_sequence_length(
                element_length_func=lambda emb, lbl: tf.shape(emb)[0],
                bucket_boundaries=bucket_boundaries,
                bucket_batch_sizes=bucket_batch_sizes,
                padded_shapes=None,
                padding_values=None,
                pad_to_bucket_boundary=False,
                no_padding=False,
                drop_remainder=False)
            )
        )

  return ds


def _reshape_full(embeddings, labels, embedding_dim):
  """Reshape to 2D."""
  embeddings.shape.assert_has_rank(2)
  emb = tf.sparse.to_dense(embeddings)
  emb = tf.reshape(emb, [tf.shape(emb)[0], -1, embedding_dim])
  emb.shape.assert_has_rank(3)
  labels.shape.assert_has_rank(1)

  return emb, labels


def _y_to_onehot(embeddings, labels, label_list):
  """Map label to int."""
  labels.shape.assert_has_rank(1)

  # Let's do some y remapping trickery.
  y_in = tf.expand_dims(labels, axis=1)
  y_out = tf.where(tf.math.equal(label_list, y_in))[:, 1]
  y_out = tf.one_hot(y_out, len(label_list))

  return embeddings, y_out


def _remove_batchdim_one(emb, lbl):
  return (tf.squeeze(emb, axis=0), tf.squeeze(lbl, axis=0))
