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

"""Data pipeline."""

import ml_collections
import tensorflow as tf


def get_datasets(
    config,
    data_config,
    batch_size,
    repeat = False,
):
  """Construct tf datasets given configs.

  Args:
    config: top level config.
    data_config: data specific config.
    batch_size: batch size to use for training.
    repeat: whether to repeat the dataset indefinitely.

  Returns:
    train_ds, test_ds: dataset objects.
  """
  if config.tuning_mode:
    # We do hparam tuning on 5% of training set. Assumes 20 shards.
    train_example_paths = tf.io.gfile.glob(data_config.train_example_path)
    train_example_paths = train_example_paths[1:]
    test_example_paths = train_example_paths[:1]
  else:
    train_example_paths = tf.io.gfile.glob(data_config.train_example_path)
    test_example_paths = tf.io.gfile.glob(data_config.test_example_path)

  def decode_fn(record_bytes):
    return tf.io.parse_single_example(
        # Data
        record_bytes,

        # Schema
        {
            'repr': tf.io.FixedLenFeature(
                [data_config.hidden_dims], dtype=tf.float32
            ),
            'label': tf.io.FixedLenFeature([], dtype=tf.int64),
        },
    )

  test_ds = tf.data.TFRecordDataset(test_example_paths)
  test_ds = test_ds.map(decode_fn)
  test_ds = test_ds.batch(batch_size)
  test_ds = test_ds.prefetch(10)

  train_ds = tf.data.TFRecordDataset(train_example_paths)
  train_ds = train_ds.map(decode_fn)
  if repeat:
    train_ds = train_ds.repeat()
  train_ds = train_ds.batch(batch_size)
  train_ds = train_ds.prefetch(10)
  return train_ds, test_ds
