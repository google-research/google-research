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

"""Set up datasets for feature selection experiments."""

import numpy as np
from sequential_attention.experiments.datasets import data_loader
import tensorflow as tf


def get_dataset(data_name, val_ratio, batch_size):
  """Get datasets split into training, validation, and test datasets."""
  # Load data.
  if data_name == "mice":
    (x_train, x_test, y_train, y_test, is_classification, num_classes) = (
        data_loader.load_mice()
    )

  elif data_name == "isolet":
    (x_train, x_test, y_train, y_test, is_classification, num_classes) = (
        data_loader.load_isolet()
    )

  elif data_name == "activity":
    (x_train, x_test, y_train, y_test, is_classification, num_classes) = (
        data_loader.load_activity()
    )

  elif data_name == "coil":
    (x_train, x_test, y_train, y_test, is_classification, num_classes) = (
        data_loader.load_coil()
    )

  elif data_name == "fashion":
    (x_train, x_test, y_train, y_test, is_classification, num_classes) = (
        data_loader.load_fashion()
    )

  elif data_name == "mnist":
    (x_train, x_test, y_train, y_test, is_classification, num_classes) = (
        data_loader.load_mnist()
    )

  else:
    raise NotImplementedError

  # Tensorflow data transform functions
  if is_classification:

    def transform(x, y):
      x = tf.cast(x, dtype=tf.float32)
      return x, tf.one_hot(y, num_classes)

  else:

    def transform(x, y):
      x = tf.cast(x, dtype=tf.float32)
      y = tf.cast(y, dtype=tf.float32)
      return x, y

  # Shuffle training data
  idx = np.random.permutation(x_train.index)
  val_size = int(np.size(idx) * val_ratio)
  x_val = x_train.reindex(idx[-val_size:])
  y_val = y_train.reindex(idx[-val_size:])
  x_train = x_train.reindex(idx[:-val_size])
  y_train = y_train.reindex(idx[:-val_size])

  # Construct tf dataset
  with tf.device("CPU"):
    assert batch_size <= x_train.shape[0], (
        f"Batch size {batch_size} is larger than training data size"
        f" {x_train.shape[0]}."
    )
    ds_train = tf.data.Dataset.from_tensor_slices(
        (x_train.values, y_train.T.values)
    )
    ds_train = ds_train.map(transform).shuffle(
        100, reshuffle_each_iteration=True
    )
    ds_train = ds_train.batch(batch_size, drop_remainder=True)

    ds_val = tf.data.Dataset.from_tensor_slices((x_val.values, y_val.T.values))
    ds_val = ds_val.map(transform)
    ds_val = ds_val.batch(batch_size, drop_remainder=False)

    ds_test = tf.data.Dataset.from_tensor_slices(
        (x_test.values, y_test.T.values)
    )
    ds_test = ds_test.map(transform)
    ds_test = ds_test.batch(batch_size, drop_remainder=False)

  return {
      "x_train": x_train,
      "y_train": y_train,
      "x_val": x_val,
      "y_val": y_val,
      "x_test": x_test,
      "y_test": y_test,
      "ds_train": ds_train,
      "ds_val": ds_val,
      "ds_test": ds_test,
      "num_features": len(x_train.columns),
      "is_classification": is_classification,
      "num_classes": num_classes,
  }
