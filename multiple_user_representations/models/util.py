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

"""Common utility functions for models."""

import os
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf


SAMPLE_WEIGHT = "sample_weight"


def update_train_dataset_with_sample_weights(
    train_dataset,
    item_counts_weights):
  """Updates sample_weight in the train dataset.

  The function adds the sample_weight to input features of the training
  dataset. The sample_weight is found by searching the item_id of the
  next_item in the dictionary, and then using the corresponding sample_weight.

  Args:
    train_dataset: tf.data.Dataset
    item_counts_weights: A dictionary mapping item_id to the tuple (item_count,
      item_weight).

  Returns:
    updated_train_dataset: Updated tf.data.Dataset with sample_weight
      corresponding to next_item.
  """

  item_keys = tf.convert_to_tensor(
      list(item_counts_weights.keys()), dtype=tf.int32)
  item_count_weights = tf.convert_to_tensor(list(item_counts_weights.values()))

  def update_sample_weights_tf_data(features):

    next_items = tf.cast(features["next_item"], tf.int32)
    matched_item_count_weights = tf.gather(
        item_count_weights,
        tf.squeeze(tf.where(tf.math.equal(next_items, item_keys))))

    features[SAMPLE_WEIGHT] = matched_item_count_weights[
        1] * 1.0 / matched_item_count_weights[0]
    return features

  return train_dataset.map(update_sample_weights_tf_data)


def save_np(file_path, np_data):
  """Saves np array at file_path."""

  tf.io.gfile.makedirs(os.path.dirname(file_path))
  with tf.io.gfile.GFile(file_path, "wb") as fout:
    np.save(fout, np_data)
