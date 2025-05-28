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

"""Data utility methods."""
import numpy as np
import tensorflow as tf


def get_default_metrics():
  return [
      tf.keras.metrics.TruePositives(name="tp"),
      tf.keras.metrics.FalsePositives(name="fp"),
      tf.keras.metrics.TrueNegatives(name="tn"),
      tf.keras.metrics.FalseNegatives(name="fn"),
      tf.keras.metrics.BinaryAccuracy(name="accuracy"),
      tf.keras.metrics.AUC(name="auc"),
  ]


def df_to_dataset(data_df, targets=None, shuffle=True, convert_to_float=False):
  data_df = data_df.copy()
  if convert_to_float:
    data_df = data_df.to_numpy(dtype=np.float32)
  else:
    data_df = dict(data_df)
  ds = tf.data.Dataset.from_tensor_slices(
      (data_df, targets) if targets is not None else data_df
  )
  if shuffle:
    ds = ds.shuffle(buffer_size=5000)  # Reasonable but arbitrary buffer_size.
  return ds


def get_sensitive_attribute_for_dataset(dataset_name):
  return {
      "adult": "sex",
      "compas": "race",
      "hsls": "racebin",
  }[dataset_name]


def csv_str_to_list(string):
  """Convert a line in a csv file into a list."""
  return [int(x) for x in string.split(",") if x.isnumeric()]
