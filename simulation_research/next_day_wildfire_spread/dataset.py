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

"""Library of utility functions for reading TF Example datasets."""

from typing import Dict, Text, Sequence

import tensorflow as tf


def get_features_dict(
    sample_size,
    features,
):
  """Creates a features dictionary for TensorFlow IO.

  Args:
    sample_size: Size of the input tiles in pixels (square).
    features: List of feature names.

  Returns:
    A features dictionary for TensorFlow IO.
  """
  sample_shape = [sample_size, sample_size]
  features = set(features)
  columns = [tf.io.FixedLenFeature(shape=sample_shape, dtype=tf.float32)
            ] * len(features)
  return dict(zip(features, columns))


def map_fire_labels(labels):
  """Remaps the raw MODIS fire labels to fire, non-fire, and uncertain.

  The raw fire labels have values spanning from 1 to 9, inclusive.
  https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/6/MYD14A1
  1, 2, 4, 6 correspond to uncertain labels.
  3, 5 correspond to non-fire labels.
  7, 8, 9 correspond to fire labels.

  Args:
    labels: Raw fire labels.

  Returns:
    Labels with values 1 for fire, 0 for non-fire, and -1 for uncertain.
  """
  non_fire = tf.where(
      tf.logical_or(tf.equal(labels, 3), tf.equal(labels, 5)),
      tf.zeros_like(labels), -1 * tf.ones_like(labels))
  fire = tf.where(tf.greater_equal(labels, 7), tf.ones_like(labels), non_fire)
  return tf.cast(fire, dtype=tf.float32)
