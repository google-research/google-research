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

# Lint as: python3
"""Definition of basic operators for the experiments."""

import gin
import tensorflow.compat.v2 as tf
from perturbations.experiments import shortest_path

# Makes it possible to use those tensorflow classes in gin files.
gin.config.external_configurable(tf.keras.losses.MeanSquaredError,
                                 module='tf.keras.losses')
gin.config.external_configurable(
    tf.keras.losses.CategoricalCrossentropy, module='tf.keras.losses')
gin.config.external_configurable(
    tf.keras.metrics.CategoricalAccuracy, module='tf.keras.metrics')


@gin.configurable
def argmax_fn(x, axis=-1):
  return tf.cast(
      tf.one_hot(tf.math.argmax(x, axis=axis), depth=tf.shape(x)[axis]),
      dtype=x.dtype)


@gin.configurable
def ranks_fn(x, axis=-1):
  return tf.cast(tf.argsort(tf.argsort(x, axis=axis), axis=axis), dtype=x.dtype)


@gin.configurable
def argsort_fn(x, axis=-1):
  return tf.cast(tf.argsort(x, axis=axis), dtype=x.dtype)


@gin.configurable
def shortest_path_fn(x):
  return tf.cast(shortest_path.Dijkstra()(x), dtype=x.dtype)
