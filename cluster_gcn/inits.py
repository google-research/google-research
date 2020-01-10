# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Implementations of different initialization methods."""

import numpy as np
import tensorflow.compat.v1 as tf


def uniform(shape, scale=0.05, name=None):
  """Uniform init."""
  initial = tf.random_uniform(
      shape, minval=-scale, maxval=scale, dtype=tf.float32)
  return tf.Variable(initial, name=name)


def glorot(shape, name=None):
  """Glorot & Bengio (AISTATS 2010) init."""
  init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
  initial = tf.random_uniform(
      shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
  return tf.Variable(initial, name=name)


def zeros(shape, name=None):
  """All zeros."""
  initial = tf.zeros(shape, dtype=tf.float32)
  return tf.Variable(initial, name=name)


def ones(shape, name=None):
  """All ones."""
  initial = tf.ones(shape, dtype=tf.float32)
  return tf.Variable(initial, name=name)
