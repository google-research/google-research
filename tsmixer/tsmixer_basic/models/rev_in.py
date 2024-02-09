# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Implementation of Reversible Instance Normalization."""

import tensorflow as tf
from tensorflow.keras import layers


class RevNorm(layers.Layer):
  """Reversible Instance Normalization."""

  def __init__(self, axis, eps=1e-5, affine=True):
    super().__init__()
    self.axis = axis
    self.eps = eps
    self.affine = affine

  def build(self, input_shape):
    if self.affine:
      self.affine_weight = self.add_weight(
          'affine_weight', shape=input_shape[-1], initializer='ones'
      )
      self.affine_bias = self.add_weight(
          'affine_bias', shape=input_shape[-1], initializer='zeros'
      )

  def call(self, x, mode, target_slice=None):
    if mode == 'norm':
      self._get_statistics(x)
      x = self._normalize(x)
    elif mode == 'denorm':
      x = self._denormalize(x, target_slice)
    else:
      raise NotImplementedError
    return x

  def _get_statistics(self, x):
    self.mean = tf.stop_gradient(
        tf.reduce_mean(x, axis=self.axis, keepdims=True)
    )
    self.stdev = tf.stop_gradient(
        tf.sqrt(
            tf.math.reduce_variance(x, axis=self.axis, keepdims=True) + self.eps
        )
    )

  def _normalize(self, x):
    x = x - self.mean
    x = x / self.stdev
    if self.affine:
      x = x * self.affine_weight
      x = x + self.affine_bias
    return x

  def _denormalize(self, x, target_slice=None):
    if self.affine:
      x = x - self.affine_bias[target_slice]
      x = x / self.affine_weight[target_slice]
    x = x * self.stdev[:, :, target_slice]
    x = x + self.mean[:, :, target_slice]
    return x
