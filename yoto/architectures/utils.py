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

"""Some simple common functions for architectures."""

import gin
import tensorflow.compat.v1 as tf


def mlp(inputs, layer_sizes):
  x = inputs
  kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
  for n, layer_size in enumerate(layer_sizes):
    x = tf.layers.dense(inputs=x, units=layer_size,
                        kernel_initializer=kernel_initializer)
    if n < len(layer_sizes) - 1:
      x = tf.nn.leaky_relu(x)
  return x


@gin.configurable("ConditioningPostprocessing")
class ConditioningPostprocessing(object):
  """Specifies the postprocessing function to apply to conditioning values."""

  def __init__(self, scales_mult, scales_add, shifts_mult, shifts_add):
    self._scales_mult = scales_mult
    self._scales_add = scales_add
    self._shifts_mult = shifts_mult
    self._shifts_add = shifts_add

  def __call__(self, scales, shifts):
    scales = self._scales_mult * scales + self._scales_add
    shifts = self._shifts_mult * shifts + self._shifts_add
    return scales, shifts
