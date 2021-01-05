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

"""Batch norm layer with optional relu activation function."""

import tensorflow.compat.v1 as tf  # tf

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs,
                    is_training,
                    relu=True,
                    init_zero=False,
                    bn_decay=BATCH_NORM_DECAY,
                    bn_epsilon=BATCH_NORM_EPSILON,
                    data_format='channels_first'):
  """Performs a batch normalization followed by a ReLU.

  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training: `bool` for whether the model is training.
    relu: `bool` if False, omits the ReLU operation.
    init_zero: `bool` if True, initializes scale parameter of batch
      normalization with 0 instead of 1 (default).
    bn_decay: `float` batch norm decay parameter to use.
    bn_epsilon: `float` batch norm epsilon parameter to use.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """

  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  assert data_format == 'channels_last'

  if data_format == 'channels_first':
    axis = 1
  else:
    axis = -1

  inputs = tf.layers.batch_normalization(
      inputs=inputs,
      axis=axis,
      momentum=bn_decay,
      epsilon=bn_epsilon,
      center=True,
      scale=True,
      training=is_training,
      fused=True,
      gamma_initializer=gamma_initializer)

  if relu:
    inputs = tf.nn.relu(inputs)
  return inputs

