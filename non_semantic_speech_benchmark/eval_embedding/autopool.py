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

"""Autopool Keras layer. https://arxiv.org/abs/1804.10070."""

import tensorflow.compat.v2 as tf


class AutoPool(tf.keras.layers.Layer):
  """Automatically tuned soft-max pooling.

  This layer automatically adapts the pooling behavior to interpolate
  between mean- and max-pooling for each dimension. See
  `https://arxiv.org/abs/1804.10070` for more details, or the unit test which
  numerically checks this layer's limiting behavior.
  """

  def __init__(self, axis=0, alpha_init=0.0, trainable=True):
    """Init for AutoPool.

    Args:
      axis: (Int) Axis along which to perform the pooling.
      alpha_init: (Int) The value to initialize alpha to.
      trainable: (Boolean) Whether `alpha` is trainable.

    From the paper: For example, when `alpha` = 0, eq. (8) reduces to an
    unweighted mean; when `alpha` = 1,it simplifies to soft-max pooling; and
    when `alpha` -> inf, it approaches the max operator.
    """
    super(AutoPool, self).__init__()

    self.axis = axis
    self.alpha_init = alpha_init
    self.trainable = trainable

  def build(self, input_shape):
    alpha_shape = input_shape.as_list()
    alpha_shape[self.axis] = 1
    alpha_shape[0] = 1  # Batch axis should be broadcast over.

    self.alpha = tf.Variable(
        initial_value=tf.ones(shape=alpha_shape) * self.alpha_init,
        trainable=self.trainable,
        name="alpha",
        dtype=tf.float32)

    super(AutoPool, self).build(input_shape)

  @property
  def average_alpha(self):
    return tf.reduce_mean(self.alpha)

  def call(self, x, keepdims=False):
    exp_input = tf.exp(self.alpha * x)
    weights = exp_input / (
        tf.reduce_sum(exp_input, self.axis, keepdims=True))
    # Weights with `NaNs` means that autopool is becoming like `min` or `max`.
    # Replacing these faulty weights helps get the right value, but might
    # interfere with gradients.
    weights = tf.where(tf.math.is_nan(weights), tf.ones_like(weights), weights)
    weights = tf.identity(weights, name="autopool_weights")
    return tf.reduce_sum(x * weights, self.axis, keepdims=keepdims)
