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

"""Implements Randomized LayerNormalization, a Batch Normalization substitute.

For every item in a batch and for every layer, we calculate the mean and
variance across the spatial dimensions, and multiply them by Gaussian noise with
a mean equal to 1.0 (at training time only). This improved the results compared
to batch normalization - more in https://arxiv.org/abs/1904.04998.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function


import tensorflow.compat.v1 as tf


def normalize(x, is_train, name='bn', stddev=0.5):
  """Applies layer normalization and applies noise on the mean and variance.

  Args:
    x: tf.Tensor to normalize, of shape [B, H, W, C].
    is_train: A boolean, True at training mode.
    name: A string, a name scope.
    stddev: Standard deviation of the Gaussian noise. Defaults to 0.5 because
      this is the largest value where the noise is guaranteed to be a
      non-negative multiplicative factor

  Returns:
    A tf.Tensor of shape [B, H, W, C], the normalized tensor.
  """

  with tf.variable_scope(name, None, [x]):
    inputs_shape = x.shape.as_list()
    params_shape = inputs_shape[-1:]
    beta = tf.get_variable(
        'beta', shape=params_shape, initializer=tf.initializers.zeros())
    gamma = tf.get_variable(
        'gamma', shape=params_shape, initializer=tf.initializers.ones())
    mean, variance = tf.nn.moments(x, [1, 2], keep_dims=True)
    if is_train:
      mean *= 1.0 + tf.random.truncated_normal(tf.shape(mean), stddev=stddev)
      variance *= 1.0 + tf.random.truncated_normal(
          tf.shape(variance), stddev=stddev)
    outputs = tf.nn.batch_normalization(
        x,
        mean,
        variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=1e-3)
    outputs.set_shape(x.shape)
  return outputs
