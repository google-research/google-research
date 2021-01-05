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

# Lint as: python2, python3
"""Utilities for initializing kernels for depthwise convolutions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow.compat.v1 as tf


def _compute_fans(shape):
  """Computes the fan-in and fan-out for a depthwise convolution's kernel."""
  if len(shape) != 4:
    raise ValueError(
        'DepthwiseVarianceScaling() is only supported for the rank-4 kernels '
        'of 2D depthwise convolutions. Bad kernel shape: {}'
        .format(str(shape)))

  receptive_field_size = shape[0] * shape[1]
  depth_multiplier = shape[3]

  fan_in = receptive_field_size
  fan_out = receptive_field_size * depth_multiplier
  return (fan_in, fan_out)


class DepthwiseVarianceScaling(tf.keras.initializers.Initializer):
  """Variant of tf.VarianceScaling() initializer for depthwise convolutions.

  A standard VarianceScaling() initializer may miscalculate the fan-in of a
  depthwise convolution by orders of magnitude. You should use this class as a
  drop-in replacement for tf.VarianceScaling() in the `depthwise_initializer`
  argument of tf.keras.layers.DepthwiseConv2D` and similar functions.

  Consider the kernel k for an ordinary 2D convolution. In general, `k` will
  have shape `(height, width, in_channels, out_channels)`, and we can estimate
  the fan-in of `k` as `height * width * in_channels`.

  On the other hand, the kernel for a depthwise convolution will have shape
  `(height, width, channels, depth_multiplier)`, and the fan-in should be
  equal to `height * width`. However, if we apply the heuristic above, we will
  estimate a fan-in of `height * width * channels` instead of `height * width`.

  In practice, this means that if `channels` is large then we can overestimate
  the fan-in of a depthwise convolution's kernel by orders of magnitude unless
  we use a corrected initializer such as this one.

  Attributes:
    scale: Scaling factor (positive float). The variance of the initialization
        distribution will be `scale / fan`, where `fan` is determined according
        to the `mode`. (The variance of the resulting initializer does not
        depend on the type of distribution that is used.)
    mode: One of "fan_in", "fan_out", "fan_avg".
    distribution: Random distribution to use. One of
        "truncated_normal", "untruncated_normal", "uniform".
    seed: A Python integer. Used to create random seeds. See
        `tf.set_random_seed` for behavior.
    dtype: The data type. Only floating point types are supported.

  Raises:
    ValueError: In case of an invalid value for the "scale" or
      "distribution" arguments.
  """

  def __init__(self,
               scale=1.0,
               mode='fan_in',
               distribution='truncated_normal',
               seed=None,
               dtype=tf.float32):
    if scale <= 0:
      raise ValueError('scale must be a positive float: {}'
                       .format(str(scale)))
    if mode not in ['fan_in', 'fan_out', 'fan_avg']:
      raise ValueError('invalid mode: {}'.format(str(mode)))
    if distribution not in ['uniform',
                            'truncated_normal',
                            'untruncated_normal']:
      raise ValueError('invalid distribution: {}'.format(str(distribution)))
    if not dtype.is_floating:
      raise ValueError('dtype must be a floating point type: {}'
                       .format(str(dtype)))

    self.scale = scale
    self.mode = mode
    self.distribution = distribution
    self.seed = seed
    self.dtype = tf.as_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype

    if not dtype.is_floating:
      raise ValueError('dtype must be a floating point type: {}'.format(dtype))

    # For a depthwise convolution with a filter multiplier of 1, we have
    #     fan_in = fan_out = receptive_field_size
    fan_in, fan_out = _compute_fans(shape)
    if self.mode == 'fan_in':
      scale = self.scale / max(1., fan_in)
    elif self.mode == 'fan_out':
      scale = self.scale / max(1., fan_out)
    else:
      assert self.mode == 'fan_avg', self.mode
      scale = self.scale / max(1., (fan_in + fan_out) / 2.)

    if self.distribution == 'truncated_normal':
      # With a true normal distribution, we'd set stddev = sqrt(scale).
      # We adjust the standard deviation to account for the fact that
      # we're drawing from a truncated normal
      stddev = math.sqrt(scale) / .87962566103423978
      return tf.truncated_normal(
          shape, 0.0, stddev, dtype, seed=self.seed)
    elif self.distribution == 'untruncated_normal':
      stddev = math.sqrt(scale)
      return tf.random_normal(
          shape, 0.0, stddev, dtype, seed=self.seed)
    else:
      # Draw samples from a uniform distribution over the interval
      # [-3.0*scale, 3.0*scale]. The resulting distribution will have
      # mean 0 and variance `scale`.
      assert self.distribution == 'uniform'
      limit = math.sqrt(3.0 * scale)
      return tf.random_uniform(
          shape, -limit, limit, dtype, seed=self.seed)

  def get_config(self):
    return {
        'scale': self.scale,
        'distribution': self.distribution,
        'seed': self.seed,
        'dtype': self.dtype.name,
    }


def depthwise_he_normal(seed=None):
  """Variant of tf.initializers.he_normal() for depthwise convolutional kernels.

  Args:
    seed: A Python integer. Used to create random seeds. See
        `tf.set_random_seed` for behavior.

  Returns:
    An instance of the DepthwiseVarianceScaling class.
  """
  return DepthwiseVarianceScaling(
      scale=2.0, mode='fan_in', distribution='truncated_normal', seed=seed)
