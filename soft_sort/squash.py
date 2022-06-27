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

"""Implements different squashing strategies before applying the Sinkhorn algorithm.

The main purpose of those functions is to map the numbers we wish to sort into
the [0, 1] segment using an increasing function, such as a logistic map.

This logistic map, when applied on the output of a neural network,
redistributes the activations into [0,1] in a smooth adaptative way, helping
the numerical stability of the Sinkhorn algorithm while maintaining a
well behaved back propagation.

In case of a logistic sigmoid, this map is exactly the CDF of a logistic
distribution. See https://en.wikipedia.org/wiki/Logistic_distribution for
details, in particular the variance of the distribution. In case of an atan
sigmoid, it is somewhat related to the CDF of a Cauchy
distribution and presents the advantage to have better behaved gradients.

In a such a logistic map, the points lying in the linear part of the map will
be well spread out on the [0, 1] segment, which will make them easier to sort.
Therefore, depending on which part of the distribution is of interest, we might
want to focus on one part of another, hence leading to different translations
before applying a squashing function.
"""

import math
import gin
import tensorflow.compat.v2 as tf


@gin.configurable
def reduce_softmax(x, tau, axis = -1):
  """Computes the softmax of a tensor along a given axis.

  Args:
   x: (tf.Tensor<float>) the input tensor of any shape.
   tau: (float) the value of the inverse softmax temperature.
     When tau is very big the obtained value is close to the maximum, when 0
     it coincides with the mean and when very negative it converges to the
     minimum.
   axis: (int) the axis along which we want to compute the softmax.

  Returns:
   a tf.Tensor<float> that has the same shape than the input tensor except for
    the reduction axis which is gone.
  """
  return tf.math.reduce_sum(tf.nn.softmax(x * tau, axis=axis) * x, axis=axis)


@gin.configurable
def whiten(x, axis = -1, min_std=1e-10):
  """Makes the input tensor zero mean and unit variance along the axis.

  Args:
   x: (tf.Tensor<float>) of any shape to be whitened.
   axis: (int) the axis along which to compute the statistics.
   min_std: (float) a minimum value of the standard deviation along the axis to
    prevent degenerated cases.

  Returns:
   A tf.Tensor<float> of the same shape as the input tensor.
  """
  mu = tf.expand_dims(tf.math.reduce_mean(x, axis=axis), axis=axis)
  min_std = 1e-6
  sigma = tf.expand_dims(
      tf.maximum(tf.math.reduce_std(x, axis=axis), min_std), axis=axis)
  return (x - mu) / sigma


@gin.configurable
def soft_stretch(
    x, axis = -1, extreme_tau = 1e12):
  """Softly rescales the values of `x` along the axis to the [0, 1] segment.

  Args:
   x: (tf.Tensor<float> of any shape) the input tensor to rescale the values of.
   axis: (int) the axis along which we want to rescale.
   extreme_tau: (float) the value of the inverse temperature to compute the
    softmax and softmin. This must be big for the output values to
    really lie in the [0, 1] segment.

  Returns:
   A tf.Tensor<float> of the same shape as the input.
  """
  min_x = tf.expand_dims(
      reduce_softmax(x, tau=-extreme_tau, axis=axis), axis=axis)
  max_x = tf.expand_dims(
      reduce_softmax(x, tau=extreme_tau, axis=axis), axis=axis)
  return (x - min_x) / (max_x - min_x)


@gin.configurable
def group_rescale(
    x,
    is_logistic = True,
    tau = 0.0,
    stretch = False):
  """Applies a sigmoid map on standardized inputs.

  By default, the inputs is centered, but it can be uncentered by playing with
  the parameters `tau` and `uncenter_towards_max`.

  Args:
   x: Tensor<float>[batch, n]
   is_logistic: (bool) uses either a logistic sigmoid or an arctan.
   tau: (float or None) inverse temperature parameter that, if not None,
    controls how much deviation we want from the mean. The bigger the closer to
    the maximum and the more negative to the minimum
   stretch: (bool) if True, stretches the values to the the full [0, 1] segment.

  Returns:
   A Tensor<float>[batch, n] after application of the sigmoid map.
  """
  x = whiten(x, axis=1)
  if is_logistic:
    x /= math.sqrt(3.0) / math.pi
  if tau != 0:
    center = reduce_softmax(x, tau=tau, axis=1)
    x = x - center[:, tf.newaxis]
  squashing_fn = tf.math.sigmoid if is_logistic else tf.math.atan
  squashed_x = squashing_fn(x)
  if stretch:
    return soft_stretch(squashed_x)
  return squashed_x
