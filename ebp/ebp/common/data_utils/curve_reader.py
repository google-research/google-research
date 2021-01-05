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

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import collections

CNPRegressionDescription = collections.namedtuple(
    "CNPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points"))


class CurveReader(object):

  def __init__(self, batch_size, max_num_context, testing=False):
    self._batch_size = batch_size
    self._max_num_context = max_num_context
    self._testing = testing
    self._x_size = 1

  def get_y_values(self, x_values, num_total_points):
    raise NotImplementedError

  def generate_curves(self):
    num_context = tf.random_uniform(
        shape=[], minval=3, maxval=self._max_num_context, dtype=tf.int32)
    num_context = self._max_num_context
    # If we are testing we want to have more targets and have them evenly
    # distributed in order to plot the function.
    if self._testing:
      num_context = self._max_num_context * 2
      num_target = 400
      num_total_points = num_target
      x_values = tf.tile(
          tf.expand_dims(tf.range(-2., 2., 1. / 100, dtype=tf.float32), axis=0),
          [self._batch_size, 1])
      x_values = tf.expand_dims(x_values, axis=-1)
    # During training the number of target points and their x-positions are
    # selected at random
    else:
      num_target = tf.random_uniform(
          shape=(), minval=2, maxval=self._max_num_context, dtype=tf.int32)
      num_target = self._max_num_context
      num_total_points = num_context + num_target
      x_values = tf.random_uniform(
          [self._batch_size, num_total_points, self._x_size], -2, 2)

    y_values = self.get_y_values(x_values, num_total_points)

    if self._testing:
      # Select the targets
      target_x = x_values
      target_y = y_values

      # Select the observations
      idx = tf.random_shuffle(tf.range(num_target))
      context_x = tf.gather(x_values, idx[:num_context], axis=1)
      context_y = tf.gather(y_values, idx[:num_context], axis=1)

    else:
      # Select the targets which will consist of the context points as well as
      # some new target points
      target_x = x_values[:, :num_target + num_context, :]
      target_y = y_values[:, :num_target + num_context, :]

      # Select the observations
      context_x = x_values[:, :num_context, :]
      context_y = y_values[:, :num_context, :]

    query = ((context_x, context_y), target_x)

    return CNPRegressionDescription(
        query=query,
        target_y=target_y,
        num_total_points=tf.shape(target_x)[1],
        num_context_points=num_context)


class SineCurvesReader(CurveReader):

  def __init__(self,
               batch_size,
               max_num_context,
               freq=2,
               scale=1,
               testing=False):
    super(SineCurvesReader, self).__init__(batch_size, max_num_context, testing)
    self._x_size = 1
    self._y_size = 1
    self._freq = freq
    self._scale = scale

  def get_y_values(self, x_values, num_total_points):
    return tf.math.sin(x_values * self._freq) * self._scale + tf.random.normal(
        shape=tf.shape(x_values), mean=0, stddev=0.3, dtype=tf.float32)


class CircleCurvesReader(CurveReader):

  def __init__(self, batch_size, max_num_context, r=2, testing=False):
    super(CircleCurvesReader, self).__init__(batch_size, max_num_context,
                                             testing)
    self._x_size = 1
    self._y_size = 1
    self.r = r

  def get_y_values(self, x_values, num_total_points):
    cluster = tf.random.uniform(shape=tf.shape(x_values)) < 0.5
    y_abs = tf.sqrt(self.r**2 - x_values**2)
    y1 = y_abs
    y2 = -y_abs
    #y2 = -tf.sqrt((2*self.r) ** 2 - x_values ** 2)
    y = tf.where(cluster, y1, y2) + tf.random.normal(
        shape=tf.shape(x_values), mean=0, stddev=0.1, dtype=tf.float32)
    return y


class MixSineCurvesReader(SineCurvesReader):

  def get_y_values(self, x_values, num_total_points):
    cluster = tf.random.uniform(shape=tf.shape(x_values)) < 0.5
    y1 = tf.math.sin(x_values * self._freq) * self._scale + tf.random.normal(
        shape=tf.shape(x_values), mean=0, stddev=0.1, dtype=tf.float32)
    y2 = tf.math.cos(x_values * self._freq) * self._scale + tf.random.normal(
        shape=tf.shape(x_values), mean=0, stddev=0.1, dtype=tf.float32)
    y = tf.where(cluster, y1, y2)
    return y


class MixLineCurvesReader(SineCurvesReader):

  def get_y_values(self, x_values, num_total_points):
    cluster = tf.random.uniform(shape=tf.shape(x_values)) < 0.5
    y1 = x_values * 0 + 0.5 + tf.random.normal(
        shape=tf.shape(x_values), mean=0, stddev=0.1, dtype=tf.float32)
    y2 = x_values * 0 - 0.5 + tf.random.normal(
        shape=tf.shape(x_values), mean=0, stddev=0.1, dtype=tf.float32)
    y = tf.where(cluster, y1, y2)
    return y


class GPCurvesReader(CurveReader):

  def __init__(self,
               batch_size,
               max_num_context,
               l1_scale=0.4,
               sigma_scale=1.0,
               testing=False):
    super(GPCurvesReader, self).__init__(batch_size, max_num_context, testing)
    self._x_size = 1
    self._y_size = 1
    self._l1_scale = l1_scale
    self._sigma_scale = sigma_scale

  def _gaussian_kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
    num_total_points = tf.shape(xdata)[1]

    # Expand and take the difference
    xdata1 = tf.expand_dims(xdata, axis=1)  # [B, 1, num_total_points, x_size]
    xdata2 = tf.expand_dims(xdata, axis=2)  # [B, num_total_points, 1, x_size]
    diff = xdata1 - xdata2  # [B, num_total_points, num_total_points, x_size]

    # [B, y_size, num_total_points, num_total_points, x_size]
    norm = tf.square(diff[:, None, :, :, :] / l1[:, :, None, None, :])

    norm = tf.reduce_sum(
        norm, -1)  # [B, data_size, num_total_points, num_total_points]

    # [B, y_size, num_total_points, num_total_points]
    kernel = tf.square(sigma_f)[:, :, None, None] * tf.exp(-0.5 * norm)

    # Add some noise to the diagonal to make the cholesky work.
    kernel += (sigma_noise**2) * tf.eye(num_total_points)

    return kernel

  def get_y_values(self, x_values, num_total_points):
    # Set kernel parameters
    l1 = (
        tf.ones(shape=[self._batch_size, self._y_size, self._x_size]) *
        self._l1_scale)
    sigma_f = tf.ones(
        shape=[self._batch_size, self._y_size]) * self._sigma_scale

    # Pass the x_values through the Gaussian kernel
    # [batch_size, y_size, num_total_points, num_total_points]
    kernel = self._gaussian_kernel(x_values, l1, sigma_f)

    # Calculate Cholesky, using double precision for better stability:
    cholesky = tf.cast(tf.cholesky(tf.cast(kernel, tf.float64)), tf.float32)

    # Sample a curve
    # [batch_size, y_size, num_total_points, 1]
    y_values = tf.matmul(
        cholesky,
        tf.random_normal([self._batch_size, self._y_size, num_total_points, 1]))

    # [batch_size, num_total_points, y_size]
    y_values = tf.transpose(tf.squeeze(y_values, 3), [0, 2, 1])
    return y_values


def get_reader(args, testing, bsize=None):
  if bsize is None:
    bsize = args.batch_size
  if args.data_name == "gp":
    return GPCurvesReader(
        batch_size=bsize, max_num_context=args.num_ctx, testing=testing)
  elif args.data_name == "sine":
    return SineCurvesReader(
        batch_size=bsize, max_num_context=args.num_ctx, testing=testing)
  elif args.data_name == "mix_sine":
    return MixSineCurvesReader(
        batch_size=bsize, max_num_context=args.num_ctx, testing=testing)
  elif args.data_name == "mix_line":
    return MixLineCurvesReader(
        batch_size=bsize, max_num_context=args.num_ctx, testing=testing)
  elif args.data_name == "circle":
    return CircleCurvesReader(
        batch_size=bsize, max_num_context=args.num_ctx, testing=testing)
  else:
    raise NotImplementedError
