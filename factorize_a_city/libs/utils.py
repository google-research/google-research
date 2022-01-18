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

"""Utils ops to support the factorize_city project."""
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

layers = tf.contrib.layers


def outlier_normalization(inp, clip_amount=3):
  """Operation for normalizing numpy images with unbounded values.

  This is used to normalize log_reflectance and log_shading images which have
  unbounded values. This function bounds the min-max of the array to be
  plus-minus clip_amount standard deviation of the mean. The clipped range is
  then shifted to [0, 1].

  Args:
    inp: [H, W, 3] A numpy array with unbounded values.
    clip_amount: (int) how many standard deviations from the mean to clip by.

  Returns:
    A tensor of shape [H, W, 3] with values ranging from [0, 1].
  """
  sigma = np.std(inp)
  mu = np.mean(inp)
  inp = np.clip(inp, mu - clip_amount * sigma, mu + clip_amount * sigma)
  m = inp - np.min(inp,)
  return m / np.max(m)


def pad_panorama_for_convolutions(tensor, ksz, mode):
  pad_top = (ksz - 1) // 2
  pad_bottom = ksz // 2
  pad_left = (ksz - 1) // 2
  pad_right = ksz // 2
  reflect_pad = [[0, 0], [pad_top, pad_bottom], [0, 0], [0, 0]]
  tensor = tf.pad(tensor, reflect_pad, mode)
  tensor = tf.concat(
      [tensor[:, :, -pad_left:,], tensor, tensor[:, :, :pad_right]], axis=-2)
  return tensor


def reduce_median(tensor, axis=0, keep_dims=False):
  return tfp.stats.percentile(tensor, 50, axis=axis, keep_dims=keep_dims)


def upsample(tensor, size=2):
  unused_b, h, w, unused_d = tensor.shape.as_list()
  return tf.compat.v1.image.resize_bilinear(
      tensor, [size * h, size * w],
      align_corners=False,
      half_pixel_centers=True)


def instance_normalization(inp, scope=""):
  with tf.compat.v1.variable_scope(scope):
    return layers.instance_norm(
        inp, center=True, scale=True, trainable=True, epsilon=1e-5)


def compute_circular_average(softmax_distribution):
  """Computes circular average of a batch of softmax_distribution.

  Args:
    softmax_distribution: [B, K] is a batch of distributions of angles over K
      bins which spans [-pi, pi]. Each bin contains the probability of an
      orientation in its corresponding angle direction.

  Returns:
    Circular average, in radians, of shape [B] for each distribution of K-bins.
  """
  unused_batch_size, k_bins = softmax_distribution.shape.as_list()
  radian_coordinates = tf.linspace(-np.pi, np.pi,
                                   k_bins + 1)[:k_bins] + (np.pi) / k_bins

  # Imagine a top-down view of the scene, where the x-axis points out the center
  # of the panorama and the +y axis is clockwise.
  x_vector_direction = tf.cos(radian_coordinates)
  y_vector_direction = tf.sin(radian_coordinates)

  expected_x_coordinate = tf.reduce_sum(
      softmax_distribution * x_vector_direction[tf.newaxis], axis=-1)
  expected_y_coordinate = tf.reduce_sum(
      softmax_distribution * y_vector_direction[tf.newaxis], axis=-1)

  # Project the circular average to the unit circle to prevent unstable
  # expoding gradients when the average is close to the origin of the
  # coordinate frame.
  dist = tf.sqrt(expected_x_coordinate * expected_x_coordinate +
                 expected_y_coordinate * expected_y_coordinate + 1e-5)
  normx = expected_x_coordinate / dist
  normy = expected_y_coordinate / dist
  return tf.atan2(normy, normx)
