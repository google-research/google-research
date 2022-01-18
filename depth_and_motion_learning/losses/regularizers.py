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

"""Regularizers for depth motion fields."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow.compat.v1 as tf


def joint_bilateral_smoothing(smoothed, reference):
  """Computes edge-aware smoothness loss.

  Args:
    smoothed: A tf.Tensor of shape [B, H, W, C1] to be smoothed.
    reference: A tf.Tensor of the shape [B, H, W, C2]. Wherever `reference` has
      more spatial variation, the strength of the smoothing of `smoothed` will
      be weaker.

  Returns:
    A scalar tf.Tensor containing the regularization, to be added to the
    training loss.
  """
  smoothed_dx = _gradient_x(smoothed)
  smoothed_dy = _gradient_y(smoothed)
  ref_dx = _gradient_x(reference)
  ref_dy = _gradient_y(reference)
  weights_x = tf.exp(-tf.reduce_mean(tf.abs(ref_dx), 3, keepdims=True))
  weights_y = tf.exp(-tf.reduce_mean(tf.abs(ref_dy), 3, keepdims=True))
  smoothness_x = smoothed_dx * weights_x
  smoothness_y = smoothed_dy * weights_y
  return tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))


def normalize_motion_map(res_motion_map, motion_map):
  """Normalizes a residual motion map by the motion map's norm."""
  with tf.name_scope('normalize_motion_map'):
    norm = tf.reduce_mean(
        tf.square(motion_map), axis=[1, 2, 3], keep_dims=True) * 3.0
    return res_motion_map / tf.sqrt(norm + 1e-12)


def l1smoothness(tensor, wrap_around=True):
  """Calculates L1 (total variation) smoothness loss of a tensor.

  Args:
    tensor: A tensor to be smoothed, of shape [B, H, W, C].
    wrap_around: True to wrap around the last pixels to the first.

  Returns:
    A scalar tf.Tensor, The total variation loss.
  """
  with tf.name_scope('l1smoothness'):
    tensor_dx = tensor - tf.roll(tensor, 1, 1)
    tensor_dy = tensor - tf.roll(tensor, 1, 2)
    # We optionally wrap around in order to impose continuity across the
    # boundary. The motivation is that there is some ambiguity between rotation
    # and spatial gradients of translation maps. We would like to discourage
    # spatial gradients of the translation field, and to absorb sich gradients
    # into the rotation as much as possible. This is why we impose continuity
    # across the spatial boundary.
    if not wrap_around:
      tensor_dx = tensor_dx[:, 1:, 1:, :]
      tensor_dy = tensor_dy[:, 1:, 1:, :]
    return tf.reduce_mean(
        tf.sqrt(1e-24 + tf.square(tensor_dx) + tf.square(tensor_dy)))


def sqrt_sparsity(motion_map):
  """A regularizer that encourages sparsity.

  This regularizer penalizes nonzero values. Close to zero it behaves like an L1
  regularizer, and far away from zero its strength decreases. The scale that
  distinguishes "close" from "far" is the mean value of the absolute of
  `motion_map`.

  Args:
     motion_map: A tf.Tensor of shape [B, H, W, C]

  Returns:
     A scalar tf.Tensor, the regularizer to be added to the training loss.
  """
  with tf.name_scope('drift'):
    tensor_abs = tf.abs(motion_map)
    mean = tf.stop_gradient(
        tf.reduce_mean(tensor_abs, axis=[1, 2], keep_dims=True))
    # We used L0.5 norm here because it's more sparsity encouraging than L1.
    # The coefficients are designed in a way that the norm asymptotes to L1 in
    # the small value limit.
    return tf.reduce_mean(2 * mean * tf.sqrt(tensor_abs / (mean + 1e-24) + 1))


def _gradient_x(img):
  return img[:, :, :-1, :] - img[:, :, 1:, :]


def _gradient_y(img):
  return img[:, :-1, :, :] - img[:, 1:, :, :]


