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

# Lint as: python3
"""Computes metrics from prediction and ground truth."""
import numpy as np
import tensorflow.compat.v2 as tf


def _get_affine_inv_wmae(prediction,
                         depth,
                         depth_conf,
                         irls_iters=5,
                         epsilon=1e-3):
  """Gets affine invariant weighted mean average error."""
  # This function returns L1 error, but does IRLS on epsilon-invariant L1 error
  # for numerical reasons.
  prediction_vec = tf.reshape(prediction, [-1])
  depth_conf_vec = tf.reshape(depth_conf, [-1])
  irls_weight = tf.ones_like(depth_conf_vec)
  for _ in range(irls_iters):
    sqrt_weight = tf.sqrt(depth_conf_vec * irls_weight)
    lhs = sqrt_weight[:, tf.newaxis] * tf.stack(
        [prediction_vec, tf.ones_like(prediction_vec)], 1)
    rhs = sqrt_weight * tf.reshape(depth, [-1])
    affine_est = tf.linalg.lstsq(
        lhs, rhs[:, tf.newaxis], l2_regularizer=1e-5, fast=False)
    prediction_affine = prediction * affine_est[0] + affine_est[1]
    resid = tf.abs(prediction_affine - depth)
    irls_weight = tf.reshape(1./tf.maximum(epsilon, resid), [-1])
  wmae = tf.reduce_sum(depth_conf * resid) / tf.reduce_sum(depth_conf)
  return wmae


def _get_affine_inv_wrmse(prediction, depth, depth_conf):
  """Gets affine invariant weighted root mean squared error."""
  prediction_vec = tf.reshape(prediction, [-1])
  depth_conf_vec = tf.reshape(depth_conf, [-1])
  lhs = tf.sqrt(depth_conf_vec)[:, tf.newaxis] * tf.stack(
      [prediction_vec, tf.ones_like(prediction_vec)], 1)
  rhs = tf.sqrt(depth_conf_vec) * tf.reshape(depth, [-1])
  affine_est = tf.linalg.lstsq(
      lhs, rhs[:, tf.newaxis], l2_regularizer=1e-5, fast=False)
  prediction_affine = prediction * affine_est[0] + affine_est[1]
  # Clip the residuals to prevent infs.
  resid_sq = tf.minimum(
      (prediction_affine - depth)**2,
      np.finfo(np.float32).max)
  wrmse = tf.sqrt(
      tf.reduce_sum(depth_conf * resid_sq) / tf.reduce_sum(depth_conf))
  return wrmse


def _pearson_correlation(x, y, w):
  """Gets Pearson correlation between `x` and `y` weighted by `w`."""
  w_sum = tf.reduce_sum(w)
  expectation = lambda z: tf.reduce_sum(w * z) / w_sum
  mu_x = expectation(x)
  mu_y = expectation(y)
  var_x = expectation(x**2) - mu_x**2
  var_y = expectation(y**2) - mu_y**2
  cov = expectation(x * y) - mu_x * mu_y
  rho = cov / tf.math.sqrt(var_x * var_y)
  return rho


def _get_spearman_rank_correlation(x, y, w):
  """Gets weighted Spearman rank correlation coefficent between `x` and `y`."""
  x = tf.reshape(x, [-1])
  y = tf.reshape(y, [-1])
  w = tf.reshape(w, [-1])
  # Argsort twice returns each item's rank.
  rank = lambda z: tf.argsort(tf.argsort(z))

  # Cast and rescale the ranks to be in [-1, 1] for better numerical stability.
  def _cast_and_rescale(z):
    return tf.cast(z - tf.shape(z)[0] // 2, tf.float32) / (
        tf.cast(tf.shape(z)[0] // 2, tf.float32))

  x_rank = _cast_and_rescale(rank(x))
  x_rank_negative = _cast_and_rescale(rank(-x))

  y_rank = _cast_and_rescale(rank(y))

  # Spearman rank correlation is just pearson correlation on
  # (any affine transformation of) rank. We take maximum in order to get
  # the absolute value of the correlation coefficient.
  return tf.maximum(
      _pearson_correlation(x_rank, y_rank, w),
      _pearson_correlation(x_rank_negative, y_rank, w))


def metrics(prediction, gt_depth, gt_depth_conf, crop_height, crop_width):
  """Computes and returns WMAE, WRMSE and Spearman's metrics."""

  def center_crop(image):
    height = image.shape[0]
    width = image.shape[1]
    offset_y = (height - crop_height) // 2
    offset_x = (width - crop_width) // 2
    end_y = offset_y + crop_height
    end_x = offset_x + crop_width
    image = image[offset_y:end_y, offset_x:end_x].astype(np.float32)
    return tf.convert_to_tensor(image)

  prediction = center_crop(prediction)
  gt_depth = center_crop(gt_depth)
  gt_depth_conf = center_crop(gt_depth_conf)
  return {
      'wmae':
          _get_affine_inv_wmae(prediction, gt_depth, gt_depth_conf),
      'wrmse':
          _get_affine_inv_wrmse(prediction, gt_depth, gt_depth_conf),
      'spearman':
          1.0 -
          _get_spearman_rank_correlation(prediction, gt_depth, gt_depth_conf),
  }
