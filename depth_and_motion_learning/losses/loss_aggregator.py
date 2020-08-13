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
"""Classes for computing a training losses given inputs and predictions."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import abc

import six
import tensorflow.compat.v1 as tf

from depth_and_motion_learning import consistency_losses
from depth_and_motion_learning import intrinsics_utils
from depth_and_motion_learning import transform_depth_map
from depth_and_motion_learning.losses import regularizers
from depth_and_motion_learning.parameter_container import ParameterContainer


class LossAggregator(six.with_metaclass(abc.ABCMeta, object)):
  """A base class for calculating losses from a set of tensors."""

  def __init__(self, endpoints, weights_overrides=None, params_overrides=None):
    """Creates an instance.

    Args:
      endpoints: A dictionary mapping strings to tf.Tensors, from which the loss
        is to be computed.
      weights_overrides: A dictionary or containing overrides for
        self._default_weights
      params_overrides: A dictionary or containing overrides for
        self._default_params
    """
    self._weights = ParameterContainer.from_defaults_and_overrides(
        self._default_weights, weights_overrides, is_strict=True)
    self._params = ParameterContainer.from_defaults_and_overrides(
        self._default_params, params_overrides, is_strict=True)
    self._losses = {k: tf.convert_to_tensor(0.0) for k in self._default_weights}
    self._endpoints = endpoints
    self._output_endpoints = {}
    self._calculate()

  @abc.abstractmethod
  def _calculate(self):
    """Populate self._losses and self._output_endpoints.

    To be implemented by subclasses.
    """
    pass

  @property
  def _default_weights(self):
    """A dictionary that maps loss names (strings) to their weights (floats)."""
    return {}

  @property
  def _default_params(self):
    """A dictionary containing other parameters, if needed bysub-classes."""
    return {}

  @property
  def losses(self):
    return self._losses

  @property
  def output_endpoints(self):
    return self._output_endpoints


class DepthMotionFieldLossAggregator(LossAggregator):
  """A LossAgregator for depth maps and 3D motion fields."""

  @property
  def _default_weights(self):
    return {
        'rgb_consistency': 1.0,
        'ssim': 3.0,
        'depth_consistency': 0.0,
        'depth_smoothing': 0.01,
        'depth_supervision': 1.0,
        'rotation_cycle_consistency': 1e-3,
        'translation_cycle_consistency': 1e-2,
        'depth_variance': 1e-6,
        'motion_smoothing': 1e-3,
        'motion_drift': 0.0
    }

  @property
  def _default_params(self):
    return {
        'target_depth_stop_gradient': True,
        'scale_normalization': False,
        'num_scales': 1,
    }

  def _calculate(self):
    # On tpu we strive to stack tensors together and perform ops once on the
    # entire stack, to save time HBM memory. We thus stack the batch-of-first-
    # frames and the batch-of-second frames, for both depth and RGB. The batch
    # dimension of rgb_stack and gt_depth_stack are thus twice the original
    # batch size.

    # Create stacks for features that need to be scaled into pyramids for
    # multi-scale training.
    rgb_stack_ = tf.concat(self._endpoints['rgb'], axis=0)
    flipped_rgb_stack_ = tf.concat(self._endpoints['rgb'][::-1], axis=0)
    predicted_depth_stack_ = tf.concat(
        self._endpoints['predicted_depth'], axis=0)
    flipped_predicted_depth_stack_ = tf.concat(
        self._endpoints['predicted_depth'][::-1], axis=0)
    residual_translation_ = tf.concat(
        self._endpoints['residual_translation'], axis=0)
    flipped_residual_translation_ = tf.concat(
        self._endpoints['residual_translation'][::-1], axis=0)
    intrinsics_mat_ = tf.concat(self._endpoints['intrinsics_mat'], axis=0)

    # Create pyramids from each stack to support multi-scale training.
    num_scales = self._params.num_scales
    rgb_pyramid = _get_pyramid(rgb_stack_, num_scales=num_scales)
    flipped_rgb_pyramid = _get_pyramid(
        flipped_rgb_stack_, num_scales=num_scales)
    predicted_depth_pyramid = _get_pyramid(
        predicted_depth_stack_, num_scales=num_scales)
    flipped_predicted_depth_pyramid = _get_pyramid(
        flipped_predicted_depth_stack_, num_scales=num_scales)
    residual_translation_pyramid = _get_pyramid(
        residual_translation_, num_scales=num_scales)
    flipped_residual_translation_pyramid = _get_pyramid(
        flipped_residual_translation_, num_scales=num_scales)
    intrinsics_mat_pyramid = _get_intrinsics_mat_pyramid(
        intrinsics_mat_, num_scales=num_scales)
    validity_mask_ = self._endpoints.get('validity_mask')
    if validity_mask_ is not None:
      validity_mask_ = tf.concat(validity_mask_, axis=0)
      validity_mask_pyramid = _get_pyramid(
          validity_mask_, num_scales, _min_pool2d)
    else:
      validity_mask_pyramid = [None] * num_scales

    if 'groundtruth_depth' in self._endpoints:
      gt_depth_stack_ = tf.concat(self._endpoints['groundtruth_depth'], axis=0)
      gt_depth_pyramid = _get_pyramid(gt_depth_stack_, num_scales=num_scales)
      if 'groundtruth_depth_weight' in self._endpoints:
        gt_depth_weight_stack_ = tf.concat(
            self._endpoints['groundtruth_depth_weight'], axis=0)
      else:
        gt_depth_weight_stack_ = tf.cast(
            tf.greater(gt_depth_stack_, 0.2), tf.float32)
      gt_depth_weight_pyramid = _get_pyramid(
          gt_depth_weight_stack_, num_scales=num_scales)

      if 'groundtruth_depth_filter' in self._endpoints:
        depth_filter_ = tf.concat(
            self._endpoints['groundtruth_depth_filter'], axis=0)
        depth_filter_ = tf.cast(depth_filter_, tf.float32)
        depth_filter_pyramid = _get_pyramid(
            gt_depth_stack_, num_scales=num_scales)

    # Calculate losses at each scale.  Iterate in reverse so that the final
    # output values are set at scale 0.
    for s in reversed(range(self._params.num_scales)):
      # Weight applied to all losses at this scale.
      scale_w = 1.0 / 2**s

      rgb_stack = rgb_pyramid[s]
      predicted_depth_stack = predicted_depth_pyramid[s]
      flipped_predicted_depth_stack = flipped_predicted_depth_pyramid[s]

      if 'groundtruth_depth' in self._endpoints:
        gt_depth_stack = gt_depth_pyramid[s]
        depth_error = tf.abs(gt_depth_stack - predicted_depth_stack)

        # Weigh the spatial loss if a weight map is provided. Otherwise, revert
        # to original behavior.
        gt_depth_weight_stack = gt_depth_weight_pyramid[s]
        depth_error = depth_error * gt_depth_weight_stack

        # Optionally filter the depth map if a boolean depth filter is provided.
        # We use a TPU-friendly equivalent of tf.boolean_mask.
        depth_filter = tf.ones_like(depth_error, tf.float32)
        if 'groundtruth_depth_filter' in self._endpoints:
          depth_filter = depth_filter_pyramid[s]

        self._losses['depth_supervision'] += scale_w * tf.reduce_mean(
            depth_error * depth_filter) / tf.reduce_mean(depth_filter)

      # In theory, the training losses should be agnostic to the global scale of
      # the predicted depth. However in reality second order effects can lead to
      # (https://en.wikipedia.org/wiki/Von_Neumann_stability_analysis) diverging
      # modes. For some reason this happens when training on TPU. Since the
      # scale is immaterial anyway, we normalize it out, and the training
      # stabilizes.
      #
      # Note that the depth supervision term, which is sensitive to the scale,
      # was applied before this normalization. Therefore the scale of the depth
      # is learned.
      mean_depth = tf.reduce_mean(predicted_depth_stack)

      # When training starts, the depth sometimes tends to collapse to a
      # constant value, which seems to be a fixed point where the trainig can
      # stuck. To discourage this collapse, we penalize the reciprocal of the
      # variance with a tiny weight. Note that the mean of predicted_depth is
      # one, hence we subtract 1.0.
      depth_var = tf.reduce_mean(
          tf.square(predicted_depth_stack / mean_depth - 1.0))
      self._losses['depth_variance'] = scale_w * 1.0 / depth_var

      if self._params.scale_normalization:
        predicted_depth_stack /= mean_depth
        flipped_predicted_depth_stack /= mean_depth

      disp = 1.0 / predicted_depth_stack

      mean_disp = tf.reduce_mean(disp, axis=[1, 2, 3], keep_dims=True)
      self._losses['depth_smoothing'] += (
          scale_w *
          regularizers.joint_bilateral_smoothing(disp / mean_disp, rgb_stack))
      self._output_endpoints['disparity'] = disp

      flipped_rgb_stack = flipped_rgb_pyramid[s]

      background_translation = tf.concat(
          self._endpoints['background_translation'], axis=0)
      flipped_background_translation = tf.concat(
          self._endpoints['background_translation'][::-1], axis=0)
      residual_translation = residual_translation_pyramid[s]
      flipped_residual_translation = flipped_residual_translation_pyramid[s]
      if self._params.scale_normalization:
        background_translation /= mean_depth
        flipped_background_translation /= mean_depth
        residual_translation /= mean_depth
        flipped_residual_translation /= mean_depth
      translation = residual_translation + background_translation
      flipped_translation = (
          flipped_residual_translation + flipped_background_translation)

      rotation = tf.concat(self._endpoints['rotation'], axis=0)
      flipped_rotation = tf.concat(self._endpoints['rotation'][::-1], axis=0)
      intrinsics_mat = intrinsics_mat_pyramid[s]
      intrinsics_mat_inv = intrinsics_utils.invert_intrinsics_matrix(
          intrinsics_mat)
      validity_mask = validity_mask_pyramid[s]

      transformed_depth = transform_depth_map.using_motion_vector(
          tf.squeeze(predicted_depth_stack, axis=-1), translation, rotation,
          intrinsics_mat, intrinsics_mat_inv)
      flipped_predicted_depth_stack = tf.squeeze(
          flipped_predicted_depth_stack, axis=-1)
      if self._params.target_depth_stop_gradient:
        flipped_predicted_depth_stack = tf.stop_gradient(
            flipped_predicted_depth_stack)
      # The first and second halves of the batch not contain Frame1's and
      # Frame2's depths transformed onto Frame2 and Frame1 respectively. Te
      # demand consistency, we need to `flip` `predicted_depth` as well.
      loss_endpoints = (
          consistency_losses.rgbd_and_motion_consistency_loss(
              transformed_depth,
              rgb_stack,
              flipped_predicted_depth_stack,
              flipped_rgb_stack,
              rotation,
              translation,
              flipped_rotation,
              flipped_translation,
              validity_mask=validity_mask))

      normalized_trans = regularizers.normalize_motion_map(
          residual_translation, translation)
      self._losses['motion_smoothing'] += scale_w * regularizers.l1smoothness(
          normalized_trans, self._weights.motion_drift == 0)
      self._losses['motion_drift'] += scale_w * regularizers.sqrt_sparsity(
          normalized_trans)
      self._losses['depth_consistency'] += (
          scale_w * loss_endpoints['depth_error'])
      self._losses['rgb_consistency'] += scale_w * loss_endpoints['rgb_error']
      self._losses['ssim'] += scale_w * 0.5 * loss_endpoints['ssim_error']

      self._losses['rotation_cycle_consistency'] += (
          scale_w * loss_endpoints['rotation_error'])
      self._losses['translation_cycle_consistency'] += (
          scale_w * loss_endpoints['translation_error'])

      self._output_endpoints['depth_proximity_weight'] = loss_endpoints[
          'depth_proximity_weight']
      self._output_endpoints['trans'] = translation
      self._output_endpoints['inv_trans'] = flipped_translation

    for k, w in self._weights.as_dict().items():
      # multiply by 2 to match the scale of the old code.
      self._losses[k] *= w * 2

    if tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
      self._losses[tf.GraphKeys.REGULARIZATION_LOSSES] = tf.add_n(
          tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))


def _get_intrinsics_mat_pyramid(intrinsics_mat, num_scales):
  """Returns multiple intrinsic matrices for different scales.

  Args:
    intrinsics_mat: <float32>[B, 3, 3] tensor containing the intrinsics matrix
      at the original scale.
    num_scales: integer indicating *total* number of matrices to return.  If
      `num_scales` is 1, the function just returns the input matrix in a list.

  Returns:
    List containing `num_scales` intrinsics matrices, each with shape
      <float32>[B, 3, 3].  The first element in the list is the input
      intrinsics matrix and the last element is the intrinsics matrix for the
      coarsest scale.
  """
  # intrinsics_mat: [B, 3, 3]
  intrinsics_mat_pyramid = [intrinsics_mat]
  # Scale the intrinsics accordingly for each scale.
  for s in range(1, num_scales):
    fx = intrinsics_mat[:, 0, 0] / 2**s
    fy = intrinsics_mat[:, 1, 1] / 2**s
    cx = intrinsics_mat[:, 0, 2] / 2**s
    cy = intrinsics_mat[:, 1, 2] / 2**s
    intrinsics_mat_pyramid.append(_make_intrinsics_matrix(fx, fy, cx, cy))
  return intrinsics_mat_pyramid


def _make_intrinsics_matrix(fx, fy, cx, cy):
  """Constructs a batch of intrinsics matrices given arguments..

  Args:
    fx: <float32>[B] tensor containing horizontal focal length.
    fy: <float32>[B] tensor containing vertical focal length.
    cx: <float32>[B] tensor containing horizontal principal offset.
    cy: <float32>[B] tensor containing vertical principal offset.

  Returns:
    <float32>[B, 3, 3] tensor containing batch of intrinsics matrices.
  """
  # fx, fy, cx, cy: [B]
  zeros = tf.zeros_like(fx)
  ones = tf.ones_like(fx)
  r1 = tf.stack([fx, zeros, cx], axis=-1)
  r2 = tf.stack([zeros, fy, cy], axis=-1)
  r3 = tf.stack([zeros, zeros, ones], axis=-1)
  intrinsics = tf.stack([r1, r2, r3], axis=1)
  return intrinsics


def _min_pool2d(input_, ksize, strides, padding):
  return -tf.nn.max_pool_v2(-input_, ksize, strides, padding)


def _get_pyramid(img, num_scales, pooling_fn=tf.nn.avg_pool2d):
  """Generates a pyramid from the input image/tensor at different scales.

  This function behaves similarly to `tfg.image.pyramid.split()`.  Instead of
  using an image resize operation, it uses average pooling to give each
  input pixel equal weight in constructing coarser scales.

  Args:
    img: [B, height, width, C] tensor, where B stands for batch size and C
      stands for number of channels.
    num_scales: integer indicating *total* number of scales to return.  If
      `num_scales` is 1, the function just returns the input image in a list.
    pooling_fn: A callable with tf.nn.avg_pool2d's signature, to be used for
      pooling `img` across scales.

  Returns:
    List containing `num_scales` tensors with shapes
      [B, height / 2^s, width / 2^s, C] where s is in [0, num_scales - 1].  The
      first element in the list is the input image and the last element is the
      resized input corresponding to the coarsest scale.
  """
  pyramid = [img]
  for _ in range(1, num_scales):
    # Scale image stack.
    last_img = pyramid[-1]
    scaled_img = pooling_fn(
        last_img, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    pyramid.append(scaled_img)
  return pyramid
