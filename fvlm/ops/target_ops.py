# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Target and sampling related ops.

This is a JAX reimplementation of:
https://github.com/tensorflow/tpu/blob/master/models/official/detection/ops/target_ops.py
"""
from typing import Tuple

import gin
from jax import lax
import jax.numpy as jnp

from ops import spatial_transform_ops
from utils import balanced_positive_negative_sampler
from utils import box_utils

Array = jnp.ndarray


def box_matching(
    boxes,
    gt_boxes,
    gt_classes):
  """Match boxes to groundtruth boxes.

  Given the proposal boxes and the groundtruth boxes and classes, perform the
  groundtruth matching by taking the argmax of the IoU between boxes and
  groundtruth boxes.

  Args:
    boxes: an array of shape of [batch_size, N, 4] representing the box
      coordinates to be matched to groundtruth boxes.
    gt_boxes: an array of shape of [batch_size, MAX_INSTANCES, 4] representing
      the groundtruth box coordinates. It is padded with -1s to indicate the
      invalid boxes.
    gt_classes: [batch_size, MAX_INSTANCES] representing the groundtruth box
      classes. It is padded with -1s to indicate the invalid classes.

  Returns:
    matched_gt_boxes: an array of shape of [batch_size, N, 4], representing
      the matched groundtruth box coordinates for each input box. If the box
      does not overlap with any groundtruth boxes, the matched boxes of it
      will be set to all 0s.
    matched_gt_classes: an array of shape of [batch_size, N], representing
      the matched groundtruth classes for each input box. If the box does not
      overlap with any groundtruth boxes, the matched box classes of it will
      be set to 0, which corresponds to the background class.
    matched_gt_indices: an array of shape of [batch_size, N], representing
      the indices of the matched groundtruth boxes in the original gt_boxes
      array. If the box does not overlap with any groundtruth boxes, the
      index of the matched groundtruth will be set to -1.
    matched_iou: an array of shape of [batch_size, N], representing the IoU
      between the box and its matched groundtruth box. The matched IoU is the
      maximum IoU of the box and all the groundtruth boxes.
    iou: an array of shape of [batch_size, N, K], representing the IoU matrix
      between boxes and the groundtruth boxes. The IoU between a box and the
      invalid groundtruth boxes whose coordinates are [-1, -1, -1, -1] is -1.
  """
  # Compute IoU between boxes and gt_boxes.
  # iou <- [batch_size, N, K]
  iou = box_utils.bbox_overlap(boxes, gt_boxes)

  # max_iou <- [batch_size, N]
  # 0.0 -> no match to gt, or -1.0 match to no gt
  matched_iou = jnp.amax(iou, axis=-1)

  # background_box_mask <- bool, [batch_size, N]
  background_box_mask = matched_iou <= 0.0

  argmax_iou_indices = jnp.argmax(iou, axis=-1).astype(jnp.int32)

  argmax_iou_indices_shape = argmax_iou_indices.shape
  batch_indices = (
      jnp.arange(argmax_iou_indices_shape[0])[Ellipsis, None] *
      jnp.ones([1, argmax_iou_indices_shape[-1]], dtype=jnp.int32))
  matched_gt_boxes = gt_boxes[batch_indices, argmax_iou_indices]
  matched_gt_boxes = jnp.where(
      jnp.tile(background_box_mask[Ellipsis, None], [1, 1, 4]),
      jnp.zeros_like(matched_gt_boxes, dtype=matched_gt_boxes.dtype),
      matched_gt_boxes)

  matched_gt_classes = gt_classes[batch_indices, argmax_iou_indices]
  matched_gt_classes = jnp.where(background_box_mask,
                                 jnp.zeros_like(matched_gt_classes),
                                 matched_gt_classes)

  matched_gt_indices = jnp.where(background_box_mask,
                                 -jnp.ones_like(argmax_iou_indices),
                                 argmax_iou_indices)

  return (matched_gt_boxes, matched_gt_classes, matched_gt_indices,
          matched_iou, iou)


@gin.configurable
def sample_box_targets(
    proposed_boxes,
    gt_boxes,
    gt_classes,
    key,
    num_samples_per_image = 512,
    fg_fraction = 0.25,
    fg_iou_threshold = 0.5,
    bg_iou_threshold_high = 0.5,
    bg_iou_threshold_low = 0.0,
    mix_gt_boxes = True,
    is_static = True,
):
  """Assigns the proposals with groundtruth classes and performs subsmpling.

  Given `proposed_boxes`, `gt_boxes`, and `gt_classes`, the function uses the
  following algorithm to generate the final `num_samples_per_image` RoIs.
    1. Calculates the IoU between each proposal box and each gt_boxes.
    2. Assigns each proposed box with a groundtruth class and box by choosing
       the largest IoU overlap.
    3. Samples `num_samples_per_image` boxes from all proposed boxes, and
       returns box_targets, class_targets, and RoIs.

  Args:
    proposed_boxes: an array of shape of [batch_size, N, 4]. N is the number
      of proposals before groundtruth assignment. The last dimension is the
      box coordinates w.r.t. the scaled images in [ymin, xmin, ymax, xmax]
      format.
    gt_boxes: an array of shape of [batch_size, MAX_NUM_INSTANCES, 4]. The
      coordinates of gt_boxes are in the pixel coordinates of the scaled
      image. This array might have padding of values -1 indicating the
      invalid box coordinates.
    gt_classes: an array with a shape of [batch_size, MAX_NUM_INSTANCES]. This
      array might have paddings with values of -1 indicating the invalid
      classes.
    key: a key representing the state of JAX random function.
    num_samples_per_image: an integer represents RoI minibatch size per image.
    fg_fraction: a float represents the target fraction of RoI minibatch that
      is labeled foreground (i.e., class > 0).
    fg_iou_threshold: a float represents the IoU overlap threshold for an RoI
      to be considered foreground (if >= fg_iou_threshold).
    bg_iou_threshold_high: a float represents the IoU overlap threshold for an
      RoI to be considered background (class = 0 if overlap in [LO, HI)).
    bg_iou_threshold_low: a float represents the IoU overlap threshold for an
      RoI to be considered background (class = 0 if overlap in [LO, HI)).
    mix_gt_boxes: a bool indicating whether to mix the groundtruth boxes
      before sampling proposals.
    is_static: If True, uses an implementation with static shape guarantees.

  Returns:
    sampled_rois: an array of shape of [batch_size, K, 4], representing the
      coordinates of the sampled RoIs, where K is the number of the sampled
      RoIs, i.e. K = num_samples_per_image.
    sampled_gt_boxes: an array of shape of [batch_size, K, 4], storing the
      box coordinates of the matched groundtruth boxes of the samples RoIs.
    sampled_gt_classes: an array of shape of [batch_size, K], storing the
      classes of the matched groundtruth boxes of the sampled RoIs.
    sampled_gt_indices: an array of shape of [batch_size, K], storing the
      indices of the sampled groudntruth boxes in the original `gt_boxes`
      array, i.e. gt_boxes[sampled_gt_indices[:, i]] = sampled_gt_boxes[:,
      i].
  """
  if mix_gt_boxes:
    boxes = jnp.concatenate([proposed_boxes, gt_boxes], axis=1)
  else:
    boxes = proposed_boxes

  (matched_gt_boxes, matched_gt_classes, matched_gt_indices, matched_iou,
   _) = box_matching(boxes, gt_boxes, gt_classes)

  positive_match = matched_iou > fg_iou_threshold
  negative_match = jnp.logical_and(
      matched_iou >= bg_iou_threshold_low,
      matched_iou < bg_iou_threshold_high)
  ignored_match = matched_iou < 0.0

  # re-assign negatively matched boxes to the background class.
  matched_gt_classes = jnp.where(negative_match,
                                 jnp.zeros_like(matched_gt_classes),
                                 matched_gt_classes)
  matched_gt_indices = jnp.where(negative_match,
                                 jnp.zeros_like(matched_gt_indices),
                                 matched_gt_indices)

  sample_candidates = jnp.logical_and(
      jnp.logical_or(positive_match, negative_match),
      jnp.logical_not(ignored_match))

  sampler = (
      balanced_positive_negative_sampler.BalancedPositiveNegativeSampler(
          positive_fraction=fg_fraction,
          is_static=is_static))

  batch_size = sample_candidates.shape[0]
  sampled_indicators = []
  for i in range(batch_size):
    sampled_indicator = sampler.subsample(sample_candidates[i],
                                          num_samples_per_image,
                                          positive_match[i],
                                          key)
    sampled_indicators.append(sampled_indicator)

  sampled_indicators = jnp.stack(sampled_indicators)
  _, sampled_indices = lax.top_k(
      sampled_indicators.astype(jnp.int32),
      k=num_samples_per_image)

  sampled_indices_shape = sampled_indices.shape
  batch_indices = (
      jnp.arange(sampled_indices_shape[0])[Ellipsis, None] *
      jnp.ones([1, sampled_indices_shape[-1]], dtype=jnp.int32))

  sampled_rois = boxes[batch_indices, sampled_indices]
  sampled_gt_boxes = matched_gt_boxes[batch_indices, sampled_indices]
  sampled_gt_classes = matched_gt_classes[batch_indices, sampled_indices]
  sampled_gt_indices = matched_gt_indices[batch_indices, sampled_indices]
  return (sampled_rois, sampled_gt_boxes, sampled_gt_classes,
          sampled_gt_indices)


@gin.configurable
def sample_mask_targets(candidate_rois,
                        candidate_gt_boxes,
                        candidate_gt_classes,
                        candidate_gt_indices,
                        gt_masks,
                        mask_target_size = 28,
                        num_mask_samples_per_image = 128):
  """Sample and create mask targets for training.

  This class samples and crops the mask targets from groundtruth masks that
  match the ROIs for training the mask prediction head.

  Args:
    candidate_rois: a tensor of shape of [batch_size, N, 4], where N is the
      number of candidate RoIs to be considered for mask sampling. It includes
      both positive and negative RoIs. The `num_mask_samples_per_image`
      positive RoIs will be sampled to create mask training targets.
    candidate_gt_boxes: a tensor of shape of [batch_size, N, 4], storing the
      corresponding groundtruth boxes to the `candidate_rois`.
    candidate_gt_classes: a tensor of shape of [batch_size, N], storing the
      corresponding groundtruth classes to the `candidate_rois`. 0 in the
      tensor corresponds to the background class, i.e. negative RoIs.
    candidate_gt_indices: a tensor of shape [batch_size, N], storing the
      corresponding groundtruth instance indices to the `candidate_gt_boxes`,
      i.e. gt_boxes[candidate_gt_indices[:, i]] = candidate_gt_boxes[:, i],
      where gt_boxes which is of shape [batch_size, MAX_INSTANCES, 4], M >= N,
      is the superset of candidate_gt_boxes.
    gt_masks: a tensor of [batch_size, MAX_INSTANCES, mask_height, mask_width]
      containing all the groundtruth masks which sample masks are drawn from.
      after sampling. The output masks are resized w.r.t the sampled RoIs.
    mask_target_size: Target mask resolution (height, width) in pixels..
    num_mask_samples_per_image: Number of mask samples per image.

  Returns:
    foreground_rois: a tensor of shape of [batch_size, K, 4] storing the RoI
      that corresponds to the sampled foreground masks, where
      K = num_mask_samples_per_image.
    foreground_classes: a tensor of shape of [batch_size, K] storing the
      classes corresponding to the sampled foreground masks.
    cropoped_foreground_masks: a tensor of shape of
      [batch_size, K, mask_target_size, mask_target_size] storing the
      cropped foreground masks used for training.
  """
  _, fg_instance_indices = lax.top_k(
      (candidate_gt_classes > 0).astype(jnp.int32),
      k=num_mask_samples_per_image)

  fg_instance_indices_shape = fg_instance_indices.shape
  batch_indices = (
      jnp.arange(fg_instance_indices_shape[0])[Ellipsis, None] *
      jnp.ones([1, fg_instance_indices_shape[-1]], dtype=jnp.int32))

  foreground_rois = candidate_rois[batch_indices, fg_instance_indices]
  foreground_boxes = candidate_gt_boxes[batch_indices, fg_instance_indices]
  foreground_classes = candidate_gt_classes[batch_indices,
                                            fg_instance_indices]
  foreground_gt_indices = candidate_gt_indices[batch_indices,
                                               fg_instance_indices]

  foreground_gt_indices_shape = foreground_gt_indices.shape
  batch_indices = (
      jnp.arange(foreground_gt_indices_shape[0])[Ellipsis, None] *
      jnp.ones([1, foreground_gt_indices_shape[-1]], dtype=jnp.int32))
  foreground_masks = gt_masks[batch_indices, foreground_gt_indices]

  cropped_foreground_masks = spatial_transform_ops.crop_mask_in_target_box(
      foreground_masks,
      foreground_boxes,
      foreground_rois,
      mask_target_size,
      sample_offset=0.5)

  return foreground_rois, foreground_classes, cropped_foreground_masks
