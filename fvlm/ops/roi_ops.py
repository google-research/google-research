# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""ROI-related ops.

This is a reimplementation of the following in JAX.
https://github.com/tensorflow/tpu/blob/master/models/official/detection/ops/roi_ops.py
"""

from typing import Dict, Tuple, Union

import gin
from jax import nn as jnn
import jax.numpy as jnp

from ops import nms
from utils import box_utils


_EPSILON = 1e-8
Array = jnp.ndarray


@gin.configurable
def multilevel_propose_rois(
    rpn_boxes,
    rpn_scores,
    anchor_boxes,
    image_shape,
    rpn_pre_nms_top_k = 2000,
    rpn_post_nms_top_k = 1000,
    rpn_nms_threshold = 0.7,
    rpn_score_threshold = 0.0,
    rpn_min_size_threshold = 0.0,
    decode_boxes = True,
    clip_boxes = True,
    apply_sigmoid_to_score = True,
    use_lrtb_boxes = False,
    output_decoded_boxes = False):
  """Proposes RoIs given a group of candidates from different FPN levels.

  The following describes the steps:
    1. For each individual level:
      a. Apply sigmoid transform if specified.
      b. Decode boxes if specified.
      c. Clip boxes if specified.
      d. Filter small boxes and those fall outside image if specified.
      e. Apply pre-NMS filtering including pre-NMS top k and score thresholding.
      f. Apply NMS.
    2. Aggregate post-NMS boxes from each level.
    3. Apply an overall top k to generate the final selected RoIs.

  Args:
    rpn_boxes: a dict with keys representing FPN levels and values representing
      box tenors of shape [batch_size, feature_h, feature_w, num_anchors * 4].
    rpn_scores: a dict with keys representing FPN levels and values representing
      logit tensors of shape [batch_size, feature_h, feature_w, num_anchors].
    anchor_boxes: a dict with keys representing FPN levels and values
      representing anchor box tensors of shape
      [batch_size, feature_h * feature_w * num_anchors, 4].
    image_shape: a tensor of shape [batch_size, 2] where the last dimension are
      [height, width] of the scaled image.
    rpn_pre_nms_top_k: an integer of top scoring RPN proposals *per level* to
      keep before applying NMS. Default: 2000.
    rpn_post_nms_top_k: an integer of top scoring RPN proposals *in total* to
      keep after applying NMS. Default: 1000.
    rpn_nms_threshold: a float between 0 and 1 representing the IoU threshold
      used for NMS. If 0.0, no NMS is applied. Default: 0.7.
    rpn_score_threshold: a float between 0 and 1 representing the minimal box
      score to keep before applying NMS. This is often used as a pre-filtering
      step for better performance. If 0, no filtering is applied. Default: 0.
    rpn_min_size_threshold: a float representing the minimal box size in each
      side (w.r.t. the scaled image) to keep before applying NMS. This is often
      used as a pre-filtering step for better performance. If 0, no filtering is
      applied. Default: 0.
    decode_boxes: a boolean indicating whether `rpn_boxes` needs to be decoded
      using `anchor_boxes`. If False, use `rpn_boxes` directly and ignore
      `anchor_boxes`. Default: True.
    clip_boxes: a boolean indicating whether boxes are first clipped to the
      scaled image size before appliying NMS. If False, no clipping is applied
      and `image_shape` is ignored. Default: True.
    apply_sigmoid_to_score: a boolean indicating whether apply sigmoid to
      `rpn_scores` before applying NMS. Default: True.
    use_lrtb_boxes: a boolean indicating whether to use LRTB format boxes.
      Defaults to False.
    output_decoded_boxes: a boolean indicating whether to output decoded boxes.

  Returns:
    selected_rois: a tensor of shape [batch_size, rpn_post_nms_top_k, 4],
      representing the box coordinates of the selected proposals w.r.t. the
      scaled image.
    selected_roi_scores: a tensor of shape [batch_size, rpn_post_nms_top_k, 1],
      representing the scores of the selected proposals.
  """
  rois = []
  roi_scores = []
  decoded_rpn_boxes = {}
  image_shape = jnp.expand_dims(image_shape, axis=1)
  for level in sorted(rpn_scores.keys()):
    _, feature_h, feature_w, num_anchors_per_location = (
        rpn_scores[level].shape)

    num_boxes = feature_h * feature_w * num_anchors_per_location
    this_level_scores = jnp.reshape(rpn_scores[level], [-1, num_boxes])
    this_level_boxes = jnp.reshape(rpn_boxes[level], [-1, num_boxes, 4])
    this_level_anchors = jnp.reshape(
        anchor_boxes[level], [-1, num_boxes, 4]).astype(this_level_scores.dtype)

    if apply_sigmoid_to_score:
      this_level_scores = jnn.sigmoid(this_level_scores)

    if decode_boxes:
      if use_lrtb_boxes:
        this_level_boxes = box_utils.decode_boxes_lrtb(
            this_level_boxes, this_level_anchors)
      else:
        this_level_boxes = box_utils.decode_boxes(
            this_level_boxes, this_level_anchors)
    if clip_boxes:
      this_level_boxes = box_utils.clip_boxes(
          this_level_boxes, image_shape)

    if rpn_min_size_threshold > 0.0:
      this_level_boxes, this_level_scores = box_utils.filter_boxes(
          this_level_boxes,
          this_level_scores,
          image_shape,
          rpn_min_size_threshold)

    if output_decoded_boxes:
      this_level_decoded_boxes = jnp.reshape(
          this_level_boxes,
          [-1, feature_h, feature_w, num_anchors_per_location * 4])
      decoded_rpn_boxes[level] = this_level_decoded_boxes

    this_level_pre_nms_top_k = min(num_boxes, rpn_pre_nms_top_k)
    this_level_post_nms_top_k = min(num_boxes, rpn_post_nms_top_k)
    if rpn_nms_threshold > 0.0:
      if rpn_score_threshold > 0.0:
        this_level_boxes, this_level_scores = (
            box_utils.filter_boxes_by_scores(
                this_level_boxes, this_level_scores, rpn_score_threshold))
      this_level_boxes, this_level_scores = box_utils.top_k_boxes(
          this_level_boxes, this_level_scores, k=this_level_pre_nms_top_k)
      this_level_roi_scores, this_level_rois = (
          nms.non_max_suppression_padded(
              this_level_scores,
              this_level_boxes,
              max_output_size=this_level_post_nms_top_k,
              iou_threshold=rpn_nms_threshold))
    else:
      this_level_rois, this_level_roi_scores = box_utils.top_k_boxes(
          this_level_boxes,
          this_level_scores,
          k=this_level_post_nms_top_k)

    rois.append(this_level_rois)
    roi_scores.append(this_level_roi_scores)

  all_rois = jnp.concatenate(rois, axis=1)
  all_roi_scores = jnp.concatenate(roi_scores, axis=1)

  _, num_valid_rois = all_roi_scores.shape
  overall_top_k = min(num_valid_rois, rpn_post_nms_top_k)

  selected_rois, selected_roi_scores = box_utils.top_k_boxes(
      all_rois, all_roi_scores, k=overall_top_k)
  if output_decoded_boxes:
    return selected_rois, selected_roi_scores, decoded_rpn_boxes
  else:
    return selected_rois, selected_roi_scores
