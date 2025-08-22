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

"""Utility functions for bounding box processing.

This is a JAX reimplementation of:
https://github.com/tensorflow/tpu/blob/master/models/official/detection/utils/box_utils.py
"""

from typing import Optional, Sequence, Tuple, Union

from jax import lax
import jax.numpy as jnp
import numpy as np


_EPSILON = 1e-7
BBOX_XFORM_CLIP = np.log(1000. / 16.)

Array = jnp.ndarray


def bbox_overlap(boxes,
                 gt_boxes,
                 is_aligned = False):
  """Calculates the overlap between proposal and ground truth boxes.

  Some `gt_boxes` may have been padded.  The returned `iou` tensor for these
  boxes will be -1.

  Args:
    boxes: an array with a shape of [batch_size, N, 4]. N is the number of
      proposals before groundtruth assignment (e.g., rpn_post_nms_topn). The
      last dimension is the pixel coordinates in [ymin, xmin, ymax, xmax] form.
    gt_boxes: an array with a shape of [batch_size, MAX_NUM_INSTANCES, 4]. This
      tensor might have paddings with a negative value.
    is_aligned: whether the number of boxes and gt_boxes is the same, and to
      perform element-wise overlap computation.

  Returns:
    iou: an array with as a shape of [batch_size, N, MAX_NUM_INSTANCES].
  """
  bb_y_min, bb_x_min, bb_y_max, bb_x_max = jnp.split(boxes, 4, axis=2)
  gt_y_min, gt_x_min, gt_y_max, gt_x_max = jnp.split(gt_boxes, 4, axis=2)

  # Calculates the intersection area.
  if is_aligned:
    i_xmin = jnp.maximum(bb_x_min, gt_x_min)
    i_xmax = jnp.minimum(bb_x_max, gt_x_max)
    i_ymin = jnp.maximum(bb_y_min, gt_y_min)
    i_ymax = jnp.minimum(bb_y_max, gt_y_max)  # [b,n,1]
    i_area = jnp.maximum(
        (i_xmax - i_xmin), 0) * jnp.maximum((i_ymax - i_ymin), 0)
  else:
    i_xmin = jnp.maximum(bb_x_min, jnp.transpose(gt_x_min, [0, 2, 1]))
    i_xmax = jnp.minimum(bb_x_max, jnp.transpose(gt_x_max, [0, 2, 1]))
    i_ymin = jnp.maximum(bb_y_min, jnp.transpose(gt_y_min, [0, 2, 1]))
    i_ymax = jnp.minimum(bb_y_max, jnp.transpose(gt_y_max, [0, 2, 1]))
    i_area = jnp.maximum(
        (i_xmax - i_xmin), 0) * jnp.maximum((i_ymax - i_ymin), 0)

  # Calculates the union area.
  bb_area = (bb_y_max - bb_y_min) * (bb_x_max - bb_x_min)
  gt_area = (gt_y_max - gt_y_min) * (gt_x_max - gt_x_min)
  # Adds a small epsilon to avoid divide-by-zero.
  if is_aligned:
    u_area = jnp.maximum(bb_area + gt_area - i_area, 0) + 1e-8
    iou = i_area / u_area
    mask = jnp.logical_and(iou >= 0.0, iou <= 1.0)  # [b,n,1]
    gt_valid_mask = jnp.amax(gt_boxes, axis=-1, keepdims=True) >= 0.0
    assert mask.shape == gt_valid_mask.shape
    return iou, jnp.logical_and(mask, gt_valid_mask)
  else:
    u_area = bb_area + jnp.transpose(gt_area, [0, 2, 1]) - i_area + 1e-8

  # Calculates IoU.
  iou = i_area / u_area

  # Fills -1 for IoU entries between the padded ground truth boxes.
  gt_invalid_mask = jnp.amax(gt_boxes, axis=-1, keepdims=True) < 0.0
  padding_mask = jnp.logical_or(
      jnp.zeros_like(bb_x_min, dtype=jnp.bool_),
      jnp.transpose(gt_invalid_mask, [0, 2, 1]))
  iou = jnp.where(padding_mask, -jnp.ones_like(iou), iou)

  return iou


def filter_boxes(boxes,
                 scores,
                 image_shape,
                 min_size_threshold):
  """Filter and remove boxes that are too small or fall outside the image.

  Args:
    boxes: a tensor whose last dimension is 4 representing the
      coordinates of boxes in ymin, xmin, ymax, xmax order.
    scores: a tensor whose shape is the same as boxes.shape[:-1]
      representing the original scores of the boxes.
    image_shape: a tensor whose shape is the same as, or `broadcastable` to
      `boxes` except the last dimension, which is 2, representing
      [height, width] of the scaled image.
    min_size_threshold: a float representing the minimal box size in each
      side (w.r.t. the scaled image). Boxes whose sides are smaller than it will
      be filtered out.

  Returns:
    filtered_boxes: a tensor whose shape is the same as `boxes` but with
      the position of the filtered boxes are filled with 0.
    filtered_scores: a tensor whose shape is the same as 'scores' but with
      the positinon of the filtered boxes filled with 0.
  """
  if boxes.shape[-1] != 4:
    raise ValueError(
        'boxes.shape[1] is {:d}, but must be 4.'.format(boxes.shape[-1]))

  if isinstance(image_shape, list) or isinstance(image_shape, tuple):
    height, width = image_shape
  else:
    image_shape = image_shape.astype(boxes.dtype)
  height = image_shape[Ellipsis, 0]
  width = image_shape[Ellipsis, 1]

  ymin = boxes[Ellipsis, 0]
  xmin = boxes[Ellipsis, 1]
  ymax = boxes[Ellipsis, 2]
  xmax = boxes[Ellipsis, 3]

  # computing height and center of boxes
  h = ymax - ymin + 1.0
  w = xmax - xmin + 1.0
  yc = ymin + 0.5 * h
  xc = xmin + 0.5 * w

  min_size = max(min_size_threshold, 1.0)

  # filtering boxes based on constraints
  filtered_size_mask = jnp.logical_and(
      h > min_size, w > min_size)
  filtered_center_mask = jnp.logical_and(
      jnp.logical_and(yc > 0.0, yc < height),
      jnp.logical_and(xc > 0.0, xc < width))
  filtered_mask = jnp.logical_and(filtered_size_mask, filtered_center_mask)

  filtered_scores = jnp.where(filtered_mask, scores, jnp.zeros_like(scores))
  filtered_boxes = jnp.expand_dims(
      filtered_mask.astype(boxes.dtype), axis=-1) * boxes

  return filtered_boxes, filtered_scores


def filter_boxes_by_scores(
    boxes,
    scores,
    min_score_threshold):
  """Filters and removes boxes whose scores are smaller than the threshold.

  Args:
    boxes: a tensor whose last dimension is 4 representing the
      coordinates of boxes in ymin, xmin, ymax, xmax order.
    scores: a tensor whose shape is the same as tf.shape(boxes)[:-1]
      representing the original scores of the boxes.
    min_score_threshold: a float representing the minimal box score threshold.
      Boxes whose score are smaller than it will be filtered out.

  Returns:
    filtered_boxes: a tensor whose shape is the same as `boxes` but with
      the position of the filtered boxes are filled with 0.
    filtered_scores: a tensor whose shape is the same as 'scores' but with
      the
  """
  if boxes.shape[-1] != 4:
    raise ValueError(
        'boxes.shape[1] is {:d}, but must be 4.'.format(boxes.shape[-1]))

  filtered_mask = scores > min_score_threshold
  filtered_scores = jnp.where(filtered_mask, scores, jnp.zeros_like(scores))
  filtered_boxes = jnp.expand_dims(
      filtered_mask, axis=-1).astype(boxes.dtype) * boxes

  return filtered_boxes, filtered_scores


def top_k_boxes(boxes,
                scores,
                k):
  """Sorts and selects top k boxes according to the scores.

  Args:
    boxes: A tensor of shape [batch_size, N, 4] representing the coordinate of
      the boxes. N is the number of boxes per image.
    scores: A tensor of shape [batch_size, N] representing the score of the
      boxes.
    k: An integer or a tensor indicating the top k number.

  Returns:
    selected_boxes: A tensor of shape [batch_size, k, 4] representing the
      selected top k box coordinates.
    selected_scores: A tensor of shape [batch_size, k] representing the selected
      top k box scores.
  """
  selected_scores, top_k_indices = lax.top_k(scores, k=k)

  batch_size, _ = scores.shape
  # preparing batch indices for Numpy-style gather
  batch_indices = jnp.tile(jnp.reshape(jnp.arange(batch_size),
                                       [batch_size, 1]), [1, k])
  selected_boxes = boxes[batch_indices, top_k_indices]

  return selected_boxes, selected_scores


def clip_boxes(boxes,
               image_shape):
  """Clips boxes to image boundaries. It's called from roi_ops.py.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    image_shape: a list of two integers, a two-element vector or a tensor such
      that all but the last dimensions are `broadcastable` to `boxes`. The last
      dimension is 2, which represents [height, width].

  Returns:
    clipped_boxes: a tensor whose shape is the same as `boxes` representing the
      clipped boxes in ymin, xmin, ymax, xmax order.

  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  if boxes.shape[-1] != 4:
    raise ValueError(
        'boxes.shape[-1] is {:d}, but must be 4.'.format(boxes.shape[-1]))

  if isinstance(image_shape, Sequence) or isinstance(image_shape, tuple):
    height, width = image_shape
    y_x_max_list = jnp.array(
        [height - 1.0, width - 1.0, height - 1.0, width - 1.0],
        dtype=boxes.dtype)
  else:
    image_shape = image_shape.astype(boxes.dtype)
    height = image_shape[Ellipsis, 0:1]
    width = image_shape[Ellipsis, 1:2]
    y_x_max_list = jnp.concatenate(
        [height - 1.0, width - 1.0, height - 1.0, width - 1.0], -1)

  return jnp.maximum(jnp.minimum(boxes, y_x_max_list), 0.0)


def encode_boxes(boxes,
                 anchors,
                 weights = None):
  """Encode boxes to targets.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    anchors: a tensor whose shape is the same as, or `broadcastable` to `boxes`,
      representing the coordinates of anchors in ymin, xmin, ymax, xmax order.
    weights: None or a list of four float numbers used to scale coordinates.

  Returns:
    encoded_boxes: a tensor whose shape is the same as `boxes` representing the
      encoded box targets.

  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  if boxes.shape[-1] != 4:
    raise ValueError(
        'boxes.shape[-1] is {:d}, but must be 4.'.format(boxes.shape[-1]))

  boxes = boxes.astype(anchors.dtype)
  ymin = boxes[Ellipsis, 0:1]
  xmin = boxes[Ellipsis, 1:2]
  ymax = boxes[Ellipsis, 2:3]
  xmax = boxes[Ellipsis, 3:4]
  box_h = ymax - ymin + 1.0
  box_w = xmax - xmin + 1.0
  box_yc = ymin + 0.5 * box_h
  box_xc = xmin + 0.5 * box_w

  anchor_ymin = anchors[Ellipsis, 0:1]
  anchor_xmin = anchors[Ellipsis, 1:2]
  anchor_ymax = anchors[Ellipsis, 2:3]
  anchor_xmax = anchors[Ellipsis, 3:4]
  anchor_h = anchor_ymax - anchor_ymin + 1.0
  anchor_w = anchor_xmax - anchor_xmin + 1.0
  anchor_yc = anchor_ymin + 0.5 * anchor_h
  anchor_xc = anchor_xmin + 0.5 * anchor_w

  encoded_dy = (box_yc - anchor_yc) / anchor_h
  encoded_dx = (box_xc - anchor_xc) / anchor_w
  encoded_dh = jnp.log(box_h / anchor_h)
  encoded_dw = jnp.log(box_w / anchor_w)
  if weights:
    encoded_dy *= weights[0]
    encoded_dx *= weights[1]
    encoded_dh *= weights[2]
    encoded_dw *= weights[3]

  encoded_boxes = jnp.concatenate(
      [encoded_dy, encoded_dx, encoded_dh, encoded_dw],
      axis=-1)
  return encoded_boxes


def decode_boxes(encoded_boxes,
                 anchors,
                 weights = None):
  """Decode boxes.

  Args:
    encoded_boxes: a tensor whose last dimension is 4 representing the
      coordinates of encoded boxes in ymin, xmin, ymax, xmax order.
    anchors: a tensor whose shape is the same as, or `broadcastable` to `boxes`,
      representing the coordinates of anchors in ymin, xmin, ymax, xmax order.
    weights: None or a list of four float numbers used to scale coordinates.

  Returns:
    encoded_boxes: a tensor whose shape is the same as `boxes` representing the
      decoded box targets.
  """
  if encoded_boxes.shape[-1] != 4:
    raise ValueError(
        'encoded_boxes.shape[-1] is {:d}, but must be 4.'
        .format(encoded_boxes.shape[-1]))

  encoded_boxes = encoded_boxes.astype(anchors.dtype)
  dy = encoded_boxes[Ellipsis, 0:1]
  dx = encoded_boxes[Ellipsis, 1:2]
  dh = encoded_boxes[Ellipsis, 2:3]
  dw = encoded_boxes[Ellipsis, 3:4]
  if weights:
    dy /= weights[0]
    dx /= weights[1]
    dh /= weights[2]
    dw /= weights[3]
  dh = jnp.minimum(dh, BBOX_XFORM_CLIP)
  dw = jnp.minimum(dw, BBOX_XFORM_CLIP)

  anchor_ymin = anchors[Ellipsis, 0:1]
  anchor_xmin = anchors[Ellipsis, 1:2]
  anchor_ymax = anchors[Ellipsis, 2:3]
  anchor_xmax = anchors[Ellipsis, 3:4]
  anchor_h = anchor_ymax - anchor_ymin + 1.0
  anchor_w = anchor_xmax - anchor_xmin + 1.0
  anchor_yc = anchor_ymin + 0.5 * anchor_h
  anchor_xc = anchor_xmin + 0.5 * anchor_w

  decoded_boxes_yc = dy * anchor_h + anchor_yc
  decoded_boxes_xc = dx * anchor_w + anchor_xc
  decoded_boxes_h = jnp.exp(dh) * anchor_h
  decoded_boxes_w = jnp.exp(dw) * anchor_w

  decoded_boxes_ymin = decoded_boxes_yc - 0.5 * decoded_boxes_h
  decoded_boxes_xmin = decoded_boxes_xc - 0.5 * decoded_boxes_w
  decoded_boxes_ymax = decoded_boxes_ymin + decoded_boxes_h - 1.0
  decoded_boxes_xmax = decoded_boxes_xmin + decoded_boxes_w - 1.0

  decoded_boxes = jnp.concatenate(
      [decoded_boxes_ymin, decoded_boxes_xmin,
       decoded_boxes_ymax, decoded_boxes_xmax],
      axis=-1)
  return decoded_boxes


def decode_boxes_lrtb(encoded_boxes_lrtb,
                      anchors,
                      weights = None):
  """Decode LRTB boxes.

  Args:
    encoded_boxes_lrtb: a tensor whose last dimension is 4 representing the
      coordinates of encoded boxes in left, right, top, bottom order.
    anchors: a tensor whose shape is the same as, or `broadcastable` to `boxes`,
      representing the coordinates of anchors in ymin, xmin, ymax, xmax order.
    weights: None or a list of four float numbers used to scale coordinates.
  Returns:
    decoded_boxes_lrtb: a tensor whose shape is the same as `boxes` representing
      the decoded box targets in lrtb (=left,right,top,bottom) format. The box
      decoded box coordinates represent the left, right, top, and bottom
      distances from an anchor location to the four borders of the matched
      groundtruth bounding box.
  """
  if encoded_boxes_lrtb.shape[-1] != 4:
    raise ValueError(
        'encoded_boxes_lrtb.shape[-1] is {:d}, but must be 4.'
        .format(encoded_boxes_lrtb.shape[-1]))

  encoded_boxes_lrtb = encoded_boxes_lrtb.astype(anchors.dtype)
  left = encoded_boxes_lrtb[Ellipsis, 0:1]
  right = encoded_boxes_lrtb[Ellipsis, 1:2]
  top = encoded_boxes_lrtb[Ellipsis, 2:3]
  bottom = encoded_boxes_lrtb[Ellipsis, 3:4]
  if weights:
    left /= weights[0]
    right /= weights[1]
    top /= weights[2]
    bottom /= weights[3]

  anchor_ymin = anchors[Ellipsis, 0:1]
  anchor_xmin = anchors[Ellipsis, 1:2]
  anchor_ymax = anchors[Ellipsis, 2:3]
  anchor_xmax = anchors[Ellipsis, 3:4]
  anchor_h = anchor_ymax - anchor_ymin
  anchor_w = anchor_xmax - anchor_xmin
  anchor_yc = anchor_ymin + 0.5 * anchor_h
  anchor_xc = anchor_xmin + 0.5 * anchor_w

  decoded_boxes_ymin = anchor_yc - top * anchor_h
  decoded_boxes_xmin = anchor_xc - left * anchor_w
  decoded_boxes_ymax = anchor_yc + bottom * anchor_h
  decoded_boxes_xmax = anchor_xc + right * anchor_w

  decoded_boxes_lrtb = jnp.concatenate(
      [decoded_boxes_ymin, decoded_boxes_xmin,
       decoded_boxes_ymax, decoded_boxes_xmax],
      axis=-1)
  return decoded_boxes_lrtb


def get_location_features(boxes,
                          image_shape,
                          add_relative_features = False,
                          top_k_relative_boxes = 5):
  """Return location features for boxes.

  For box i, the per-ROI feature fi =
  [y0 / H, x0 / W, y1 / H, x1 / W, h * w / (H * W)].
  , where H and W are the image size. The relative ROI features between box i
  and j are [(y0i - y0j) / H, (x0i - x0j) / W, (y1i - y1j) / H, (x1i - x1j) / W,
  hj * wj / (hi * wi)].

  Args:
    boxes: Boxes of shape [batch, num_rois, 4] representing the
      coordinates of boxes in ymin, xmin, ymax, xmax order in last dimension.
    image_shape: Image shape array of shape [batch, 2] representing
      [height, width] of the scaled image in the last dimension.
    add_relative_features: Bool to add relative ROI features or not.
    top_k_relative_boxes: Number of ROIs to use for relative features.

  Returns:
    location_features: Array of location features of each box,
      The shape is [batch, num_rois, 5 + top_k_relative_boxes * 5], where the
      first 5 elements of the last dimension are the ROI-specific features.
  """
  num_rois = boxes.shape[1]
  if add_relative_features and top_k_relative_boxes > num_rois:
    raise ValueError(f'Number of relative boxes must be smaller than ROIs:'
                     f'{top_k_relative_boxes} > {num_rois}')
  position_features = boxes / jnp.tile(
      image_shape[:, None, :], (1, 1, 2))
  area_features = (
      position_features[:, :, 2:3] - position_features[:, :, 0:1]) * (
          position_features[:, :, 3:4] - position_features[:, :, 1:2])

  if add_relative_features:
    relative_position_features = (
        position_features[:, :, None, :] -
        position_features[:, None, :top_k_relative_boxes, :])
    relative_position_features = relative_position_features.reshape(
        boxes.shape[:-1] + (4 * top_k_relative_boxes,))
    relative_area_features = (
        area_features[:, None, :top_k_relative_boxes, :] /
        (area_features[:, :, None, :] + _EPSILON))
    relative_area_features = relative_area_features.reshape(
        boxes.shape[:-1] + (top_k_relative_boxes,))
    location_features = jnp.concatenate(
        [position_features, area_features,
         relative_position_features, relative_area_features],
        axis=-1)
  else:
    location_features = jnp.concatenate([position_features, area_features],
                                        axis=-1)
  return location_features
