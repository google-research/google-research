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

"""Non-max Suppression example.

This script does non-max suppression used in models like SSD
"""

from jax import lax
import jax.numpy as jnp
import numpy as np

_NMS_TILE_SIZE = 256


def _bbox_overlap(boxes, gt_boxes):
  """Find Bounding box overlap.

  Args:
    boxes: first set of bounding boxes
    gt_boxes: second set of boxes to compute IOU

  Returns:
    iou: Intersection over union matrix of all input bounding boxes
  """
  bb_y_min, bb_x_min, bb_y_max, bb_x_max = jnp.split(
      ary=boxes, indices_or_sections=4, axis=2)
  gt_y_min, gt_x_min, gt_y_max, gt_x_max = jnp.split(
      ary=gt_boxes, indices_or_sections=4, axis=2)

  # Calculates the intersection area.
  i_xmin = jnp.maximum(bb_x_min, jnp.transpose(gt_x_min, [0, 2, 1]))
  i_xmax = jnp.minimum(bb_x_max, jnp.transpose(gt_x_max, [0, 2, 1]))
  i_ymin = jnp.maximum(bb_y_min, jnp.transpose(gt_y_min, [0, 2, 1]))
  i_ymax = jnp.minimum(bb_y_max, jnp.transpose(gt_y_max, [0, 2, 1]))
  i_area = jnp.maximum((i_xmax - i_xmin), 0) * jnp.maximum((i_ymax - i_ymin), 0)

  # Calculates the union area.
  bb_area = (bb_y_max - bb_y_min) * (bb_x_max - bb_x_min)
  gt_area = (gt_y_max - gt_y_min) * (gt_x_max - gt_x_min)
  # Adds a small epsilon to avoid divide-by-zero.
  u_area = bb_area + jnp.transpose(gt_area, [0, 2, 1]) - i_area + 1e-8

  # Calculates IoU.
  iou = i_area / u_area

  return iou


def _self_suppression(in_args):
  iou, _, iou_sum = in_args
  batch_size = iou.shape[0]
  can_suppress_others = jnp.reshape(
      jnp.max(iou, 1) <= 0.5, [batch_size, -1, 1]).astype(iou.dtype)
  iou_suppressed = jnp.reshape(
      (jnp.max(can_suppress_others * iou, 1) <= 0.5).astype(iou.dtype),
      [batch_size, -1, 1]) * iou
  iou_sum_new = jnp.sum(iou_suppressed, [1, 2])
  return iou_suppressed, jnp.any(iou_sum - iou_sum_new > 0.5), iou_sum_new


def _cross_suppression(in_args):
  boxes, box_slice, iou_threshold, inner_idx = in_args
  batch_size = boxes.shape[0]
  new_slice = lax.dynamic_slice(boxes, [0, inner_idx * _NMS_TILE_SIZE, 0],
                                [batch_size, _NMS_TILE_SIZE, 4])
  iou = _bbox_overlap(new_slice, box_slice)
  ret_slice = jnp.expand_dims(
      (jnp.all(iou < iou_threshold, [1])).astype(box_slice.dtype),
      2) * box_slice
  return boxes, ret_slice, iou_threshold, inner_idx + 1


def _suppression_loop_body(in_args):
  """Process boxes in the range [idx*_NMS_TILE_SIZE, (idx+1)*_NMS_TILE_SIZE).

  Args:
    in_args: A tuple of arguments: boxes, iou_threshold, output_size, idx

  Returns:
    boxes: updated boxes.
    iou_threshold: pass down iou_threshold to the next iteration.
    output_size: the updated output_size.
    idx: the updated induction variable.
  """
  boxes, iou_threshold, output_size, idx = in_args
  num_tiles = boxes.shape[1] // _NMS_TILE_SIZE
  batch_size = boxes.shape[0]

  # Iterates over tiles that can possibly suppress the current tile.
  box_slice = lax.dynamic_slice(boxes, [0, idx * _NMS_TILE_SIZE, 0],
                                [batch_size, _NMS_TILE_SIZE, 4])
  def _loop_cond(in_args):
    _, _, _, inner_idx = in_args
    return inner_idx < idx

  _, box_slice, _, _ = lax.while_loop(
      _loop_cond,
      _cross_suppression, (boxes, box_slice, iou_threshold,
                           0))

  # Iterates over the current tile to compute self-suppression.
  iou = _bbox_overlap(box_slice, box_slice)
  mask = jnp.expand_dims(
      jnp.reshape(jnp.arange(_NMS_TILE_SIZE), [1, -1]) > jnp.reshape(
          jnp.arange(_NMS_TILE_SIZE), [-1, 1]), 0)
  iou *= (jnp.logical_and(mask, iou >= iou_threshold)).astype(iou.dtype)

  def _loop_cond2(in_args):
    _, loop_condition, _ = in_args
    return loop_condition

  suppressed_iou, _, _ = lax.while_loop(
      _loop_cond2, _self_suppression,
      (iou, True,
       jnp.sum(iou, [1, 2])))
  suppressed_box = jnp.sum(suppressed_iou, 1) > 0
  box_slice *= jnp.expand_dims(1.0 - suppressed_box.astype(box_slice.dtype), 2)

  # Uses box_slice to update the input boxes.
  mask = jnp.reshape(
      (jnp.equal(jnp.arange(num_tiles), idx)).astype(boxes.dtype),
      [1, -1, 1, 1])
  boxes = jnp.tile(jnp.expand_dims(
      box_slice, 1), [1, num_tiles, 1, 1]) * mask + jnp.reshape(
          boxes, [batch_size, num_tiles, _NMS_TILE_SIZE, 4]) * (1 - mask)
  boxes = jnp.reshape(boxes, [batch_size, -1, 4])

  # Updates output_size.
  output_size += jnp.sum(
      jnp.any(box_slice > 0, [2]).astype(jnp.int32), [1])
  return boxes, iou_threshold, output_size, idx + 1


def non_max_suppression_padded(scores,
                               boxes,
                               max_output_size,
                               iou_threshold):
  """A wrapper that handles non-maximum suppression.

  Assumption:
    * The boxes are sorted by scores unless the box is a dot (all coordinates
      are zero).
    * Boxes with higher scores can be used to suppress boxes with lower scores.

  The overal design of the algorithm is to handle boxes tile-by-tile:

  boxes = boxes.pad_to_multiply_of(tile_size)
  num_tiles = len(boxes) // tile_size
  output_boxes = []
  for i in range(num_tiles):
    box_tile = boxes[i*tile_size : (i+1)*tile_size]
    for j in range(i - 1):
      suppressing_tile = boxes[j*tile_size : (j+1)*tile_size]
      iou = _bbox_overlap(box_tile, suppressing_tile)
      # if the box is suppressed in iou, clear it to a dot
      box_tile *= _update_boxes(iou)
    # Iteratively handle the diagnal tile.
    iou = _box_overlap(box_tile, box_tile)
    iou_changed = True
    while iou_changed:
      # boxes that are not suppressed by anything else
      suppressing_boxes = _get_suppressing_boxes(iou)
      # boxes that are suppressed by suppressing_boxes
      suppressed_boxes = _get_suppressed_boxes(iou, suppressing_boxes)
      # clear iou to 0 for boxes that are suppressed, as they cannot be used
      # to suppress other boxes any more
      new_iou = _clear_iou(iou, suppressed_boxes)
      iou_changed = (new_iou != iou)
      iou = new_iou
    # remaining boxes that can still suppress others, are selected boxes.
    output_boxes.append(_get_suppressing_boxes(iou))
    if len(output_boxes) >= max_output_size:
      break

  Args:
    scores: a tensor with a shape of [batch_size, anchors].
    boxes: a tensor with a shape of [batch_size, anchors, 4].
    max_output_size: a scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non max suppression.
    iou_threshold: a float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
  Returns:
    nms_scores: a tensor with a shape of [batch_size, max_output_size].
      It has the same dtype as input scores.
    nms_proposals: a tensor with a shape of [batch_size, max_output_size, 4].
      It has the same dtype as input boxes.
  """
  batch_size = boxes.shape[0]
  num_boxes = boxes.shape[1]
  pad = int(np.ceil(float(num_boxes) / _NMS_TILE_SIZE)
           ) * _NMS_TILE_SIZE - num_boxes
  boxes = jnp.pad(boxes.astype(jnp.float32), [[0, 0], [0, pad], [0, 0]])
  scores = jnp.pad(scores.astype(jnp.float32), [[0, 0], [0, pad]])
  num_boxes += pad

  def _loop_cond(in_args):
    unused_boxes, unused_threshold, output_size, idx = in_args
    return jnp.logical_and(
        jnp.min(output_size) < max_output_size,
        idx < num_boxes // _NMS_TILE_SIZE)

  selected_boxes, _, output_size, _ = lax.while_loop(
      _loop_cond, _suppression_loop_body, (
          boxes, iou_threshold,
          jnp.zeros([batch_size], jnp.int32),
          0
      ))
  idx = num_boxes - lax.top_k(
      jnp.any(selected_boxes > 0, [2]).astype(jnp.int32) *
      jnp.expand_dims(jnp.arange(num_boxes, 0, -1), 0),
      max_output_size)[0].astype(jnp.int32)
  idx = jnp.minimum(idx, num_boxes - 1)
  idx = jnp.reshape(
      idx + jnp.reshape(jnp.arange(batch_size) * num_boxes, [-1, 1]), [-1])
  boxes = jnp.reshape(
      (jnp.reshape(boxes, [-1, 4]))[idx],
      [batch_size, max_output_size, 4])
  boxes = boxes * (
      jnp.reshape(jnp.arange(max_output_size), [1, -1, 1]) < jnp.reshape(
          output_size, [-1, 1, 1])).astype(boxes.dtype)
  scores = jnp.reshape(
      jnp.reshape(scores, [-1, 1])[idx],
      [batch_size, max_output_size])
  scores = scores * (
      jnp.reshape(jnp.arange(max_output_size), [1, -1]) < jnp.reshape(
          output_size, [-1, 1])).astype(scores.dtype)
  return scores, boxes
