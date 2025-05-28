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

"""Transform functions for preprocessing."""
from typing import Any, Optional, Tuple

import tensorflow as tf


SizeTuple = Tuple[tf.Tensor, tf.Tensor]  # (height, width).
Self = Any

PADDING_VALUE = -1
PADDING_VALUE_STR = b""

NOTRACK_BOX = (0., 0., 0., 0.)  # No-track bounding box for padding.
NOTRACK_RBOX = (0., 0., 0., 0., 0.)  # No-track bounding rbox for padding.


def crop_or_pad_boxes(boxes, top, left, height,
                      width, h_orig, w_orig,
                      min_cropped_area = None):
  """Transforms the relative box coordinates according to the frame crop.

  Note that, if height/width are larger than h_orig/w_orig, this function
  implements the equivalent of padding.

  Args:
    boxes: Tensor of bounding boxes with shape (..., 4).
    top: Top of crop box in absolute pixel coordinates.
    left: Left of crop box in absolute pixel coordinates.
    height: Height of crop box in absolute pixel coordinates.
    width: Width of crop box in absolute pixel coordinates.
    h_orig: Original image height in absolute pixel coordinates.
    w_orig: Original image width in absolute pixel coordinates.
    min_cropped_area: If set, remove cropped boxes whose area relative to the
      original box is less than min_cropped_area or that covers the entire
      image.

  Returns:
    Boxes tensor with same shape as input boxes but updated values.
  """
  # Video track bound boxes: [num_instances, num_tracks, 4]
  # Image bounding boxes: [num_instances, 4]
  assert boxes.shape[-1] == 4
  seq_len = tf.shape(boxes)[0]
  not_padding = tf.reduce_any(tf.not_equal(boxes, PADDING_VALUE), axis=-1)
  has_tracks = len(boxes.shape) == 3
  if has_tracks:
    num_tracks = tf.shape(boxes)[1]
  else:
    assert len(boxes.shape) == 2
    num_tracks = 1

  # Transform the box coordinates.
  a = tf.cast(tf.stack([h_orig, w_orig]), tf.float32)
  b = tf.cast(tf.stack([top, left]), tf.float32)
  c = tf.cast(tf.stack([height, width]), tf.float32)
  boxes = tf.reshape(
      (tf.reshape(boxes, (seq_len, num_tracks, 2, 2)) * a - b) / c,
      (seq_len, num_tracks, len(NOTRACK_BOX)),
  )

  # Filter the valid boxes.
  areas_uncropped = tf.reduce_prod(
      tf.maximum(boxes[Ellipsis, 2:] - boxes[Ellipsis, :2], 0), axis=-1
  )
  boxes = tf.minimum(tf.maximum(boxes, 0.0), 1.0)
  if has_tracks:
    cond = tf.reduce_all((boxes[:, :, 2:] - boxes[:, :, :2]) > 0.0, axis=-1)
    boxes = tf.where(cond[:, :, tf.newaxis], boxes, NOTRACK_BOX)
    if min_cropped_area is not None:
      areas_cropped = tf.reduce_prod(
          tf.maximum(boxes[Ellipsis, 2:] - boxes[Ellipsis, :2], 0), axis=-1
      )
      boxes = tf.where(
          tf.logical_and(
              tf.reduce_max(areas_cropped, axis=0, keepdims=True)
              > min_cropped_area * areas_uncropped,
              tf.reduce_min(areas_cropped, axis=0, keepdims=True) < 1,
          )[Ellipsis, tf.newaxis],
          boxes,
          tf.constant(NOTRACK_BOX)[tf.newaxis, tf.newaxis],
      )
  else:
    boxes = tf.reshape(boxes, (seq_len, 4))
    # Image ops use `-1``, whereas video ops above use `NOTRACK_BOX`.
    boxes = tf.where(not_padding[Ellipsis, tf.newaxis], boxes, PADDING_VALUE)

  return boxes


def cxcywha_to_corners(cxcywha):
  """Convert [cx, cy, w, h, a] to four corners of [x, y].

  TF version of cxcywha_to_corners in
  third_party/py/scenic/model_lib/base_models/box_utils.py.

  Args:
    cxcywha: [..., 5]-tf.Tensor of [center-x, center-y, width, height, angle]
    representation of rotated boxes. Angle is in radians and center of rotation
    is defined by [center-x, center-y] point.

  Returns:
    [..., 4, 2]-tf.Tensor of four corners of the rotated box as [x, y] points.
  """
  assert cxcywha.shape[-1] == 5, "Expected [..., [cx, cy, w, h, a] input."
  bs = cxcywha.shape[:-1]
  cx, cy, w, h, a = tf.split(cxcywha, num_or_size_splits=5, axis=-1)
  xs = tf.constant([.5, .5, -.5, -.5]) * w
  ys = tf.constant([-.5, .5, .5, -.5]) * h
  pts = tf.stack([xs, ys], axis=-1)
  sin = tf.sin(a)
  cos = tf.cos(a)
  rot = tf.reshape(tf.concat([cos, -sin, sin, cos], axis=-1), (*bs, 2, 2))
  offset = tf.reshape(tf.concat([cx, cy], -1), (*bs, 1, 2))
  corners = pts @ rot + offset
  return corners


def corners_to_cxcywha(corners):
  """Convert four corners of [x, y] to [cx, cy, w, h, a].

  Args:
    corners: [..., 4, 2]-tf.Tensor of four corners of the rotated box as [x, y]
      points.

  Returns:
    [..., 5]-tf.Tensor of [center-x, center-y, width, height, angle]
    representation of rotated boxes. Angle is in radians and center of rotation
    is defined by [center-x, center-y] point.
  """
  assert corners.shape[-2] == 4 and corners.shape[-1] == 2, (
      "Expected [..., [cx, cy, w, h, a] input.")

  cornersx, cornersy = tf.unstack(corners, axis=-1)
  cx = tf.reduce_mean(cornersx, axis=-1)
  cy = tf.reduce_mean(cornersy, axis=-1)
  wcornersx = (
      cornersx[Ellipsis, 0] + cornersx[Ellipsis, 1] - cornersx[Ellipsis, 2] - cornersx[Ellipsis, 3])
  wcornersy = (
      cornersy[Ellipsis, 0] + cornersy[Ellipsis, 1] - cornersy[Ellipsis, 2] - cornersy[Ellipsis, 3])
  hcornersy = (-cornersy[Ellipsis, 0,] + cornersy[Ellipsis, 1] + cornersy[Ellipsis, 2] -
               cornersy[Ellipsis, 3])
  a = -tf.atan2(wcornersy, wcornersx)
  cos = tf.cos(a)
  w = wcornersx / (2 * cos)
  h = hcornersy / (2 * cos)
  cxcywha = tf.stack([cx, cy, w, h, a], axis=-1)

  return cxcywha
