# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Utilities for bounding box manipulations."""
# TODO: b/483281963 - Utilise standard implementations for these functions.

from collections.abc import MutableSequence, Sequence
import torch


def box_cxcywh_to_xyxy(x):
  """Converts a bounding box from center-width-height format to xyxy format."""

  x_c, y_c, w, h = x.unbind(-1)
  return torch.stack(
      [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)],
      dim=-1,
  )


def box_xyxy_to_cxcywh(x):
  """Converts a bounding box from xyxy format to center-width-height format."""

  x0, y0, x1, y1 = x.unbind(-1)
  return torch.stack(
      [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)], dim=-1
  )


def coco_to_xyxy(coco_bbox):
  """Converts a COCO-format bbox [x, y, w, h] to [x1, y1, x2, y2]."""
  x, y, width, height = coco_bbox
  return [x, y, x + width, y + height]


def rescale_bboxes(
    out_bbox, size
):
  """Rescales bounding boxes from normalized coords to original pixel coords."""

  img_h, img_w = size
  b = box_cxcywh_to_xyxy(out_bbox)
  rescaled_b = b * torch.tensor(
      [img_w, img_h, img_w, img_h], dtype=torch.float32
  )
  return rescaled_b


def box_area(boxes):
  """Computes the area of a set of bounding boxes."""
  return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(
    *, boxes1, boxes2
):
  """Computes the intersection over union (IoU) between two sets of boxes.

  Args:
    boxes1: A tensor of shape (N, 4) containing N bounding boxes in [x1, y1, x2,
      y2] format.
    boxes2: A tensor of shape (M, 4) containing M bounding boxes in [x1, y1, x2,
      y2] format.

  Returns:
    A tuple (iou, union), where iou is a tensor of shape (N, M) with the IoU
    values and union is a tensor of shape (N, M) with the union areas.
  """
  area1 = box_area(boxes1)
  area2 = box_area(boxes2)

  lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  ## [N,M,2]
  rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  ## [N,M,2]

  wh = (rb - lt).clamp(min=0)  ## [N,M,2]
  inter = wh[:, :, 0] * wh[:, :, 1]  ## [N,M]

  union = area1[:, None] + area2 - inter

  return inter / union, union


def generalized_box_iou(
    *, boxes1, boxes2
):
  """Generalized IoU from https://giou.stanford.edu/."""
  assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
  assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
  iou, union = box_iou(boxes1=boxes1, boxes2=boxes2)

  lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
  rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

  wh = (rb - lt).clamp(min=0)  ## [N,M,2]
  area = wh[:, :, 0] * wh[:, :, 1]

  return iou - (area - union) / area


def is_valid_box(
    box, box_format = "xywh"
):
  """Checks if a bounding box is valid."""
  if len(box) != 4:
    return False

  if box_format == "xywh":
    _, _, w, h = box
    return w > 0 and h > 0
  elif box_format == "xyxy":
    x0, y0, x1, y1 = box
    return x1 > x0 and y1 > y0
  else:
    raise ValueError(f"Unsupported box format: {box_format}")

