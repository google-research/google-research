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

"""Generates detections from network predictions.

This is a JAX reimplementation of:
https://github.com/tensorflow/tpu/blob/master/models/official/detection/ops/postprocess_ops.py
"""

import functools
from typing import Dict, Tuple

import gin
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from ops import nms
from utils import box_utils

BBOX_XFORM_CLIP = np.log(1000. / 16.)
Array = jnp.ndarray


@functools.partial(jax.vmap, in_axes=[0, 0], out_axes=0)
def batch_gather(x, idx):
  """Performs a batched gather of the data.

  Args:
    x: A [batch, num_in, ...] JTensor of data to gather from.
    idx: A [batch, num_out] JTensor of dtype int32 or int64 specifying which
      elements to gather. Every value is expected to be in the range of [0,
      num_in].

  Returns:
    A [batch, num_out, ...] JTensor of gathered data.
  """
  return x[idx]


def generate_detections_vmap(
    class_outputs,
    box_outputs,
    pre_nms_num_detections = 5000,
    post_nms_num_detections = 100,
    nms_threshold = 0.3,
    score_threshold = 0.05,
    class_box_regression = True,
):
  """Generates the detections given anchor boxes and predictions.

  Args:
    class_outputs: An array with shape [batch, num_boxes, num_classes] of class
      logits for each box.
    box_outputs: An array with shape [batch, num_boxes, num_classes, 4] of
      predicted boxes in [ymin, xmin, ymax, xmax] order. Also accept num_classes
      = 1 for class agnostic box outputs.
    pre_nms_num_detections: An integer that specifies the number of candidates
      before NMS.
    post_nms_num_detections: An integer that specifies the number of candidates
      after NMS.
    nms_threshold: A float number to specify the IOU threshold of NMS.
    score_threshold: A float representing the threshold for deciding when to
      remove boxes based on score.
    class_box_regression: Whether to use class-specific box regression or not.
      Default True is to assume box_outputs are class-specific.

  Returns:
    A tuple of arrays corresponding to
      (box coordinates, object categories for each boxes, and box scores).
  """
  _, _, num_classes = jnp.shape(class_outputs)

  if not class_box_regression:
    if num_classes == 1:
      raise ValueError(
          'If using `class_box_regression=False` we expect num_classes = 1'
      )
    box_outputs = jnp.tile(box_outputs, [1, 1, num_classes, 1])

  box_outputs = box_outputs[:, :, 1:, :]
  box_scores = class_outputs[:, :, 1:]

  def batched_per_class_nms_fn(per_class_boxes, per_class_scores):
    # Transpose the data so the class dim is now a batch dim.
    per_class_boxes = jnp.transpose(per_class_boxes, (1, 0, 2))
    per_class_scores = jnp.transpose(per_class_scores, (1, 0))

    above_threshold = per_class_scores > score_threshold
    per_class_scores = jnp.where(
        above_threshold, per_class_scores, per_class_scores * 0 - 1
    )
    # Obtains pre_nms_num_boxes before running NMS.
    per_class_scores, indices = lax.top_k(
        per_class_scores,
        k=min(pre_nms_num_detections, per_class_scores.shape[-1]),
    )
    per_class_boxes = batch_gather(per_class_boxes, indices)

    # Run NMS where the [num_classes, ...] dim is the batch dim.
    nmsed_scores, nmsed_boxes = nms.non_max_suppression_padded(
        scores=per_class_scores,
        boxes=per_class_boxes,
        max_output_size=post_nms_num_detections,
        iou_threshold=nms_threshold,
    )

    nmsed_classes = jnp.ones([num_classes - 1, post_nms_num_detections])
    nmsed_classes *= jnp.arange(1, num_classes, dtype=jnp.int32)[:, None]

    nmsed_boxes = jnp.reshape(nmsed_boxes, [-1, 4])
    nmsed_scores = jnp.reshape(nmsed_scores, [-1])
    nmsed_classes = jnp.reshape(nmsed_classes, [-1])

    nmsed_scores, indices = lax.top_k(nmsed_scores, k=post_nms_num_detections)
    nmsed_boxes = nmsed_boxes[indices]
    nmsed_classes = nmsed_classes[indices]
    valid_detections = jnp.sum((nmsed_scores > 0.0).astype(jnp.int32))
    return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections

  batched_per_class_nms = jax.vmap(
      batched_per_class_nms_fn, in_axes=0, out_axes=0
  )

  final_boxes, final_scores, final_classes, valid_detections = (
      batched_per_class_nms(box_outputs, box_scores)
  )
  return final_boxes, final_scores, final_classes, valid_detections


def generate_detections(
    class_outputs,
    box_outputs,
    pre_nms_num_detections = 5000,
    post_nms_num_detections = 100,
    nms_threshold = 0.3,
    score_threshold = 0.05,
    class_box_regression = True,
):
  """Generates the detections given anchor boxes and predictions.

  Args:
    class_outputs: An array with shape [batch, num_boxes, num_classes] of
      class logits for each box.
    box_outputs: An array with shape [batch, num_boxes, num_classes, 4] of
      predicted boxes in [ymin, xmin, ymax, xmax] order. Also accept
      num_classes = 1 for class agnostic box outputs.
    pre_nms_num_detections: An integer that specifies the number of candidates
      before NMS.
    post_nms_num_detections: An integer that specifies the number of candidates
      after NMS.
    nms_threshold: A float number to specify the IOU threshold of NMS.
    score_threshold: A float representing the threshold for deciding when to
      remove boxes based on score.
    class_box_regression: Whether to use class-specific box regression or not.
      Default True is to assume box_outputs are class-specific.

  Returns:
    A tuple of arrays corresponding to
      (box coordinates, object categories for each boxes, and box scores).
  """
  batch_size, _, num_classes = jnp.shape(class_outputs)

  final_boxes = []
  final_scores = []
  final_classes = []
  all_valid = []
  for b in range(batch_size):
    nmsed_boxes = []
    nmsed_scores = []
    nmsed_classes = []
    # Skips the background class.
    for i in range(1, num_classes):
      box_idx = i if class_box_regression else 0
      boxes_i = box_outputs[b, :, box_idx]
      scores_i = class_outputs[b, :, i]
      # Filter by threshold.
      above_threshold = scores_i > score_threshold
      scores_i = jnp.where(above_threshold, scores_i, scores_i*0 - 1)

      # Obtains pre_nms_num_boxes before running NMS.
      scores_i, indices = lax.top_k(
          scores_i, k=min(pre_nms_num_detections, scores_i.shape[-1]))
      boxes_i = boxes_i[indices]

      nmsed_scores_i, nmsed_boxes_i = nms.non_max_suppression_padded(
          scores=scores_i[None, Ellipsis],
          boxes=boxes_i[None, Ellipsis],
          max_output_size=post_nms_num_detections,
          iou_threshold=nms_threshold)

      nmsed_classes_i = jnp.ones([post_nms_num_detections]) * i
      nmsed_boxes.append(nmsed_boxes_i[0])
      nmsed_scores.append(nmsed_scores_i[0])
      nmsed_classes.append(nmsed_classes_i)

    # Concats results from all classes and sort them.
    nmsed_boxes = jnp.concatenate(nmsed_boxes, axis=0)
    nmsed_scores = jnp.concatenate(nmsed_scores, axis=0)
    nmsed_classes = jnp.concatenate(nmsed_classes, axis=0)
    nmsed_scores, indices = lax.top_k(nmsed_scores, k=post_nms_num_detections)
    nmsed_boxes = nmsed_boxes[indices]
    nmsed_classes = nmsed_classes[indices]
    valid_detections = jnp.sum((nmsed_scores > 0.0).astype(jnp.int32))

    all_valid.append(valid_detections)
    final_classes.append(nmsed_classes)
    final_scores.append(nmsed_scores)
    final_boxes.append(nmsed_boxes)

  return (jnp.stack(final_boxes, axis=0), jnp.stack(final_scores, axis=0),
          jnp.stack(final_classes, axis=0), jnp.stack(all_valid, axis=0))


@gin.configurable
def process_and_generate_detections(
    box_outputs,
    class_outputs,
    anchor_boxes,
    image_shape,
    pre_nms_num_detections = 5000,
    post_nms_num_detections = 100,
    nms_threshold = 0.5,
    score_threshold = 0.05,
    class_box_regression = True,
    class_is_logit = True,
    use_vmap = False,
):
  """Generate final detections.

  Args:
    box_outputs: An array of shape of [batch_size, K, num_classes * 4]
      representing the class-specific box coordinates relative to anchors.
    class_outputs: An array of shape of [batch_size, K, num_classes]
      representing the class logits before applying score activiation.
    anchor_boxes: An array of shape of [batch_size, K, 4] representing the
      corresponding anchor boxes w.r.t `box_outputs`.
    image_shape: An array of shape of [batch_size, 2] storing the image height
      and width w.r.t. the scaled image, i.e. the same image space as
      `box_outputs` and `anchor_boxes`.
    pre_nms_num_detections: An integer that specifies the number of candidates
      before NMS.
    post_nms_num_detections: An integer that specifies the number of candidates
      after NMS.
    nms_threshold: A float number to specify the IOU threshold of NMS.
    score_threshold: A float representing the threshold for deciding when to
      remove boxes based on score.
    class_box_regression: Whether to use class-specific box regression or not.
      Default True is to assume box_outputs are class-specific.
    class_is_logit: Whether the class outputs are logits.
    use_vmap: Whether to use a vmapped version of the generate_detections
      functions. This is very helpful to speed up the compile time when the
      number of categories is large.

  Returns:
    A dictionary with the following key-value pairs:
      nmsed_boxes: `float` array of shape [batch_size, max_total_size, 4]
        representing top detected boxes in [y1, x1, y2, x2].
      nmsed_scores: `float` array of shape [batch_size, max_total_size]
        representing sorted confidence scores for detected boxes. The values are
        between [0, 1].
      nmsed_classes: `int` array of shape [batch_size, max_total_size]
        representing classes for detected boxes.
      valid_detections: `int` array of shape [batch_size] only the top
        `valid_detections` boxes are valid detections.
  """
  if class_is_logit:
    class_outputs = jax.nn.softmax(class_outputs, axis=-1)

  _, num_locations, num_classes = class_outputs.shape
  if class_box_regression:
    num_detections = num_locations * num_classes
    box_outputs = box_outputs.reshape(-1, num_detections, 4)
    anchor_boxes = jnp.tile(
        jnp.expand_dims(anchor_boxes, axis=2), [1, 1, num_classes, 1])
    anchor_boxes = anchor_boxes.reshape(-1, num_detections, 4)

  decoded_boxes = box_utils.decode_boxes(
      box_outputs, anchor_boxes, weights=[10.0, 10.0, 5.0, 5.0])
  decoded_boxes = box_utils.clip_boxes(decoded_boxes, image_shape[Ellipsis, :2])
  if class_box_regression:
    decoded_boxes = decoded_boxes.reshape(-1, num_locations, num_classes, 4)
  else:
    decoded_boxes = decoded_boxes[:, :, None, :]

  generate_detections_fn = (
      generate_detections_vmap
      if use_vmap
      else generate_detections
  )

  nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = (
      generate_detections_fn(
          class_outputs,
          decoded_boxes,
          pre_nms_num_detections,
          post_nms_num_detections,
          nms_threshold,
          score_threshold,
          class_box_regression,
      )
  )

  return {
      'num_detections': valid_detections,
      'detection_boxes': nmsed_boxes,
      'detection_classes': nmsed_classes,
      'detection_scores': nmsed_scores,
  }
