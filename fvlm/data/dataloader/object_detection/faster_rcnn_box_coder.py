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

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Faster RCNN box coder.

Faster RCNN box coder follows the coding schema described below:
  ty = (y - ya) / ha
  tx = (x - xa) / wa
  th = log(h / ha)
  tw = log(w / wa)
  where x, y, w, h denote the box's center coordinates, width and height
  respectively. Similarly, xa, ya, wa, ha denote the anchor's center
  coordinates, width and height. tx, ty, tw and th denote the anchor-encoded
  center, width and height respectively.

  See http://arxiv.org/abs/1506.01497 for details.
"""

from data.dataloader.object_detection import box_coder
from data.dataloader.object_detection import box_list
import tensorflow.compat.v1 as tf

EPSILON = 1e-8


class FasterRcnnBoxCoder(box_coder.BoxCoder):
  """Faster RCNN box coder."""

  def __init__(self, scale_factors=None):
    """Constructor for FasterRcnnBoxCoder.

    Args:
      scale_factors: List of 4 positive scalars to scale ty, tx, th and tw.
        If set to None, does not perform scaling. For Faster RCNN,
        the open-source implementation recommends using [10.0, 10.0, 5.0, 5.0].
    """
    if scale_factors:
      assert len(scale_factors) == 4
      for scalar in scale_factors:
        assert scalar > 0
    self._scale_factors = scale_factors

  @property
  def code_size(self):
    return 4

  def _encode(self, boxes, anchors):
    """Encode a box collection with respect to anchor collection.

    Args:
      boxes: BoxList holding N boxes to be encoded.
      anchors: BoxList of anchors.

    Returns:
      a tensor representing N anchor-encoded boxes of the format
      [ty, tx, th, tw].
    """
    # Convert anchors to the center coordinate representation.
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    ycenter, xcenter, h, w = boxes.get_center_coordinates_and_sizes()
    # Avoid NaN in division and log below.
    ha += EPSILON
    wa += EPSILON
    h += EPSILON
    w += EPSILON

    tx = (xcenter - xcenter_a) / wa
    ty = (ycenter - ycenter_a) / ha
    tw = tf.log(w / wa)
    th = tf.log(h / ha)
    # Scales location targets as used in paper for joint training.
    if self._scale_factors:
      ty *= self._scale_factors[0]
      tx *= self._scale_factors[1]
      th *= self._scale_factors[2]
      tw *= self._scale_factors[3]
    return tf.transpose(tf.stack([ty, tx, th, tw]))

  def _decode(self, rel_codes, anchors):
    """Decode relative codes to boxes.

    Args:
      rel_codes: a tensor representing N anchor-encoded boxes.
      anchors: BoxList of anchors.

    Returns:
      boxes: BoxList holding N bounding boxes.
    """
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()

    ty, tx, th, tw = tf.unstack(tf.transpose(rel_codes))
    if self._scale_factors:
      ty /= self._scale_factors[0]
      tx /= self._scale_factors[1]
      th /= self._scale_factors[2]
      tw /= self._scale_factors[3]
    w = tf.exp(tw) * wa
    h = tf.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.
    return box_list.BoxList(tf.transpose(tf.stack([ymin, xmin, ymax, xmax])))


class LRTBBoxCoder(box_coder.BoxCoder):
  """Left-right-top-bottom format box coder.

  It encodes the distance between anchor locations and the left, right, top,
  bottom boundaries of the box.
  """

  def __init__(self, normalizer=1.0):
    """Constructor for LRTBBoxCoder."""
    self.normalizer = normalizer

  @property
  def code_size(self):
    return 4

  def _encode(self, boxes, anchors):
    """Encode a box collection with respect to anchor collection.

    Args:
      boxes: BoxList holding N boxes to be encoded.
      anchors: BoxList of anchors.

    Returns:
      center_targets: a centerness target of N boxes.
      lrtb_targets: a tensor representing N anchor-encoded boxes of the format
        [left, right, top, bottom].
    """
    # Convert anchors to the center coordinate representation.
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    _, _, h, w, ymin, xmin, ymax, xmax = (
        boxes.get_center_coordinates_and_sizes_and_corners())

    # Avoid NaN in division and log below.
    ha += EPSILON
    wa += EPSILON
    h += EPSILON
    w += EPSILON

    left = (xcenter_a - xmin) / wa
    right = (xmax - xcenter_a) / wa
    top = (ycenter_a - ymin) / ha
    bottom = (ymax - ycenter_a) / ha

    left /= self.normalizer
    right /= self.normalizer
    top /= self.normalizer
    bottom /= self.normalizer

    # Create lrtb targets.
    lrtb_targets = tf.transpose(tf.stack([left, right, top, bottom]))
    valid_match = tf.greater(tf.reduce_min(lrtb_targets, -1), 0.0)
    lrtb_targets = tf.where(
        valid_match, lrtb_targets, tf.zeros_like(lrtb_targets))

    # Centerness score.
    left_right = tf.stack([left, right], axis=-1)

    left_right = tf.where(tf.stack([valid_match, valid_match], -1),
                          left_right, tf.zeros_like(left_right))
    top_bottom = tf.stack([top, bottom], axis=-1)
    top_bottom = tf.where(tf.stack([valid_match, valid_match], -1),
                          top_bottom, tf.zeros_like(top_bottom))
    center_targets = tf.sqrt(
        (tf.reduce_min(left_right, -1) /
         (tf.reduce_max(left_right, -1) + EPSILON)) *
        (tf.reduce_min(top_bottom, -1) /
         (tf.reduce_max(top_bottom, -1) + EPSILON)))
    center_targets = tf.where(valid_match,
                              center_targets,
                              tf.zeros_like(center_targets))
    return center_targets, lrtb_targets

  def _decode(self, rel_codes, anchors):
    """Decode relative codes to boxes.

    Args:
      rel_codes: a tensor representing N anchor-encoded boxes.
      anchors: BoxList of anchors.

    Returns:
      boxes: BoxList holding N bounding boxes.
    """
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()

    left, right, top, bottom = tf.unstack(tf.transpose(rel_codes))
    left *= self.normalizer
    right *= self.normalizer
    top *= self.normalizer
    bottom *= self.normalizer

    left *= wa
    right *= wa
    top *= ha
    bottom *= ha

    ymin = ycenter_a - top
    xmin = xcenter_a - left
    ymax = ycenter_a - bottom
    xmax = xcenter_a - right
    return box_list.BoxList(tf.transpose(tf.stack([ymin, xmin, ymax, xmax])))
