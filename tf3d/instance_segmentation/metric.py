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

"""Instance Segmentation metric."""
import gin
import gin.tf
import tensorflow as tf
from tf3d import base_ap_metric
from tf3d import standard_fields
from tf3d.utils import instance_segmentation_utils


@gin.configurable
class InstanceSegmentationMetric(base_ap_metric.BaseAPMetric):
  """Instance segmentation average precision metric."""

  def __init__(self,
               iou_threshold=0.5,
               num_classes=None,
               label_map=None,
               label_map_path=None,
               eval_prefix='eval',
               name='instance_segmentation_metric'):
    """Instance segmentation AP metric.

    Args:
      iou_threshold: Intersection-over-union threshold at which the average
        precision is computed.
      num_classes: Number of classes.
      label_map: A dictionary mapping label ids to label names.
      label_map_path: path to labelmap (could be None).
      eval_prefix: Prefix for eval name; separates scalar values in Tensorboard.
      name: class name.
    """
    super(InstanceSegmentationMetric, self).__init__(
        iou_threshold=iou_threshold,
        num_classes=num_classes,
        label_map=label_map,
        label_map_path=label_map_path,
        eval_prefix=eval_prefix,
        name=name)

  def update_state(self, inputs, outputs):
    """Function that updates the metric state at each example.

    Args:
      inputs: A dictionary containing input tensors.
      outputs: A dictionary containing output tensors.

    Returns:
      Update op.
    """
    detections_score = tf.reshape(
        outputs[standard_fields.DetectionResultFields.objects_score], [-1])
    detections_class = tf.reshape(
        outputs[standard_fields.DetectionResultFields.objects_class], [-1])
    num_detections = tf.shape(detections_score)[0]
    detections_instance_mask = tf.reshape(
        outputs[
            standard_fields.DetectionResultFields.instance_segments_voxel_mask],
        [num_detections, -1])
    gt_class = tf.reshape(inputs[standard_fields.InputDataFields.objects_class],
                          [-1])
    num_gt = tf.shape(gt_class)[0]
    gt_voxel_instance_ids = tf.reshape(
        inputs[standard_fields.InputDataFields.object_instance_id_voxels], [-1])
    gt_instance_masks = tf.transpose(
        tf.one_hot(gt_voxel_instance_ids - 1, depth=num_gt, dtype=tf.float32))
    for c in self.class_range:
      gt_mask_c = tf.equal(gt_class, c)
      num_gt_c = tf.math.reduce_sum(tf.cast(gt_mask_c, dtype=tf.int32))
      gt_instance_masks_c = tf.boolean_mask(gt_instance_masks, gt_mask_c)
      detections_mask_c = tf.equal(detections_class, c)
      num_detections_c = tf.math.reduce_sum(
          tf.cast(detections_mask_c, dtype=tf.int32))
      if num_detections_c == 0:
        continue
      det_scores_c = tf.boolean_mask(detections_score, detections_mask_c)
      det_instance_mask_c = tf.boolean_mask(detections_instance_mask,
                                            detections_mask_c)
      det_scores_c, sorted_indices = tf.math.top_k(
          det_scores_c, k=num_detections_c)
      det_instance_mask_c = tf.gather(det_instance_mask_c, sorted_indices)
      tp_c = tf.zeros([num_detections_c], dtype=tf.int32)
      if num_gt_c > 0:
        ious_c = instance_segmentation_utils.points_mask_iou(
            masks1=gt_instance_masks_c, masks2=det_instance_mask_c)
        max_overlap_gt_ids = tf.cast(
            tf.math.argmax(ious_c, axis=0), dtype=tf.int32)
        is_gt_box_detected = tf.zeros([num_gt_c], dtype=tf.int32)
        for i in tf.range(num_detections_c):
          gt_id = max_overlap_gt_ids[i]
          if (ious_c[gt_id, i] > self.iou_threshold and
              is_gt_box_detected[gt_id] == 0):
            tp_c = tf.maximum(
                tf.one_hot(i, num_detections_c, dtype=tf.int32), tp_c)
            is_gt_box_detected = tf.maximum(
                tf.one_hot(gt_id, num_gt_c, dtype=tf.int32), is_gt_box_detected)
      self.tp[c] = tf.concat([self.tp[c], tp_c], axis=0)
      self.scores[c] = tf.concat([self.scores[c], det_scores_c], axis=0)
      self.num_gt[c] += num_gt_c
    return tf.no_op()
