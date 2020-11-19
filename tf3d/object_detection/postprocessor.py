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

"""3D Object Detection postprocessing functions."""

import tensorflow as tf
from tf3d import standard_fields
from tf3d.object_detection.box_utils import box_ops
from tf3d.utils import instance_sampling_utils


def _get_top_k_indices(outputs, k):
  max_score = tf.reduce_max(
      outputs[standard_fields.DetectionResultFields.objects_score], axis=1)
  num_boxes = tf.shape(max_score)[0]
  [_, indices] = tf.nn.top_k(max_score, k=tf.minimum(num_boxes, k))
  return indices


def _remove_low_score_boxes(outputs,
                            score_thresh,
                            min_num_boxes=100,
                            epsilon=0.0001):
  """remove the boxes that have an score lower than threshold.

  Args:
    outputs: Output dictionary.
    score_thresh: A float corresponding to score threshold.
    min_num_boxes: Minimum number of boxes.
    epsilon: A very small value.
  """
  max_scores = tf.reduce_max(
      outputs[standard_fields.DetectionResultFields.objects_score], axis=1)
  max_score = tf.reduce_max(max_scores)
  score_mask = tf.greater(max_scores,
                          tf.minimum(score_thresh, max_score - epsilon))
  high_score_indices = tf.cast(
      tf.reshape(tf.where(score_mask), [-1]), dtype=tf.int32)
  top_k_indices = _get_top_k_indices(outputs=outputs, k=min_num_boxes)
  high_score_indices = tf.cond(
      tf.shape(high_score_indices)[0] > min_num_boxes,
      lambda: high_score_indices, lambda: top_k_indices)

  for tensor_name in standard_fields.get_output_object_fields():
    if tensor_name in outputs and outputs[tensor_name] is not None:
      outputs[tensor_name] = tf.gather(outputs[tensor_name], high_score_indices)


def _keep_top_k_boxes(outputs, k):
  """Keeps the top k highest score boxes.

  Args:
    outputs: Output dictionary.
    k: Number of boxes.
  """
  indices = _get_top_k_indices(outputs, k)
  for tensor_name in standard_fields.get_output_object_fields():
    if tensor_name in outputs and outputs[tensor_name] is not None:
      outputs[tensor_name] = tf.gather(outputs[tensor_name], indices)


def _sample_furthest_voxels(outputs, num_furthest_voxel_samples,
                            sampler_score_vs_distance_coef):
  """Samples voxels based on distance and scores."""
  num_furthest_voxel_samples = tf.minimum(
      num_furthest_voxel_samples,
      tf.shape(
          outputs[standard_fields.DetectionResultFields.objects_center])[0])
  _, seed_indices = instance_sampling_utils.sample_based_on_scores_and_distances(
      inputs=outputs[standard_fields.DetectionResultFields.objects_center],
      scores=tf.reduce_max(
          outputs[standard_fields.DetectionResultFields.objects_score], axis=1),
      num_samples=num_furthest_voxel_samples,
      scores_coef=sampler_score_vs_distance_coef)
  for tensor_name in standard_fields.get_output_object_fields():
    if tensor_name in outputs and outputs[tensor_name] is not None:
      outputs[tensor_name] = tf.gather(outputs[tensor_name], seed_indices)


def postprocess(outputs,
                score_thresh,
                iou_thresh,
                max_output_size,
                use_furthest_voxel_sampling=True,
                num_furthest_voxel_samples=1000,
                sampler_score_vs_distance_coef=2.0,
                apply_nms=True,
                max_num_boxes_before_nms=4000):
  """Postprocess the outputs of our network, including untiling.

  Args:
    outputs: A dict of `Tensor` objects with network outputs.
    score_thresh: scalar threshold for score (low scoring boxes are removed).
    iou_thresh: scalar threshold for IOU (boxes that that high IOU overlap with
      previously selected boxes are removed).
    max_output_size: maximum number of retained boxes per class.
    use_furthest_voxel_sampling: If True, it will sample voxels based on
      their scores and distance from each other.
    num_furthest_voxel_samples: Number of voxels to be sampled using furthest
      voxel sampling.
    sampler_score_vs_distance_coef: The coefficient that balances the weight
      between furthest voxel sampling and highest score sampling.
    apply_nms: If True, performs NMS.
    max_num_boxes_before_nms: Maximum number of boxes to keep before NMS.

  Returns:
    outputs: Our dict of `Tensor` objects with outputs post processed.
  """
  # Softmax semantic scores, removing backgorund scores
  outputs[standard_fields.DetectionResultFields.objects_score] = tf.nn.softmax(
      outputs[standard_fields.DetectionResultFields.objects_score])
  outputs[standard_fields.DetectionResultFields.objects_score] = outputs[
      standard_fields.DetectionResultFields.objects_score][:, 1:]

  # Remove low score boxes
  _remove_low_score_boxes(outputs=outputs, score_thresh=score_thresh)

  if use_furthest_voxel_sampling:
    _sample_furthest_voxels(outputs, num_furthest_voxel_samples,
                            sampler_score_vs_distance_coef)
  else:
    # Keep only top k before NMS
    _keep_top_k_boxes(outputs=outputs, k=max_num_boxes_before_nms)

  if apply_nms:
    # Non maximum suppression
    (outputs[standard_fields.DetectionResultFields.objects_length],
     outputs[standard_fields.DetectionResultFields.objects_height],
     outputs[standard_fields.DetectionResultFields.objects_width],
     outputs[standard_fields.DetectionResultFields.objects_center],
     outputs[standard_fields.DetectionResultFields.objects_rotation_matrix],
     outputs[standard_fields.DetectionResultFields.objects_class],
     outputs[standard_fields.DetectionResultFields.objects_score]
    ) = box_ops.nms(
        boxes_length=outputs[
            standard_fields.DetectionResultFields.objects_length],
        boxes_height=outputs[
            standard_fields.DetectionResultFields.objects_height],
        boxes_width=outputs[
            standard_fields.DetectionResultFields.objects_width],
        boxes_center=outputs[
            standard_fields.DetectionResultFields.objects_center],
        boxes_rotation_matrix=outputs[
            standard_fields.DetectionResultFields.objects_rotation_matrix],
        boxes_score=outputs[
            standard_fields.DetectionResultFields.objects_score],
        score_thresh=score_thresh,
        iou_thresh=iou_thresh,
        max_output_size=max_output_size)
  else:
    outputs[standard_fields.DetectionResultFields.objects_class] = tf.cast(
        tf.expand_dims(
            tf.math.argmax(
                outputs[standard_fields.DetectionResultFields.objects_score],
                axis=-1),
            axis=-1),
        dtype=tf.int32)
    outputs[standard_fields.DetectionResultFields
            .objects_score] = tf.math.reduce_max(
                outputs[standard_fields.DetectionResultFields.objects_score],
                axis=-1,
                keepdims=True)
  outputs[standard_fields.DetectionResultFields.objects_class] += 1
