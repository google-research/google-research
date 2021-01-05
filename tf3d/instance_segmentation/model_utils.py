# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Instance segmentation model utility functions."""

import tensorflow as tf
from tf3d import standard_fields
from tf3d.instance_segmentation import postprocessor
from tf3d.utils import mask_utils
from tf3d.utils import voxel_utils


def mask_valid_voxels(inputs, outputs):
  """Mask the voxels that are valid and in image view."""
  valid_mask = mask_utils.num_voxels_mask(inputs=inputs)
  mask_utils.apply_mask_to_output_voxel_tensors(
      outputs=outputs, valid_mask=valid_mask)


def mask_valid_points(inputs, outputs):
  """Mask the voxels that are valid and in image view."""
  valid_mask = mask_utils.num_points_mask(inputs=inputs)
  mask_utils.apply_mask_to_output_point_tensors(
      outputs=outputs, valid_mask=valid_mask)


def postprocess(inputs, outputs, is_training, num_furthest_voxel_samples,
                sampler_score_vs_distance_coef, embedding_similarity_strategy,
                embedding_similarity_threshold, score_threshold, apply_nms,
                nms_iou_threshold):
  """Post-processor function.

  Args:
    inputs: A dictionary containing input tensors.
    outputs: A dictionary containing predicted tensors.
    is_training: If during training stage or not.
    num_furthest_voxel_samples: Number of voxels to be sampled using furthest
      voxel sampling in the postprocessor.
    sampler_score_vs_distance_coef: The coefficient that balances the weight
      between furthest voxel sampling and highest score sampling in the
      postprocessor.
    embedding_similarity_strategy: Embedding similarity strategy.
    embedding_similarity_threshold: Similarity threshold used to decide if two
      point embedding vectors belong to the same instance.
    score_threshold: Instance score threshold used throughout postprocessing.
    apply_nms: If True, it will apply non-maximum suppression to the final
      predictions.
    nms_iou_threshold: Intersection over union threshold used in non-maximum
      suppression.
  """
  if not is_training:

    # Squeeze output voxel properties.
    for key in standard_fields.get_output_voxel_fields():
      if key in outputs and outputs[key] is not None:
        outputs[key] = tf.squeeze(outputs[key], axis=0)

    # Squeeze output point properties.
    for key in standard_fields.get_output_point_fields():
      if key in outputs and outputs[key] is not None:
        outputs[key] = tf.squeeze(outputs[key], axis=0)

    # Squeeze output object properties.
    for key in standard_fields.get_output_object_fields():
      if key in outputs and outputs[key] is not None:
        outputs[key] = tf.squeeze(outputs[key], axis=0)

    # Mask the valid voxels
    mask_valid_voxels(inputs=inputs, outputs=outputs)

    # Mask the valid points
    mask_valid_points(inputs=inputs, outputs=outputs)

    # NMS
    postprocessor.postprocess(
        outputs=outputs,
        num_furthest_voxel_samples=num_furthest_voxel_samples,
        sampler_score_vs_distance_coef=sampler_score_vs_distance_coef,
        embedding_similarity_strategy=embedding_similarity_strategy,
        embedding_similarity_threshold=embedding_similarity_threshold,
        apply_nms=apply_nms,
        nms_score_threshold=score_threshold,
        nms_iou_threshold=nms_iou_threshold)

    # Add instance segment point masks at eval time
    if standard_fields.InputDataFields.points_to_voxel_mapping in inputs:
      instance_segments_point_mask = (
          voxel_utils.sparse_voxel_grid_to_pointcloud(
              voxel_features=tf.expand_dims(
                  tf.transpose(outputs[standard_fields.DetectionResultFields
                                       .instance_segments_voxel_mask]),
                  axis=0),
              segment_ids=inputs[
                  standard_fields.InputDataFields.points_to_voxel_mapping],
              num_valid_voxels=inputs[
                  standard_fields.InputDataFields.num_valid_voxels],
              num_valid_points=inputs[
                  standard_fields.InputDataFields.num_valid_points]))
      outputs[standard_fields.DetectionResultFields
              .instance_segments_point_mask] = tf.transpose(
                  tf.squeeze(instance_segments_point_mask, axis=0))
