# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Implements a 3d instance segmentation model."""

import gin
import gin.tf
import tensorflow as tf
from tf3d import base_model
from tf3d import standard_fields
from tf3d.instance_segmentation import model_utils
from tf3d.layers import sparse_voxel_unet
from tf3d.utils import voxel_utils


@gin.configurable
class InstanceSegmentationModel(base_model.BaseModel):
  """3D instance segmentation model.

  An embedding vector is learned for each point which is used to group points
  into instances.
  """

  def __init__(self,
               num_classes,
               loss_names_to_functions=None,
               loss_names_to_weights=None,
               embedding_dims=64,
               embedding_similarity_strategy='distance',
               embedding_similarity_threshold=0.5,
               apply_nms=True,
               nms_score_threshold=0.1,
               nms_iou_threshold=0.3,
               num_furthest_voxel_samples=1000,
               sampler_score_vs_distance_coef=0.5,
               train_dir='/tmp/model/train',
               summary_log_freq=100):
    """An object detection model based on 3D UNet sparse voxel network.

    Args:
      num_classes: A int indicating the number of semantic classes to predict
        logits.
      loss_names_to_functions: A dictionary mapping loss names to
        loss functions.
      loss_names_to_weights: A dictionary mapping loss names to loss weights.
      embedding_dims: An integer determining per voxels embeddings with the
        specified dimensionality are added to the outputs dictionary.
      embedding_similarity_strategy: Defines the method for computing similarity
        between embedding vectors. Possible values are 'dotproduct'
        and 'distance'.
      embedding_similarity_threshold: Similarity threshold used to decide if two
        point embedding vectors belong to the same instance.
      apply_nms: If True, it will apply non-maximum suppression to the final
        predictions.
      nms_score_threshold: Score threshold used in non-maximum suppression.
      nms_iou_threshold: Intersection over union threshold used in
        non-maximum suppression.
      num_furthest_voxel_samples: Number of voxels to be sampled using furthest
        voxel sampling in the postprocessor.
      sampler_score_vs_distance_coef: The coefficient that balances the weight
        between furthest voxel sampling and highest score sampling in the
        postprocessor.
      train_dir: A directory path to write tensorboard summary for losses.
      summary_log_freq: A int of the frequency (as batches) to log summary.

    Returns:
      A dictionary containing tensors that contain predicted object properties.
    """
    super().__init__(
        loss_names_to_functions=loss_names_to_functions,
        loss_names_to_weights=loss_names_to_weights,
        train_dir=train_dir,
        summary_log_freq=summary_log_freq)

    self.num_classes = num_classes
    self.embedding_dims = embedding_dims
    self.embedding_similarity_strategy = embedding_similarity_strategy
    self.embedding_similarity_threshold = embedding_similarity_threshold
    self.apply_nms = apply_nms
    self.nms_score_threshold = nms_score_threshold
    self.nms_iou_threshold = nms_iou_threshold
    self.num_furthest_voxel_samples = num_furthest_voxel_samples
    self.sampler_score_vs_distance_coef = sampler_score_vs_distance_coef
    task_names_to_num_output_channels = {
        standard_fields.DetectionResultFields.object_semantic_voxels:
            num_classes,
        standard_fields.DetectionResultFields.instance_embedding_voxels:
            embedding_dims,
    }
    task_names_to_use_relu_last_conv = {
        standard_fields.DetectionResultFields.object_semantic_voxels: False,
        standard_fields.DetectionResultFields.instance_embedding_voxels: False,
    }
    task_names_to_use_batch_norm_in_last_layer = {}
    for key in task_names_to_num_output_channels:
      task_names_to_use_batch_norm_in_last_layer[key] = False
    self.sparse_conv_unet = sparse_voxel_unet.SparseConvUNet(
        task_names_to_num_output_channels=task_names_to_num_output_channels,
        task_names_to_use_relu_last_conv=task_names_to_use_relu_last_conv,
        task_names_to_use_batch_norm_in_last_layer=(
            task_names_to_use_batch_norm_in_last_layer))

  def __call__(self, inputs, training=True):
    """Runs the model and returns the semantic logits prediction.

    Args:
      inputs: A dictionary of tensors containing inputs for the model.
      training: Whether the model runs in training mode.

    Returns:
      A dictionary of tensors containing semantic logits prediction.
    """
    inputs[standard_fields.InputDataFields.num_valid_voxels] = tf.reshape(
        inputs[standard_fields.InputDataFields.num_valid_voxels], [-1])
    voxel_inputs = (inputs[standard_fields.InputDataFields.voxel_features],
                    inputs[standard_fields.InputDataFields.voxel_xyz_indices],
                    inputs[standard_fields.InputDataFields.num_valid_voxels])
    outputs = self.sparse_conv_unet(voxel_inputs, training=training)
    # If at eval time, transfer voxel features to points
    if ((not training) and
        (standard_fields.InputDataFields.points_to_voxel_mapping in inputs)):
      voxel_to_point_mapping = (
          standard_fields.get_output_voxel_to_point_field_mapping())
      point_tensor_outputs = {}
      for task_name in outputs:
        if task_name in voxel_to_point_mapping and outputs[
            task_name] is not None:
          point_tensor_outputs[voxel_to_point_mapping[task_name]] = (
              voxel_utils.sparse_voxel_grid_to_pointcloud(
                  voxel_features=outputs[task_name],
                  segment_ids=inputs[
                      standard_fields.InputDataFields.points_to_voxel_mapping],
                  num_valid_voxels=inputs[
                      standard_fields.InputDataFields.num_valid_voxels],
                  num_valid_points=inputs[
                      standard_fields.InputDataFields.num_valid_points]))
      outputs.update(point_tensor_outputs)

    model_utils.postprocess(
        inputs=inputs,
        outputs=outputs,
        is_training=training,
        num_furthest_voxel_samples=self.num_furthest_voxel_samples,
        sampler_score_vs_distance_coef=self.sampler_score_vs_distance_coef,
        embedding_similarity_strategy=self.embedding_similarity_strategy,
        embedding_similarity_threshold=self.embedding_similarity_threshold,
        score_threshold=self.nms_score_threshold,
        apply_nms=self.apply_nms,
        nms_iou_threshold=self.nms_iou_threshold)

    if training:
      self.calculate_losses(inputs=inputs, outputs=outputs)

    return outputs
