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

"""Implements a 3d object detection model."""

import gin
import gin.tf
import tensorflow as tf
from tf3d import base_model
from tf3d import standard_fields
from tf3d.layers import sparse_voxel_hourglass
from tf3d.object_detection import model_utils
from tf3d.utils import voxel_utils


@gin.configurable
class ObjectDetectionModel(base_model.BaseModel):
  """3D object detection model.

  Please refer to the following paper for more details:
  M. Najibi, G. Lai, A. Kundu, Z. Lu, V. Rathod, T. Funkhouser, C. Pantofaru,
  D. Ross, L. S. Davis, A. Fathi,
  'DOPS: Learning to Detect 3D Objects and Predict Their 3D Shapes', CVPR 2020.
  """

  def __init__(self,
               num_classes,
               loss_names_to_functions=None,
               loss_names_to_weights=None,
               predict_rotation_x=False,
               predict_rotation_y=False,
               predict_rotation_z=True,
               apply_nms=True,
               nms_score_threshold=0.1,
               nms_iou_threshold=0.3,
               nms_max_num_predicted_boxes=200,
               use_furthest_voxel_sampling=False,
               num_furthest_voxel_samples=1000,
               sampler_score_vs_distance_coef=10.0,
               train_dir='/tmp/model/train',
               summary_log_freq=100):
    """An object detection model based on 3D UNet sparse voxel network.

    Args:
      num_classes: A int indicating the number of semantic classes to predict
        logits.
      loss_names_to_functions: A dictionary mapping loss names to
        loss functions.
      loss_names_to_weights: A dictionary mapping loss names to loss weights.
      predict_rotation_x: If True, it will predict rotation cos and sin
        around x.
      predict_rotation_y: If True, it will predict rotation cos and sin
        around y.
      predict_rotation_z: If True, it will predict rotation cos and sin
        around z.
      apply_nms: If True, it will apply non-maximum suppression to the final
        predictions.
      nms_score_threshold: Score threshold used in non-maximum suppression.
      nms_iou_threshold: Intersection over union threshold used in
        non-maximum suppression.
      nms_max_num_predicted_boxes: Maximum number of predicted boxes after
        non-maximum suppression.
      use_furthest_voxel_sampling: If True, postprocessor will sample voxels
        based on their scores and distance from each other.
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

    self.apply_nms = apply_nms
    self.nms_score_threshold = nms_score_threshold
    self.nms_iou_threshold = nms_iou_threshold
    self.nms_max_num_predicted_boxes = nms_max_num_predicted_boxes
    self.use_furthest_voxel_sampling = use_furthest_voxel_sampling
    self.num_furthest_voxel_samples = num_furthest_voxel_samples
    self.sampler_score_vs_distance_coef = sampler_score_vs_distance_coef
    task_names_to_num_output_channels = {
        standard_fields.DetectionResultFields.object_length_voxels:
            1,
        standard_fields.DetectionResultFields.object_height_voxels:
            1,
        standard_fields.DetectionResultFields.object_width_voxels:
            1,
        standard_fields.DetectionResultFields.object_center_voxels:
            3,
        standard_fields.DetectionResultFields.object_weight_voxels:
            1,
        standard_fields.DetectionResultFields.object_semantic_voxels:
            num_classes,
    }
    task_names_to_use_relu_last_conv = {
        standard_fields.DetectionResultFields.object_length_voxels: True,
        standard_fields.DetectionResultFields.object_height_voxels: True,
        standard_fields.DetectionResultFields.object_width_voxels: True,
        standard_fields.DetectionResultFields.object_center_voxels: False,
        standard_fields.DetectionResultFields.object_weight_voxels: True,
        standard_fields.DetectionResultFields.object_semantic_voxels: False,
    }
    if predict_rotation_x:
      task_names_to_num_output_channels[standard_fields.DetectionResultFields
                                        .object_rotation_x_cos_voxels] = 1
      task_names_to_num_output_channels[standard_fields.DetectionResultFields
                                        .object_rotation_x_sin_voxels] = 1
      task_names_to_use_relu_last_conv[standard_fields.DetectionResultFields
                                       .object_rotation_x_cos_voxels] = False
      task_names_to_use_relu_last_conv[standard_fields.DetectionResultFields
                                       .object_rotation_x_sin_voxels] = False
    if predict_rotation_y:
      task_names_to_num_output_channels[standard_fields.DetectionResultFields
                                        .object_rotation_y_cos_voxels] = 1
      task_names_to_num_output_channels[standard_fields.DetectionResultFields
                                        .object_rotation_y_sin_voxels] = 1
      task_names_to_use_relu_last_conv[standard_fields.DetectionResultFields
                                       .object_rotation_y_cos_voxels] = False
      task_names_to_use_relu_last_conv[standard_fields.DetectionResultFields
                                       .object_rotation_y_sin_voxels] = False
    if predict_rotation_z:
      task_names_to_num_output_channels[standard_fields.DetectionResultFields
                                        .object_rotation_z_cos_voxels] = 1
      task_names_to_num_output_channels[standard_fields.DetectionResultFields
                                        .object_rotation_z_sin_voxels] = 1
      task_names_to_use_relu_last_conv[standard_fields.DetectionResultFields
                                       .object_rotation_z_cos_voxels] = False
      task_names_to_use_relu_last_conv[standard_fields.DetectionResultFields
                                       .object_rotation_z_sin_voxels] = False
    task_names_to_use_batch_norm_in_last_layer = {}
    for key in task_names_to_num_output_channels:
      task_names_to_use_batch_norm_in_last_layer[key] = False
    self.num_classes = num_classes
    self.sparse_conv_hourglass = sparse_voxel_hourglass.SparseConvHourGlass(
        task_names_to_num_output_channels=task_names_to_num_output_channels,
        task_names_to_use_relu_last_conv=task_names_to_use_relu_last_conv,
        task_names_to_use_batch_norm_in_last_layer=(
            task_names_to_use_batch_norm_in_last_layer))

  def __call__(self, inputs, training=True):
    """Runs the model and returns the semantic logits prediciton.

    Args:
      inputs: A dictionary of tensors containing inputs for the model.
      training: Whether the model runs in training mode.

    Returns:
      A dictionary of tensors containing semantic logits prediciton.
    """
    # when not using custom train step, this field becomes 2 dim.
    inputs[standard_fields.InputDataFields.num_valid_voxels] = tf.reshape(
        inputs[standard_fields.InputDataFields.num_valid_voxels], [-1])
    voxel_inputs = (inputs[standard_fields.InputDataFields.voxel_features],
                    inputs[standard_fields.InputDataFields.voxel_xyz_indices],
                    inputs[standard_fields.InputDataFields.num_valid_voxels])
    outputs = self.sparse_conv_hourglass(voxel_inputs, training=training)
    outputs[
        standard_fields.DetectionResultFields.object_center_voxels] += inputs[
            standard_fields.InputDataFields.voxel_positions]
    model_utils.rectify_outputs(outputs=outputs)
    # If at eval time, transfer voxel features to points
    if ((not training) and
        (standard_fields.InputDataFields.points_to_voxel_mapping in inputs)):
      inputs[standard_fields.InputDataFields.num_valid_points] = tf.reshape(
          inputs[standard_fields.InputDataFields.num_valid_points], [-1])
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
        apply_nms=self.apply_nms,
        nms_score_threshold=self.nms_score_threshold,
        nms_iou_threshold=self.nms_iou_threshold,
        nms_max_num_predicted_boxes=self.nms_max_num_predicted_boxes,
        use_furthest_voxel_sampling=self.use_furthest_voxel_sampling,
        num_furthest_voxel_samples=self.num_furthest_voxel_samples,
        sampler_score_vs_distance_coef=self.sampler_score_vs_distance_coef)

    if training:
      self.calculate_losses(inputs=inputs, outputs=outputs)

    return outputs
