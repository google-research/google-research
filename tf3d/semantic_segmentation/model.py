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

"""Implements a 3d semantic segmentation model."""
import gin
import gin.tf
import tensorflow as tf
from tf3d import base_model
from tf3d import standard_fields
from tf3d.layers import sparse_voxel_unet
from tf3d.losses import classification_losses
from tf3d.utils import voxel_utils


@gin.configurable
class SemanticSegmentationModel(base_model.BaseModel):
  """3D UNet sparse voxel network for semantic segmentation.

  Please refer to the following paper for more details:
  M. Najibi, G. Lai, A. Kundu, Z. Lu, V. Rathod, T. Funkhouser, C. Pantofaru,
  D. Ross, L. S. Davis, A. Fathi,
  'DOPS: Learning to Detect 3D Objects and Predict Their 3D Shapes', CVPR 2020.
  """

  def __init__(self,
               num_classes,
               train_dir='/tmp/model/train',
               summary_log_freq=100):
    """A semantic segmentation model based on 3D UNet sparse voxel network.

    Args:
      num_classes: A int indicating the number of semantic classes to predict
        logits.
      train_dir: A directory path to write tensorboard summary for losses.
      summary_log_freq: A int of the frequency (as batches) to log summary.

    Returns:
      A dictionary containing a predicted tensor per task. The predicted tensors
        are of size [batch_size, num_voxels, num_task_channels].
    """
    super().__init__(
        loss_names_to_functions={
            'semantic_loss': classification_losses.classification_loss
        },
        loss_names_to_weights={'semantic_loss': 1.0},
        train_dir=train_dir,
        summary_log_freq=summary_log_freq)

    task_names_to_num_output_channels = {
        standard_fields.DetectionResultFields.object_semantic_voxels:
            num_classes
    }

    self.num_classes = num_classes
    self.sparse_conv_unet = sparse_voxel_unet.SparseConvUNet(
        task_names_to_num_output_channels=task_names_to_num_output_channels)

  def call(self, inputs, training=True):
    """Runs the model and returns the semantic logits prediction.

    Args:
      inputs: A dictionary of tensors containing inputs for the model.
      training: Whether the model runs in training mode.

    Returns:
      A dictionary of tensors containing semantic logits prediction.
    """
    # when not using custom train step, this field becomes 2 dim.
    inputs[standard_fields.InputDataFields.num_valid_voxels] = tf.reshape(
        inputs[standard_fields.InputDataFields.num_valid_voxels], [-1])
    voxel_inputs = (inputs[standard_fields.InputDataFields.voxel_features],
                    inputs[standard_fields.InputDataFields.voxel_xyz_indices],
                    inputs[standard_fields.InputDataFields.num_valid_voxels])
    outputs = self.sparse_conv_unet(voxel_inputs, training=training)

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
      # include fields used by tensorboard visualization call back.
      outputs.update(point_tensor_outputs)

    if training:
      self.calculate_losses(inputs=inputs, outputs=outputs)

    return outputs
