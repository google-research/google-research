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

"""Object detection model utility functions."""
import tensorflow as tf
from tf3d import standard_fields
from tf3d.object_detection import postprocessor
from tf3d.utils import mask_utils
from tf3d.utils import rotation_matrix


def rotation_matrix_from_cos_sin(outputs):
  """Compute rotation in radians from cos and sin predictions."""
  for key in [
      standard_fields.DetectionResultFields.object_rotation_x_cos_voxels,
      standard_fields.DetectionResultFields.object_rotation_x_sin_voxels,
      standard_fields.DetectionResultFields.object_rotation_y_cos_voxels,
      standard_fields.DetectionResultFields.object_rotation_y_sin_voxels,
      standard_fields.DetectionResultFields.object_rotation_z_cos_voxels,
      standard_fields.DetectionResultFields.object_rotation_z_sin_voxels
  ]:
    if key not in outputs:
      outputs[key] = None
  outputs[standard_fields.DetectionResultFields
          .object_rotation_matrix_voxels] = rotation_matrix.from_euler_cos_sin(
              cos_x=outputs[standard_fields.DetectionResultFields
                            .object_rotation_x_cos_voxels],
              sin_x=outputs[standard_fields.DetectionResultFields
                            .object_rotation_x_sin_voxels],
              cos_y=outputs[standard_fields.DetectionResultFields
                            .object_rotation_y_cos_voxels],
              sin_y=outputs[standard_fields.DetectionResultFields
                            .object_rotation_y_sin_voxels],
              cos_z=outputs[standard_fields.DetectionResultFields
                            .object_rotation_z_cos_voxels],
              sin_z=outputs[standard_fields.DetectionResultFields
                            .object_rotation_z_sin_voxels])


def mask_valid_voxels(inputs, outputs):
  """Mask the voxels that are valid."""
  if standard_fields.DetectionResultFields.objects_center in outputs:
    return outputs
  valid_mask = mask_utils.num_voxels_mask(inputs=inputs)
  mask_utils.apply_mask_to_output_voxel_tensors(
      outputs=outputs, valid_mask=valid_mask)
  for key, value in standard_fields.get_output_voxel_to_object_field_mapping(
  ).items():
    if key in outputs:
      outputs[value] = outputs[key]


def normalize_cos_sin_rotation(outputs, cos_key, sin_key, epsilon=0.01):
  """Normalize the cos and sin of rotation.

  Args:
    outputs: A dictionary containing predicted tensors.
    cos_key: A string corresponding to the key in outputs dictionary pointing
      to the tensor of cos.
    sin_key: A string corresponding to the key in outputs dictionary pointing
      to the tensor of sin.
    epsilon: Epsilon. A very small number.

  Raises:
    ValueError: If last dimension of the cos / sin tensors are not 1.
    ValueError: If tensor ranks are smaller than 1.
    ValueError: If the cos and sin tensors do not have the same rank.
  """
  tensor_rank = len(outputs[cos_key].get_shape().as_list())
  if tensor_rank < 1:
    raise ValueError('Tensor rank is smaller than 1.')
  if outputs[cos_key].get_shape().as_list()[tensor_rank - 1] != 1:
    raise ValueError('Last dimension of the cos tensor should be 1.')
  if len(outputs[sin_key].get_shape().as_list()) != tensor_rank:
    raise ValueError('Cos and sin tensors should have the same rank.')
  if outputs[sin_key].get_shape().as_list()[tensor_rank - 1] != 1:
    raise ValueError('Last dimension of the sin tensor should be 1.')

  cos_sin_norm = tf.expand_dims(
      tf.norm(
          tf.concat([outputs[cos_key], outputs[sin_key]],
                    axis=(tensor_rank - 1)),
          axis=(tensor_rank - 1)),
      axis=(tensor_rank - 1))
  cos_sin_norm = tf.maximum(cos_sin_norm, epsilon)
  outputs[cos_key] /= cos_sin_norm
  outputs[sin_key] /= cos_sin_norm


def normalize_cos_sin_rotations_in_outputs(outputs):
  """Normalize the cos and sin rotations."""
  if ((standard_fields.DetectionResultFields.object_rotation_x_cos_voxels in
       outputs) and
      (standard_fields.DetectionResultFields.object_rotation_x_sin_voxels in
       outputs) and (outputs[standard_fields.DetectionResultFields
                             .object_rotation_x_cos_voxels] is not None) and
      (outputs[standard_fields.DetectionResultFields
               .object_rotation_x_sin_voxels] is not None)):
    normalize_cos_sin_rotation(
        outputs=outputs,
        cos_key=standard_fields.DetectionResultFields
        .object_rotation_x_cos_voxels,
        sin_key=standard_fields.DetectionResultFields
        .object_rotation_x_sin_voxels)
  if ((standard_fields.DetectionResultFields.object_rotation_y_cos_voxels in
       outputs) and
      (standard_fields.DetectionResultFields.object_rotation_y_sin_voxels in
       outputs) and (outputs[standard_fields.DetectionResultFields
                             .object_rotation_y_cos_voxels] is not None) and
      (outputs[standard_fields.DetectionResultFields
               .object_rotation_y_sin_voxels] is not None)):
    normalize_cos_sin_rotation(
        outputs=outputs,
        cos_key=standard_fields.DetectionResultFields
        .object_rotation_y_cos_voxels,
        sin_key=standard_fields.DetectionResultFields
        .object_rotation_y_sin_voxels)
  if ((standard_fields.DetectionResultFields.object_rotation_z_cos_voxels in
       outputs) and
      (standard_fields.DetectionResultFields.object_rotation_z_sin_voxels in
       outputs) and (outputs[standard_fields.DetectionResultFields
                             .object_rotation_z_cos_voxels] is not None) and
      (outputs[standard_fields.DetectionResultFields
               .object_rotation_z_sin_voxels] is not None)):
    normalize_cos_sin_rotation(
        outputs=outputs,
        cos_key=standard_fields.DetectionResultFields
        .object_rotation_z_cos_voxels,
        sin_key=standard_fields.DetectionResultFields
        .object_rotation_z_sin_voxels)


def make_box_sizes_positive(outputs,
                            length_key,
                            height_key,
                            width_key,
                            min_object_length=0.01,
                            min_object_height=0.01,
                            min_object_width=0.01):
  """Make height, width, and length positive.

  Args:
    outputs: A dictionary containing predicted tensors.
    length_key: A string corresponding to the key in outputs dictionary pointing
      to the tensor of object lengths.
    height_key: A string corresponding to the key in outputs dictionary pointing
      to the tensor of object heights.
    width_key: A string corresponding to the key in outputs dictionary pointing
      to the tensor of object widths.
    min_object_length: Minimum object length.
    min_object_height: Minimum object height.
    min_object_width: Minimum object width.
  """
  outputs[length_key] = tf.maximum(min_object_length,
                                   tf.abs(outputs[length_key]))
  outputs[height_key] = tf.maximum(min_object_height,
                                   tf.abs(outputs[height_key]))
  outputs[width_key] = tf.maximum(min_object_width, tf.abs(outputs[width_key]))


def make_box_sizes_positive_in_outputs(outputs):
  """Make the predicted box sizes positive."""
  make_box_sizes_positive(
      outputs=outputs,
      length_key=standard_fields.DetectionResultFields.object_length_voxels,
      height_key=standard_fields.DetectionResultFields.object_height_voxels,
      width_key=standard_fields.DetectionResultFields.object_width_voxels)


def rectify_outputs(outputs):
  """Apply fixes to output values."""
  make_box_sizes_positive_in_outputs(outputs=outputs)
  normalize_cos_sin_rotations_in_outputs(outputs=outputs)
  rotation_matrix_from_cos_sin(outputs=outputs)
  if outputs[standard_fields.DetectionResultFields
             .object_rotation_matrix_voxels] is None:
    tensor_shape = outputs[standard_fields.DetectionResultFields
                           .object_length_voxels].get_shape().as_list()
    for i in range(len(tensor_shape)):
      if tensor_shape[i] is None:
        tensor_shape[i] = tf.shape(outputs[
            standard_fields.DetectionResultFields.object_length_voxels])[i]
    tensor_shape.append(1)
    identity_matrix = tf.eye(3, dtype=tf.float32)
    for _ in range(len(tensor_shape) - 2):
      identity_matrix = tf.expand_dims(identity_matrix, axis=0)
    outputs[standard_fields.DetectionResultFields
            .object_rotation_matrix_voxels] = tf.tile(identity_matrix,
                                                      tensor_shape)


def postprocess(inputs, outputs, is_training, apply_nms, nms_score_threshold,
                nms_iou_threshold, nms_max_num_predicted_boxes,
                use_furthest_voxel_sampling, num_furthest_voxel_samples,
                sampler_score_vs_distance_coef):
  """Post-processor function."""
  if not is_training:

    # Squeeze voxel properties.
    for key in standard_fields.get_output_voxel_fields():
      if key in outputs and outputs[key] is not None:
        outputs[key] = tf.squeeze(outputs[key], axis=0)
    for key in standard_fields.get_output_point_fields():
      if key in outputs and outputs[key] is not None:
        outputs[key] = tf.squeeze(outputs[key], axis=0)
    for key in standard_fields.get_output_object_fields():
      if key in outputs and outputs[key] is not None:
        outputs[key] = tf.squeeze(outputs[key], axis=0)

    # Mask the valid voxels
    mask_valid_voxels(inputs=inputs, outputs=outputs)

    # NMS
    postprocessor.postprocess(
        outputs=outputs,
        score_thresh=nms_score_threshold,
        iou_thresh=nms_iou_threshold,
        max_output_size=nms_max_num_predicted_boxes,
        use_furthest_voxel_sampling=use_furthest_voxel_sampling,
        num_furthest_voxel_samples=num_furthest_voxel_samples,
        sampler_score_vs_distance_coef=sampler_score_vs_distance_coef,
        apply_nms=apply_nms)
