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

"""Object detection box prediction losses."""

import gin
import gin.tf
import tensorflow as tf
from tf3d import standard_fields
from tf3d.losses import utils as loss_utils
from tf3d.utils import batch_utils
from tf3d.utils import box_utils
from tf3d.utils import mask_utils


def _box_rotation_regression_loss(loss_type, is_balanced,
                                  input_boxes_rotation_matrix,
                                  input_boxes_instance_id,
                                  output_boxes_rotation_matrix, delta):
  """Computes regression loss on object rotations."""

  def fn():
    """Loss function for when number of input and output boxes is positive."""
    if is_balanced:
      weights = loss_utils.get_balanced_loss_weights_multiclass(
          labels=input_boxes_instance_id)
    else:
      weights = tf.ones([tf.shape(input_boxes_instance_id)[0], 1],
                        dtype=tf.float32)
    gt_rotation_matrix = tf.reshape(input_boxes_rotation_matrix, [-1, 9])
    predicted_rotation_matrix = tf.reshape(output_boxes_rotation_matrix,
                                           [-1, 9])
    if loss_type == 'huber':
      loss_fn = tf.keras.losses.Huber(
          delta=delta, reduction=tf.keras.losses.Reduction.NONE)
    elif loss_type == 'absolute_difference':
      loss_fn = tf.keras.losses.MeanAbsoluteError(
          reduction=tf.keras.losses.Reduction.NONE)
    else:
      raise ValueError(('Unknown loss type %s.' % loss_type))
    rotation_losses = loss_fn(
        y_true=gt_rotation_matrix, y_pred=predicted_rotation_matrix)
    return tf.reduce_mean(rotation_losses * tf.reshape(weights, [-1]))

  cond_input = tf.greater(tf.shape(input_boxes_rotation_matrix)[0], 0)
  cond_output = tf.greater(tf.shape(output_boxes_rotation_matrix)[0], 0)
  cond = tf.logical_and(cond_input, cond_output)
  return tf.cond(cond, fn, lambda: tf.constant(0.0, dtype=tf.float32))


def _box_size_regression_loss(loss_type, is_balanced, input_boxes_length,
                              input_boxes_height, input_boxes_width,
                              input_boxes_instance_id, output_boxes_length,
                              output_boxes_height, output_boxes_width, delta):
  """Computes regression loss on object sizes."""

  def fn():
    """Loss function for when number of input and output boxes is positive."""
    if is_balanced:
      weights = loss_utils.get_balanced_loss_weights_multiclass(
          labels=input_boxes_instance_id)
    else:
      weights = tf.ones([tf.shape(input_boxes_instance_id)[0], 1],
                        dtype=tf.float32)
    gt_length = tf.reshape(input_boxes_length, [-1, 1])
    gt_height = tf.reshape(input_boxes_height, [-1, 1])
    gt_width = tf.reshape(input_boxes_width, [-1, 1])
    predicted_length = tf.reshape(output_boxes_length, [-1, 1])
    predicted_height = tf.reshape(output_boxes_height, [-1, 1])
    predicted_width = tf.reshape(output_boxes_width, [-1, 1])
    predicted_length /= gt_length
    predicted_height /= gt_height
    predicted_width /= gt_width
    predicted_size = tf.concat(
        [predicted_length, predicted_height, predicted_width], axis=1)
    gt_size = tf.ones_like(predicted_size)
    if loss_type == 'huber':
      loss_fn = tf.keras.losses.Huber(
          delta=delta, reduction=tf.keras.losses.Reduction.NONE)
    elif loss_type == 'absolute_difference':
      loss_fn = tf.keras.losses.MeanAbsoluteError(
          reduction=tf.keras.losses.Reduction.NONE)
    else:
      raise ValueError(('Unknown loss type %s.' % loss_type))
    size_losses = loss_fn(y_true=gt_size, y_pred=predicted_size)
    return tf.reduce_mean(size_losses * tf.reshape(weights, [-1]))

  cond_input = tf.greater(tf.shape(input_boxes_length)[0], 0)
  cond_output = tf.greater(tf.shape(output_boxes_length)[0], 0)
  cond = tf.logical_and(cond_input, cond_output)
  return tf.cond(cond, fn, lambda: tf.constant(0.0, dtype=tf.float32))


def _box_center_distance_loss(loss_type, is_balanced, input_boxes_center,
                              input_boxes_instance_id, output_boxes_center,
                              delta):
  """Computes regression loss on object center locations."""

  def fn():
    """Loss function for when number of input and output boxes is positive."""
    if is_balanced:
      weights = loss_utils.get_balanced_loss_weights_multiclass(
          labels=input_boxes_instance_id)
    else:
      weights = tf.ones([tf.shape(input_boxes_instance_id)[0], 1],
                        dtype=tf.float32)
    gt_center = tf.reshape(input_boxes_center, [-1, 3])
    predicted_center = tf.reshape(output_boxes_center, [-1, 3])
    if loss_type == 'huber':
      loss_fn = tf.keras.losses.Huber(
          delta=delta, reduction=tf.keras.losses.Reduction.NONE)
    elif loss_type == 'absolute_difference':
      loss_fn = tf.keras.losses.MeanAbsoluteError(
          reduction=tf.keras.losses.Reduction.NONE)
    else:
      raise ValueError(('Unknown loss type %s.' % loss_type))
    center_losses = loss_fn(y_true=gt_center, y_pred=predicted_center)
    return tf.reduce_mean(center_losses * tf.reshape(weights, [-1]))

  cond_input = tf.greater(tf.shape(input_boxes_center)[0], 0)
  cond_output = tf.greater(tf.shape(output_boxes_center)[0], 0)
  cond = tf.logical_and(cond_input, cond_output)
  return tf.cond(cond, fn, lambda: tf.constant(0.0, dtype=tf.float32))


def _box_corner_distance_loss(
    loss_type, is_balanced, input_boxes_length, input_boxes_height,
    input_boxes_width, input_boxes_center, input_boxes_rotation_matrix,
    input_boxes_instance_id, output_boxes_length, output_boxes_height,
    output_boxes_width, output_boxes_center, output_boxes_rotation_matrix,
    delta):
  """Computes regression loss on object corner locations."""

  def fn():
    """Loss function for when number of input and output boxes is positive."""
    if is_balanced:
      weights = loss_utils.get_balanced_loss_weights_multiclass(
          labels=input_boxes_instance_id)
    else:
      weights = tf.ones([tf.shape(input_boxes_instance_id)[0], 1],
                        dtype=tf.float32)
    normalized_box_size = 5.0
    predicted_boxes_length = output_boxes_length
    predicted_boxes_height = output_boxes_height
    predicted_boxes_width = output_boxes_width
    predicted_boxes_center = output_boxes_center
    predicted_boxes_rotation_matrix = output_boxes_rotation_matrix
    gt_boxes_length = input_boxes_length
    gt_boxes_height = input_boxes_height
    gt_boxes_width = input_boxes_width
    gt_boxes_center = input_boxes_center
    gt_boxes_rotation_matrix = input_boxes_rotation_matrix
    if loss_type in ['normalized_huber', 'normalized_euclidean']:
      predicted_boxes_length /= (gt_boxes_length / normalized_box_size)
      predicted_boxes_height /= (gt_boxes_height / normalized_box_size)
      predicted_boxes_width /= (gt_boxes_width / normalized_box_size)
      gt_boxes_length = tf.ones_like(
          gt_boxes_length, dtype=tf.float32) * normalized_box_size
      gt_boxes_height = tf.ones_like(
          gt_boxes_height, dtype=tf.float32) * normalized_box_size
      gt_boxes_width = tf.ones_like(
          gt_boxes_width, dtype=tf.float32) * normalized_box_size
    gt_box_corners = box_utils.get_box_corners_3d(
        boxes_length=gt_boxes_length,
        boxes_height=gt_boxes_height,
        boxes_width=gt_boxes_width,
        boxes_rotation_matrix=gt_boxes_rotation_matrix,
        boxes_center=gt_boxes_center)
    predicted_box_corners = box_utils.get_box_corners_3d(
        boxes_length=predicted_boxes_length,
        boxes_height=predicted_boxes_height,
        boxes_width=predicted_boxes_width,
        boxes_rotation_matrix=predicted_boxes_rotation_matrix,
        boxes_center=predicted_boxes_center)
    corner_weights = tf.tile(weights, [1, 8])
    if loss_type in ['huber', 'normalized_huber']:
      loss_fn = tf.keras.losses.Huber(
          delta=delta, reduction=tf.keras.losses.Reduction.NONE)
    elif loss_type in ['normalized_absolute_difference', 'absolute_difference']:
      loss_fn = tf.keras.losses.MeanAbsoluteError(
          reduction=tf.keras.losses.Reduction.NONE)
    else:
      raise ValueError(('Unknown loss type %s.' % loss_type))
    box_corner_losses = loss_fn(
        y_true=tf.reshape(gt_box_corners, [-1, 3]),
        y_pred=tf.reshape(predicted_box_corners, [-1, 3]))
    return tf.reduce_mean(box_corner_losses * tf.reshape(corner_weights, [-1]))

  cond_input = tf.greater(tf.shape(input_boxes_length)[0], 0)
  cond_output = tf.greater(tf.shape(output_boxes_length)[0], 0)
  cond = tf.logical_and(cond_input, cond_output)
  return tf.cond(cond, fn, lambda: tf.constant(0.0, dtype=tf.float32))


def _get_voxels_valid_mask(inputs_1):
  """Returns the mask that removes voxels that are outside objects."""
  num_voxels_mask = mask_utils.num_voxels_mask(inputs=inputs_1)
  within_objects_mask = mask_utils.voxels_within_objects_mask(inputs=inputs_1)
  return tf.logical_and(within_objects_mask, num_voxels_mask)


def _get_voxels_valid_inputs_outputs(inputs_1, outputs_1):
  """Applies the valid mask to input and output voxel tensors."""
  valid_mask = _get_voxels_valid_mask(inputs_1=inputs_1)
  inputs_1 = mask_utils.apply_mask_to_input_voxel_tensors(
      inputs=inputs_1, valid_mask=valid_mask)
  mask_utils.apply_mask_to_output_voxel_tensors(
      outputs=outputs_1, valid_mask=valid_mask)
  return inputs_1, outputs_1, valid_mask


def _box_rotation_regression_loss_on_voxel_tensors_unbatched(
    inputs_1, outputs_1, loss_type, delta, is_balanced, is_intermediate):
  """Computes regression loss on predicted object rotation for each voxel."""
  inputs_1, outputs_1, valid_mask = _get_voxels_valid_inputs_outputs(
      inputs_1=inputs_1, outputs_1=outputs_1)

  def loss_fn_unbatched():
    """Loss function."""
    if is_intermediate:
      output_boxes_rotation_matrix = outputs_1[
          standard_fields.DetectionResultFields
          .intermediate_object_rotation_matrix_voxels]
    else:
      output_boxes_rotation_matrix = outputs_1[
          standard_fields.DetectionResultFields.object_rotation_matrix_voxels]
    return _box_rotation_regression_loss(
        loss_type=loss_type,
        is_balanced=is_balanced,
        input_boxes_rotation_matrix=inputs_1[
            standard_fields.InputDataFields.object_rotation_matrix_voxels],
        input_boxes_instance_id=inputs_1[
            standard_fields.InputDataFields.object_instance_id_voxels],
        output_boxes_rotation_matrix=output_boxes_rotation_matrix,
        delta=delta)

  return tf.cond(
      tf.reduce_any(valid_mask),
      loss_fn_unbatched, lambda: tf.constant(0.0, dtype=tf.float32))


@gin.configurable(
    'box_rotation_regression_loss_on_voxel_tensors',
    blacklist=['inputs', 'outputs'])
def box_rotation_regression_loss_on_voxel_tensors(inputs,
                                                  outputs,
                                                  loss_type,
                                                  delta=0.5,
                                                  is_balanced=False,
                                                  is_intermediate=False):
  """Computes regression loss on object size.

  Args:
    inputs: A dictionary of tf.Tensors with our input data.
    outputs: A dictionary of tf.Tensors with the network output.
    loss_type: Loss type.
    delta: float, the voxel where the huber loss function changes from a
      quadratic to linear.
    is_balanced: If True, the per-voxel losses are re-weighted to have equal
      total weight for each object instance.
    is_intermediate: If True, intermediate tensors are used for computing
      the loss.

  Returns:
    localization_loss: A tf.float32 scalar corresponding to localization loss.
  """
  standard_fields.check_input_voxel_fields(inputs=inputs)
  standard_fields.check_output_voxel_fields(outputs=outputs)

  def fn(inputs_1, outputs_1):
    return _box_rotation_regression_loss_on_voxel_tensors_unbatched(
        inputs_1=inputs_1,
        outputs_1=outputs_1,
        loss_type=loss_type,
        delta=delta,
        is_balanced=is_balanced,
        is_intermediate=is_intermediate)

  return loss_utils.apply_unbatched_loss_on_voxel_tensors(
      inputs=inputs, outputs=outputs, unbatched_loss_fn=fn)


def _box_size_regression_loss_on_voxel_tensors_unbatched(
    inputs_1, outputs_1, loss_type, delta, is_balanced, is_intermediate):
  """Computes regression loss on predicted object size for each voxel."""
  inputs_1, outputs_1, valid_mask = _get_voxels_valid_inputs_outputs(
      inputs_1=inputs_1, outputs_1=outputs_1)

  def loss_fn_unbatched():
    """Loss function."""
    if is_intermediate:
      output_boxes_length = outputs_1[standard_fields.DetectionResultFields
                                      .intermediate_object_length_voxels]
      output_boxes_height = outputs_1[standard_fields.DetectionResultFields
                                      .intermediate_object_height_voxels]
      output_boxes_width = outputs_1[standard_fields.DetectionResultFields
                                     .intermediate_object_width_voxels]
    else:
      output_boxes_length = outputs_1[
          standard_fields.DetectionResultFields.object_length_voxels]
      output_boxes_height = outputs_1[
          standard_fields.DetectionResultFields.object_height_voxels]
      output_boxes_width = outputs_1[
          standard_fields.DetectionResultFields.object_width_voxels]
    return _box_size_regression_loss(
        loss_type=loss_type,
        is_balanced=is_balanced,
        input_boxes_length=inputs_1[
            standard_fields.InputDataFields.object_length_voxels],
        input_boxes_height=inputs_1[
            standard_fields.InputDataFields.object_height_voxels],
        input_boxes_width=inputs_1[
            standard_fields.InputDataFields.object_width_voxels],
        input_boxes_instance_id=inputs_1[
            standard_fields.InputDataFields.object_instance_id_voxels],
        output_boxes_length=output_boxes_length,
        output_boxes_height=output_boxes_height,
        output_boxes_width=output_boxes_width,
        delta=delta)

  return tf.cond(
      tf.reduce_any(valid_mask),
      loss_fn_unbatched, lambda: tf.constant(0.0, dtype=tf.float32))


@gin.configurable(
    'box_size_regression_loss_on_voxel_tensors',
    blacklist=['inputs', 'outputs'])
def box_size_regression_loss_on_voxel_tensors(inputs,
                                              outputs,
                                              loss_type,
                                              delta=0.5,
                                              is_balanced=False,
                                              is_intermediate=False):
  """Computes regression loss on object size.

  Args:
    inputs: A dictionary of tf.Tensors with our input data.
    outputs: A dictionary of tf.Tensors with the network output.
    loss_type: Loss type.
    delta: float, the voxel where the huber loss function changes from a
      quadratic to linear.
    is_balanced: If True, the per-voxel losses are re-weighted to have equal
      total weight for each object instance.
    is_intermediate: If True, intermediate tensors are used for computing
      the loss.

  Returns:
    localization_loss: A tf.float32 scalar corresponding to localization loss.
  """
  standard_fields.check_input_voxel_fields(inputs=inputs)
  standard_fields.check_output_voxel_fields(outputs=outputs)

  def fn(inputs_1, outputs_1):
    return _box_size_regression_loss_on_voxel_tensors_unbatched(
        inputs_1=inputs_1,
        outputs_1=outputs_1,
        loss_type=loss_type,
        delta=delta,
        is_balanced=is_balanced,
        is_intermediate=is_intermediate)

  return loss_utils.apply_unbatched_loss_on_voxel_tensors(
      inputs=inputs, outputs=outputs, unbatched_loss_fn=fn)


def _box_center_distance_loss_on_voxel_tensors_unbatched(
    inputs_1, outputs_1, loss_type, delta, is_balanced, is_intermediate):
  """Computes huber loss on predicted object centers for each voxel."""
  inputs_1, outputs_1, valid_mask = _get_voxels_valid_inputs_outputs(
      inputs_1=inputs_1, outputs_1=outputs_1)

  def loss_fn_unbatched():
    """Loss function."""
    if is_intermediate:
      output_boxes_center = outputs_1[standard_fields.DetectionResultFields
                                      .intermediate_object_center_voxels]
    else:
      output_boxes_center = outputs_1[
          standard_fields.DetectionResultFields.object_center_voxels]
    return _box_center_distance_loss(
        loss_type=loss_type,
        is_balanced=is_balanced,
        input_boxes_center=inputs_1[
            standard_fields.InputDataFields.object_center_voxels],
        input_boxes_instance_id=inputs_1[
            standard_fields.InputDataFields.object_instance_id_voxels],
        output_boxes_center=output_boxes_center,
        delta=delta)

  return tf.cond(
      tf.reduce_any(valid_mask),
      loss_fn_unbatched, lambda: tf.constant(0.0, dtype=tf.float32))


@gin.configurable(
    'box_center_distance_loss_on_voxel_tensors',
    blacklist=['inputs', 'outputs'])
def box_center_distance_loss_on_voxel_tensors(inputs,
                                              outputs,
                                              loss_type,
                                              delta=1.0,
                                              is_balanced=False,
                                              is_intermediate=False):
  """Computes huber loss on object center locations.

  Args:
    inputs: A dictionary of tf.Tensors with our input data.
    outputs: A dictionary of tf.Tensors with the network output.
    loss_type: Loss type.
    delta: float, the voxel where the huber loss function changes from a
      quadratic to linear.
    is_balanced: If True, the per-voxel losses are re-weighted to have equal
      total weight for each object instance.
    is_intermediate: If True, intermediate tensors are used for computing
      the loss.

  Returns:
    localization_loss: A tf.float32 scalar corresponding to localization loss.
  """
  standard_fields.check_input_voxel_fields(inputs=inputs)
  standard_fields.check_output_voxel_fields(outputs=outputs)

  def fn(inputs_1, outputs_1):
    return _box_center_distance_loss_on_voxel_tensors_unbatched(
        inputs_1=inputs_1,
        outputs_1=outputs_1,
        loss_type=loss_type,
        delta=delta,
        is_balanced=is_balanced,
        is_intermediate=is_intermediate)

  return loss_utils.apply_unbatched_loss_on_voxel_tensors(
      inputs=inputs, outputs=outputs, unbatched_loss_fn=fn)


def _box_corner_distance_loss_on_voxel_tensors_unbatched(
    inputs_1, outputs_1, loss_type, delta, is_balanced, is_intermediate):
  """Computes huber loss on predicted objects for each voxel."""
  inputs_1, outputs_1, valid_mask = _get_voxels_valid_inputs_outputs(
      inputs_1=inputs_1, outputs_1=outputs_1)

  def loss_fn_unbatched():
    """Loss function."""
    if is_intermediate:
      output_boxes_length = outputs_1[standard_fields.DetectionResultFields
                                      .intermediate_object_length_voxels]
      output_boxes_height = outputs_1[standard_fields.DetectionResultFields
                                      .intermediate_object_height_voxels]
      output_boxes_width = outputs_1[standard_fields.DetectionResultFields
                                     .intermediate_object_width_voxels]
      output_boxes_center = outputs_1[standard_fields.DetectionResultFields
                                      .intermediate_object_center_voxels]
      output_boxes_rotation_matrix = outputs_1[
          standard_fields.DetectionResultFields
          .intermediate_object_rotation_matrix_voxels]
    else:
      output_boxes_length = outputs_1[
          standard_fields.DetectionResultFields.object_length_voxels]
      output_boxes_height = outputs_1[
          standard_fields.DetectionResultFields.object_height_voxels]
      output_boxes_width = outputs_1[
          standard_fields.DetectionResultFields.object_width_voxels]
      output_boxes_center = outputs_1[
          standard_fields.DetectionResultFields.object_center_voxels]
      output_boxes_rotation_matrix = outputs_1[
          standard_fields.DetectionResultFields.object_rotation_matrix_voxels]
    return _box_corner_distance_loss(
        loss_type=loss_type,
        is_balanced=is_balanced,
        input_boxes_length=inputs_1[
            standard_fields.InputDataFields.object_length_voxels],
        input_boxes_height=inputs_1[
            standard_fields.InputDataFields.object_height_voxels],
        input_boxes_width=inputs_1[
            standard_fields.InputDataFields.object_width_voxels],
        input_boxes_center=inputs_1[
            standard_fields.InputDataFields.object_center_voxels],
        input_boxes_rotation_matrix=inputs_1[
            standard_fields.InputDataFields.object_rotation_matrix_voxels],
        input_boxes_instance_id=inputs_1[
            standard_fields.InputDataFields.object_instance_id_voxels],
        output_boxes_length=output_boxes_length,
        output_boxes_height=output_boxes_height,
        output_boxes_width=output_boxes_width,
        output_boxes_center=output_boxes_center,
        output_boxes_rotation_matrix=output_boxes_rotation_matrix,
        delta=delta)

  return tf.cond(
      tf.reduce_any(valid_mask),
      loss_fn_unbatched, lambda: tf.constant(0.0, dtype=tf.float32))


@gin.configurable(
    'box_corner_distance_loss_on_voxel_tensors',
    blacklist=['inputs', 'outputs'])
def box_corner_distance_loss_on_voxel_tensors(
    inputs,
    outputs,
    loss_type,
    delta=1.0,
    is_balanced=False,
    is_intermediate=False):
  """Computes regression loss on object corner locations using object tensors.

  Args:
    inputs: A dictionary of tf.Tensors with our input data.
    outputs: A dictionary of tf.Tensors with the network output.
    loss_type: Loss type.
    delta: float, the voxel where the huber loss function changes from a
      quadratic to linear.
    is_balanced: If True, the per-voxel losses are re-weighted to have equal
      total weight for each object instance.
    is_intermediate: If True, intermediate tensors are used for computing
      the loss.

  Returns:
    localization_loss: A tf.float32 scalar corresponding to localization loss.
  """
  standard_fields.check_input_voxel_fields(inputs=inputs)
  standard_fields.check_output_voxel_fields(outputs=outputs)

  def fn(inputs_1, outputs_1):
    return _box_corner_distance_loss_on_voxel_tensors_unbatched(
        inputs_1=inputs_1,
        outputs_1=outputs_1,
        loss_type=loss_type,
        delta=delta,
        is_balanced=is_balanced,
        is_intermediate=is_intermediate)

  return loss_utils.apply_unbatched_loss_on_voxel_tensors(
      inputs=inputs, outputs=outputs, unbatched_loss_fn=fn)


def _box_corner_distance_loss_on_object_tensors(
    inputs, outputs, loss_type, delta, is_balanced):
  """Computes huber loss on object corner locations."""
  valid_mask_class = tf.greater(
      tf.reshape(inputs[standard_fields.InputDataFields.objects_class], [-1]),
      0)
  valid_mask_instance = tf.greater(
      tf.reshape(inputs[standard_fields.InputDataFields.objects_instance_id],
                 [-1]), 0)
  valid_mask = tf.logical_and(valid_mask_class, valid_mask_instance)

  def fn():
    for field in standard_fields.get_input_object_fields():
      if field in inputs:
        inputs[field] = tf.boolean_mask(inputs[field], valid_mask)
    for field in standard_fields.get_output_object_fields():
      if field in outputs:
        outputs[field] = tf.boolean_mask(outputs[field], valid_mask)
    return _box_corner_distance_loss(
        loss_type=loss_type,
        is_balanced=is_balanced,
        input_boxes_length=inputs[
            standard_fields.InputDataFields.objects_length],
        input_boxes_height=inputs[
            standard_fields.InputDataFields.objects_height],
        input_boxes_width=inputs[standard_fields.InputDataFields.objects_width],
        input_boxes_center=inputs[
            standard_fields.InputDataFields.objects_center],
        input_boxes_rotation_matrix=inputs[
            standard_fields.InputDataFields.objects_rotation_matrix],
        input_boxes_instance_id=inputs[
            standard_fields.InputDataFields.objects_instance_id],
        output_boxes_length=outputs[
            standard_fields.DetectionResultFields.objects_length],
        output_boxes_height=outputs[
            standard_fields.DetectionResultFields.objects_height],
        output_boxes_width=outputs[
            standard_fields.DetectionResultFields.objects_width],
        output_boxes_center=outputs[
            standard_fields.DetectionResultFields.objects_center],
        output_boxes_rotation_matrix=outputs[
            standard_fields.DetectionResultFields.objects_rotation_matrix],
        delta=delta)

  return tf.cond(
      tf.reduce_any(valid_mask), fn, lambda: tf.constant(0.0, dtype=tf.float32))


@gin.configurable(
    'box_corner_distance_loss_on_object_tensors',
    blacklist=['inputs', 'outputs'])
def box_corner_distance_loss_on_object_tensors(
    inputs,
    outputs,
    loss_type,
    delta=1.0,
    is_balanced=False):
  """Computes regression loss on object corner locations using object tensors.

  Args:
    inputs: A dictionary of tf.Tensors with our input data.
    outputs: A dictionary of tf.Tensors with the network output.
    loss_type: Loss type.
    delta: float, the voxel where the huber loss function changes from a
      quadratic to linear.
    is_balanced: If True, the per-voxel losses are re-weighted to have equal
      total weight for each object instance.

  Returns:
    localization_loss: A tf.float32 scalar corresponding to localization loss.
  """
  def fn(inputs_1, outputs_1):
    return _box_corner_distance_loss_on_object_tensors(
        inputs=inputs_1,
        outputs=outputs_1,
        loss_type=loss_type,
        delta=delta,
        is_balanced=is_balanced)

  batch_size = len(inputs[standard_fields.InputDataFields.objects_length])
  losses = []
  for b in range(batch_size):
    inputs_1 = batch_utils.get_batch_size_1_input_objects(inputs=inputs, b=b)
    outputs_1 = batch_utils.get_batch_size_1_output_objects(
        outputs=outputs, b=b)
    cond_input = tf.greater(
        tf.shape(inputs_1[standard_fields.InputDataFields.objects_length])[0],
        0)
    cond_output = tf.greater(
        tf.shape(
            outputs_1[standard_fields.DetectionResultFields.objects_length])[0],
        0)
    cond = tf.logical_and(cond_input, cond_output)
    # pylint: disable=cell-var-from-loop
    loss = tf.cond(cond, lambda: fn(inputs_1=inputs_1, outputs_1=outputs_1),
                   lambda: tf.constant(0.0, dtype=tf.float32))
    # pylint: enable=cell-var-from-loop
    losses.append(loss)
  return tf.reduce_mean(tf.stack(losses))
