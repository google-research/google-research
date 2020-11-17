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

"""Utility functions used in losses."""

import tensorflow as tf
from tf3d import standard_fields
from tf3d.utils import batch_utils
from tf3d.utils import box_utils
from tf3d.utils import instance_segmentation_utils as isu


def sample_from_labels_balanced(labels, num_samples, epsilon=1e-5):
  """Samples from labels inverse proportional to their frequency.

  Args:
    labels: A tf.int32 tensor of size [N].
    num_samples: Number of samples.
    epsilon: A very small number.

  Returns:
    indices: A tf.int32 tensor of size [num_samples].
  """
  labels = isu.map_labels_to_0_to_n(labels)
  max_label = tf.reduce_max(labels)
  labels_onehot = tf.one_hot(labels, depth=(max_label + 1), dtype=tf.float32)
  frequencies = tf.reduce_sum(labels_onehot, axis=0)
  inv_frequencies = 1.0 / (frequencies + epsilon)
  inv_frequencies = tf.gather(inv_frequencies, labels)
  return tf.squeeze(
      tf.random.categorical(
          tf.math.log(tf.expand_dims(inv_frequencies, axis=0)), num_samples),
      axis=0)


def sample_from_instances(inputs, outputs, num_samples):
  """Samples equally from instances."""
  input_tensors = {}
  output_tensors = {}
  input_tensors[
      standard_fields.InputDataFields.object_instance_id_image] = tf.reshape(
          inputs[standard_fields.InputDataFields.object_instance_id_image],
          [-1])
  input_tensors[standard_fields.InputDataFields
                .object_instance_id_image] = isu.map_labels_to_0_to_n(
                    input_tensors[standard_fields.InputDataFields
                                  .object_instance_id_image])
  seed_indices = isu.randomly_select_n_points_per_segment(
      labels=input_tensors[
          standard_fields.InputDataFields.object_instance_id_image],
      num_points=num_samples,
      include_ignore_label=False)
  seed_indices = tf.reshape(seed_indices, [-1])
  for field in standard_fields.get_input_image_fields():
    if field in inputs:
      input_tensors[field] = tf.gather(inputs[field], seed_indices)
  for field in standard_fields.get_output_image_fields():
    if field in outputs:
      output_tensors[field] = tf.gather(outputs[field], seed_indices)
  return input_tensors, output_tensors


def get_linear_loss_coef(loss_end_step):
  current_step = tf.train.get_or_create_global_step()
  return tf.to_float(current_step) / tf.to_float(loss_end_step)


def get_normalized_center_distances(predicted_boxes_center, gt_boxes_center,
                                    gt_boxes_length, gt_boxes_height,
                                    gt_boxes_width):
  """Computes the normalized distance of predicted and ground-truth centers.

  Args:
    predicted_boxes_center: A tf.float32 tensor of size [N, 3].
    gt_boxes_center: A tf.float32 tensor of size [N, 3].
    gt_boxes_length: A tf.float32 tensor of size [N, 1].
    gt_boxes_height: A tf.float32 tensor of size [N, 1].
    gt_boxes_width: A tf.float32 tensor of size [N, 1].

  Returns:
    A tf.float32 tensor of size [N] of normalized center distances.
  """
  center_distance = tf.norm(
      tf.reshape(gt_boxes_center - predicted_boxes_center, [-1, 3]), axis=1)
  box_size = tf.norm(
      tf.concat([
          tf.reshape(gt_boxes_length, [-1, 1]),
          tf.reshape(gt_boxes_height, [-1, 1]),
          tf.reshape(gt_boxes_width, [-1, 1])
      ],
                axis=1),
      axis=1)
  return center_distance / box_size


def get_normalized_corner_distances(predicted_boxes_center,
                                    predicted_boxes_length,
                                    predicted_boxes_height,
                                    predicted_boxes_width,
                                    predicted_boxes_rotation_matrix,
                                    gt_boxes_center,
                                    gt_boxes_length,
                                    gt_boxes_height,
                                    gt_boxes_width,
                                    gt_boxes_rotation_matrix,
                                    min_box_size=0.1):
  """Returns the average corner distance between prediction and ground-truth.

  Args:
    predicted_boxes_center: A tf.float32 tensor of size [N, 3].
    predicted_boxes_length: A tf.float32 tensor of size [N, 1].
    predicted_boxes_height: A tf.float32 tensor of size [N, 1].
    predicted_boxes_width: A tf.float32 tensor of size [N, 1].
    predicted_boxes_rotation_matrix: A tf.float32 tensor of size [N, 3, 3].
    gt_boxes_center: A tf.float32 tensor of size [N, 3].
    gt_boxes_length: A tf.float32 tensor of size [N, 1].
    gt_boxes_height: A tf.float32 tensor of size [N, 1].
    gt_boxes_width: A tf.float32 tensor of size [N, 1].
    gt_boxes_rotation_matrix: A tf.float32 tensor of size [N, 3, 3].
    min_box_size: Minimum box size.

  Returns:
    A tf.float32 tensor of size [N] of average normalized corner distances.
  """
  predicted_boxes_corners = box_utils.get_box_corners_3d(
      boxes_length=tf.reshape(predicted_boxes_length, [-1, 1]),
      boxes_height=tf.reshape(predicted_boxes_height, [-1, 1]),
      boxes_width=tf.reshape(predicted_boxes_width, [-1, 1]),
      boxes_center=tf.reshape(predicted_boxes_center, [-1, 3]),
      boxes_rotation_matrix=tf.reshape(predicted_boxes_rotation_matrix,
                                       [-1, 3, 3]))
  gt_boxes_corners = box_utils.get_box_corners_3d(
      boxes_length=tf.reshape(gt_boxes_length, [-1, 1]),
      boxes_height=tf.reshape(gt_boxes_height, [-1, 1]),
      boxes_width=tf.reshape(gt_boxes_width, [-1, 1]),
      boxes_center=tf.reshape(gt_boxes_center, [-1, 3]),
      boxes_rotation_matrix=tf.reshape(gt_boxes_rotation_matrix, [-1, 3, 3]))
  corner_distances = tf.reduce_mean(
      tf.norm(predicted_boxes_corners - gt_boxes_corners, axis=2), axis=1)
  box_size = tf.norm(
      tf.concat([
          tf.reshape(gt_boxes_length, [-1, 1]),
          tf.reshape(gt_boxes_height, [-1, 1]),
          tf.reshape(gt_boxes_width, [-1, 1])
      ],
                axis=1),
      axis=1)
  return corner_distances / tf.maximum(box_size, min_box_size)


def get_closest_predicted_center_per_positive_instance(
    predicted_boxes_center, gt_boxes_center, gt_boxes_length, gt_boxes_height,
    gt_boxes_width, gt_object_instance_id):
  """Returns the indices of closest predicted centers to each instance.

  Assumes the indices in `gt_object_instance_id` range from 1 to K.

  Args:
    predicted_boxes_center: A tf.float32 tensor of size [N, 3].
    gt_boxes_center: A tf.float32 tensor of size [N, 3].
    gt_boxes_length: A tf.float32 tensor of size [N, 1].
    gt_boxes_height: A tf.float32 tensor of size [N, 1].
    gt_boxes_width: A tf.float32 tensor of size [N, 1].
    gt_object_instance_id: A tf.int32 tensor of size [N].

  Returns:
    A tf.int32 tensor of size [K] containing index of the voxel with the closest
    normalized center distance to ground-truth for each instance. K is the
    number of instances.
  """
  normalized_center_distances = get_normalized_center_distances(
      predicted_boxes_center=predicted_boxes_center,
      gt_boxes_center=gt_boxes_center,
      gt_boxes_length=gt_boxes_length,
      gt_boxes_height=gt_boxes_height,
      gt_boxes_width=gt_boxes_width)
  instance_ids = tf.reshape(gt_object_instance_id, [-1])
  max_id = tf.reduce_max(instance_ids)
  ids_onehot = tf.one_hot(instance_ids - 1, max_id, dtype=tf.float32)
  inverse_instance_distance = ids_onehot * tf.expand_dims(
      (1.0 / (1.0 + normalized_center_distances)), axis=1)
  closest_indices = tf.math.argmax(inverse_instance_distance, axis=0)
  return closest_indices


def get_balanced_loss_weights_foreground_background(labels):
  """Returns weights for balancing the positive and negative loss."""
  positive_labels = tf.cast(tf.greater(labels, 0), dtype=tf.float32)
  num_positive = tf.reduce_sum(positive_labels)
  num_negative = tf.reduce_sum(1.0 - positive_labels)
  positive_weight = (num_positive + num_negative) / (
      tf.maximum(1.0, num_positive) * 2.0)
  negative_weight = (num_positive + num_negative) / (
      tf.maximum(1.0, num_negative) * 2.0)
  weights = positive_labels * positive_weight + (
      1.0 - positive_labels) * negative_weight
  return tf.reshape(weights, [-1, 1])


def get_balanced_loss_weights_multiclass(labels):
  """Returns weights to balance the loss between multiple classes of labels."""
  _, idx, count = tf.unique_with_counts(tf.reshape(labels, [-1]))
  count = tf.gather(count, idx)
  num_labels = tf.shape(count)[0]
  weights = 1.0 / tf.cast(count, dtype=tf.float32)
  total_weights = tf.reduce_sum(weights)
  weights *= (tf.cast(num_labels, dtype=tf.float32) / total_weights)
  return tf.reshape(weights, [-1, 1])


def apply_unbatched_loss_on_voxel_tensors(inputs, outputs, unbatched_loss_fn):
  """Applies the `unbatched_loss_fn` to each example in the batch."""
  batch_size = inputs[standard_fields.InputDataFields
                      .object_length_voxels].get_shape().as_list()[0]
  losses = []
  for b in range(batch_size):
    inputs_1 = batch_utils.get_batch_size_1_input_voxels(inputs=inputs, b=b)
    outputs_1 = batch_utils.get_batch_size_1_output_voxels(
        outputs=outputs, b=b)
    cond_input = tf.greater(
        tf.shape(
            inputs_1[standard_fields.InputDataFields.object_length_voxels])[0],
        0)
    detection_fields = standard_fields.DetectionResultFields
    if detection_fields.object_center_voxels in outputs_1:
      cond_output = tf.greater(
          tf.shape(outputs_1[detection_fields.object_center_voxels])[0], 0)
    elif detection_fields.intermediate_object_center_voxels in outputs_1:
      cond_output = tf.greater(
          tf.shape(
              outputs_1[detection_fields.intermediate_object_center_voxels])[0],
          0)
    else:
      cond_output = cond_input
    cond = tf.logical_and(cond_input, cond_output)
    # pylint: disable=cell-var-from-loop
    loss = tf.cond(
        cond,
        lambda: unbatched_loss_fn(inputs_1=inputs_1, outputs_1=outputs_1),
        lambda: tf.constant(0.0, dtype=tf.float32))
    # pylint: enable=cell-var-from-loop
    losses.append(loss)
  return tf.reduce_mean(tf.stack(losses))


def apply_unbatched_loss_on_object_tensors(inputs, outputs, unbatched_loss_fn):
  """Applies the `unbatched_loss_fn` to each example in the batch."""
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
    loss = tf.cond(
        cond,
        lambda: unbatched_loss_fn(inputs_1=inputs_1, outputs_1=outputs_1),
        lambda: tf.constant(0.0, dtype=tf.float32))
    # pylint: enable=cell-var-from-loop
    losses.append(loss)
  return tf.reduce_mean(tf.stack(losses))
