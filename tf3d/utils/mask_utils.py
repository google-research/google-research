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

"""Mask utility functions."""

import tensorflow as tf
from tf3d import standard_fields


def num_points_mask(inputs):
  """Returns a boolean mask that will keep the first num_points values."""
  num_points = tf.squeeze(
      inputs[standard_fields.InputDataFields.num_valid_points])
  if standard_fields.InputDataFields.point_positions in inputs:
    example_size = tf.shape(
        tf.reshape(inputs[standard_fields.InputDataFields.point_positions],
                   [-1, 3]))[0]
  elif standard_fields.InputDataFields.object_length_points in inputs:
    example_size = tf.shape(
        tf.reshape(inputs[standard_fields.InputDataFields.object_length_points],
                   [-1]))[0]
  else:
    raise ValueError('Could not find a key to compute example size from.')

  valid_mask_num_points = tf.concat([
      tf.ones([num_points], dtype=tf.int32),
      tf.zeros([example_size - num_points], dtype=tf.int32)
  ],
                                    axis=0)
  return tf.cast(valid_mask_num_points, dtype=tf.bool)


def num_voxels_mask(inputs):
  """Returns a boolean mask that will keep the first num_voxels values."""
  num_voxels = tf.squeeze(
      inputs[standard_fields.InputDataFields.num_valid_voxels])
  if standard_fields.InputDataFields.voxel_positions in inputs:
    example_size = tf.shape(
        tf.reshape(inputs[standard_fields.InputDataFields.voxel_positions],
                   [-1, 3]))[0]
  elif standard_fields.InputDataFields.object_length_voxels in inputs:
    example_size = tf.shape(
        tf.reshape(inputs[standard_fields.InputDataFields.object_length_voxels],
                   [-1]))[0]
  else:
    raise ValueError('Could not find a key to compute example size from.')

  valid_mask_num_voxels = tf.concat([
      tf.ones([num_voxels], dtype=tf.int32),
      tf.zeros([example_size - num_voxels], dtype=tf.int32)
  ],
                                    axis=0)
  return tf.cast(valid_mask_num_voxels, dtype=tf.bool)


def points_within_objects_mask(inputs):
  """Returns a boolean mask indicating points that fall within objects."""
  return tf.greater(
      tf.reshape(
          inputs[standard_fields.InputDataFields.object_instance_id_points],
          [-1]), 0)


def voxels_within_objects_mask(inputs):
  """Returns a boolean mask indicating voxels that fall within objects."""
  return tf.greater(
      tf.reshape(
          inputs[standard_fields.InputDataFields.object_instance_id_voxels],
          [-1]), 0)


def apply_mask_to_input_point_tensors(inputs, valid_mask):
  """Applies mask to input point tensors."""
  masked_tensors = {}
  for field in standard_fields.get_input_point_fields():
    if field in inputs:
      if field != standard_fields.InputDataFields.num_valid_points:
        masked_tensors[field] = tf.boolean_mask(inputs[field], valid_mask)
  return masked_tensors


def apply_mask_to_input_voxel_tensors(inputs, valid_mask):
  """Applies mask to input voxel tensors."""
  masked_tensors = {}
  for field in standard_fields.get_input_voxel_fields():
    if field in inputs:
      if field != standard_fields.InputDataFields.num_valid_voxels:
        masked_tensors[field] = tf.boolean_mask(inputs[field], valid_mask)
  return masked_tensors


def apply_mask_to_output_point_tensors(outputs, valid_mask):
  """Applies mask to output point tensors."""
  for field in standard_fields.get_output_point_fields():
    if field in outputs and outputs[field] is not None:
      outputs[field] = tf.boolean_mask(outputs[field], valid_mask)


def apply_mask_to_output_voxel_tensors(outputs, valid_mask):
  """Applies mask to output voxel tensors."""
  for field in standard_fields.get_output_voxel_fields():
    if field in outputs and outputs[field] is not None:
      outputs[field] = tf.boolean_mask(outputs[field], valid_mask)
