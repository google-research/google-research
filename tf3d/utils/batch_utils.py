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

"""Utility functions used in losses."""

from tf3d import standard_fields


def get_batch_size_1_input_images(inputs, b):
  """Returns input dictionary containing tensors with batch size of 1.

  Note that this function only applies its example selection to the image
  tensors.

  Args:
    inputs: A dictionary of tf.Tensors with our input data.
    b: Example index in the batch.

  Returns:
    inputs_1:  A dictionary of tf.Tensors with batch size of one.
  """
  b_1_inputs = {}
  for field in standard_fields.get_input_image_fields():
    if field in inputs:
      b_1_inputs[field] = inputs[field][b:b + 1, Ellipsis]
  return b_1_inputs


def get_batch_size_1_input_points(inputs, b):
  """Returns input dictionary containing tensors with batch size of 1.

  Note that this function only applies its example selection to the point
  tensors.

  Args:
    inputs: A dictionary of tf.Tensors with our input data.
    b: Example index in the batch.

  Returns:
    inputs_1:  A dictionary of tf.Tensors with batch size of one.
  """
  b_1_inputs = {}
  for field in standard_fields.get_input_point_fields():
    if field in inputs:
      b_1_inputs[field] = inputs[field][b]
  return b_1_inputs


def get_batch_size_1_input_voxels(inputs, b):
  """Returns input dictionary containing tensors with batch size of 1.

  Note that this function only applies its example selection to the voxel
  tensors.

  Args:
    inputs: A dictionary of tf.Tensors with our input data.
    b: Example index in the batch.

  Returns:
    inputs_1:  A dictionary of tf.Tensors with batch size of one.
  """
  b_1_inputs = {}
  for field in standard_fields.get_input_voxel_fields():
    if field in inputs:
      b_1_inputs[field] = inputs[field][b]
  return b_1_inputs


def get_batch_size_1_input_objects(inputs, b):
  """Returns input dictionary containing tensors with batch size of 1.

  Note that this function only applies its example selection to the object
  tensors.

  Args:
    inputs: A dictionary of tf.Tensors with our input data.
    b: Example index in the batch.

  Returns:
    inputs_1:  A dictionary of tf.Tensors with batch size of one.
  """
  b_1_inputs = {}
  for field in standard_fields.get_input_object_fields():
    if field in inputs:
      b_1_inputs[field] = inputs[field][b]
  return b_1_inputs


def get_batch_size_1_output_images(outputs, b):
  """Returns output dictionary containing tensors with batch size of 1.

  Note that this function only applies its example selection to the image
  tensors.

  Args:
    outputs: A dictionary of tf.Tensors with the network output.
    b: Example index in the batch.

  Returns:
    outputs_1:  A dictionary of tf.Tensors with batch size of one.
  """
  b_1_outputs = {}
  for field in standard_fields.get_output_image_fields():
    if field in outputs:
      b_1_outputs[field] = outputs[field][b:b + 1, Ellipsis]
  return b_1_outputs


def get_batch_size_1_output_points(outputs, b):
  """Returns output dictionary containing tensors with batch size of 1.

  Note that this function only applies its example selection to the point
  tensors.

  Args:
    outputs: A dictionary of tf.Tensors with the network output.
    b: Example index in the batch.

  Returns:
    outputs_1:  A dictionary of tf.Tensors with batch size of one.
  """
  b_1_outputs = {}
  for field in standard_fields.get_output_point_fields():
    if field in outputs and outputs[field] is not None:
      b_1_outputs[field] = outputs[field][b]
  return b_1_outputs


def get_batch_size_1_output_voxels(outputs, b):
  """Returns output dictionary containing tensors with batch size of 1.

  Note that this function only applies its example selection to the voxel
  tensors.

  Args:
    outputs: A dictionary of tf.Tensors with the network output.
    b: Example index in the batch.

  Returns:
    outputs_1:  A dictionary of tf.Tensors with batch size of one.
  """
  b_1_outputs = {}
  for field in standard_fields.get_output_voxel_fields():
    if field in outputs and outputs[field] is not None:
      b_1_outputs[field] = outputs[field][b]
  return b_1_outputs


def get_batch_size_1_output_objects(outputs, b):
  """Returns output dictionary containing tensors with batch size of 1.

  Note that this function only applies its example selection to the object
  tensors.

  Args:
    outputs: A dictionary of tf.Tensors with the network output.
    b: Example index in the batch.

  Returns:
    outputs_1:  A dictionary of tf.Tensors with batch size of one.
  """
  b_1_outputs = {}
  for field in standard_fields.get_output_object_fields():
    if field in outputs and outputs[field] is not None:
      b_1_outputs[field] = outputs[field][b]
  return b_1_outputs


def get_batch_size_1_output_anchors(outputs, b):
  """Returns output dictionary containing tensors with batch size of 1.

  Note that this function only applies its example selection to the anchor
  tensors.

  Args:
    outputs: A dictionary of tf.Tensors with the network output.
    b: Example index in the batch.

  Returns:
    outputs_1:  A dictionary of tf.Tensors with batch size of one.
  """
  b_1_outputs = {}
  for field in standard_fields.get_output_anchor_fields():
    if field in outputs and outputs[field] is not None:
      b_1_outputs[field] = outputs[field][b]
  return b_1_outputs
