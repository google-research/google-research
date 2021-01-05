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

"""Tests for ...tf3d.losses.box_prediction_losses."""

import tensorflow as tf
from tf3d import standard_fields
from tf3d.losses import box_prediction_losses


class BoxPredictionLossesTest(tf.test.TestCase):

  def _get_random_inputs(self):
    return {
        standard_fields.InputDataFields.object_rotation_matrix_voxels:
            tf.random.uniform([1, 100, 3, 3],
                              minval=-1.0,
                              maxval=1.0,
                              dtype=tf.float32),
        standard_fields.InputDataFields.object_length_voxels:
            tf.random.uniform([1, 100, 1],
                              minval=0.1,
                              maxval=2.0,
                              dtype=tf.float32),
        standard_fields.InputDataFields.object_height_voxels:
            tf.random.uniform([1, 100, 1],
                              minval=0.1,
                              maxval=2.0,
                              dtype=tf.float32),
        standard_fields.InputDataFields.object_width_voxels:
            tf.random.uniform([1, 100, 1],
                              minval=0.1,
                              maxval=2.0,
                              dtype=tf.float32),
        standard_fields.InputDataFields.object_center_voxels:
            tf.random.uniform([1, 100, 3],
                              minval=-5.0,
                              maxval=5.0,
                              dtype=tf.float32),
        standard_fields.InputDataFields.object_class_voxels:
            tf.random.uniform([1, 100, 1], minval=0, maxval=7, dtype=tf.int32),
        standard_fields.InputDataFields.object_instance_id_voxels:
            tf.random.uniform([1, 100, 1], minval=0, maxval=20, dtype=tf.int32),
    }

  def _get_empty_inputs(self):
    inputs = self._get_random_inputs()
    for key in inputs:
      if key in inputs:
        tensor_shape = inputs[key].shape.as_list()
        tensor_shape[1] = 0
        inputs[key] = tf.zeros(tensor_shape, dtype=inputs[key].dtype)
    inputs[standard_fields.InputDataFields.num_valid_voxels] = tf.constant(
        [0], dtype=tf.int32)
    return inputs

  def _get_dictionaries_for_distance_loss_relative(self):
    gt_box_center = tf.reshape(
        tf.constant([10.0, -20.0, 30.0], dtype=tf.float32), [1, 1, 3])
    gt_box_length = tf.reshape(
        tf.constant([1.0], dtype=tf.float32), [1, 1, 1])
    gt_box_height = tf.reshape(
        tf.constant([2.0], dtype=tf.float32), [1, 1, 1])
    gt_box_width = tf.reshape(
        tf.constant([3.0], dtype=tf.float32), [1, 1, 1])
    gt_box_r = tf.reshape(tf.eye(3, dtype=tf.float32), [1, 1, 3, 3])
    gt_box_class = tf.reshape(tf.constant([1], dtype=tf.int32), [1, 1, 1])
    gt_instance_ids = tf.reshape(tf.constant([1], dtype=tf.int32), [1, 1, 1])
    pred_box_center1 = tf.reshape(
        tf.constant([10.1, -20.1, 30.1], dtype=tf.float32), [1, 1, 3])
    pred_box_length1 = tf.reshape(
        tf.constant([1.1], dtype=tf.float32), [1, 1, 1])
    pred_box_height1 = tf.reshape(
        tf.constant([2.1], dtype=tf.float32), [1, 1, 1])
    pred_box_width1 = tf.reshape(
        tf.constant([3.1], dtype=tf.float32), [1, 1, 1])
    pred_box_r1 = tf.reshape(tf.eye(3, dtype=tf.float32), [1, 1, 3, 3])
    pred_box_center2 = tf.reshape(
        tf.constant([10.1, -20.2, 30.2], dtype=tf.float32), [1, 1, 3])
    pred_box_length2 = tf.reshape(
        tf.constant([1.11], dtype=tf.float32), [1, 1, 1])
    pred_box_height2 = tf.reshape(
        tf.constant([2.11], dtype=tf.float32), [1, 1, 1])
    pred_box_width2 = tf.reshape(
        tf.constant([3.11], dtype=tf.float32), [1, 1, 1])
    pred_box_r2 = tf.reshape(tf.eye(3, dtype=tf.float32), [1, 1, 3, 3])
    inputs = {
        standard_fields.InputDataFields.object_rotation_matrix_voxels:
            gt_box_r,
        standard_fields.InputDataFields.object_length_voxels:
            gt_box_length,
        standard_fields.InputDataFields.object_height_voxels:
            gt_box_height,
        standard_fields.InputDataFields.object_width_voxels:
            gt_box_width,
        standard_fields.InputDataFields.object_center_voxels:
            gt_box_center,
        standard_fields.InputDataFields.object_class_voxels:
            gt_box_class,
        standard_fields.InputDataFields.object_instance_id_voxels:
            gt_instance_ids,
    }
    outputs1 = {
        standard_fields.DetectionResultFields.object_rotation_matrix_voxels:
            pred_box_r1,
        standard_fields.DetectionResultFields.object_length_voxels:
            pred_box_length1,
        standard_fields.DetectionResultFields.object_height_voxels:
            pred_box_height1,
        standard_fields.DetectionResultFields.object_width_voxels:
            pred_box_width1,
        standard_fields.DetectionResultFields.object_center_voxels:
            pred_box_center1,
    }
    outputs2 = {
        standard_fields.DetectionResultFields.object_rotation_matrix_voxels:
            pred_box_r2,
        standard_fields.DetectionResultFields.object_length_voxels:
            pred_box_length2,
        standard_fields.DetectionResultFields.object_height_voxels:
            pred_box_height2,
        standard_fields.DetectionResultFields.object_width_voxels:
            pred_box_width2,
        standard_fields.DetectionResultFields.object_center_voxels:
            pred_box_center2,
    }
    return inputs, outputs1, outputs2

  def test_box_size_regression_loss_on_voxel_tensors_empty_inputs(self):
    inputs = self._get_empty_inputs()
    outputs = {
        standard_fields.DetectionResultFields.object_length_voxels:
            tf.zeros([1, 0, 3], dtype=tf.float32),
        standard_fields.DetectionResultFields.object_height_voxels:
            tf.zeros([1, 0, 3], dtype=tf.float32),
        standard_fields.DetectionResultFields.object_width_voxels:
            tf.zeros([1, 0, 3], dtype=tf.float32),
    }
    loss = box_prediction_losses.box_size_regression_loss_on_voxel_tensors(
        inputs=inputs,
        outputs=outputs,
        loss_type='huber')
    self.assertAllClose(loss.numpy(), 0.0)

  def test_box_size_regression_loss_on_voxel_tensors_correct_prediction(self):
    inputs = self._get_random_inputs()
    inputs[standard_fields.InputDataFields.num_valid_voxels] = tf.constant(
        [100], dtype=tf.int32)
    outputs = {
        standard_fields.DetectionResultFields.object_length_voxels:
            inputs[standard_fields.InputDataFields.object_length_voxels],
        standard_fields.DetectionResultFields.object_height_voxels:
            inputs[standard_fields.InputDataFields.object_height_voxels],
        standard_fields.DetectionResultFields.object_width_voxels:
            inputs[standard_fields.InputDataFields.object_width_voxels],
    }
    loss = box_prediction_losses.box_size_regression_loss_on_voxel_tensors(
        inputs=inputs,
        outputs=outputs,
        loss_type='huber')
    self.assertAllClose(loss.numpy(), 0.0)

  def test_box_size_regression_loss_on_voxel_tensors_relative(self):
    (inputs, outputs1,
     outputs2) = self._get_dictionaries_for_distance_loss_relative()
    inputs[standard_fields.InputDataFields.num_valid_voxels] = tf.constant(
        [1], dtype=tf.int32)
    loss1 = box_prediction_losses.box_size_regression_loss_on_voxel_tensors(
        inputs=inputs,
        outputs=outputs1,
        loss_type='huber')
    loss2 = box_prediction_losses.box_size_regression_loss_on_voxel_tensors(
        inputs=inputs,
        outputs=outputs2,
        loss_type='huber')
    self.assertGreater(loss2.numpy(), loss1.numpy())

  def test_box_center_distance_loss_on_voxel_tensors_empty_inputs(self):
    inputs = self._get_empty_inputs()
    outputs = {
        standard_fields.DetectionResultFields.object_center_voxels:
            tf.zeros([1, 0, 3], dtype=tf.float32),
    }
    loss = box_prediction_losses.box_center_distance_loss_on_voxel_tensors(
        inputs=inputs,
        outputs=outputs,
        loss_type='huber')
    self.assertAllClose(loss.numpy(), 0.0)

  def test_box_center_distance_loss_on_voxel_tensors_correct_prediction(self):
    inputs = self._get_random_inputs()
    inputs[standard_fields.InputDataFields.num_valid_voxels] = tf.constant(
        [100], dtype=tf.int32)
    outputs = {
        standard_fields.DetectionResultFields.object_center_voxels:
            inputs[standard_fields.InputDataFields.object_center_voxels],
    }
    loss = box_prediction_losses.box_center_distance_loss_on_voxel_tensors(
        inputs=inputs,
        outputs=outputs,
        loss_type='huber')
    self.assertAllClose(loss.numpy(), 0.0)

  def test_box_center_distance_loss_on_voxel_tensors_relative(self):
    (inputs, outputs1,
     outputs2) = self._get_dictionaries_for_distance_loss_relative()
    inputs[standard_fields.InputDataFields.num_valid_voxels] = tf.constant(
        [1], dtype=tf.int32)
    loss1 = box_prediction_losses.box_center_distance_loss_on_voxel_tensors(
        inputs=inputs,
        outputs=outputs1,
        loss_type='huber')
    loss2 = box_prediction_losses.box_center_distance_loss_on_voxel_tensors(
        inputs=inputs,
        outputs=outputs2,
        loss_type='huber')
    self.assertGreater(loss2.numpy(), loss1.numpy())

  def test_box_corner_distance_loss_on_voxel_tensors_empty_inputs(self):
    inputs = self._get_empty_inputs()
    inputs[standard_fields.InputDataFields.num_valid_voxels] = tf.constant(
        [0], dtype=tf.int32)
    outputs = {
        standard_fields.DetectionResultFields.object_rotation_matrix_voxels:
            tf.zeros([1, 0, 3, 3], dtype=tf.float32),
        standard_fields.DetectionResultFields.object_length_voxels:
            tf.zeros([1, 0, 1], dtype=tf.float32),
        standard_fields.DetectionResultFields.object_height_voxels:
            tf.zeros([1, 0, 1], dtype=tf.float32),
        standard_fields.DetectionResultFields.object_width_voxels:
            tf.zeros([1, 0, 1], dtype=tf.float32),
        standard_fields.DetectionResultFields.object_center_voxels:
            tf.zeros([1, 0, 3], dtype=tf.float32),
    }
    loss = box_prediction_losses.box_corner_distance_loss_on_voxel_tensors(
        inputs=inputs,
        outputs=outputs,
        loss_type='normalized_huber')
    self.assertAllClose(loss.numpy(), 0.0)

  def test_box_corner_distance_loss_on_voxel_tensors_correct_prediction(self):
    inputs = self._get_random_inputs()
    inputs[standard_fields.InputDataFields.num_valid_voxels] = tf.constant(
        [100], dtype=tf.int32)
    outputs = {
        standard_fields.DetectionResultFields.object_rotation_matrix_voxels:
            inputs[standard_fields.InputDataFields.object_rotation_matrix_voxels
                  ],
        standard_fields.DetectionResultFields.object_length_voxels:
            inputs[standard_fields.InputDataFields.object_length_voxels],
        standard_fields.DetectionResultFields.object_height_voxels:
            inputs[standard_fields.InputDataFields.object_height_voxels],
        standard_fields.DetectionResultFields.object_width_voxels:
            inputs[standard_fields.InputDataFields.object_width_voxels],
        standard_fields.DetectionResultFields.object_center_voxels:
            inputs[standard_fields.InputDataFields.object_center_voxels],
    }
    loss = box_prediction_losses.box_corner_distance_loss_on_voxel_tensors(
        inputs=inputs,
        outputs=outputs,
        loss_type='normalized_huber')
    self.assertAllClose(loss.numpy(), 0.0)

  def test_box_corner_distance_loss_on_voxel_tensors_relative(self):
    (inputs, outputs1,
     outputs2) = self._get_dictionaries_for_distance_loss_relative()
    inputs[standard_fields.InputDataFields.num_valid_voxels] = tf.constant(
        [1], dtype=tf.int32)
    loss1 = box_prediction_losses.box_corner_distance_loss_on_voxel_tensors(
        inputs=inputs,
        outputs=outputs1,
        loss_type='normalized_huber')
    loss2 = box_prediction_losses.box_corner_distance_loss_on_voxel_tensors(
        inputs=inputs,
        outputs=outputs2,
        loss_type='normalized_huber')
    self.assertGreater(loss2.numpy(), loss1.numpy())

  def test_box_corner_distance_loss_on_object_tensors_correct_prediction(self):
    voxel_inputs = self._get_random_inputs()
    inputs = {}
    for key, value in standard_fields.get_input_voxel_to_object_field_mapping(
    ).items():
      if key in voxel_inputs:
        inputs[value] = [voxel_inputs[key][0, Ellipsis]]
    outputs = {
        standard_fields.DetectionResultFields.objects_rotation_matrix:
            inputs[standard_fields.InputDataFields.objects_rotation_matrix],
        standard_fields.DetectionResultFields.objects_length:
            inputs[standard_fields.InputDataFields.objects_length],
        standard_fields.DetectionResultFields.objects_height:
            inputs[standard_fields.InputDataFields.objects_height],
        standard_fields.DetectionResultFields.objects_width:
            inputs[standard_fields.InputDataFields.objects_width],
        standard_fields.DetectionResultFields.objects_center:
            inputs[standard_fields.InputDataFields.objects_center],
    }
    loss = box_prediction_losses.box_corner_distance_loss_on_object_tensors(
        inputs=inputs,
        outputs=outputs,
        loss_type='normalized_huber')
    self.assertAllClose(loss.numpy(), 0.0)

  def test_box_corner_distance_loss_on_object_tensors_relative(self):
    (voxel_inputs, voxel_outputs1,
     voxel_outputs2) = self._get_dictionaries_for_distance_loss_relative()
    inputs = {}
    outputs1 = {}
    outputs2 = {}
    for key, value in standard_fields.get_input_voxel_to_object_field_mapping(
    ).items():
      if key in voxel_inputs:
        inputs[value] = [voxel_inputs[key][0, Ellipsis]]
    for key, value in standard_fields.get_output_voxel_to_object_field_mapping(
    ).items():
      if key in voxel_outputs1:
        outputs1[value] = [voxel_outputs1[key][0, Ellipsis]]
    for key, value in standard_fields.get_output_voxel_to_object_field_mapping(
    ).items():
      if key in voxel_outputs2:
        outputs2[value] = [voxel_outputs2[key][0, Ellipsis]]
    loss1 = box_prediction_losses.box_corner_distance_loss_on_object_tensors(
        inputs=inputs,
        outputs=outputs1,
        loss_type='normalized_huber')
    loss2 = box_prediction_losses.box_corner_distance_loss_on_object_tensors(
        inputs=inputs,
        outputs=outputs2,
        loss_type='normalized_huber')
    self.assertGreater(loss2.numpy(), loss1.numpy())


if __name__ == '__main__':
  tf.test.main()
