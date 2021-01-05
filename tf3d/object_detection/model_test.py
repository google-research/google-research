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

"""Tests for ...tf3d.object_detection.model."""

import functools
import tensorflow as tf
from tf3d import standard_fields
from tf3d.losses import box_prediction_losses
from tf3d.losses import classification_losses
from tf3d.object_detection import model


class ObjectDetectionModelTest(tf.test.TestCase):

  def get_inputs(self, num_voxels, num_classes):
    return {
        standard_fields.InputDataFields.num_valid_voxels:
            tf.constant([num_voxels - 20], dtype=tf.int32),
        standard_fields.InputDataFields.voxel_positions:
            tf.random.uniform(
                shape=(1, num_voxels, 3), minval=-10.0, maxval=10.0),
        standard_fields.InputDataFields.voxel_features:
            tf.random.uniform(
                shape=(1, num_voxels, 10), minval=-2.0, maxval=2.0),
        standard_fields.InputDataFields.voxel_xyz_indices:
            tf.random.uniform(
                shape=(1, num_voxels, 3),
                minval=-100,
                maxval=100,
                dtype=tf.int32),
        standard_fields.InputDataFields.object_center_voxels:
            tf.random.uniform(
                shape=(1, num_voxels, 3), minval=-10.0, maxval=10.0),
        standard_fields.InputDataFields.object_length_voxels:
            tf.random.uniform(
                shape=(1, num_voxels, 1), minval=0.01, maxval=10.0),
        standard_fields.InputDataFields.object_height_voxels:
            tf.random.uniform(
                shape=(1, num_voxels, 1), minval=0.01, maxval=10.0),
        standard_fields.InputDataFields.object_width_voxels:
            tf.random.uniform(
                shape=(1, num_voxels, 1), minval=0.01, maxval=10.0),
        standard_fields.InputDataFields.object_rotation_matrix_voxels:
            tf.random.uniform(
                shape=(1, num_voxels, 3, 3), minval=-1.0, maxval=1.0),
        standard_fields.InputDataFields.object_class_voxels:
            tf.random.uniform(
                shape=(1, num_voxels, 1),
                minval=0,
                maxval=num_classes,
                dtype=tf.int32),
        standard_fields.InputDataFields.object_instance_id_voxels:
            tf.random.uniform(
                shape=(1, num_voxels, 1), minval=0, maxval=10, dtype=tf.int32),
    }

  def test_call_train(self):
    num_classes = 5
    loss_fn_box_corner_distance_on_voxel_tensors = functools.partial(
        box_prediction_losses.box_corner_distance_loss_on_voxel_tensors,
        is_intermediate=False,
        loss_type='absolute_difference',
        is_balanced=True)
    loss_fn_box_classification_using_center_distance = functools.partial(
        classification_losses.box_classification_using_center_distance_loss,
        is_intermediate=False,
        is_balanced=True,
        max_positive_normalized_distance=0.3)
    loss_fn_hard_negative_classification = functools.partial(
        classification_losses.hard_negative_classification_loss,
        is_intermediate=False,
        gamma=1.0)
    loss_names_to_functions = {
        'box_corner_distance_loss_on_voxel_tensors':
            loss_fn_box_corner_distance_on_voxel_tensors,
        'box_classification_using_center_distance_loss':
            loss_fn_box_classification_using_center_distance,
        'hard_negative_classification_loss':
            loss_fn_hard_negative_classification,
    }
    loss_names_to_weights = {
        'box_corner_distance_loss_on_voxel_tensors': 5.0,
        'box_classification_using_center_distance_loss': 1.0,
        'hard_negative_classification_loss': 1.0,
    }
    object_detection_model = model.ObjectDetectionModel(
        loss_names_to_functions=loss_names_to_functions,
        loss_names_to_weights=loss_names_to_weights,
        num_classes=num_classes,
        predict_rotation_x=True,
        predict_rotation_y=True,
        predict_rotation_z=True)
    num_voxels = 100
    inputs = self.get_inputs(num_voxels=num_voxels, num_classes=num_classes)
    outputs = object_detection_model(inputs, training=True)
    self.assertEqual(
        outputs[standard_fields.DetectionResultFields.object_semantic_voxels]
        .get_shape(), (1, num_voxels, num_classes))
    self.assertEqual(
        outputs[standard_fields.DetectionResultFields.object_center_voxels]
        .get_shape(), (1, num_voxels, 3))
    self.assertEqual(
        outputs[standard_fields.DetectionResultFields.object_length_voxels]
        .get_shape(), (1, num_voxels, 1))
    self.assertEqual(
        outputs[standard_fields.DetectionResultFields.object_height_voxels]
        .get_shape(), (1, num_voxels, 1))
    self.assertEqual(
        outputs[standard_fields.DetectionResultFields.object_width_voxels]
        .get_shape(), (1, num_voxels, 1))
    self.assertEqual(
        outputs[standard_fields.DetectionResultFields
                .object_rotation_x_cos_voxels].get_shape(),
        (1, num_voxels, 1))
    self.assertEqual(
        outputs[standard_fields.DetectionResultFields
                .object_rotation_x_sin_voxels].get_shape(),
        (1, num_voxels, 1))
    self.assertEqual(
        outputs[standard_fields.DetectionResultFields
                .object_rotation_y_cos_voxels].get_shape(),
        (1, num_voxels, 1))
    self.assertEqual(
        outputs[standard_fields.DetectionResultFields
                .object_rotation_y_sin_voxels].get_shape(),
        (1, num_voxels, 1))
    self.assertEqual(
        outputs[standard_fields.DetectionResultFields
                .object_rotation_z_cos_voxels].get_shape(),
        (1, num_voxels, 1))
    self.assertEqual(
        outputs[standard_fields.DetectionResultFields
                .object_rotation_z_sin_voxels].get_shape(),
        (1, num_voxels, 1))

  def test_call_eval(self):
    num_classes = 5
    object_detection_model = model.ObjectDetectionModel(
        num_classes=num_classes,
        predict_rotation_x=True,
        predict_rotation_y=True,
        predict_rotation_z=True)
    num_voxels = 100
    inputs = self.get_inputs(num_voxels=num_voxels, num_classes=num_classes)
    outputs = object_detection_model(inputs, training=False)
    num_valid_voxels = inputs[
        standard_fields.InputDataFields.num_valid_voxels].numpy()[0]
    self.assertEqual(
        outputs[standard_fields.DetectionResultFields.object_semantic_voxels]
        .get_shape(), (num_valid_voxels, num_classes))
    self.assertEqual(
        outputs[standard_fields.DetectionResultFields.object_center_voxels]
        .get_shape(), (num_valid_voxels, 3))
    self.assertEqual(
        outputs[standard_fields.DetectionResultFields.object_length_voxels]
        .get_shape(), (num_valid_voxels, 1))
    self.assertEqual(
        outputs[standard_fields.DetectionResultFields.object_height_voxels]
        .get_shape(), (num_valid_voxels, 1))
    self.assertEqual(
        outputs[standard_fields.DetectionResultFields.object_width_voxels]
        .get_shape(), (num_valid_voxels, 1))
    self.assertEqual(
        outputs[standard_fields.DetectionResultFields
                .object_rotation_x_cos_voxels].get_shape(),
        (num_valid_voxels, 1))
    self.assertEqual(
        outputs[standard_fields.DetectionResultFields
                .object_rotation_x_sin_voxels].get_shape(),
        (num_valid_voxels, 1))
    self.assertEqual(
        outputs[standard_fields.DetectionResultFields
                .object_rotation_y_cos_voxels].get_shape(),
        (num_valid_voxels, 1))
    self.assertEqual(
        outputs[standard_fields.DetectionResultFields
                .object_rotation_y_sin_voxels].get_shape(),
        (num_valid_voxels, 1))
    self.assertEqual(
        outputs[standard_fields.DetectionResultFields
                .object_rotation_z_cos_voxels].get_shape(),
        (num_valid_voxels, 1))
    self.assertEqual(
        outputs[standard_fields.DetectionResultFields
                .object_rotation_z_sin_voxels].get_shape(),
        (num_valid_voxels, 1))
    self.assertIn(standard_fields.DetectionResultFields.objects_center, outputs)
    self.assertIn(standard_fields.DetectionResultFields.objects_length, outputs)
    self.assertIn(standard_fields.DetectionResultFields.objects_height, outputs)
    self.assertIn(standard_fields.DetectionResultFields.objects_width, outputs)
    self.assertIn(standard_fields.DetectionResultFields.objects_rotation_matrix,
                  outputs)
    self.assertIn(standard_fields.DetectionResultFields.objects_rotation_x_cos,
                  outputs)
    self.assertIn(standard_fields.DetectionResultFields.objects_rotation_x_sin,
                  outputs)
    self.assertIn(standard_fields.DetectionResultFields.objects_rotation_y_cos,
                  outputs)
    self.assertIn(standard_fields.DetectionResultFields.objects_rotation_y_sin,
                  outputs)
    self.assertIn(standard_fields.DetectionResultFields.objects_rotation_z_cos,
                  outputs)
    self.assertIn(standard_fields.DetectionResultFields.objects_rotation_z_sin,
                  outputs)


if __name__ == '__main__':
  tf.test.main()
