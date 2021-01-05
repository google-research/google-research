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

"""Tests for ...instance_segmentation.model."""

import functools
import tensorflow as tf
from tf3d import standard_fields
from tf3d.instance_segmentation import model
from tf3d.losses import classification_losses
from tf3d.losses import metric_learning_losses


class ModelTest(tf.test.TestCase):

  def _get_inputs(self, num_voxels, num_points):
    return {
        standard_fields.InputDataFields.num_valid_points:
            tf.constant([num_points - 20], dtype=tf.int32),
        standard_fields.InputDataFields.point_positions:
            tf.random.uniform(
                shape=(1, num_points, 3), minval=-10.0, maxval=10.0),
        standard_fields.InputDataFields.num_valid_voxels:
            tf.constant([num_voxels - 20], dtype=tf.int32),
        standard_fields.InputDataFields.voxel_positions:
            tf.random.uniform(
                shape=(1, num_voxels, 3), minval=-10.0, maxval=10.0),
        standard_fields.InputDataFields.voxel_features:
            tf.random.uniform(
                shape=(1, num_voxels, 10), minval=-10.0, maxval=10.0),
        standard_fields.InputDataFields.voxel_xyz_indices:
            tf.random.uniform(
                shape=(1, num_voxels, 3),
                minval=-100,
                maxval=100,
                dtype=tf.int32),
        standard_fields.InputDataFields.object_instance_id_voxels:
            tf.random.uniform(
                shape=(1, num_voxels), minval=0, maxval=5, dtype=tf.int32),
        standard_fields.InputDataFields.object_class_voxels:
            tf.random.uniform(
                shape=(1, num_voxels, 1), minval=0, maxval=10, dtype=tf.int32),
    }

  def test_call_train(self):
    loss_names_to_functions = {
        'npair_loss':
            functools.partial(
                metric_learning_losses.npair_loss, num_samples=10),
        'regularization_loss':
            metric_learning_losses.embedding_regularization_loss,
        'classification_loss':
            functools.partial(
                classification_losses.classification_loss_using_mask_iou,
                num_samples=10),
    }
    loss_names_to_weights = {
        'npair_loss': 10.0,
        'regularization_loss': 10.0,
        'classification_loss': 1.0,
    }
    num_points = 130
    num_voxels = 100
    num_classes = 5
    inputs = self._get_inputs(num_voxels=num_voxels, num_points=num_points)
    instance_segmentation_model = model.InstanceSegmentationModel(
        loss_names_to_functions=loss_names_to_functions,
        loss_names_to_weights=loss_names_to_weights,
        num_classes=num_classes)
    outputs = instance_segmentation_model(inputs, training=True)
    self.assertAllEqual(
        outputs[standard_fields.DetectionResultFields.instance_embedding_voxels]
        .shape[-1], 64)
    self.assertAllEqual(
        outputs[standard_fields.DetectionResultFields.object_semantic_voxels]
        .shape[-1], 5)

  def test_call_eval(self):
    num_points = 130
    num_voxels = 100
    num_classes = 5
    inputs = self._get_inputs(num_voxels=num_voxels, num_points=num_points)
    instance_segmentation_model = model.InstanceSegmentationModel(
        loss_names_to_functions=None,
        loss_names_to_weights=None,
        num_classes=num_classes)
    outputs = instance_segmentation_model(inputs, training=False)
    self.assertAllEqual(
        outputs[standard_fields.DetectionResultFields.instance_embedding_voxels]
        .shape[-1], 64)
    self.assertAllEqual(
        outputs[standard_fields.DetectionResultFields.object_semantic_voxels]
        .shape[-1], 5)


if __name__ == '__main__':
  tf.test.main()
