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

"""Tests for ...tf3d.semantic_segmentation.model."""

import tensorflow as tf
from tf3d import standard_fields
from tf3d.semantic_segmentation import model


class ModelTest(tf.test.TestCase):

  def test_call(self):
    num_classes = 5
    segmentation_model = model.SemanticSegmentationModel(
        num_classes=num_classes)

    voxel_features = tf.constant(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]],
        dtype=tf.float32)
    voxel_xyz_indices = tf.constant(
        [[[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]], dtype=tf.int32)
    num_valid_voxels = tf.constant([4], dtype=tf.int32)
    num_valid_points = tf.constant([7], dtype=tf.int32)
    object_class_voxels = tf.constant([[[0], [1], [2], [3]]], dtype=tf.int32)
    points_to_voxel_mapping = tf.constant([[0, 1, 1, 2, 2, 3, 3, 0]],
                                          dtype=tf.int32)
    inputs = {
        standard_fields.InputDataFields.voxel_features:
            voxel_features,
        standard_fields.InputDataFields.voxel_xyz_indices:
            voxel_xyz_indices,
        standard_fields.InputDataFields.num_valid_voxels:
            num_valid_voxels,
        standard_fields.InputDataFields.object_class_voxels:
            object_class_voxels,
        standard_fields.InputDataFields.num_valid_points:
            num_valid_points,
        standard_fields.InputDataFields.points_to_voxel_mapping:
            points_to_voxel_mapping,
    }
    outputs = segmentation_model(inputs, training=True)
    self.assertAllEqual(
        outputs[
            standard_fields.DetectionResultFields.object_semantic_voxels].shape,
        [1, 4, num_classes])

    eval_outputs = segmentation_model(inputs, training=False)
    self.assertAllEqual(
        eval_outputs[
            standard_fields.DetectionResultFields.object_semantic_points].shape,
        [1, 8, num_classes])


if __name__ == '__main__':
  tf.test.main()
