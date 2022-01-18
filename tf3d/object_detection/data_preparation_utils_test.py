# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Tests for ...object_detection.data_preparation_utils."""

import numpy as np
import tensorflow as tf
from tf3d import standard_fields
from tf3d.object_detection import data_preparation_utils


class DataPreparationUtilsTest(tf.test.TestCase):

  def test_prepare_waymo_open_dataset(self):
    inputs = {}
    inputs[standard_fields.InputDataFields.point_positions] = tf.random.uniform(
        [1000, 3], minval=-1.0, maxval=1.0, dtype=tf.float32)
    inputs[
        standard_fields.InputDataFields.point_intensities] = tf.random.uniform(
            [1000, 1], minval=0.0, maxval=100.0, dtype=tf.float32)
    inputs[
        standard_fields.InputDataFields.point_elongations] = tf.random.uniform(
            [1000, 1], minval=-1.0, maxval=1.0, dtype=tf.float32)
    inputs[standard_fields.InputDataFields.point_normals] = tf.random.uniform(
        [1000, 3], minval=-1.0, maxval=1.0, dtype=tf.float32)
    inputs['cameras/front/intrinsics/K'] = tf.random.uniform([3, 3],
                                                             minval=0.0,
                                                             maxval=1.0,
                                                             dtype=tf.float32)
    inputs['cameras/front/extrinsics/R'] = tf.random.uniform([3, 3],
                                                             minval=-1.0,
                                                             maxval=1.0,
                                                             dtype=tf.float32)
    inputs['cameras/front/extrinsics/t'] = tf.random.uniform([3],
                                                             minval=-1.0,
                                                             maxval=1.0,
                                                             dtype=tf.float32)
    inputs['cameras/front/image'] = tf.random.uniform([200, 300, 3],
                                                      minval=0,
                                                      maxval=255,
                                                      dtype=tf.int32)
    inputs['objects/pose/R'] = tf.random.uniform([10, 3, 3],
                                                 minval=-1.0,
                                                 maxval=1.0,
                                                 dtype=tf.float32)
    inputs['objects/pose/t'] = tf.random.uniform([10, 3],
                                                 minval=-1.0,
                                                 maxval=1.0,
                                                 dtype=tf.float32)
    inputs['objects/shape/dimension'] = tf.random.uniform([10, 3],
                                                          minval=0.01,
                                                          maxval=1.0,
                                                          dtype=tf.float32)
    inputs['objects/category/label'] = tf.random.uniform([10],
                                                         minval=0,
                                                         maxval=20,
                                                         dtype=tf.int32)
    prepared_inputs = data_preparation_utils.prepare_waymo_open_dataset(
        inputs, valid_object_classes=[1, 2, 3, 4, 5])
    for key in [
        standard_fields.InputDataFields.point_positions,
        standard_fields.InputDataFields.point_intensities,
        standard_fields.InputDataFields.point_elongations,
        standard_fields.InputDataFields.camera_intrinsics,
        standard_fields.InputDataFields.camera_rotation_matrix,
        standard_fields.InputDataFields.camera_translation,
        standard_fields.InputDataFields.camera_image,
        standard_fields.InputDataFields.camera_raw_image,
        standard_fields.InputDataFields.camera_original_image,
        standard_fields.InputDataFields.objects_rotation_matrix,
        standard_fields.InputDataFields.objects_center,
        standard_fields.InputDataFields.objects_length,
        standard_fields.InputDataFields.objects_width,
        standard_fields.InputDataFields.objects_height,
        standard_fields.InputDataFields.objects_class
    ]:
      self.assertIn(key, prepared_inputs)

  def test_prepare_kitti_dataset(self):
    inputs = {}
    inputs[standard_fields.InputDataFields.point_positions] = tf.random.uniform(
        [1000, 3], minval=-1.0, maxval=1.0, dtype=tf.float32)
    inputs[
        standard_fields.InputDataFields.point_intensities] = tf.random.uniform(
            [1000, 1], minval=0.0, maxval=100.0, dtype=tf.float32)
    inputs['cameras/cam02/intrinsics/K'] = tf.random.uniform([3, 3],
                                                             minval=0.0,
                                                             maxval=1.0,
                                                             dtype=tf.float32)
    inputs['cameras/cam02/extrinsics/R'] = tf.random.uniform([3, 3],
                                                             minval=-1.0,
                                                             maxval=1.0,
                                                             dtype=tf.float32)
    inputs['cameras/cam02/extrinsics/t'] = tf.random.uniform([3],
                                                             minval=-1.0,
                                                             maxval=1.0,
                                                             dtype=tf.float32)
    inputs['cameras/cam02/image'] = tf.random.uniform([200, 300, 3],
                                                      minval=0,
                                                      maxval=255,
                                                      dtype=tf.int32)
    inputs['objects/pose/R'] = tf.random.uniform([10, 3, 3],
                                                 minval=-1.0,
                                                 maxval=1.0,
                                                 dtype=tf.float32)
    inputs['objects/pose/t'] = tf.random.uniform([10, 3],
                                                 minval=-1.0,
                                                 maxval=1.0,
                                                 dtype=tf.float32)
    inputs['objects/shape/dimension'] = tf.random.uniform([10, 3],
                                                          minval=0.01,
                                                          maxval=1.0,
                                                          dtype=tf.float32)
    inputs['objects/category/label'] = tf.random.uniform([10],
                                                         minval=0,
                                                         maxval=20,
                                                         dtype=tf.int32)
    prepared_inputs = data_preparation_utils.prepare_kitti_dataset(
        inputs, valid_object_classes=[1, 2])
    for key in [
        standard_fields.InputDataFields.point_positions,
        standard_fields.InputDataFields.point_intensities,
        standard_fields.InputDataFields.camera_intrinsics,
        standard_fields.InputDataFields.camera_rotation_matrix,
        standard_fields.InputDataFields.camera_translation,
        standard_fields.InputDataFields.camera_image,
        standard_fields.InputDataFields.camera_raw_image,
        standard_fields.InputDataFields.camera_original_image,
        standard_fields.InputDataFields.objects_rotation_matrix,
        standard_fields.InputDataFields.objects_center,
        standard_fields.InputDataFields.objects_length,
        standard_fields.InputDataFields.objects_width,
        standard_fields.InputDataFields.objects_height,
        standard_fields.InputDataFields.objects_class
    ]:
      self.assertIn(key, prepared_inputs)

  def test_proxy_dataset(self):
    inputs = {}
    inputs[standard_fields.InputDataFields.point_positions] = tf.random.uniform(
        [1000, 3], minval=-1.0, maxval=1.0, dtype=tf.float32)
    inputs[
        standard_fields.InputDataFields.point_intensities] = tf.random.uniform(
            [1000, 1], minval=0.0, maxval=100.0, dtype=tf.float32)

    inputs['camera_intrinsics'] = tf.random.uniform([3, 3], minval=0.0,
                                                    maxval=1.0,
                                                    dtype=tf.float32)

    inputs['camera_rotation_matrix'] = tf.random.uniform([3, 3], minval=0.0,
                                                         maxval=1.0,
                                                         dtype=tf.float32)

    inputs['camera_translation'] = tf.random.uniform([3], minval=0.0,
                                                     maxval=1.0,
                                                     dtype=tf.float32)

    inputs['image'] = tf.random.uniform([640, 512, 3],
                                        minval=0,
                                        maxval=255,
                                        dtype=tf.int32)

    inputs['objects_rotation'] = tf.random.uniform([10, 3, 3], minval=0.0,
                                                   maxval=1.0,
                                                   dtype=tf.float32)
    inputs['objects_center'] = tf.random.uniform([10, 3], minval=0.0,
                                                 maxval=1.0,
                                                 dtype=tf.float32)

    inputs['objects_length'] = tf.random.uniform([10, 1], minval=0.0,
                                                 maxval=1.0,
                                                 dtype=tf.float32)
    inputs['objects_width'] = tf.random.uniform([10, 1], minval=0.0,
                                                maxval=1.0,
                                                dtype=tf.float32)
    inputs['objects_height'] = tf.random.uniform([10, 1], minval=0.0,
                                                 maxval=1.0,
                                                 dtype=tf.float32)

    inputs['objects_class'] = tf.random.uniform([10],
                                                minval=0,
                                                maxval=20,
                                                dtype=tf.int32)

    prepared_inputs = data_preparation_utils.prepare_proxy_dataset(inputs)
    for key in [
        standard_fields.InputDataFields.point_positions,
        standard_fields.InputDataFields.point_intensities,
        standard_fields.InputDataFields.camera_intrinsics,
        standard_fields.InputDataFields.camera_rotation_matrix,
        standard_fields.InputDataFields.camera_translation,
        standard_fields.InputDataFields.camera_image,
        standard_fields.InputDataFields.camera_raw_image,
        standard_fields.InputDataFields.camera_original_image,
        standard_fields.InputDataFields.objects_rotation_matrix,
        standard_fields.InputDataFields.objects_center,
        standard_fields.InputDataFields.objects_length,
        standard_fields.InputDataFields.objects_width,
        standard_fields.InputDataFields.objects_height,
        standard_fields.InputDataFields.objects_class
    ]:
      self.assertIn(key, prepared_inputs)

  def test_compute_kitti_difficulty(self):
    image_height = 240
    boxes = tf.constant([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0],
                         [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0],
                         [0.0, 0.0, 0.15, 1.0], [0.0, 0.0, 0.1, 1.0]])
    occlusions = tf.constant([[0], [1], [1], [1], [0], [1]])
    truncations = tf.constant([[0.1], [0.2], [0.5], [0.8], [0.1], [0.1]])

    difficulty = data_preparation_utils.compute_kitti_difficulty(
        boxes, occlusions, truncations, image_height)

    expected_difficulty = np.array([[3], [2], [1], [0], [2], [0]])
    self.assertAllEqual(difficulty.numpy(), expected_difficulty)

  def test_get_waymo_per_frame_with_prediction_feature_spec(self):
    feature_spec = (
        data_preparation_utils.get_waymo_per_frame_with_prediction_feature_spec(
            num_object_classes=10, encoded_features_dimension=64))
    self.assertIn('predictions', feature_spec)


if __name__ == '__main__':
  tf.test.main()
