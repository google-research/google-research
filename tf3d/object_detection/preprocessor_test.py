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

"""Tests for ...object_detection.preprocessor."""

import tensorflow as tf
from tf3d import standard_fields
from tf3d.object_detection import preprocessor


class ObjectDetectionPreprocessorTest(tf.test.TestCase):

  def _image_correspondence_fn(self, inputs):
    return {
        'view_images': {
            'rgb_view':
                tf.cast(
                    tf.zeros([5, 200, 300, 3], dtype=tf.int32), dtype=tf.uint8),
        },
        'view_indices_2d': {
            'rgb_view':
                tf.random.uniform([5, 100, 2],
                                  minval=-10,
                                  maxval=1000,
                                  dtype=tf.int32)
        }
    }

  def _get_input_dict(self, height=240, width=320):
    return {
        standard_fields.InputDataFields.camera_image:
            tf.zeros((height, width, 3), dtype=tf.uint8),
        standard_fields.InputDataFields.point_positions:
            tf.random.uniform((100, 3), minval=-1, maxval=1),
        standard_fields.InputDataFields.camera_intrinsics:
            tf.constant([
                [160.0, 0.0, 160.0],  # fx,  s, cx
                [0.0, 160.0, 120.0],  #  0, fy, cy
                [0.0, 0.0, 1.0],  #  0,  0,  1
            ]),
        standard_fields.InputDataFields.camera_rotation_matrix:
            tf.eye(3),
        standard_fields.InputDataFields.camera_translation:
            tf.constant([0., 0., 2.]),
        standard_fields.InputDataFields.objects_class:
            tf.constant([1, 4, 5]),
        standard_fields.InputDataFields.objects_length:
            tf.constant([[4.0], [1.0], [1.0]]),
        standard_fields.InputDataFields.objects_height:
            tf.constant([[2.0], [1.0], [4.0]]),
        standard_fields.InputDataFields.objects_width:
            tf.constant([[2.0], [1.0], [1.0]]),
        standard_fields.InputDataFields.objects_rotation_matrix:
            tf.constant([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                         [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                         [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]),
        standard_fields.InputDataFields.objects_center:
            tf.constant([[4.0, 4.0, 4.0], [2.5, 2.5, 2.5], [0.5, 1.5, 9.5]]),
        standard_fields.InputDataFields.objects_difficulty:
            tf.constant([[1], [1], [1]]),
        standard_fields.InputDataFields.objects_instance_id:
            tf.constant([[1], [2], [1]]),
        standard_fields.InputDataFields.objects_has_3d_info:
            tf.constant([1, 1, 0]),
        standard_fields.InputDataFields.camera_image_name:
            tf.convert_to_tensor('image', tf.string),
    }

  def test_preprocess_output_shapes(self):
    height, width = (240, 320)
    input_dict = self._get_input_dict(height, width)
    object_keys = preprocessor._OBJECT_KEYS
    output_keys = [
        standard_fields.InputDataFields.camera_intrinsics,
        standard_fields.InputDataFields.camera_rotation_matrix,
        standard_fields.InputDataFields.camera_translation,
        standard_fields.InputDataFields.point_positions,
        standard_fields.InputDataFields.num_valid_points,
        standard_fields.InputDataFields.object_class_points,
        standard_fields.InputDataFields.object_center_points,
        standard_fields.InputDataFields.object_height_points,
        standard_fields.InputDataFields.object_width_points,
        standard_fields.InputDataFields.object_rotation_matrix_points,
        standard_fields.InputDataFields.object_length_points,
        standard_fields.InputDataFields.object_instance_id_points,
    ]
    output_dict = preprocessor.preprocess(
        inputs=input_dict,
        images_points_correspondence_fn=self._image_correspondence_fn,
        image_preprocess_fn_dic=None)
    for key in output_keys:
      self.assertIn(key, output_dict)
    self.assertEqual(
        output_dict[standard_fields.InputDataFields.camera_intrinsics].shape,
        (3, 3))
    self.assertEqual(
        output_dict[
            standard_fields.InputDataFields.camera_rotation_matrix].shape,
        (3, 3))
    self.assertEqual(
        output_dict[standard_fields.InputDataFields.camera_translation].shape,
        (3,))
    self.assertEqual(
        output_dict[standard_fields.InputDataFields.point_positions].shape,
        (100, 3))
    self.assertEqual(
        output_dict[standard_fields.InputDataFields.num_valid_points].numpy(),
        100)
    self.assertEqual(
        output_dict[standard_fields.InputDataFields.object_class_points].shape,
        (100,))
    self.assertEqual(
        output_dict[standard_fields.InputDataFields.object_center_points].shape,
        (100, 3))
    self.assertEqual(
        output_dict[standard_fields.InputDataFields.object_height_points].shape,
        (100, 1))
    self.assertEqual(
        output_dict[standard_fields.InputDataFields.object_width_points].shape,
        (100, 1))
    self.assertEqual(
        output_dict[standard_fields.InputDataFields.object_length_points].shape,
        (100, 1))
    self.assertEqual(
        output_dict[standard_fields.InputDataFields
                    .object_rotation_matrix_points].shape, (100, 3, 3))
    self.assertEqual(
        output_dict[
            standard_fields.InputDataFields.object_instance_id_points].shape,
        (100,))
    for key in object_keys:
      self.assertEqual(output_dict[key].shape[0], 2)

  def test_preprocess_output_keys(self):
    height, width = (240, 320)
    input_dict = self._get_input_dict(height, width)
    output_dict = preprocessor.preprocess(
        inputs=input_dict,
        images_points_correspondence_fn=self._image_correspondence_fn,
        output_keys=[standard_fields.InputDataFields.camera_image],
        image_preprocess_fn_dic=None)
    self.assertIn(standard_fields.InputDataFields.camera_image, output_dict)
    self.assertEqual(len(output_dict.keys()), 1)

  def test_preprocess_missing_input_raises(self):
    with self.assertRaises(ValueError):
      empty_input = {}
      preprocessor.preprocess(inputs=empty_input)


if __name__ == '__main__':
  tf.test.main()
