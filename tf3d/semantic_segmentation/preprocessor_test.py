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

"""Tests for ...tf3d.semantic_segmentation.preprocessor."""

import math
import numpy as np
import six
import tensorflow as tf
from tf3d import standard_fields
from tf3d.semantic_segmentation import preprocessor


class PreprocessTest(tf.test.TestCase):

  def test_rotate_points_and_normals_around_axis(self):
    points = tf.constant([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]], dtype=tf.float32)
    normals = tf.constant([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=tf.float32)
    motions = tf.constant([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=tf.float32)
    rotation_angle = 90.0 * math.pi / 180.0
    (rotated_points, rotated_normals, rotated_motions
    ) = preprocessor.rotate_points_and_normals_motions_around_axis(
        points=points,
        normals=normals,
        motions=motions,
        rotation_angle=rotation_angle,
        axis=2)
    expected_rotated_points = np.array([[-1.0, 1.0, 1.0],
                                        [0.0, 1.0, 0.0]], dtype=np.float32)
    expected_rotated_normals = np.array([[-1.0, 1.0, 0.0],
                                         [0.0, 1.0, 1.0]], dtype=np.float32)
    expected_rotated_motions = np.array([[-1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
                                        dtype=np.float32)
    self.assertAllClose(rotated_points.numpy(), expected_rotated_points)
    self.assertAllClose(rotated_normals.numpy(), expected_rotated_normals)
    self.assertAllClose(rotated_motions.numpy(), expected_rotated_motions)

  def test_rotate_randomly(self):
    points = tf.constant([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]], dtype=tf.float32)
    normals = tf.constant([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=tf.float32)
    motions = tf.constant([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=tf.float32)
    (rotated_points, rotated_normals,
     rotated_motions) = preprocessor.rotate_randomly(
         points=points,
         normals=normals,
         motions=motions,
         x_min_degree_rotation=-10,
         x_max_degree_rotation=10,
         y_min_degree_rotation=-10,
         y_max_degree_rotation=10,
         z_min_degree_rotation=-180,
         z_max_degree_rotation=180)
    points_norm = tf.norm(points, axis=1)
    normals_norm = tf.norm(normals, axis=1)
    motions_norm = tf.norm(motions, axis=1)
    rotated_points_norm = tf.norm(rotated_points, axis=1)
    rotated_normals_norm = tf.norm(rotated_normals, axis=1)
    rotated_motions_norm = tf.norm(rotated_motions, axis=1)
    self.assertAllClose(points_norm.numpy(), rotated_points_norm.numpy())
    self.assertAllClose(normals_norm.numpy(), rotated_normals_norm.numpy())
    self.assertAllClose(motions_norm.numpy(), rotated_motions_norm.numpy())
    self.assertAllEqual(rotated_points.shape, [2, 3])
    self.assertAllEqual(rotated_normals.shape, [2, 3])
    self.assertAllEqual(rotated_motions.shape, [2, 3])

  def test_flip_points_and_normals(self):
    points = tf.constant([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]], dtype=tf.float32)
    normals = tf.constant([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=tf.float32)
    motions = tf.constant([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=tf.float32)
    (rotated_points, rotated_normals,
     rotated_motions) = preprocessor.flip_points_and_normals_motions(
         points=points,
         normals=normals,
         motions=motions,
         x_rotate=tf.convert_to_tensor(-1.0, dtype=tf.float32),
         y_rotate=tf.convert_to_tensor(-1.0, dtype=tf.float32))
    expected_rotated_points = np.array([[-1.0, -1.0, 1.0],
                                        [-1.0, 0.0, 0.0]], dtype=np.float32)
    expected_rotated_normals = np.array([[-1.0, -1.0, 0.0],
                                         [-1.0, 0.0, 1.0]], dtype=np.float32)
    expected_rotated_motions = np.array([[-1.0, -1.0, 0.0], [-1.0, 0.0, 1.0]],
                                        dtype=np.float32)
    self.assertAllClose(rotated_points.numpy(), expected_rotated_points)
    self.assertAllClose(rotated_normals.numpy(), expected_rotated_normals)
    self.assertAllClose(rotated_motions.numpy(), expected_rotated_motions)

  def test_flip_randomly_points_and_normals(self):
    points = tf.constant([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]], dtype=tf.float32)
    normals = tf.constant([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=tf.float32)
    motions = tf.constant([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=tf.float32)
    (rotated_points, rotated_normals,
     rotated_motions) = preprocessor.flip_randomly_points_and_normals_motions(
         points=points, normals=normals, motions=motions, is_training=True)
    points_norm = tf.norm(points, axis=1)
    normals_norm = tf.norm(normals, axis=1)
    motions_norm = tf.norm(motions, axis=1)
    rotated_points_norm = tf.norm(rotated_points, axis=1)
    rotated_normals_norm = tf.norm(rotated_normals, axis=1)
    rotated_motions_norm = tf.norm(rotated_motions, axis=1)
    self.assertAllClose(points_norm, rotated_points_norm.numpy())
    self.assertAllClose(normals_norm, rotated_normals_norm.numpy())
    self.assertAllClose(motions_norm, rotated_motions_norm.numpy())
    self.assertAllEqual(rotated_points.shape, [2, 3])
    self.assertAllEqual(rotated_normals.shape, [2, 3])
    self.assertAllEqual(rotated_motions.shape, [2, 3])

  def test_randomly_crop_points(self):
    mesh_inputs = {
        standard_fields.InputDataFields.point_positions:
            tf.random.uniform([100000, 3],
                              minval=-10.0,
                              maxval=10.0,
                              dtype=tf.float32)
    }
    preprocessor.randomly_crop_points(
        mesh_inputs=mesh_inputs,
        view_indices_2d_inputs={},
        x_random_crop_size=1.0,
        y_random_crop_size=2.0)
    cropped_points = mesh_inputs[
        standard_fields.InputDataFields.point_positions]
    min_cropped_points = tf.reduce_min(cropped_points, axis=0)
    max_cropped_points = tf.reduce_max(cropped_points, axis=0)
    self.assertLessEqual(max_cropped_points.numpy()[0] - 1.0,
                         min_cropped_points.numpy()[0])
    self.assertLessEqual(max_cropped_points.numpy()[1] - 2.0,
                         min_cropped_points.numpy()[1])

  def test_pick_labeled_image(self):
    view_image_inputs = {
        'rgb_image':
            tf.random.uniform([10, 200, 300, 3],
                              minval=0.0,
                              maxval=255.0,
                              dtype=tf.float32),
        'depth_image':
            tf.random.uniform([4, 100, 150, 1],
                              minval=0.0,
                              maxval=10.0,
                              dtype=tf.float32),
    }
    mesh_inputs = {
        standard_fields.InputDataFields.point_loss_weights:
            tf.random.uniform([10000, 1],
                              minval=0.0,
                              maxval=1.0,
                              dtype=tf.float32),
        standard_fields.InputDataFields.point_positions:
            tf.random.uniform([10000, 3],
                              minval=-2.0,
                              maxval=2.0,
                              dtype=tf.float32),
    }
    view_indices_2d_inputs = {
        'rgb_image':
            tf.random.uniform([10, 10000, 2],
                              minval=0,
                              maxval=10,
                              dtype=tf.int32),
        'depth_image':
            tf.random.uniform([4, 10000, 2],
                              minval=-1,
                              maxval=10,
                              dtype=tf.int32)
    }
    preprocessor.pick_labeled_image(
        mesh_inputs=mesh_inputs,
        view_image_inputs=view_image_inputs,
        view_indices_2d_inputs=view_indices_2d_inputs,
        view_name='rgb_image')
    self.assertAllEqual(view_image_inputs['rgb_image'].shape,
                        np.array([1, 200, 300, 3]))
    self.assertAllEqual(view_image_inputs['depth_image'].shape,
                        np.array([4, 100, 150, 1]))
    self.assertEqual(
        mesh_inputs[
            standard_fields.InputDataFields.point_loss_weights].shape[1], 1)
    self.assertEqual(
        mesh_inputs[
            standard_fields.InputDataFields.point_positions].shape[1], 3)
    self.assertEqual(
        mesh_inputs[
            standard_fields.InputDataFields.point_loss_weights].shape[0],
        mesh_inputs[
            standard_fields.InputDataFields.point_positions].shape[0])
    self.assertEqual(
        mesh_inputs[
            standard_fields.InputDataFields.point_loss_weights].shape[0],
        view_indices_2d_inputs['rgb_image'].shape[1])
    self.assertEqual(view_indices_2d_inputs['rgb_image'].shape[0], 1)
    self.assertEqual(view_indices_2d_inputs['rgb_image'].shape[2], 2)

  def test_empty_inputs_raises_value_error(self):
    with self.assertRaises(ValueError):
      empty_input = {}
      preprocessor.preprocess(inputs=empty_input)

  def test_inputs_missing_image_raises_value_error(self):
    inputs = {
        'depth': tf.ones((50, 50, 1)),
        'ignore_label': 255,
    }
    with self.assertRaises(ValueError):
      preprocessor.preprocess(inputs=inputs)

  def test_points_with_wrong_dimension_raises_value_error(self):
    inputs = {
        'points': tf.zeros((1000, 1, 3)),
        'normals': tf.zeros((1000, 1, 3)),
        'colors': tf.zeros((1000, 3), dtype=tf.uint8),
        'semantic_labels': tf.zeros((1000, 1), dtype=tf.int32),
    }
    with self.assertRaises(ValueError):
      preprocessor.preprocess(inputs=inputs)

  def test_preprocess_points(self):
    for is_training in [True, False]:
      points = tf.random.uniform(
          (1000, 3), minval=10.0, maxval=50.0, dtype=tf.float32)
      normals = tf.random.uniform(
          (1000, 3), minval=-0.5, maxval=0.5, dtype=tf.float32)
      colors = tf.random.uniform((1000, 3),
                                 minval=0,
                                 maxval=255,
                                 dtype=tf.int32)
      colors = tf.cast(colors, dtype=tf.uint8)
      semantic_labels = tf.random.uniform((1000, 1),
                                          minval=0,
                                          maxval=10,
                                          dtype=tf.int32)
      points_centered = points - tf.expand_dims(
          tf.reduce_mean(points, axis=0), axis=0)
      inputs = {
          'points': points,
          'normals': normals,
          'colors': colors,
          'semantic_labels': semantic_labels,
          'ignore_label': 255,
      }

      inputs = preprocessor.preprocess(
          inputs=inputs,
          z_min_degree_rotation=-50.0,
          z_max_degree_rotation=50.0,
          is_training=is_training)
      self.assertEqual(inputs['ignore_label'], 255)
      inputs = {
          k: v for k, v in six.iteritems(inputs) if isinstance(v, tf.Tensor)
      }
      self.assertEqual(
          [1000, 3],
          list(inputs[standard_fields.InputDataFields.point_positions].shape))
      self.assertAllClose(
          points_centered.numpy()[:, 2],
          inputs[standard_fields.InputDataFields.point_positions].numpy()[:, 2])

  def test_preprocess_points_with_padding(self):
    for is_training in [True, False]:
      points = tf.random.uniform(
          (1000, 3), minval=10.0, maxval=50.0, dtype=tf.float32)
      normals = tf.random.uniform(
          (1000, 3), minval=-0.5, maxval=0.5, dtype=tf.float32)
      colors = tf.random.uniform((1000, 3),
                                 minval=0,
                                 maxval=255,
                                 dtype=tf.int32)
      semantic_labels = tf.random.uniform((1000, 1),
                                          minval=0,
                                          maxval=10,
                                          dtype=tf.int32)
      colors = tf.cast(colors, dtype=tf.uint8)
      points_centered = points - tf.expand_dims(
          tf.reduce_mean(points, axis=0), axis=0)
      points_centered = tf.pad(points_centered, paddings=[[0, 1000], [0, 0]])
      inputs = {
          'points': points,
          'normals': normals,
          'colors': colors,
          'semantic_labels': semantic_labels,
          'ignore_label': 255,
      }
      inputs = preprocessor.preprocess(
          inputs=inputs,
          z_min_degree_rotation=-50.0,
          z_max_degree_rotation=50.0,
          is_training=is_training,
          points_pad_or_clip_size=2000)
      self.assertEqual(inputs['ignore_label'], 255)
      inputs = {
          k: v for k, v in six.iteritems(inputs) if isinstance(v, tf.Tensor)
      }
      self.assertEqual(
          [2000, 3],
          list(inputs[standard_fields.InputDataFields.point_positions].shape))
      self.assertEqual(
          [2000, 3],
          list(inputs[standard_fields.InputDataFields.point_normals].shape))
      self.assertEqual(
          [2000, 3],
          list(inputs[standard_fields.InputDataFields.point_colors].shape))
      self.assertEqual(
          [2000, 1],
          list(
              inputs[standard_fields.InputDataFields.point_loss_weights].shape))
      self.assertEqual(1000,
                       inputs[standard_fields.InputDataFields.num_valid_points])
      self.assertAllClose(
          points_centered.numpy()[:, 2],
          inputs[standard_fields.InputDataFields.point_positions].numpy()[:, 2])

  def test_preprocess_points_without_normals_and_colors(self):
    for is_training in [True, False]:
      points = tf.random.uniform(
          (1000, 3), minval=10.0, maxval=50.0, dtype=tf.float32)
      semantic_labels = tf.random.uniform((1000, 1),
                                          minval=0,
                                          maxval=10,
                                          dtype=tf.int32)
      points_centered = points - tf.expand_dims(
          tf.reduce_mean(points, axis=0), axis=0)
      points_centered = tf.pad(points_centered, paddings=[[0, 1000], [0, 0]])
      inputs = {
          'points': points,
          'semantic_labels': semantic_labels,
          'ignore_label': 255,
      }
      inputs = preprocessor.preprocess(
          inputs=inputs,
          z_min_degree_rotation=-50.0,
          z_max_degree_rotation=50.0,
          is_training=is_training,
          points_pad_or_clip_size=2000)
      self.assertEqual(inputs['ignore_label'], 255)
      inputs = {
          k: v for k, v in six.iteritems(inputs) if isinstance(v, tf.Tensor)
      }
      self.assertEqual(
          [2000, 3],
          list(inputs[standard_fields.InputDataFields.point_positions].shape))
      self.assertEqual(
          [2000, 1],
          list(
              inputs[standard_fields.InputDataFields.point_loss_weights].shape))
      self.assertEqual(
          1000,
          inputs[standard_fields.InputDataFields.num_valid_points].numpy())
      self.assertAllClose(
          points_centered.numpy()[:, 2],
          inputs[standard_fields.InputDataFields.point_positions].numpy()[:, 2])


if __name__ == '__main__':
  tf.test.main()
