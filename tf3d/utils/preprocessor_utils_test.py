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

"""Tests for tf3d.utils.preprocessor_utils."""

import math
import numpy as np
import tensorflow as tf
from tf3d import standard_fields
from tf3d.utils import preprocessor_utils
from tf3d.utils import rotation_matrix


class PreprocessorUtilsTest(tf.test.TestCase):

  def test_randomly_sample_points(self):
    mesh_inputs = {
        'point_positions':
            tf.random.uniform([100000, 3],
                              minval=-10.0,
                              maxval=10.0,
                              dtype=tf.float32)
    }
    preprocessor_utils.randomly_sample_points(
        mesh_inputs=mesh_inputs,
        view_indices_2d_inputs={},
        target_num_points=1000)
    sampled_points = mesh_inputs['point_positions']
    self.assertAllEqual(sampled_points.shape, np.array([1000, 3]))

  def test_rotate_points_around_axis(self):
    points = tf.constant([[1.0, 1.0, 1.0],
                          [0.0, 0.0, 0.0],
                          [-1.0, -1.0, -1.0],
                          [0.0, 1.0, 2.0]], dtype=tf.float32)
    rotation_angle = math.pi / 4.0
    points_x = preprocessor_utils.rotate_points_around_axis(
        points=points,
        rotation_angle=rotation_angle,
        axis=0,
        rotation_center=(0.0, 0.0, 0.0))
    points_y = preprocessor_utils.rotate_points_around_axis(
        points=points,
        rotation_angle=rotation_angle,
        axis=1,
        rotation_center=(0.0, 0.0, 0.0))
    points_z = preprocessor_utils.rotate_points_around_axis(
        points=points,
        rotation_angle=rotation_angle,
        axis=2,
        rotation_center=(0.0, 0.0, 0.0))
    expected_points_x = tf.constant(
        [[1.0, 0.0, math.sqrt(2.0)],
         [0.0, 0.0, 0.0],
         [-1.0, 0.0, -math.sqrt(2.0)],
         [0.0, -math.sqrt(2.0) / 2.0, math.sqrt(2.0) * 3.0 / 2.0]],
        dtype=tf.float32)
    expected_points_y = tf.constant(
        [[math.sqrt(2.0), 1.0, 0.0],
         [0.0, 0.0, 0.0],
         [-math.sqrt(2.0), -1.0, 0.0],
         [math.sqrt(2.0), 1.0, math.sqrt(2.0)]],
        dtype=tf.float32)
    expected_points_z = tf.constant(
        [[0.0, math.sqrt(2.0), 1.0],
         [0.0, 0.0, 0.0],
         [0.0, -math.sqrt(2.0), -1.0],
         [-math.sqrt(2.0) / 2.0, math.sqrt(2.0) / 2.0, 2.0]],
        dtype=tf.float32)
    self.assertAllClose(points_x.numpy(), expected_points_x.numpy())
    self.assertAllClose(points_y.numpy(), expected_points_y.numpy())
    self.assertAllClose(points_z.numpy(), expected_points_z.numpy())

  def test_rotate_objects_around_axis(self):
    object_centers = tf.constant([[1.0, 1.0, 1.0],
                                  [0.0, 0.0, 0.0],
                                  [-1.0, -1.0, -1.0],
                                  [0.0, 1.0, 2.0]], dtype=tf.float32)
    object_rotation_matrices = tf.tile(
        tf.expand_dims(tf.eye(3), axis=0), [4, 1, 1])
    object_rotations_axis = tf.zeros([4, 1], dtype=tf.float32)
    rotation_angle = math.pi / 4.0
    (object_centers_x, object_rotation_matrices_x,
     object_rotations_axis_x) = preprocessor_utils.rotate_objects_around_axis(
         object_centers=object_centers,
         object_rotation_matrices=object_rotation_matrices,
         object_rotations_axis=object_rotations_axis,
         rotation_angle=rotation_angle,
         axis=0,
         rotation_center=(0.0, 0.0, 0.0))
    (object_centers_y, object_rotation_matrices_y,
     object_rotations_axis_y) = preprocessor_utils.rotate_objects_around_axis(
         object_centers=object_centers,
         object_rotation_matrices=object_rotation_matrices,
         object_rotations_axis=object_rotations_axis,
         rotation_angle=rotation_angle,
         axis=1,
         rotation_center=(0.0, 0.0, 0.0))
    (object_centers_z, object_rotation_matrices_z,
     object_rotations_axis_z) = preprocessor_utils.rotate_objects_around_axis(
         object_centers=object_centers,
         object_rotation_matrices=object_rotation_matrices,
         object_rotations_axis=object_rotations_axis,
         rotation_angle=rotation_angle,
         axis=2,
         rotation_center=(0.0, 0.0, 0.0))
    expected_object_centers_x = tf.constant(
        [[1.0, 0.0, math.sqrt(2.0)], [0.0, 0.0, 0.0],
         [-1.0, 0.0, -math.sqrt(2.0)],
         [0.0, -math.sqrt(2.0) / 2.0,
          math.sqrt(2.0) * 3.0 / 2.0]],
        dtype=tf.float32)
    expected_object_centers_y = tf.constant(
        [[math.sqrt(2.0), 1.0, 0.0], [0.0, 0.0, 0.0],
         [-math.sqrt(2.0), -1.0, 0.0], [math.sqrt(2.0), 1.0,
                                        math.sqrt(2.0)]],
        dtype=tf.float32)
    expected_object_centers_z = tf.constant(
        [[0.0, math.sqrt(2.0), 1.0], [0.0, 0.0, 0.0],
         [0.0, -math.sqrt(2.0), -1.0],
         [-math.sqrt(2.0) / 2.0,
          math.sqrt(2.0) / 2.0, 2.0]],
        dtype=tf.float32)
    expected_object_rotation_matrices_x = tf.constant(
        [[1.0, 0.0, 0.0],
         [0.0, math.sqrt(2.0) / 2.0, -math.sqrt(2.0) / 2.0],
         [0.0, math.sqrt(2.0) / 2.0, math.sqrt(2.0) / 2.0]], dtype=tf.float32)
    expected_object_rotation_matrices_x = tf.tile(
        tf.expand_dims(expected_object_rotation_matrices_x, axis=0), [4, 1, 1])
    expected_object_rotation_matrices_y = tf.constant(
        [[math.sqrt(2.0) / 2.0, 0.0, math.sqrt(2.0) / 2.0],
         [0.0, 1.0, 0.0],
         [-math.sqrt(2.0) / 2.0, 0.0, math.sqrt(2.0) / 2.0]], dtype=tf.float32)
    expected_object_rotation_matrices_y = tf.tile(
        tf.expand_dims(expected_object_rotation_matrices_y, axis=0), [4, 1, 1])
    expected_object_rotation_matrices_z = tf.constant(
        [[math.sqrt(2.0) / 2.0, -math.sqrt(2.0) / 2.0, 0.0],
         [math.sqrt(2.0) / 2.0, math.sqrt(2.0) / 2.0, 0.0],
         [0.0, 0.0, 1.0]], dtype=tf.float32)
    expected_object_rotation_matrices_z = tf.tile(
        tf.expand_dims(expected_object_rotation_matrices_z, axis=0), [4, 1, 1])
    expected_object_rotations_axis_x = tf.ones([4, 1],
                                               dtype=tf.float32) * math.pi / 4.0
    expected_object_rotations_axis_y = tf.ones([4, 1],
                                               dtype=tf.float32) * math.pi / 4.0
    expected_object_rotations_axis_z = tf.ones([4, 1],
                                               dtype=tf.float32) * math.pi / 4.0
    self.assertAllClose(object_centers_x.numpy(),
                        expected_object_centers_x.numpy())
    self.assertAllClose(object_rotation_matrices_x.numpy(),
                        expected_object_rotation_matrices_x.numpy())
    self.assertAllClose(object_rotations_axis_x.numpy(),
                        expected_object_rotations_axis_x.numpy())
    self.assertAllClose(object_centers_y.numpy(),
                        expected_object_centers_y.numpy())
    self.assertAllClose(object_rotation_matrices_y.numpy(),
                        expected_object_rotation_matrices_y.numpy())
    self.assertAllClose(object_rotations_axis_y.numpy(),
                        expected_object_rotations_axis_y.numpy())
    self.assertAllClose(object_centers_z.numpy(),
                        expected_object_centers_z.numpy())
    self.assertAllClose(object_rotation_matrices_z.numpy(),
                        expected_object_rotation_matrices_z.numpy())
    self.assertAllClose(object_rotations_axis_z.numpy(),
                        expected_object_rotations_axis_z.numpy())

  def test_rotate_randomly(self):
    mesh_inputs = {
        standard_fields.InputDataFields.point_positions:
            tf.random.uniform([100, 3],
                              minval=-10.0,
                              maxval=10.0,
                              dtype=tf.float32),
    }
    object_inputs = {
        standard_fields.InputDataFields.objects_center:
            tf.random.uniform([20, 3],
                              minval=-10.0,
                              maxval=10.0,
                              dtype=tf.float32),
        standard_fields.InputDataFields.objects_rotation_matrix:
            tf.random.uniform([20, 3, 3],
                              minval=-1.0,
                              maxval=1.0,
                              dtype=tf.float32),
    }
    preprocessor_utils.rotate_randomly(
        mesh_inputs=mesh_inputs,
        object_inputs=object_inputs,
        x_min_degree_rotation=-10.0,
        x_max_degree_rotation=10.0,
        y_min_degree_rotation=-180.0,
        y_max_degree_rotation=180.0,
        z_min_degree_rotation=-10.0,
        z_max_degree_rotation=10.0,
        rotation_center=(0.0, 0.0, 0.0))
    self.assertAllEqual(
        mesh_inputs[standard_fields.InputDataFields.point_positions].shape,
        [100, 3])
    self.assertAllEqual(
        object_inputs[standard_fields.InputDataFields.objects_center].shape,
        [20, 3])
    self.assertAllEqual(
        object_inputs[
            standard_fields.InputDataFields.objects_rotation_matrix].shape,
        [20, 3, 3])

  def test_random_uniformly_sample_a_seed_point(self):
    point_indices = tf.constant([0, -1, 3, 5, 4, 9])
    random_seed_index = (
        preprocessor_utils._random_uniformly_sample_a_seed_point(
            object_instance_id_points=point_indices))
    self.assertLess(random_seed_index.numpy(), 6)
    self.assertGreater(random_seed_index.numpy(), -1)

  def test_get_closest_points_to_random_seed_point(self):
    point_locations = tf.random.uniform([10000, 3],
                                        minval=-100.0,
                                        maxval=100.0,
                                        dtype=tf.float32)
    object_instance_id_points = tf.random.uniform([10000, 1],
                                                  minval=0,
                                                  maxval=20,
                                                  dtype=tf.int32)
    num_closest_points = 1000
    cropped_point_indices, cropped_point_distances, point_distances = (
        preprocessor_utils._get_closest_points_to_random_seed_point(
            point_locations=point_locations,
            object_instance_id_points=object_instance_id_points,
            num_closest_points=num_closest_points,
            max_distance=None))
    self.assertAllEqual(cropped_point_indices.shape, [1000])
    self.assertAllEqual(cropped_point_distances.shape, [1000])
    self.assertLess(np.amax(cropped_point_indices.numpy()), 10000)
    self.assertGreater(np.amin(cropped_point_indices.numpy()), -1)
    self.assertGreater(cropped_point_distances.numpy()[999],
                       cropped_point_distances.numpy()[0])
    self.assertGreaterEqual(np.amin(cropped_point_distances.numpy()), 0.0)
    self.assertGreaterEqual(np.amin(point_distances.numpy()), 0.0)
    self.assertAllEqual(point_distances.shape, [10000])

  def test_complete_partial_objects(self):
    cropped_point_indices = tf.constant([0, 5, 2, 7, 4, 9], dtype=tf.int32)
    object_instance_id_points = tf.constant([0, 0, 2, 2, 1, 2, 1, 1, 3, 1, 1],
                                            dtype=tf.int64)
    completed_point_indices = preprocessor_utils._complete_partial_objects(
        cropped_point_indices=cropped_point_indices,
        object_instance_id_points=object_instance_id_points)
    expected_completed_point_indices = tf.constant(
        [0, 5, 3, 7, 4, 9, 2, 6, 10], dtype=tf.int32)
    self.assertAllEqual(
        np.sort(completed_point_indices.numpy()),
        np.sort(expected_completed_point_indices.numpy()))

  def test_add_closest_background_points(self):
    cropped_indices = tf.constant([1, 3, 5, 2], dtype=tf.int32)
    distances = tf.constant([0.0, 2.0, 1.0, 5.0, 4.0, 3.0, 7.0, 8.0, 6.0],
                            dtype=tf.float32)
    instance_id = tf.constant([0, 0, 1, 1, 2, 0, 0, 0, 0], dtype=tf.int32)
    target_num_background_points = 3
    added_indices = preprocessor_utils._add_closest_background_points(
        cropped_point_indices=cropped_indices,
        object_instance_id_points=instance_id,
        point_distances=distances,
        target_num_background_points=target_num_background_points)
    expected_indices = tf.constant([1, 3, 5, 2, 0], dtype=tf.int32)
    self.assertAllEqual(
        np.sort(added_indices.numpy()), np.sort(expected_indices.numpy()))

  def test_crop_points_around_random_seed_point(self):
    mesh_inputs = {}
    view_indices_2d_inputs = {}
    mesh_inputs[
        standard_fields.InputDataFields.point_positions] = tf.random.uniform(
            [1000, 3], minval=-100.0, maxval=100.0, dtype=tf.float32)
    mesh_inputs[standard_fields.InputDataFields
                .object_instance_id_points] = tf.random.uniform([1000, 1],
                                                                minval=0,
                                                                maxval=20,
                                                                dtype=tf.int32)
    mesh_inputs[standard_fields.InputDataFields
                .object_length_points] = tf.random.uniform([1000, 1],
                                                           minval=5.0,
                                                           maxval=20.0,
                                                           dtype=tf.float32)
    mesh_inputs[standard_fields.InputDataFields
                .object_height_points] = tf.random.uniform([1000, 1],
                                                           minval=5.0,
                                                           maxval=20.0,
                                                           dtype=tf.float32)
    mesh_inputs[standard_fields.InputDataFields
                .object_width_points] = tf.random.uniform([1000, 1],
                                                          minval=5.0,
                                                          maxval=20.0,
                                                          dtype=tf.float32)
    view_indices_2d_inputs['rgb_view'] = tf.random.uniform([5, 1000, 2],
                                                           minval=-10,
                                                           maxval=1000,
                                                           dtype=tf.int32)
    preprocessor_utils.crop_points_around_random_seed_point(
        mesh_inputs=mesh_inputs,
        view_indices_2d_inputs=view_indices_2d_inputs,
        num_closest_points=100,
        max_distance=30.0,
        num_background_points=10)
    n = mesh_inputs[standard_fields.InputDataFields.point_positions].shape[0]
    self.assertGreaterEqual(n, 100)
    self.assertAllEqual(
        mesh_inputs[standard_fields.InputDataFields.point_positions].shape,
        np.array([n, 3]))
    self.assertAllEqual(
        mesh_inputs[
            standard_fields.InputDataFields.object_instance_id_points].shape,
        np.array([n, 1]))
    self.assertAllEqual(
        mesh_inputs[standard_fields.InputDataFields.object_length_points].shape,
        np.array([n, 1]))
    self.assertAllEqual(
        mesh_inputs[standard_fields.InputDataFields.object_height_points].shape,
        np.array([n, 1]))
    self.assertAllEqual(
        mesh_inputs[standard_fields.InputDataFields.object_width_points].shape,
        np.array([n, 1]))
    self.assertAllEqual(view_indices_2d_inputs['rgb_view'].shape,
                        np.array([5, n, 2]))

  def test_remove_objects_by_num_points(self):
    mesh_inputs = {
        standard_fields.InputDataFields.point_positions:
            tf.constant([[0.0, 0.0, 0.0], [0.1, 0.0, 0.2], [-0.1, 0.1, 0.1],
                         [2.1, 2.0, 2.0], [5.0, 5.0, 5.0]],
                        dtype=tf.float32)
    }
    object_inputs = {
        standard_fields.InputDataFields.objects_length:
            tf.ones([2, 1], dtype=tf.float32),
        standard_fields.InputDataFields.objects_height:
            tf.ones([2, 1], dtype=tf.float32),
        standard_fields.InputDataFields.objects_width:
            tf.ones([2, 1], dtype=tf.float32),
        standard_fields.InputDataFields.objects_rotation_matrix:
            tf.tile(
                tf.expand_dims(tf.eye(3, dtype=tf.float32), axis=0),
                [2, 1, 1]),
        standard_fields.InputDataFields.objects_center:
            tf.constant([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], dtype=tf.float32),
    }
    preprocessor_utils.remove_objects_by_num_points(
        mesh_inputs=mesh_inputs,
        object_inputs=object_inputs,
        min_num_points_in_objects=2)
    self.assertAllClose(
        object_inputs[standard_fields.InputDataFields.objects_center].numpy(),
        np.array([[0.0, 0.0, 0.0]]))
    self.assertAllClose(
        object_inputs[standard_fields.InputDataFields.objects_length].numpy(),
        np.array([[1.0]]))
    self.assertAllClose(
        object_inputs[standard_fields.InputDataFields.objects_height].numpy(),
        np.array([[1.0]]))
    self.assertAllClose(
        object_inputs[standard_fields.InputDataFields.objects_width].numpy(),
        np.array([[1.0]]))
    self.assertAllClose(
        object_inputs[
            standard_fields.InputDataFields.objects_rotation_matrix].numpy(),
        np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]))

  def test_preprocess_one_image(self):
    image = tf.random.uniform([100, 200, 3],
                              minval=-128,
                              maxval=128,
                              dtype=tf.int32)
    points_in_image_frame = tf.random.uniform([100, 2],
                                              minval=-10,
                                              maxval=200,
                                              dtype=tf.int32)

    def image_preprocess_fn(inputs, is_training):
      if is_training:
        for key in inputs:
          inputs[key] = inputs[key][20:80, 50:150, :]
      return inputs

    processed_image, processed_points_in_image_frame = (
        preprocessor_utils.preprocess_one_image(
            image=image,
            image_preprocess_fn=image_preprocess_fn,
            points_in_image_frame=points_in_image_frame,
            is_training=True))
    self.assertAllEqual(processed_image.shape, np.array([60, 100, 3]))
    expected_points_in_image_frame = points_in_image_frame.numpy() - np.array(
        [20, 50])
    expected_points_in_image_frame[
        points_in_image_frame.numpy()[:, 0] < 20, :] = -1
    expected_points_in_image_frame[
        points_in_image_frame.numpy()[:, 1] < 50, :] = -1
    expected_points_in_image_frame[
        points_in_image_frame.numpy()[:, 0] >= 80, :] = -1
    expected_points_in_image_frame[
        points_in_image_frame.numpy()[:, 1] >= 150, :] = -1
    self.assertAllEqual(processed_points_in_image_frame.numpy(),
                        expected_points_in_image_frame)

  def test_preprocess_images(self):
    view_image_inputs = {
        'rgb_image':
            tf.random.uniform([4, 100, 200, 3],
                              minval=-128,
                              maxval=128,
                              dtype=tf.int32),
        'depth_image':
            tf.random.uniform([4, 120, 220, 1],
                              minval=0.0,
                              maxval=10.0,
                              dtype=tf.float32),
    }
    view_indices_2d_inputs = {
        'rgb_image':
            tf.random.uniform([4, 100, 2],
                              minval=-10,
                              maxval=200,
                              dtype=tf.int32),
        'depth_image':
            tf.random.uniform([4, 100, 2],
                              minval=-10,
                              maxval=200,
                              dtype=tf.int32)
    }

    def rgb_image_preprocess_fn(inputs, is_training):
      if is_training:
        for key in inputs:
          inputs[key] = inputs[key][20:80, 50:150, :]
      return inputs

    def depth_image_preprocess_fn(inputs, is_training):
      if is_training:
        for key in inputs:
          inputs[key] = inputs[key][10:80, 20:150, :]
      return inputs

    image_preprocess_fn_dic = {
        'rgb_image': rgb_image_preprocess_fn,
        'depth_image': depth_image_preprocess_fn
    }
    points_in_rgb_image_frame = view_indices_2d_inputs['rgb_image']
    points_in_depth_image_frame = view_indices_2d_inputs['depth_image']
    preprocessor_utils.preprocess_images(
        view_image_inputs=view_image_inputs,
        view_indices_2d_inputs=view_indices_2d_inputs,
        image_preprocess_fn_dic=image_preprocess_fn_dic,
        is_training=True)
    self.assertAllEqual(view_image_inputs['rgb_image'].shape,
                        np.array([4, 60, 100, 3]))
    self.assertAllEqual(view_image_inputs['depth_image'].shape,
                        np.array([4, 70, 130, 1]))
    expected_points_in_rgb_image_frame = (
        points_in_rgb_image_frame.numpy() - np.array([20, 50]))
    expected_points_in_rgb_image_frame[
        points_in_rgb_image_frame.numpy()[:, :, 0] < 20, :] = -1
    expected_points_in_rgb_image_frame[
        points_in_rgb_image_frame.numpy()[:, :, 1] < 50, :] = -1
    expected_points_in_rgb_image_frame[
        points_in_rgb_image_frame.numpy()[:, :, 0] >= 80, :] = -1
    expected_points_in_rgb_image_frame[
        points_in_rgb_image_frame.numpy()[:, :, 1] >= 150, :] = -1
    self.assertAllEqual(view_indices_2d_inputs['rgb_image'].numpy(),
                        expected_points_in_rgb_image_frame)
    expected_points_in_depth_image_frame = (
        points_in_depth_image_frame.numpy() - np.array([10, 20]))
    expected_points_in_depth_image_frame[
        points_in_depth_image_frame.numpy()[:, :, 0] < 10, :] = -1
    expected_points_in_depth_image_frame[
        points_in_depth_image_frame.numpy()[:, :, 1] < 20, :] = -1
    expected_points_in_depth_image_frame[
        points_in_depth_image_frame.numpy()[:, :, 0] >= 80, :] = -1
    expected_points_in_depth_image_frame[
        points_in_depth_image_frame.numpy()[:, :, 1] >= 150, :] = -1
    self.assertAllEqual(view_indices_2d_inputs['depth_image'].numpy(),
                        expected_points_in_depth_image_frame)

  def test_make_objects_axis_aligned(self):
    rotation1 = rotation_matrix.from_rotation_around_x(-math.pi / 6.0)
    rotation2 = tf.constant([[0.0, 0.0, 1.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0]],
                            dtype=tf.float32)
    inputs = {
        standard_fields.InputDataFields.objects_length:
            tf.constant([[1.0], [2.0]], dtype=tf.float32),
        standard_fields.InputDataFields.objects_width:
            tf.constant([[3.0], [3.0]], dtype=tf.float32),
        standard_fields.InputDataFields.objects_height:
            tf.constant([[2.0], [1.0]], dtype=tf.float32),
        standard_fields.InputDataFields.objects_center:
            tf.constant([[1.0, 1.0, 1.0], [-1.0, 2.0, 1.0]]),
        standard_fields.InputDataFields.objects_rotation_matrix:
            tf.stack([rotation1, rotation2], axis=0),
    }
    expected_inputs = {
        standard_fields.InputDataFields.objects_length:
            tf.constant([[1.0], [1.0]], dtype=tf.float32),
        standard_fields.InputDataFields.objects_width:
            tf.constant([[3.5980759], [2.0]], dtype=tf.float32),
        standard_fields.InputDataFields.objects_height:
            tf.constant([[3.232051], [3.0]], dtype=tf.float32),
        standard_fields.InputDataFields.objects_center:
            tf.constant([[1.0, 1.0, 1.0], [-1.0, 2.0, 1.0]]),
        standard_fields.InputDataFields.objects_rotation_matrix:
            tf.tile(
                tf.expand_dims(tf.eye(3, dtype=tf.float32), axis=0), [2, 1, 1]),
    }
    preprocessor_utils.make_objects_axis_aligned(inputs)
    for key in expected_inputs:
      self.assertAllClose(inputs[key].numpy(), expected_inputs[key].numpy())

  def test_fit_objects_to_instance_id_points(self):
    mesh_inputs = {}
    object_inputs = {}
    mesh_inputs[standard_fields.InputDataFields
                .object_instance_id_points] = tf.constant(
                    [0, 2, 5, 3, 3, 1, 1, 1], dtype=tf.int32)
    mesh_inputs[
        standard_fields.InputDataFields.object_class_points] = tf.constant(
            [[0], [0], [3], [3], [3], [2], [2], [2]], dtype=tf.int32)
    mesh_inputs[standard_fields.InputDataFields.point_positions] = tf.constant(
        [[0.0, 0.0, 0.0], [-1.0, -1.0, -1.0], [3.0, 3.0, 3.0], [5.0, 5.0, 5.0],
         [7.0, 7.0, 7.0], [-3.0, -3.0, 0.0], [1.0, 2.0, 3.0], [4.0, 1.0, 4.0]],
        dtype=tf.float32)
    object_inputs = {}
    preprocessor_utils.fit_objects_to_instance_id_points(
        mesh_inputs=mesh_inputs, object_inputs=object_inputs, epsilon=0.001)
    self.assertAllEqual(
        object_inputs[standard_fields.InputDataFields.objects_class].numpy(),
        np.array([[3], [3], [2]]))
    self.assertAllClose(
        object_inputs[standard_fields.InputDataFields.objects_center].numpy(),
        np.array([[3.0, 3.0, 3.0], [6.0, 6.0, 6.0], [0.5, -0.5, 2.0]]))
    self.assertAllClose(
        object_inputs[standard_fields.InputDataFields.objects_length].numpy(),
        np.array([[0.001], [2.0], [7.0]]))
    self.assertAllClose(
        object_inputs[standard_fields.InputDataFields.objects_width].numpy(),
        np.array([[0.001], [2.0], [5.0]]))
    self.assertAllClose(
        object_inputs[standard_fields.InputDataFields.objects_height].numpy(),
        np.array([[0.001], [2.0], [4.0]]))
    self.assertAllClose(
        object_inputs[
            standard_fields.InputDataFields.objects_rotation_matrix].numpy(),
        np.tile(np.expand_dims(np.eye(3), axis=0), [3, 1, 1]))
    self.assertAllClose(
        object_inputs[
            standard_fields.InputDataFields.objects_rotation_matrix].numpy(),
        np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                  [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                  [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]))
    self.assertAllEqual(
        mesh_inputs[
            standard_fields.InputDataFields.object_instance_id_points].numpy(),
        np.array([0, 0, 1, 2, 2, 3, 3, 3]))

  def test_add_point_offsets(self):
    inputs = {
        standard_fields.InputDataFields.point_positions:
            tf.random.uniform([100, 3],
                              minval=-1.0,
                              maxval=1.0,
                              dtype=tf.float32)
    }
    preprocessor_utils.add_point_offsets(
        inputs=inputs, voxel_grid_cell_size=(0.1, 0.1, 0.1))
    self.assertAllEqual(
        inputs[standard_fields.InputDataFields.point_offsets].shape,
        [100, 3])

  def test_add_point_offset_bins(self):
    inputs = {
        standard_fields.InputDataFields.point_positions:
            tf.random.uniform([100, 3],
                              minval=-1.0,
                              maxval=1.0,
                              dtype=tf.float32)
    }
    preprocessor_utils.add_point_offset_bins(
        inputs=inputs,
        voxel_grid_cell_size=(0.1, 0.1, 0.1),
        num_bins_x=4,
        num_bins_y=4,
        num_bins_z=2)
    self.assertAllEqual(
        inputs[standard_fields.InputDataFields.point_offset_bins].shape,
        [100, 32])


if __name__ == '__main__':
  tf.test.main()
