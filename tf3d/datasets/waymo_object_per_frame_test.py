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

"""Tests for ...datasets.waymo_object_per_frame."""

from absl.testing import parameterized
import numpy as np
from six.moves import range
import tensorflow as tf

from tf3d import data_provider
from tf3d.datasets import waymo_object_per_frame


class WaymoObjectPerFrameTest(parameterized.TestCase, tf.test.TestCase):

  def test_tf_data_feature_label_keys(self):
    """Tests the ability of a get_tf_data_datasets to have extra labels/key.

    Test is done here because TAP is off in specific dataset tests.
    """
    features_data = data_provider.get_tf_data_dataset(
        dataset_name='waymo_object_per_frame',
        split_name='val',
        batch_size=1,
        preprocess_fn=None,
        is_training=True,
        num_readers=2,
        num_parallel_batches=2,
        shuffle_buffer_size=2)
    features = next(iter(features_data))

    cameras = ['front', 'front_left', 'front_right', 'side_left', 'side_right']
    lidars = ['top', 'front', 'side_left', 'side_right', 'rear']
    for camera in cameras:
      self.assertAllEqual(
          features[('cameras/%s/extrinsics/t' % camera)].get_shape().as_list(),
          np.array([1, 3]))
      self.assertAllEqual(
          features[('cameras/%s/extrinsics/R' % camera)].get_shape().as_list(),
          np.array([1, 3, 3]))
      self.assertAllEqual(
          features[('cameras/%s/intrinsics/distortion' %
                    camera)].get_shape().as_list(), np.array([1, 5]))
      self.assertAllEqual(
          features[('cameras/%s/intrinsics/K' % camera)].get_shape().as_list(),
          np.array([1, 3, 3]))
      self.assertAllEqual(
          features[('cameras/%s/image' % camera)].get_shape().as_list()[3], 3)
    for lidar in lidars:
      self.assertEqual(
          features[('lidars/%s/pointcloud/positions' %
                    lidar)].get_shape().as_list()[2], 3)
      self.assertEqual(
          features[('lidars/%s/pointcloud/intensity' %
                    lidar)].get_shape().as_list()[2], 1)
      self.assertEqual(
          features[('lidars/%s/pointcloud/elongation' %
                    lidar)].get_shape().as_list()[2], 1)
      self.assertAllEqual(
          features[('lidars/%s/extrinsics/R' % lidar)].get_shape().as_list(),
          np.array([1, 3, 3]))
      self.assertAllEqual(
          features[('lidars/%s/extrinsics/t' % lidar)].get_shape().as_list(),
          np.array([1, 3]))
      self.assertEqual(
          features['lidars/%s/camera_projections/positions' %
                   lidar].get_shape().as_list()[2], 2)
      self.assertEqual(
          features['lidars/%s/camera_projections/ids' %
                   lidar].get_shape().as_list()[2], 1)
    self.assertEqual(features['objects/pose/R'].get_shape().as_list()[2], 3)
    self.assertEqual(features['objects/pose/R'].get_shape().as_list()[3], 3)
    self.assertEqual(features['objects/pose/t'].get_shape().as_list()[2], 3)
    self.assertEqual(
        features['objects/shape/dimension'].get_shape().as_list()[2], 3)
    self.assertLen(features['objects/category/label'].get_shape().as_list(), 2)

  def test_get_feature_keys(self):
    feature_keys = waymo_object_per_frame.get_feature_keys()
    self.assertIsNotNone(feature_keys)

  def test_get_label_keys(self):
    label_keys = waymo_object_per_frame.get_label_keys()
    self.assertIsNotNone(label_keys)

  def test_get_file_pattern(self):
    file_pattern = waymo_object_per_frame.get_file_pattern('trainval')
    self.assertIsNotNone(file_pattern)

  def test_get_decode_fn(self):
    decode_fn = waymo_object_per_frame.get_decode_fn()
    self.assertIsNotNone(decode_fn)

  def test_prepare_lidar_images_and_correspondences(self):
    lidar_to_num_points = {
        'top': 1000,
        'front': 100,
        'side_left': 500,
        'side_right': 200,
        'rear': 2000,
    }
    camera_to_image_size = {
        'front': (300, 400, 3),
        'front_left': (600, 800, 3),
        'front_right': (800, 1000, 3),
        'side_left': (200, 400, 3),
        'side_right': (800, 1200, 3),
    }
    resized_image_height = 100
    resized_image_width = 300
    inputs = {}
    for lidar_name in lidar_to_num_points:
      inputs[('lidars/%s/extrinsics/R' % lidar_name)] = (
          tf.random.uniform([3, 3],
                            minval=-1.0,
                            maxval=1.0,
                            dtype=tf.float32))
      inputs[('lidars/%s/extrinsics/t' % lidar_name)] = (
          tf.random.uniform([3],
                            minval=-5.0,
                            maxval=5.0,
                            dtype=tf.float32))
      inputs[('lidars/%s/pointcloud/positions' % lidar_name)] = (
          tf.random.uniform([lidar_to_num_points[lidar_name], 3],
                            minval=-5.0,
                            maxval=5.0,
                            dtype=tf.float32))
      inputs[('lidars/%s/pointcloud/intensity' % lidar_name)] = (
          tf.random.uniform([lidar_to_num_points[lidar_name], 1],
                            minval=-5.0,
                            maxval=5.0,
                            dtype=tf.float32))
      inputs[('lidars/%s/pointcloud/elongation' % lidar_name)] = (
          tf.random.uniform([lidar_to_num_points[lidar_name], 1],
                            minval=-5.0,
                            maxval=5.0,
                            dtype=tf.float32))
      inputs[('lidars/%s/pointcloud/inside_nlz' % lidar_name)] = tf.cast(
          tf.random.uniform([lidar_to_num_points[lidar_name], 1],
                            minval=0,
                            maxval=2,
                            dtype=tf.int32),
          dtype=tf.bool)
      inputs[('lidars/%s/camera_projections/positions' % lidar_name)] = (
          tf.random.uniform([lidar_to_num_points[lidar_name], 2],
                            minval=0,
                            maxval=1000,
                            dtype=tf.int64))
      inputs[('lidars/%s/camera_projections/ids' % lidar_name)] = (
          tf.random.uniform([lidar_to_num_points[lidar_name], 1],
                            minval=0,
                            maxval=5,
                            dtype=tf.int64))

    camera_names = list(camera_to_image_size.keys())
    for i in range(len(camera_names)):
      camera_name = camera_names[i]
      inputs[('cameras/%s/image' % camera_name)] = (
          tf.random.uniform(
              camera_to_image_size[camera_name],
              minval=-1.0,
              maxval=1.0,
              dtype=tf.float32))
      inputs[('cameras/%s/id' % camera_name)] = tf.constant(i, dtype=tf.int64)

    outputs = waymo_object_per_frame.prepare_lidar_images_and_correspondences(
        inputs=inputs,
        resized_image_height=resized_image_height,
        resized_image_width=resized_image_width)
    num_points = outputs['points_position'].get_shape().as_list()[0]
    self.assertAllEqual(outputs['points_position'].get_shape().as_list(),
                        np.array([num_points, 3]))
    self.assertAllEqual(outputs['points_intensity'].get_shape().as_list(),
                        np.array([num_points, 1]))
    self.assertAllEqual(outputs['points_elongation'].get_shape().as_list(),
                        np.array([num_points, 1]))
    self.assertAllEqual(outputs['points_normal'].get_shape().as_list(),
                        np.array([num_points, 3]))
    self.assertAllEqual(
        outputs['view_images']['rgb_view'].get_shape().as_list(),
        np.array([5, 100, 300, 3]))
    self.assertAllEqual(
        outputs['view_indices_2d']['rgb_view'].get_shape().as_list(),
        np.array([5, num_points, 2]))

  def test_compute_semantic_labels(self):
    inputs = {}
    points = tf.constant([[0.0, 0.0, 0.0],
                          [1.0, 1.0, 1.0],
                          [1.1, 1.0, 1.0],
                          [1.0, 1.1, 1.1],
                          [3.0, 3.0, 3.0],
                          [3.1, 3.0, 3.0],
                          [3.0, 3.0, 3.1],
                          [5.0, 5.0, 5.0]], dtype=tf.float32)
    inputs['points'] = points
    inputs['objects/shape/dimension'] = tf.constant([[0.5, 0.5, 0.5],
                                                     [0.5, 0.5, 0.5],
                                                     [0.5, 0.5, 0.5]],
                                                    dtype=tf.float32)
    inputs['objects/pose/R'] = tf.constant([[[1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 1.0]],
                                            [[1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 1.0]],
                                            [[1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, 1.0]]],
                                           dtype=tf.float32)
    inputs['objects/pose/t'] = tf.constant([[0.0, 0.0, 0.0],
                                            [1.0, 1.0, 1.0],
                                            [3.0, 3.0, 3.0]], dtype=tf.float32)
    inputs['objects/category/label'] = tf.constant([1, 2, 3], dtype=tf.int32)
    point_labels = waymo_object_per_frame.compute_semantic_labels(
        inputs=inputs, points_key='points', box_margin=0.1)
    expected_point_labels = tf.constant(
        [[1], [2], [2], [2], [3], [3], [3], [0]], dtype=tf.int32)
    self.assertAllEqual(point_labels.numpy(), expected_point_labels.numpy())

  def test_compute_semantic_labels_value_error(self):
    num_points = 100
    num_objects = 10
    points = tf.random.uniform([num_points, 3],
                               minval=-5.0,
                               maxval=5.0,
                               dtype=tf.float32)
    objects_dimension = tf.random.uniform([num_objects, 3],
                                          minval=0.01,
                                          maxval=10.0,
                                          dtype=tf.float32)
    objects_rotation = tf.random.uniform([num_objects, 3, 3],
                                         minval=-1.0,
                                         maxval=1.0,
                                         dtype=tf.float32)
    objects_center = tf.random.uniform([num_objects, 3],
                                       minval=1 - .0,
                                       maxval=10.0,
                                       dtype=tf.float32)
    objects_label = tf.random.uniform([num_objects],
                                      minval=0,
                                      maxval=10,
                                      dtype=tf.int32)

    inputs = {
        'objects/shape/dimension': objects_dimension,
        'objects/pose/R': objects_rotation,
        'objects/pose/t': objects_center,
        'objects/category/label': objects_label,
    }
    with self.assertRaises(ValueError):
      waymo_object_per_frame.compute_semantic_labels(
          inputs=inputs, points_key='points', box_margin=0.1)

    inputs = {
        'points': points,
        'objects/pose/R': objects_rotation,
        'objects/pose/t': objects_center,
        'objects/category/label': objects_label,
    }
    with self.assertRaises(ValueError):
      waymo_object_per_frame.compute_semantic_labels(
          inputs=inputs, points_key='points', box_margin=0.1)

    inputs = {
        'points': points,
        'objects/shape/dimension': objects_dimension,
        'objects/pose/t': objects_center,
        'objects/category/label': objects_label,
    }
    with self.assertRaises(ValueError):
      waymo_object_per_frame.compute_semantic_labels(
          inputs=inputs, points_key='points', box_margin=0.1)

    inputs = {
        'points': points,
        'objects/shape/dimension': objects_dimension,
        'objects/pose/R': objects_rotation,
        'objects/category/label': objects_label,
    }
    with self.assertRaises(ValueError):
      waymo_object_per_frame.compute_semantic_labels(
          inputs=inputs, points_key='points', box_margin=0.1)

    inputs = {
        'points': points,
        'objects/shape/dimension': objects_dimension,
        'objects/pose/R': objects_rotation,
        'objects/pose/t': objects_center,
    }
    with self.assertRaises(ValueError):
      waymo_object_per_frame.compute_semantic_labels(
          inputs=inputs, points_key='points', box_margin=0.1)

  @parameterized.named_parameters(
      {'testcase_name': 'static_frames',
       'frame_translation': [[0., 0., 0.,], [0., 0., 0.,]],
       'frame_rotation': [[[1., 0., 0.,],
                           [0., 1., 0.,],
                           [0., 0., 1.,]],
                          [[1., 0., 0.,],
                           [0., 1., 0.,],
                           [0., 0., 1.,]]],
       'motion_labels_vec': [[1., 0., 0.,],
                             [1., 0., 0.,],
                             [.75, .25, 0.,],
                             [1.25, -.25, 0.,],
                             [0., 0., 0.,],
                             [0., 0., 0.,],
                             [0., 0., 0.,],
                             [0., 0., 0.,],]},

      {'testcase_name': 'translated_frames',
       'frame_translation': [[0., 0., 0.,], [1., 0., 0.,]],
       'frame_rotation': [[[1., 0., 0.,],
                           [0., 1., 0.,],
                           [0., 0., 1.,]],
                          [[1., 0., 0.,],
                           [0., 1., 0.,],
                           [0., 0., 1.,]]],
       'motion_labels_vec': [[2., 0., 0.,],
                             [2., 0., 0.,],
                             [1.75, .25, 0.,],
                             [2.25, -.25, 0.,],
                             [1., 0., 0.,],
                             [1., 0., 0.,],
                             [1., 0., 0.,],
                             [0., 0., 0.,],]},

      {'testcase_name': 'rotated_translated_frames',
       'frame_translation': [[0., 0., 0.,], [1., 0., 0.,]],
       'frame_rotation': [[[1., 0., 0.,],
                           [0., 1., 0.,],
                           [0., 0., 1.,]],
                          [[0., -1., 0.,],
                           [1., 0., 0.,],
                           [0., 0., 1.,]]],
       'motion_labels_vec': [[1., 1., 0.,],
                             [-1., 1., 0.,],
                             [-1.5, 1., 0.,],
                             [-.5, 1., 0.,],
                             [-5., 0., 0.,],
                             [-5.1, .1, 0.,],
                             [-5., 0., 0.,],
                             [0., 0., 0.,]]},
      )
  def test_compute_motion_labels(self, frame_translation, frame_rotation,
                                 motion_labels_vec):
    scene = {}
    frame_start_index = 0
    scene = {
        'frames/pose/R': tf.constant(frame_rotation, dtype=tf.float32),
        'frames/pose/t': tf.constant(frame_translation, dtype=tf.float32)
    }

    # prepare frame0
    points_key = 'points_position'
    frame0 = {
        points_key: tf.constant(
            [[0.0, 0.0, 0.0],  # in box 0
             [1.0, 1.0, 1.0],  # in box 1
             [1.25, 1.0, 1.0],  # in box 1
             [.75, 1.0, 1.0],  # in box 1
             [3.0, 3.0, 3.0],  # in box 2
             [3.1, 3.0, 3.0],  # in box 2
             [3.0, 3.0, 3.1],  # in box 2
             [5.0, 5.0, 5.0]],  # in box -1
            dtype=tf.float32),
        'objects/name': tf.constant(['obj1', 'obj2', 'obj3'],
                                    dtype=tf.string),
        'objects/shape/dimension': tf.constant(
            [[0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5]], dtype=tf.float32),
        'objects/pose/R': tf.constant(
            [[[1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]],
             [[1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]],
             [[1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]]],
            dtype=tf.float32),
        'objects/pose/t': tf.constant(
            [[0.0, 0.0, 0.0],
             [1.0, 1.0, 1.0],
             [3.0, 3.0, 3.0]], dtype=tf.float32)
    }

    # prepare frame1
    frame1 = {
        'objects/name': tf.constant(['obj1', 'obj2', 'obj3'],
                                    dtype=tf.string),
        'objects/shape/dimension': tf.constant(
            [[0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5]], dtype=tf.float32),
        'objects/pose/R': tf.constant(
            [[[1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]],
             [[.0, -1.0, 0.0],  # second box is rotated 90 degree
              [1.0, .0, 0.0],  # along z axis. counterclockwise
              [0.0, 0.0, 1.0]],
             [[1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]]],
            dtype=tf.float32),
        'objects/pose/t': tf.constant(
            [[1.0, 0.0, 0.0],  # first box: translated (1,0,0)
             [2.0, 1.0, 1.0],  # second box: translated (1,0,0)
             [3.0, 3.0, 3.0]],  # third box: static
            dtype=tf.float32)
    }
    motion_vec = waymo_object_per_frame.compute_motion_labels(
        scene,
        frame0,
        frame1,
        frame_start_index,
        points_key,
        box_margin=0.1)
    expected_motion_labels_vec = tf.constant(
        motion_labels_vec, dtype=tf.float32)
    self.assertTrue(
        (1e-6 >
         (motion_vec.numpy() - expected_motion_labels_vec.numpy())).all())

  def test_transform_pointcloud(self):
    scene = {
        'frames/pose/R': tf.constant(
            [[[0., -1., 0.,],
              [1., 0., 0.,],
              [0., 0., 1.,]],  # counter-clockwise 90 degrees
             [[0., 1., 0.,],
              [-1., 0., 0.,],
              [0., 0., 1.,]]], dtype=tf.float32),  # clockwise 90 degrees
        'frames/pose/t': tf.constant(
            [[1., 0., 0.,],
             [0., 1., 0.,]], dtype=tf.float32),
    }
    point_positions = tf.constant(
        [[0.0, 0.0, 0.0],
         [0.0, 1.0, 1.0],
         [1.0, 1.0, 1.0]], dtype=tf.float32)

    transformed_positions = waymo_object_per_frame.transform_pointcloud_to_another_frame(
        scene, point_positions, 0, 1)
    np_expected_transformed_positions = np.array([[1.0, 1.0, 0.0],
                                                  [1.0, 0.0, 1.0],
                                                  [0.0, 0.0, 1.0]],
                                                 dtype=np.float32)
    self.assertAllClose(transformed_positions.numpy(),
                        np_expected_transformed_positions)


if __name__ == '__main__':
  tf.test.main()
