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

"""Tests for tf3d.utils.projections."""

import numpy as np
import tensorflow as tf
from tf3d.utils import projections


class ProjectionsTest(tf.test.TestCase):

  def _get_transformation_matrices(self):
    # pylint: disable=bad-whitespace
    rotate_world_to_camera = tf.constant([
        [  2.34773703e-04,  -9.99944150e-01,  -1.05634779e-02],
        [  1.04494076e-02,   1.05653536e-02,  -9.99889553e-01],
        [  9.99945402e-01,   1.24365382e-04,   1.04513029e-02]
    ], dtype=tf.float32)

    translate_world_to_camera = tf.constant([0.05705245,
                                             -0.07546672,
                                             -0.26938692], dtype=tf.float32)

    tr_camera_to_image = tf.constant([
        [ 721.53771973,    0.        ,  609.55932617],
        [   0.        ,  721.53771973,  172.85400391],
        [   0.        ,    0.        ,    1.        ]], dtype=tf.float32)
    # pylint: enable=bad-whitespace
    return (rotate_world_to_camera,
            translate_world_to_camera,
            tr_camera_to_image)

  def test_to_camera_frame(self):
    num_points = 1000
    original_wf_points = tf.random.uniform((num_points, 3),
                                           minval=-10.0,
                                           maxval=10.0)
    (rotate_world_to_camera,
     translate_world_to_camera,
     _) = self._get_transformation_matrices()
    cf_points = projections.to_camera_frame(original_wf_points,
                                            rotate_world_to_camera,
                                            translate_world_to_camera)
    computed_wf_points = projections.to_world_frame(cf_points,
                                                    rotate_world_to_camera,
                                                    translate_world_to_camera)
    self.assertAllClose(original_wf_points.numpy(), computed_wf_points.numpy())
    self.assertEqual(original_wf_points.shape, (num_points, 3))
    self.assertEqual(cf_points.shape, (num_points, 3))
    self.assertEqual(computed_wf_points.shape, (num_points, 3))

  def test_to_image_frame(self):
    height = 375
    width = 1242
    cf_points = tf.convert_to_tensor([
        [0.0, 0.0, 10.0],
        [0.0, 0.2, 12.0],
        [0.0, -0.2, 20.0],
        [1.0, 0.0, 20.0],
        [-1.0, 0.0, 20.0],
        [0, 0, -10],
        [-1, 1, -20]], dtype=tf.float32)

    _, _, camera_intrinsics = self._get_transformation_matrices()
    image_points, within_image = projections.to_image_frame(cf_points,
                                                            height,
                                                            width,
                                                            camera_intrinsics)
    np_image_points = image_points.numpy()
    np_within_image = within_image.numpy()
    self.assertEqual(np_within_image.sum(), 5)
    self.assertTrue((np_image_points[np_within_image] >= 0).all())
    self.assertTrue((np_image_points[np_within_image, 0] < width).all())
    self.assertTrue((np_image_points[np_within_image, 1] < height).all())

  def test_image_frame_to_camera_frame(self):
    num_points = 10
    height, width = 480, 640
    image_frame = tf.concat([
        tf.random.uniform(
            [num_points, 1], minval=0.0, maxval=width, dtype=tf.float32),
        tf.random.uniform(
            [num_points, 1], minval=0.0, maxval=height, dtype=tf.float32)
    ],
                            axis=1)
    camera_intrinsics = tf.constant(
        [[500, 0., 320.], [0., 500., 120.], [0., 0., 1.]], dtype=tf.float32)
    camera_frame = projections.image_frame_to_camera_frame(
        image_frame=image_frame, camera_intrinsics=camera_intrinsics)
    self.assertEqual(camera_frame.shape, (num_points, 3))

  def _test_create_image_from_rank1_values_unbatched(self, use_sparse_tensor):
    image_height, image_width = (240, 320)
    xx = tf.random.uniform((10,), minval=0, maxval=image_width, dtype=tf.int32)
    yy = tf.random.uniform((10,), minval=0, maxval=image_height, dtype=tf.int32)
    pixel_locations = tf.stack([yy, xx], axis=1)
    pixel_values = tf.ones((tf.shape(pixel_locations)[0],), dtype=tf.float32)
    created_image = projections.create_image_from_point_values_unbatched(
        pixel_locations=pixel_locations,
        pixel_values=pixel_values,
        image_height=image_height,
        image_width=image_width,
        default_value=255,
        use_sparse_tensor=use_sparse_tensor)
    self.assertAllClose(pixel_values.numpy(), np.ones((10,), dtype=np.uint8))
    np_expected_image = np.full((image_height, image_width, 1), 255, np.uint8)
    np_yy = pixel_locations.numpy()[:, 0]
    np_xx = pixel_locations.numpy()[:, 1]
    np_expected_image[np_yy, np_xx, Ellipsis] = [1]
    self.assertAllClose(created_image.numpy(), np_expected_image)

  def _test_create_image_from_rank1_values(self, use_sparse_tensor):
    image_height, image_width = (240, 320)
    xx = tf.random.uniform((4, 10),
                           minval=0,
                           maxval=image_width,
                           dtype=tf.int32)
    yy = tf.random.uniform((4, 10),
                           minval=0,
                           maxval=image_height,
                           dtype=tf.int32)
    pixel_locations = tf.stack([yy, xx], axis=2)
    pixel_values = tf.random.uniform([4, 10, 3],
                                     minval=-2.0,
                                     maxval=2.0,
                                     dtype=tf.float32)
    num_valid_points = tf.constant([10, 3, 5, 7], dtype=tf.int32)
    created_image = projections.create_image_from_point_values(
        pixel_locations=pixel_locations,
        pixel_values=pixel_values,
        num_valid_points=num_valid_points,
        image_height=image_height,
        image_width=image_width,
        default_value=255.0,
        use_sparse_tensor=use_sparse_tensor)
    self.assertAllEqual(created_image.shape, np.array([4, 240, 320, 3]))

  def _test_create_image_from_rank2_values_unbatched(self, use_sparse_tensor):
    image_height, image_width = (240, 320)
    xx = tf.random.uniform((10,), minval=0, maxval=image_width, dtype=tf.int32)
    yy = tf.random.uniform((10,), minval=0, maxval=image_height, dtype=tf.int32)
    pixel_locations = tf.stack([yy, xx], axis=1)
    pixel_values = tf.random.uniform((10, 3),
                                     minval=0,
                                     maxval=255,
                                     dtype=tf.float32)
    created_image = projections.create_image_from_point_values_unbatched(
        pixel_locations=pixel_locations,
        pixel_values=pixel_values,
        image_height=image_height,
        image_width=image_width,
        default_value=0,
        use_sparse_tensor=use_sparse_tensor)
    self.assertEqual(created_image.shape, (image_height, image_width, 3))
    np_pixel_locations = pixel_locations.numpy().round().astype(np.int32)
    np_yy = np_pixel_locations[:, 0]
    np_xx = np_pixel_locations[:, 1]
    self.assertAllClose(
        created_image.numpy()[np_yy, np_xx], pixel_values.numpy(), atol=1e-3)

  def test_create_image_rank1_unbatched_sparse(self):
    self._test_create_image_from_rank1_values_unbatched(use_sparse_tensor=True)

  def test_create_image_rank1_unbatched_scatter(self):
    self._test_create_image_from_rank1_values_unbatched(use_sparse_tensor=False)

  def test_create_image_rank1_sparse(self):
    self._test_create_image_from_rank1_values(use_sparse_tensor=True)

  def test_create_image_rank1_scatter(self):
    self._test_create_image_from_rank1_values(use_sparse_tensor=False)

  def test_create_image_rank2_unbatched_sparse(self):
    self._test_create_image_from_rank2_values_unbatched(use_sparse_tensor=True)

  def test_create_image_rank2_unbatched_scatter(self):
    self._test_create_image_from_rank2_values_unbatched(use_sparse_tensor=False)

  def _test_move_image_values_to_points(self, use_sparse_tensor):
    image_values = tf.constant(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]],
         [[9.0, 10.0], [11.0, 12.0]]],
        dtype=tf.float32)
    image_point_indices = tf.constant([[[1], [-1]], [[6], [-1]], [[0], [3]]],
                                      dtype=tf.int32)
    num_points = 10
    point_values = projections.move_image_values_to_points(
        image_values=image_values,
        image_point_indices=image_point_indices,
        num_points=num_points,
        default_value=-1.0,
        use_sparse_tensor=use_sparse_tensor)
    expected_point_values = tf.constant(
        [[9.0, 10.0],
         [1.0, 2.0],
         [-1.0, -1.0],
         [11.0, 12.0],
         [-1.0, -1.0],
         [-1.0, -1.0],
         [5.0, 6.0],
         [-1.0, -1.0],
         [-1.0, -1.0],
         [-1.0, -1.0]],
        dtype=tf.float32)
    self.assertAllClose(point_values.numpy(), expected_point_values.numpy())

  def test_move_image_values_to_points_sparse(self):
    self._test_move_image_values_to_points(use_sparse_tensor=True)

  def test_move_image_values_to_points_scatter(self):
    self._test_move_image_values_to_points(use_sparse_tensor=False)

  def test_update_pixel_locations_given_deformed_meshgrid(self):
    pixel_locations = tf.constant([[0, 1],
                                   [1, 1],
                                   [2, 0],
                                   [2, 2],
                                   [0, 2],
                                   [0, 1],
                                   [1, 1],
                                   [3, 3]], dtype=tf.int32)
    meshgrid_y = tf.constant([[1, 1, 1, 1, 1],
                              [2, 2, 2, 2, 2],
                              [3, 3, 3, 3, 3],
                              [4, 4, 4, 4, 4]])
    meshgrid_x = tf.constant([[1, 2, 3, 4, 5],
                              [1, 2, 3, 4, 5],
                              [1, 2, 3, 4, 5],
                              [1, 2, 3, 4, 5]])
    meshgrid_yx = tf.stack([meshgrid_y, meshgrid_x], axis=2)
    deformed_meshgrid_y = tf.constant([[-1, 3, 2, 2],
                                       [-1, -1, 3, 2],
                                       [1, -1, 2, -1]])
    deformed_meshgrid_x = tf.constant([[2, -2, 2, -1],
                                       [-1, 3, 2, 3],
                                       [2, -1, 3, -1]])
    deformed_meshgrid_yx = tf.stack([deformed_meshgrid_y, deformed_meshgrid_x],
                                    axis=2)
    updated_pixel_locations = (
        projections.update_pixel_locations_given_deformed_meshgrid(
            pixel_locations=pixel_locations,
            original_meshgrid=meshgrid_yx,
            deformed_meshgrid=deformed_meshgrid_yx))
    expected_updated_pixel_locations = tf.constant([[2, 0],
                                                    [0, 2],
                                                    [-1, -1],
                                                    [-1, -1],
                                                    [-1, -1],
                                                    [2, 0],
                                                    [0, 2],
                                                    [-1, -1]], dtype=tf.int32)
    self.assertAllEqual(updated_pixel_locations.numpy(),
                        expected_updated_pixel_locations.numpy())

  def test_project_points_with_depth_visibility_check(self):
    point_positions = tf.constant([[-1.0, -1.0, 1.0],
                                   [-1.0, 1.0, 1.0],
                                   [1.0, -1.0, 1.0],
                                   [1.0, 1.0, 1.0]], dtype=tf.float32)
    camera_intrinsics = tf.constant([[1.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0],
                                     [0.0, 0.0, 1.0]], dtype=tf.float32)
    camera_rotation_matrix = tf.constant([[1.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0],
                                          [0.0, 0.0, 1.0]], dtype=tf.float32)
    camera_translation = tf.constant([5.0, 5.0, 0.0], dtype=tf.float32)
    image_width = 10
    image_height = 10
    depth_image_00 = tf.ones([5, 5, 1], dtype=tf.float32) * 1.0
    depth_image_01 = tf.ones([5, 5, 1], dtype=tf.float32) * 2.0
    depth_image_10 = tf.ones([5, 5, 1], dtype=tf.float32) * 1.0
    depth_image_11 = tf.ones([5, 5, 1], dtype=tf.float32) * 4.0
    depth_image = tf.concat([
        tf.concat([depth_image_00, depth_image_01], axis=1),
        tf.concat([depth_image_10, depth_image_11], axis=1)
    ],
                            axis=0)
    depth_threshold = 0.1
    (points_in_image_frame,
     visibility) = projections.project_points_with_depth_visibility_check(
         point_positions=point_positions,
         camera_intrinsics=camera_intrinsics,
         camera_rotation_matrix=camera_rotation_matrix,
         camera_translation=camera_translation,
         image_width=image_width,
         image_height=image_height,
         depth_image=depth_image,
         depth_intrinsics=camera_intrinsics,
         depth_threshold=depth_threshold)
    self.assertAllEqual(
        visibility.numpy().astype(np.int32), np.array([1, 1, 0, 0]))
    self.assertAllEqual(points_in_image_frame.numpy(), np.array([[4, 4],
                                                                 [4, 6],
                                                                 [6, 4],
                                                                 [6, 6]]))


if __name__ == '__main__':
  tf.test.main()
