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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from single_view_mpi.libs import geometry
from single_view_mpi.libs import utils

# Because there are tables of numbers that line up to make them readable.
# pylint: disable=bad-whitespace


def _random_camera(b):
  angles = tf.random.uniform([b, 3], minval=-math.pi / 10, maxval=math.pi / 10)
  translate = tf.random.uniform([b, 3], minval=-10, maxval=10)
  pose = geometry.pose_from_6dof(tf.concat([translate, angles], axis=-1))
  fxfy = tf.random.uniform([b, 2], minval=.1, maxval=2)
  cxcy = tf.random.uniform([b, 2], minval=0, maxval=1)
  intrinsics = tf.concat([fxfy, cxcy], axis=-1)
  return pose, intrinsics


class GeometryTest(tf.test.TestCase):

  # ========== MATRICES, PLANES, POINTS ==========

  def test_broadcasting_matmul(self):
    a = tf.constant([[[1.0, 2.0]], [[10.0, 20.0]]])
    b = tf.constant([[[[3.0], [4.0]]]])
    # a has shape    [2, 1, 2],
    # b has shape [1, 1, 2, 1].
    # With broadcasting, ab will have shape [1, 2, 1, 1]
    # and ba will have shape [1, 2, 2, 2]
    prod = geometry.broadcasting_matmul(a, b)
    prod2 = geometry.broadcasting_matmul(b, a)
    self.assertAllEqual(prod, [[[[11.0]], [[110.0]]]])
    self.assertAllEqual(
        prod2, [[[[3.0, 6.0], [4.0, 8.0]], [[30.0, 60.0], [40.0, 80.0]]]])

  def test_mat34_to_mat44(self):
    m34 = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
    m44 = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0],
           [0.0, 0.0, 0.0, 1.0]]
    # Non-batched case
    m = geometry.mat34_to_mat44(tf.constant(m34))
    self.assertAllEqual(m, m44)
    # Batched with an extra dimension of size 3
    m = geometry.mat34_to_mat44(tf.constant([m34, m34, m34]))
    self.assertAllEqual(m, [m44, m44, m44])
    # Batched with an two extra dimensions of size 2 and 1
    m = geometry.mat34_to_mat44(tf.constant([[m34], [m34]]))
    self.assertAllEqual(m, [[m44], [m44]])
    # Batched with lots of dimensions
    m34_lots = [[[[m34] * 2] * 3] * 4] * 5
    m44_lots = [[[[m44] * 2] * 3] * 4] * 5
    m = geometry.mat34_to_mat44(tf.constant(m34_lots))
    self.assertAllEqual(m, m44_lots)

  def test_mat33_to_mat44(self):
    m33 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    m44 = [[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 6.0, 0.0], [7.0, 8.0, 9.0, 0.0],
           [0.0, 0.0, 0.0, 1.0]]
    # Non-batched case
    m = geometry.mat33_to_mat44(tf.constant(m33))
    self.assertAllEqual(m, m44)
    # Batched with an extra dimension of size 3
    m = geometry.mat33_to_mat44(tf.constant([m33, m33, m33]))
    self.assertAllEqual(m, [m44, m44, m44])
    # Batched with an two extra dimensions of size 2 and 1
    m = geometry.mat33_to_mat44(tf.constant([[m33], [m33]]))
    self.assertAllEqual(m, [[m44], [m44]])

  def test_mat34_product(self):
    # Identity
    i = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    a = [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]]
    b = [[2.0, 1.0, 0.0, 3.0], [1.0, 0.0, 2.0, -4.0], [0.0, 2.0, -1.0, 1.0]]
    # ab = a * b
    ab = [[4.0, 7.0, 1.0, 2.0], [7.0, 10.0, 2.0, 3.0], [10.0, 13.0, 3.0, 4.0]]
    a_id = geometry.mat34_product(tf.constant(a), tf.constant(i))
    id_a = geometry.mat34_product(tf.constant(i), tf.constant(a))
    self.assertAllEqual(a, a_id)
    self.assertAllEqual(a, id_a)
    prod = geometry.mat34_product(tf.constant(a), tf.constant(b))
    self.assertAllEqual(ab, prod)
    # Broadcasting
    abab = tf.constant([[a, i], [a, i]])
    bid = tf.constant([[b], [i]])
    prod = geometry.mat34_product(abab, bid)
    self.assertAllEqual([[ab, b], [a, i]], prod)

  def test_mat34_transform(self):
    i = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    m = [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]]
    # A few vectors and their images under m:
    zero = [0.0, 0.0, 0.0]
    mzero = [4.0, 5.0, 6.0]
    x = [1.0, 0.0, 0.0]
    mx = [5.0, 7.0, 9.0]
    y = [0.0, 1.0, 0.0]
    my = [6.0, 8.0, 10.0]
    z = [0.0, 0.0, 1.0]
    mz = [7.0, 9.0, 11.0]
    foo = [-10.0, 2.0, -3.0]
    mfoo = [-11.0, -21.0, -31.0]
    # Four vectors
    input_1 = tf.constant([zero, x, y, z, foo])
    identity_1 = geometry.mat34_transform(tf.constant(i), input_1)
    result_1 = geometry.mat34_transform(tf.constant(m), input_1)
    self.assertAllEqual([zero, x, y, z, foo], identity_1)
    self.assertAllEqual([mzero, mx, my, mz, mfoo], result_1)
    # A 2x2 matrix of vectors
    input_2 = tf.constant([[x, y], [z, foo]])
    identity_2 = geometry.mat34_transform(tf.constant(i), input_2)
    result_2 = geometry.mat34_transform(tf.constant(m), input_2)
    self.assertAllEqual([[x, y], [z, foo]], identity_2)
    self.assertAllEqual([[mx, my], [mz, mfoo]], result_2)
    # Batched transforms (2 batches of 3 vectors).
    matrices = tf.constant([i, m])
    input_3 = tf.constant([[zero, x, y], [z, foo, foo]])
    result_3 = geometry.mat34_transform(matrices, input_3)
    self.assertAllEqual([[zero, x, y], [mz, mfoo, mfoo]], result_3)

  def test_mat34_transform_planes(self):
    # Some planes...
    planes = tf.convert_to_tensor(
        [[1.0, 2.0, 3.0, 4.0], [0.5, 0.3, 0.1, 0.0],
         [0.0, 1.0, 0.0, 100.0], [-1.0, 0.0, -1.0, 0.25]])
    # ...and some points
    points = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    # (Since we know it's all linear, four planes and three
    # points are enough to span all possibilities!)
    i = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    m = [[1.0, 0.0, 0.0, 4.0], [0.0, 0.0, -1.0, 5.0], [0.0, 1.0, 0.0, 6.0]]
    # Identity transform doesn't change the planes:
    input_planes = tf.constant(planes)
    identity_planes = geometry.mat34_transform_planes(
        tf.constant(i), input_planes)
    self.assertAllEqual(identity_planes, planes)
    # Transform and inverse transform is the identity:
    self.assertAllEqual(
        geometry.mat34_transform_planes(
            tf.constant(m),
            geometry.mat34_transform_planes(
                geometry.mat34_pose_inverse(tf.constant(m)), input_planes)),
        planes)
    # Dot products between planes and points are preserved (since
    # m doesn't change scale):
    input_points = tf.constant(points)
    dot_products = tf.matmul(
        geometry.homogenize(input_points), planes, transpose_b=True)
    transformed_planes = geometry.mat34_transform_planes(
        tf.constant(m), input_planes)
    transformed_points = geometry.mat34_transform(tf.constant(m), input_points)
    transformed_dot_products = tf.matmul(
        geometry.homogenize(transformed_points),
        transformed_planes,
        transpose_b=True)
    self.assertAllClose(dot_products, transformed_dot_products)

  def test_mat34_pose_inverse(self):
    identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0]]
    translate = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0],
                 [0.0, 0.0, 1.0, 3.0]]
    untranslate = [[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, -2.0],
                   [0.0, 0.0, 1.0, -3.0]]
    # Rotate around Y axis and translate along X axis.
    c = math.cos(1.0)
    s = math.sin(1.0)
    rotate = [[c, 0.0, -s, 10.0], [0.0, 1.0, 0.0, 0.0], [s, 0.0, c, 0.0]]
    unrotate = [[c, 0.0, s, -10.0 * c], [0.0, 1.0, 0.0, 0.0],
                [-s, 0.0, c, 10 * s]]
    data = [
        # A series of (matrix, inverse) pairs to check. We check them both ways.
        (identity, identity),
        (translate, untranslate),
        (rotate, unrotate),
        # Batched examples
        ([identity, translate, rotate], [identity, untranslate, unrotate]),
        ([[identity, translate], [rotate,
                                  untranslate]], [[identity, untranslate],
                                                  [unrotate, translate]])
    ]
    for (matrix, inverse) in data:
      result = geometry.mat34_pose_inverse(tf.constant(matrix))
      self.assertAllClose(inverse, result)
      reverse = geometry.mat34_pose_inverse(tf.constant(inverse))
      self.assertAllClose(matrix, reverse)
      # Additionally, check that the product with the inverse is the identity
      product = geometry.mat34_product(
          tf.constant(matrix), geometry.mat34_pose_inverse(tf.constant(matrix)))
      (_, identities) = utils.broadcast_to_match(product, tf.constant(identity))
      self.assertAllClose(product, identities)

  def test_pose_from_6dof(self):
    parms_translate = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
    mat_translate = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]]
    # Use 90-degree rotations, so we can figure out the results easily
    # X: y -> z, z -> -y
    # Y: z -> x, x -> -z
    # Z: x -> y, y -> -x
    parms_rotate_x = [1.0, 2.0, 3.0, math.pi / 2, 0.0, 0.0]
    mat_rotate_x = [[1, 0, 0, 1], [0, 0, -1, 2], [0, 1, 0, 3]]
    parms_rotate_y = [1.0, 2.0, 3.0, 0.0, math.pi / 2, 0.0]
    mat_rotate_y = [[0, 0, 1, 1], [0, 1, 0, 2], [-1, 0, 0, 3]]
    parms_rotate_z = [1.0, 2.0, 3.0, 0.0, 0.0, math.pi / 2]
    mat_rotate_z = [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3]]
    # Z applied first, then Y, then X.
    parms_rotate_xy = [0.0, 0.0, 0.0, math.pi / 2, math.pi / 2, 0.0]
    mat_rotate_xy = [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]]
    parms_rotate_xz = [0.0, 0.0, 0.0, math.pi / 2, 0.0, math.pi / 2]
    mat_rotate_xz = [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0]]
    parms_rotate_xyz = [0.0, 0.0, 0.0, math.pi / 2, math.pi / 2, math.pi / 2]
    mat_rotate_xyz = [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]]
    # Invididual translation and rotations
    self.assertAllClose(mat_translate,
                        geometry.pose_from_6dof(tf.constant(parms_translate)))
    self.assertAllClose(mat_rotate_x,
                        geometry.pose_from_6dof(tf.constant(parms_rotate_x)))
    self.assertAllClose(mat_rotate_y,
                        geometry.pose_from_6dof(tf.constant(parms_rotate_y)))
    self.assertAllClose(mat_rotate_z,
                        geometry.pose_from_6dof(tf.constant(parms_rotate_z)))
    # Combinations
    self.assertAllClose(mat_rotate_xy,
                        geometry.pose_from_6dof(tf.constant(parms_rotate_xy)))
    self.assertAllClose(mat_rotate_xz,
                        geometry.pose_from_6dof(tf.constant(parms_rotate_xz)))
    self.assertAllClose(mat_rotate_xyz,
                        geometry.pose_from_6dof(tf.constant(parms_rotate_xyz)))

    # Batching (1 level)
    self.assertAllClose(
        [mat_rotate_xy, mat_translate, mat_rotate_xyz],
        geometry.pose_from_6dof(
            tf.constant([parms_rotate_xy, parms_translate, parms_rotate_xyz])))

    # Batching (2 levels)
    self.assertAllClose(
        [[mat_rotate_xy, mat_translate], [mat_rotate_xyz, mat_rotate_y]],
        geometry.pose_from_6dof(
            tf.constant([[parms_rotate_xy, parms_translate],
                         [parms_rotate_xyz, parms_rotate_y]])))

  # ========== CAMERAS ==========

  def test_intrinsics(self):
    intrinsics = tf.constant([[1.0, 1.0, 0.5, 0.5], [0.5, 0.4, 0.25, 0.75]])
    intrinsics_matrix = [[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
                         [[0.5, 0.0, 0.25], [0.0, 0.4, 0.75], [0.0, 0.0, 1.0]]]
    identity = tf.eye(3)
    self.assertAllClose(
        geometry.intrinsics_matrix(intrinsics), intrinsics_matrix)
    self.assertAllClose([identity, identity],
                        tf.matmul(
                            intrinsics_matrix,
                            geometry.inverse_intrinsics_matrix(intrinsics)))

  def test_homogenize(self):
    coords = [[1.0, 1.0], [0.5, -0.25], [100.0, -50.0]]
    hcoords = [[1.0, 1.0, 1.0], [0.5, -0.25, 1.0], [100.0, -50.0, 1.0]]
    homogenized = geometry.homogenize(tf.constant(coords))
    dehomogenized = geometry.dehomogenize(homogenized)
    scale = tf.constant([10.0, 0.2, .5], shape=[3, 1])
    dehomogenized2 = geometry.dehomogenize(homogenized * scale)
    self.assertAllClose(homogenized, hcoords)
    self.assertAllClose(dehomogenized, coords)
    self.assertAllClose(dehomogenized2, coords)

  def test_texture_to_camera_coordinates(self):
    # [fx, fy, cx, cy] for a camera with 90-degree field of view in x and y.
    intrinsics_a = [0.5, 0.5, 0.5, 0.5]
    # and for a ~53 degree field of view with principal point at bottom left.
    intrinsics_b = [1.0, 1.0, 0.0, 1.0]
    # Four corners of an image. Reminder: both in image coordinates and
    # in camera coordinates, X axis points right and Y axis points down.
    corners = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    coords_a = geometry.texture_to_camera_coordinates(
        tf.constant(corners), tf.constant(intrinsics_a))
    self.assertAllClose([[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0],
                         [1.0, 1.0, 1.0]], coords_a)
    coords_b = geometry.texture_to_camera_coordinates(
        tf.constant(corners), tf.constant(intrinsics_b))
    self.assertAllClose(
        [[0.0, -1.0, 1.0], [1.0, -1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
        coords_b)
    # Use the same data to test the reverse conversion. First multiply
    # by some different z-values to check that it projects back correctly.
    z_values = tf.constant([1.0, 2.0, 0.5, 0.75], shape=[4, 1])
    coords_a *= z_values
    coords_a = geometry.camera_to_texture_coordinates(coords_a,
                                                      tf.constant(intrinsics_a))
    self.assertAllClose(corners, coords_a)
    coords_b *= z_values
    coords_b = geometry.camera_to_texture_coordinates(coords_b,
                                                      tf.constant(intrinsics_b))
    self.assertAllClose(corners, coords_b)

  def test_get_camera_relative_points(self):
    # We'll have a batch of 2, with 0: identity pose, 1:translation.
    poses = tf.constant([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0]],
                         [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0],
                          [0.0, 0.0, 1.0, 3.0]]])
    points = tf.constant([[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 0.0, 0.0],
                           [3.0, 0.0, 0.0]],
                          [[1.0, 0.0, 0.0], [4.0, 5.0, 6.0], [5.0, 0.0, 0.0],
                           [6.0, 0.0, 0.0]]])
    indices = tf.constant([[0, 1, 1], [2, 3, 0]])
    self.assertAllClose(
        [
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],  # [0, 1, 1]
            [[6.0, 2.0, 3.0], [7.0, 2.0, 3.0], [2.0, 2.0, 3.0]]
        ],  # [2, 3, 0]
        geometry.get_camera_relative_points(indices, points, poses))

  # ========== IMAGES AND SAMPLING ==========

  def test_pixel_center_grid(self):

    def grid(height, width):
      return [[[(x + 0.5) / width, (y + 0.5) / height]
               for x in range(width)]
              for y in range(height)]

    for height in range(1, 5):
      for width in range(1, 5):
        self.assertAllClose(
            grid(height, width), geometry.pixel_center_grid(height, width))

  def clip_texture_coords_to_corner_pixels(self):
    width = 5
    height = 4
    # Corner pixel centers should be at (0.1, 0.125), (0.9, 0.875).
    # Each of x and y might be (a) less than the minimum, (b) in the valid
    # range, or (c) more than the maximum. That gives us 9 combinations, so
    # let's test them all:
    coords = [
        [0.0, -1.0],
        [-1.0, 0.4],
        [0.05, 3.14],  # x too small
        [0.1, 0.1],
        [0.2, 0.126],
        [0.88, 0.88],  # x ok
        [0.91, 0.0],
        [1.5, 0.65],
        [1.0, 1.0]
    ]  # x too big
    target = [[0.1, 0.125], [0.1, 0.4], [0.1, 0.875], [0.1, 0.125],
              [0.2, 0.126], [0.88, 0.875], [0.9, 0.125], [0.9, 0.65],
              [0.9, 0.875]]
    self.assertAllClose(
        geometry.clip_texture_coords_to_corner_pixels(
            tf.constant(coords), height, width), tf.constant(target))

  def test_sample_image(self):
    # A 2x2 image:
    red = [1.0, 0.0, 0.0]
    green = [0.0, 1.0, 0.0]
    blue = [0.0, 0.0, 1.0]
    white = [1.0, 1.0, 1.0]
    black = [0.0, 0.0, 0.0]
    image = tf.constant([[red, green], [blue, white]])
    # A list of points:
    points = []
    targets = []
    unclamped_targets = []

    # Add a point and the expected values when we sample it
    # with and without clamping the coordinates:
    def add_point(point, target, unclamped_target):
      points.append(point)
      targets.append(target)
      unclamped_targets.append(unclamped_target)

    # A point way outside the image:
    add_point([-1.0, -1.0], red, black)
    # The top left corner of the image:
    add_point([0.0, 0.0], red, [0.25, 0.0, 0.0])
    # The top-left pixel center:
    add_point([0.25, 0.25], red, red)
    # Half-way along the top edge:
    add_point([0.5, 0.0], [0.5, 0.5, 0.0], [0.25, 0.25, 0.0])
    # Half-way between the two left pixels:
    add_point([0.25, 0.5], [0.5, 0.0, 0.5], [0.5, 0.0, 0.5])
    # Centre of the image
    add_point([0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # Way off to the right
    add_point([2.0, 0.5], [0.5, 1.0, 0.5], black)

    self.assertAllClose(
        geometry.sample_image(image, tf.constant(points)), tf.constant(targets))
    self.assertAllClose(
        geometry.sample_image(image, tf.constant(points), clamp=False),
        tf.constant(unclamped_targets))

    # Now a very simple test with extra batch axes. We make four 1x1 images
    # and sample a single point from each one in various batch configurations.
    red_image = [[red]]
    green_image = [[green]]
    blue_image = [[blue]]
    white_image = [[white]]
    image_batch = [red_image, green_image, blue_image, white_image]
    points_batch = [[0.0, 0.0], [0.5, 0.25], [0.5, 0.5], [-1.0, -1.0]]
    target_batch = [red, green, blue, white]

    for dims in [[2, 2], [4, 1, 1], [4], [2, 1, 2, 1]]:
      self.assertAllClose(
          geometry.sample_image(
              tf.reshape(image_batch, dims + [1, 1, 3]),
              tf.reshape(points_batch, dims + [1, 2])),
          tf.reshape(target_batch, dims + [1, 3]))

  # ========== WARPS AND HOMOGRAPHIES ==========

  def test_homography(self):
    intrinsics = tf.constant([1.0, 1.0, 0.5, 0.5])
    pose_identity = tf.constant([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0]])
    pose_shift_left = tf.constant([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0],
                                   [0.0, 0.0, 1.0, 0.0]])
    # Rotate about Z axis by 0.1 radians.
    c = math.cos(0.1)
    s = math.sin(0.1)
    pose_rotated = tf.constant([[c, 0.0, -s, 0.0], [0.0, 1.0, 0.0, 0.0],
                                [s, 0.0, c, 0.0]])
    plane_far = tf.constant([0.0, 0.0, 1.0, -1e10])
    plane_near = tf.constant([0.0, 0.0, 1.0, -1.0])
    identity = tf.eye(3)
    # Homography when translating the camera along X axis:
    homography_left_far = geometry.inverse_homography(pose_identity, intrinsics,
                                                      pose_shift_left,
                                                      intrinsics, plane_far)
    homography_left_near = geometry.inverse_homography(pose_identity,
                                                       intrinsics,
                                                       pose_shift_left,
                                                       intrinsics, plane_near)
    homography_right_far = geometry.inverse_homography(pose_shift_left,
                                                       intrinsics,
                                                       pose_identity,
                                                       intrinsics, plane_far)
    homography_right_near = geometry.inverse_homography(pose_shift_left,
                                                        intrinsics,
                                                        pose_identity,
                                                        intrinsics, plane_near)
    # Far plane is so distant that homography should be the identity.
    self.assertAllClose(homography_left_far, identity)
    self.assertAllClose(homography_right_far, identity)
    # New plane shifts, and shifts back.
    self.assertAllClose(homography_left_near,
                        [[1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # Technically when we compose two homographies and check that the result
    # is the identity, we should do so up to a scale factor. But for these
    # simple cases it is not necessary.
    self.assertAllClose(
        tf.matmul(homography_left_near, homography_right_near), identity)

    # Now let's try a rotation.
    homography_rotate = geometry.inverse_homography(pose_identity, intrinsics,
                                                    pose_rotated, intrinsics,
                                                    plane_far)
    homography_unrotate = geometry.inverse_homography(pose_rotated, intrinsics,
                                                      pose_identity, intrinsics,
                                                      plane_far)
    homography_rotate_near = geometry.inverse_homography(
        pose_identity, intrinsics, pose_rotated, intrinsics, plane_near)
    self.assertAllClose(
        tf.matmul(homography_rotate, homography_unrotate), identity)
    # No translation so homographies for the two planes should be the same.
    self.assertAllClose(homography_rotate, homography_rotate_near)

    # Now we'll test that points move in suitable ways when the homographies
    # are applied.
    # 4 points: [[center, center-right], [top-left, bottom-left]],
    points = [[[0.5, 0.5], [1.0, 0.5]], [[0.0, 0.0], [0.0, 1.0]]]
    points_left_far = geometry.apply_homography(homography_left_far,
                                                tf.constant(points))
    points_left_near = geometry.apply_homography(homography_left_near,
                                                 tf.constant(points))
    points_rotated = geometry.apply_homography(homography_rotate,
                                               tf.constant(points))
    # Distant points don't move without camera rotation:
    self.assertAllClose(points_left_far, points)
    # Nearby points translate
    self.assertAllClose((points_left_near + [1.0, 0.0]), points)
    # Rotation around Z axis moves principal point by 0.1 radians:
    shift = math.tan(0.1) * 1.0  # 1.0 = fx
    self.assertAllClose(points_rotated[0, 0], [0.5 + shift, 0.5])

  def test_random_homography(self):
    # Test homographies as follows:
    # 1. generate three random points
    # 2. generate two random camera poses (source, target) and intrinsics
    # 3. compute the plane containing the three points
    # 4. project each point using each camera, to get texture coordinates
    # 5. generate the homography for this plane between the two cameras
    # 6. apply the homography to the target texture coords and we should get
    #    the source texture coords.
    # We do this for (a batch of) 1000 different random homographies.
    batch = 1000
    tf.random.set_seed(2345)
    points = tf.random.uniform([batch, 3, 3], minval=-100, maxval=100)
    source_pose, source_intrinsics = _random_camera(batch)
    target_pose, target_intrinsics = _random_camera(batch)
    source_points = geometry.mat34_transform(source_pose, points)
    target_points = geometry.mat34_transform(target_pose, points)
    # Compute the plane equation in source camera space.
    p0, p1, p2 = tf.unstack(source_points, axis=-2)
    normal = tf.linalg.cross(p1 - p0, p2 - p0)
    offset = -tf.math.reduce_sum(normal * p0, axis=-1, keepdims=True)
    plane = tf.concat([normal, offset], axis=-1)
    # Now we're ready for the homography.
    homography = geometry.inverse_homography(source_pose, source_intrinsics,
                                             target_pose, target_intrinsics,
                                             plane)
    source_coords = geometry.camera_to_texture_coordinates(
        source_points, source_intrinsics[Ellipsis, tf.newaxis, :])
    target_coords = geometry.camera_to_texture_coordinates(
        target_points, target_intrinsics[Ellipsis, tf.newaxis, :])
    # Apply-homography expects a 2D grid of coords, so add a dimension:
    source_coords = source_coords[Ellipsis, tf.newaxis, :]
    target_coords = target_coords[Ellipsis, tf.newaxis, :]
    result = geometry.apply_homography(homography, target_coords)
    # Every now and then we get a point very close to a camera plane, which
    # means accuracy will be lower and the test can fail. So we'll zero-out
    # all those points.
    source_bad = tf.abs(source_points[Ellipsis, -1]) < 1e-1
    target_bad = tf.abs(target_points[Ellipsis, -1]) < 1e-1
    valid = 1.0 - tf.cast(
        tf.math.logical_or(source_bad, target_bad), tf.float32)
    valid = valid[Ellipsis, tf.newaxis, tf.newaxis]
    self.assertAllClose((valid * source_coords), (valid * result),
                        atol=1e-03,
                        rtol=1e-03)


# pylint: enable=bad-whitespace

if __name__ == '__main__':
  tf.test.main()
