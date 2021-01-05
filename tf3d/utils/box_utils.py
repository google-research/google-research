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

"""Functions to compute overlaps between set of oriented boxes and points."""

import tensorflow as tf


def point_features_to_box_features(points, point_features, num_bins_per_axis,
                                   boxes_length, boxes_height, boxes_width,
                                   boxes_rotation_matrix, boxes_center):
  """Creates box features from the feature of the points inside each box.

  Given a set of points and their features, and a set of oriented
  boxes (paremeterized as dimension + rotation + translation), this function
  computes a feature for each box.

  Args:
    points: A tf.Tensor of shape (N, 3) with N point positions of dimension 3.
    point_features: A tf.Tensor of shape (N, feature_dim).
    num_bins_per_axis: Number of bins per axis of the box.
    boxes_length: A tf.Tensor of shape (B, 1) with B box sizes.
    boxes_height: A tf.Tensor of shape (B, 1) with B box sizes.
    boxes_width: A tf.Tensor of shape (B, 1) with B box sizes.
    boxes_rotation_matrix: A tf.Tensor of shape (B, 3, 3) with rotation for B
      boxes.
    boxes_center: A tf.Tensor of shape (B, 3) with each row containing 3D
      translation component (t) of the box pose, pointing to center of the box.

  Returns:
    box_features: A tf.Tensor of shape (B, num_bins_per_axis^3 * feature_dim).
  """
  if len(points.shape) != 2:
    raise ValueError('Points should be rank 2.')
  if len(point_features.shape) != 2:
    raise ValueError('Point_features should be rank 2.')
  if len(boxes_length.shape) != 2:
    raise ValueError('Box lengths should be rank 2.')
  if len(boxes_height.shape) != 2:
    raise ValueError('Box heights should be rank 2.')
  if len(boxes_width.shape) != 2:
    raise ValueError('Box widths should be rank 2.')
  if len(boxes_rotation_matrix.shape) != 3:
    raise ValueError('Box rotation matrices should be rank 3.')
  if len(boxes_center.shape) != 2:
    raise ValueError('Box centers should be rank 2.')

  num_boxes = tf.shape(boxes_length)[0]
  num_segments = num_bins_per_axis * num_bins_per_axis * num_bins_per_axis
  feature_dim = point_features.get_shape().as_list()[1]

  def body_fn(box_id):
    """While loop body function to update box indices."""
    points_in_box_frame = tf.tensordot(
        points - boxes_center[box_id],
        tf.transpose(boxes_rotation_matrix[box_id]),
        axes=(1, 1))
    box_size = tf.concat(
        [boxes_length[box_id], boxes_width[box_id], boxes_height[box_id]],
        axis=0)
    points_in_normalized_box_corner_frame = (
        points_in_box_frame / box_size) + 0.5
    point_indices = tf.cast(
        tf.math.floor(points_in_normalized_box_corner_frame *
                      tf.cast(num_bins_per_axis, dtype=tf.float32)),
        dtype=tf.int32)
    valid_mask = tf.reduce_all(
        tf.logical_and(
            tf.greater_equal(point_indices, 0),
            tf.less(point_indices, num_bins_per_axis)),
        axis=1)
    point_indices_valid = tf.boolean_mask(point_indices, valid_mask)
    point_segments = tf.reduce_sum(
        point_indices_valid * tf.constant(
            [[1, num_bins_per_axis, num_bins_per_axis * num_bins_per_axis]],
            dtype=tf.int32),
        axis=1)
    point_features_valid = tf.boolean_mask(point_features, valid_mask)
    return tf.math.unsorted_segment_mean(
        data=point_features_valid,
        segment_ids=point_segments,
        num_segments=num_segments)

  box_features = tf.map_fn(
      fn=body_fn, elems=tf.range(num_boxes), dtype=tf.float32)
  box_features = tf.stop_gradient(box_features)
  return tf.reshape(box_features, [num_boxes, num_segments * feature_dim])


def map_points_to_boxes(points, boxes_length, boxes_height, boxes_width,
                        boxes_rotation_matrix, boxes_center, box_margin):
  """Given pointcloud and oriented 3D boxes computes per point box id.

  Given a set of points, and a set of oriented boxes (paremeterized as dimension
  + rotation + translation), this function computes for each point, the box
  index it belongs to. If a point is not contained in any box a -1 value
  is stored. if a point is contained in multiple boxes, output box_indices will
  have index corresponding to the last box containing the point.

  We expect points and box orientation/translation are provided in same
  reference frame. The points and box can be either 2 or 3 dimensional.

  Args:
    points: A tf.Tensor of shape (N, 3) with N point positions of dimension 3.
    boxes_length: A tf.Tensor of shape (B, 1) with B box sizes.
    boxes_height: A tf.Tensor of shape (B, 1) with B box sizes.
    boxes_width: A tf.Tensor of shape (B, 1) with B box sizes.
    boxes_rotation_matrix: A tf.Tensor of shape (B, 3, 3) with rotation for B
      boxes.
    boxes_center: A tf.Tensor of shape (B, 3) with each row containing 3D
      translation component (t) of the box pose, pointing to center of the box.
    box_margin: A margin to be added to the box radius.

  Returns:
    box_indices: A tf.Tensor of shape (N,) with box ids.
  """
  if len(points.shape) != 2:
    raise ValueError('Points should be rank 2.')
  if len(boxes_length.shape) != 2:
    raise ValueError('Box lengths should be rank 2.')
  if len(boxes_height.shape) != 2:
    raise ValueError('Box heights should be rank 2.')
  if len(boxes_width.shape) != 2:
    raise ValueError('Box widths should be rank 2.')
  if len(boxes_rotation_matrix.shape) != 3:
    raise ValueError('Box rotation matrices should be rank 3.')
  if len(boxes_center.shape) != 2:
    raise ValueError('Box centers should be rank 2.')

  num_points = tf.shape(points)[0]
  num_boxes = tf.shape(boxes_length)[0]

  def body_fn(box_indices, box_id):
    """While loop body function to update box indices."""

    # Transform points to box frame
    points_in_box_frame = tf.tensordot(
        points - boxes_center[box_id],
        tf.transpose(boxes_rotation_matrix[box_id]),
        axes=(1, 1))

    # Find out which points are within box.
    box_radius = tf.concat(
        [boxes_length[box_id], boxes_width[box_id], boxes_height[box_id]],
        axis=0) / 2.0
    box_radius += box_margin
    within_box = tf.reduce_all(
        tf.less_equal(tf.abs(points_in_box_frame), box_radius), axis=1)

    # Set box indices for within box points. We set box_id + 1 and finally use
    # return box_indices - 1. This allows us to use tf.maximum.
    box_id += 1
    box_indices = tf.maximum(box_indices,
                             tf.cast(within_box, dtype=tf.int32) * box_id)
    return box_indices, box_id

  # Loop over all boxes and update box_indices.
  box_indices, _ = tf.while_loop(
      cond=lambda box_indices, box_id: box_id < num_boxes,
      body=body_fn,
      loop_vars=[tf.zeros([
          num_points,
      ], dtype=tf.int32),
                 tf.constant(0)])
  return box_indices - 1


def get_box_corners_3d(boxes_length, boxes_height, boxes_width,
                       boxes_rotation_matrix, boxes_center):
  """Given 3D oriented boxes, computes the box corner positions.

  A 6dof oriented box is fully described by the size (dimension) of the box, and
  its 6DOF pose (R|t). We expect each box pose to be given as 3x3 rotation (R)
  and 3D translation (t) vector pointing to center of the box.

  We expect box pose given as rotation (R) and translation (t) are provided in
  same reference frame we expect to get the box corners. In other words, a point
  in box frame x_box becomes: x = R * x_box + t.

  box_sizes describe size (dimension) of each box along x, y and z direction of
  the box reference frame. Typically these are described as length, width and
  height respectively. So each row of box_sizes encodes [length, width, height].

                 z
                 ^
                 |

             2 --------- 1
            /|          /|
           / |         / |
          3 --------- 0  |
          |  |        |  |    --> y
          |  6 -------|- 5
          | /         | /
          |/          |/
          7 --------- 4


            /
           x

  Args:
    boxes_length: A tf.Tensor of shape (B, 1) with B box sizes.
    boxes_height: A tf.Tensor of shape (B, 1) with B box sizes.
    boxes_width: A tf.Tensor of shape (B, 1) with B box sizes.
    boxes_rotation_matrix: A tf.Tensor of shape (B, 3, 3) with rotation for B
      boxes.
    boxes_center: A tf.Tensor of shape (B, 3) with each row containing 3D
      translation component (t) of the box pose, pointing to center of the box.

  Returns:
    A tf.Tensor of shape (B, 8, 3) containing box corners.
  """
  if len(boxes_length.shape) != 2:
    raise ValueError('Box lengths should be rank 2.')
  if len(boxes_height.shape) != 2:
    raise ValueError('Box heights should be rank 2.')
  if len(boxes_width.shape) != 2:
    raise ValueError('Box widths should be rank 2.')
  if len(boxes_rotation_matrix.shape) != 3:
    raise ValueError('Box rotation matrices should be rank 3.')
  if len(boxes_center.shape) != 2:
    raise ValueError('Box centers should be rank 2.')

  num_boxes = tf.shape(boxes_length)[0]

  # Corners in normalized box frame (unit cube centered at origin)
  corners = tf.constant([
      [0.5, 0.5, 0.5],  # top
      [-0.5, 0.5, 0.5],  # top
      [-0.5, -0.5, 0.5],  # top
      [0.5, -0.5, 0.5],  # top
      [0.5, 0.5, -0.5],  # bottom
      [-0.5, 0.5, -0.5],  # bottom
      [-0.5, -0.5, -0.5],  # bottom
      [0.5, -0.5, -0.5],  # bottom
  ])
  # corners in box frame
  corners = tf.einsum(
      'bi,ji->bji', tf.concat([boxes_length, boxes_width, boxes_height],
                              axis=1), corners)
  # corners after rotation
  corners = tf.einsum('bij,bkj->bki', boxes_rotation_matrix, corners)
  # corners after translation
  corners = corners + tf.reshape(boxes_center, (num_boxes, 1, 3))

  return corners


def get_box_as_dotted_lines(box_corners, num_of_points_per_line=100):
  """Convert box corners into points representing dotted lines.

  The order of the box corners are expected as follows:

             2 --------- 1
            /|          /|
           / |         / |
          3 --------- 0  |
          |  |        |  |
          |  6 -------|- 5
          | /         | /
          |/          |/
          7 --------- 4

  Args:
    box_corners: A tf.Tensor of shape (B, 8, 3) containing with B set of
      ordered box corners.
    num_of_points_per_line: An int indicating number of points per dotted line.
  Returns:
    A tf.Tensor of shape (B, 12 * num_of_points_per_line, 3) representing
    point positions for B boxes.
  """
  lines = tf.constant([[0, 1], [1, 2], [2, 3], [3, 0],  # top
                       [4, 5], [5, 6], [6, 7], [7, 4],  # bottom
                       [0, 4], [1, 5], [2, 6], [3, 7]],  # vertical
                      dtype=tf.int32)
  line_indices = tf.reshape(lines, [-1])

  line_endpoints = tf.gather(box_corners, line_indices, axis=1)
  line_endpoints = tf.reshape(line_endpoints, [-1, 12, 2, 3])
  coef = tf.linspace(0., 1., num_of_points_per_line)
  coef = coef[tf.newaxis, tf.newaxis, :, tf.newaxis]
  interpolated_points = ((1. - coef) * line_endpoints[:, :, 0, tf.newaxis, :]
                         + coef * line_endpoints[:, :, 1, tf.newaxis, :])
  interpolated_points = tf.reshape(interpolated_points,
                                   [-1, 12 * num_of_points_per_line, 3])
  return interpolated_points


def ray_to_box_coordinate_frame(box_center, box_rotation_matrix,
                                rays_start_point, rays_end_point):
  """Moves a set of rays into a box's coordinate frame.

  Args:
    box_center: A tf.float32 tensor of size [3] or [r, 3].
    box_rotation_matrix: A tf.float32 tensor of size [3, 3] or [r, 3, 3].
    rays_start_point: A tf.float32 tensor of size [r, 3] where r is the number
      of rays.
    rays_end_point: A tf.float32 tensor of size [r, 3] where r is the number of
      rays.

  Returns:
    rays_start_point_in_box_frame: A tf.float32 tensor of size [r, 3].
    rays_end_point_in_box_frame: A tf.float32 tensor of size [r, 3].
  """
  r = tf.shape(rays_start_point)[0]
  box_center = tf.broadcast_to(box_center, [r, 3])
  box_rotation_matrix = tf.broadcast_to(box_rotation_matrix, [r, 3, 3])
  rays_start_point_in_box_frame = tf.linalg.matmul(
      tf.expand_dims(rays_start_point - box_center, axis=1),
      box_rotation_matrix)
  rays_end_point_in_box_frame = tf.linalg.matmul(
      tf.expand_dims(rays_end_point - box_center, axis=1),
      box_rotation_matrix)
  return (tf.reshape(rays_start_point_in_box_frame, [-1, 3]),
          tf.reshape(rays_end_point_in_box_frame, [-1, 3]))


def ray_box_intersection(box_center, box_rotation_matrix, box_length,
                         box_width, box_height, rays_start_point,
                         rays_end_point, epsilon=0.000001):
  """Intersects a set of rays with a box.

  Note: The intersection points are returned in the box coordinate frame.
  Note: Make sure the start and end point of the rays are not the same.
  Note: Even though a start and end point is passed for each ray, rays are
    never ending and can intersect a box beyond their start / end points.

  Args:
    box_center: A tf.float32 tensor of size [3]. or [r, 3]
    box_rotation_matrix: A tf.float32 tensor of size [3, 3]or [r, 3, 3].
    box_length: A scalar tf.float32 tensor or of size [r].
    box_width: A scalar tf.float32 tensor or of size [r].
    box_height: A scalar tf.float32 tensor or of size [r].
    rays_start_point: A tf.float32 tensor of size [r, 3] where r is the number
      of rays.
    rays_end_point: A tf.float32 tensor of size [r, 3] where r is the number of
      rays.
    epsilon: A very small number.

  Returns:
    intersection_points_in_box_frame: A tf.float32 tensor of size [r', 2, 3]
      that contains intersection points in box coordinate frame.
    indices_of_intersecting_rays: A tf.int32 tensor of size [r'].
  """
  r = tf.shape(rays_start_point)[0]
  box_length = tf.broadcast_to(box_length, [r])
  box_height = tf.broadcast_to(box_height, [r])
  box_width = tf.broadcast_to(box_width, [r])
  box_center = tf.broadcast_to(box_center, [r, 3])
  box_rotation_matrix = tf.broadcast_to(box_rotation_matrix, [r, 3, 3])
  rays_start_point_in_box_frame, rays_end_point_in_box_frame = (
      ray_to_box_coordinate_frame(
          box_center=box_center,
          box_rotation_matrix=box_rotation_matrix,
          rays_start_point=rays_start_point,
          rays_end_point=rays_end_point))
  rays_a = rays_end_point_in_box_frame - rays_start_point_in_box_frame
  intersection_masks = []
  intersection_points = []
  box_size = [box_length, box_width, box_height]
  for axis in range(3):
    plane_value = box_size[axis] / 2.0
    for _ in range(2):
      plane_value = -plane_value
      t = ((plane_value - rays_start_point_in_box_frame[:, axis]) /
           rays_a[:, axis])
      intersection_points_i = []
      intersection_masks_i = tf.cast(tf.ones_like(t, dtype=tf.int32), tf.bool)
      for axis2 in range(3):
        intersection_points_i_axis2 = (
            rays_start_point_in_box_frame[:, axis2] + t * rays_a[:, axis2])
        intersection_points_i.append(intersection_points_i_axis2)
        intersection_masks_i = tf.logical_and(
            intersection_masks_i,
            tf.logical_and(
                intersection_points_i_axis2 <=
                (box_size[axis2] / 2.0 + epsilon),
                intersection_points_i_axis2 >=
                (-box_size[axis2] / 2.0 - epsilon)))
      intersection_points_i = tf.stack(intersection_points_i, axis=1)
      intersection_masks.append(intersection_masks_i)
      intersection_points.append(intersection_points_i)
  intersection_masks = tf.stack(intersection_masks, axis=1)
  intersection_points = tf.stack(intersection_points, axis=1)
  intersection_masks_any = tf.equal(
      tf.reduce_sum(tf.cast(intersection_masks, dtype=tf.int32), axis=1), 2)
  indices = tf.cast(
      tf.range(tf.shape(intersection_masks_any)[0]), dtype=tf.int32)
  indices = tf.boolean_mask(indices, intersection_masks_any)
  intersection_masks = tf.boolean_mask(intersection_masks,
                                       intersection_masks_any)
  intersection_points = tf.boolean_mask(intersection_points,
                                        intersection_masks_any)
  intersection_points = tf.reshape(
      tf.boolean_mask(intersection_points, intersection_masks), [-1, 2, 3])
  return intersection_points, indices
