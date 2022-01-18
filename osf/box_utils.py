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

"""Utility functions for bounding box computation."""
import time
from absl import logging
import numpy as np
import tensorflow as tf
from osf import ray_utils


def ray_to_box_coordinate_frame(box_center, box_rotation_matrix,
                                rays_start_point, rays_end_point):
  """Moves a set of rays into a box's coordinate frame.

  Args:
    box_center: A tf.float32 tensor of size [3].
    box_rotation_matrix: A tf.float32 tensor of size [3, 3].
    rays_start_point: A tf.float32 tensor of size [r, 3] where r is the number
      of rays.
    rays_end_point: A tf.float32 tensor of size [r, 3] where r is the number of
      rays.

  Returns:
    rays_start_point_in_box_frame: A tf.float32 tensor of size [r, 3].
    rays_end_point_in_box_frame: A tf.float32 tensor of size [r, 3].
  """
  rays_start_point_in_box_frame = tf.tensordot(
      rays_start_point - box_center,
      tf.transpose(box_rotation_matrix),
      axes=(1, 1))
  rays_end_point_in_box_frame = tf.tensordot(
      rays_end_point - box_center,
      tf.transpose(box_rotation_matrix),
      axes=(1, 1))
  return rays_start_point_in_box_frame, rays_end_point_in_box_frame


def ray_to_box_coordinate_frame_pairwise(box_center, box_rotation_matrix,
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
      tf.expand_dims(rays_end_point - box_center, axis=1), box_rotation_matrix)
  return (tf.reshape(rays_start_point_in_box_frame,
                     [-1, 3]), tf.reshape(rays_end_point_in_box_frame, [-1, 3]))


def ray_box_intersection(box_center,
                         box_rotation_matrix,
                         box_size,
                         rays_start_point,
                         rays_end_point,
                         epsilon=0.000001):
  """Intersects a set of rays with a box.

  Note: The intersection points are returned in the box coordinate frame.
  Note: Make sure the start and end point of the rays are not the same.
  Note: Even though a start and end point is passed for each ray, rays are
    never ending and can intersect a box beyond their start / end points.

  Args:
    box_center: A tf.float32 tensor of size [3].
    box_rotation_matrix: A tf.float32 tensor of size [3, 3].
    box_size: A tf.float32 tensor of size [3].
    rays_start_point: A tf.float32 tensor of size [r, 3] where r is the number
      of rays.
    rays_end_point: A tf.float32 tensor of size [r, 3] where r is the number of
      rays.
    epsilon: A very small number.

  Returns:
    rays_start_point_in_box_frame:
    intersection_masks_any:
    intersection_points_in_box_frame: A tf.float32 tensor of size [r', 2, 3]
      that contains intersection points in box coordinate frame.
    indices_of_intersecting_rays: A tf.int32 tensor of size [r'].
    intersection_ts: A tf.float32 tensor of size [r'].
  """
  rays_start_point_in_box_frame, rays_end_point_in_box_frame = (
      ray_to_box_coordinate_frame(
          box_center=box_center,
          box_rotation_matrix=box_rotation_matrix,
          rays_start_point=rays_start_point,
          rays_end_point=rays_end_point))
  rays_a = rays_end_point_in_box_frame - rays_start_point_in_box_frame
  normalized_rays_a = ray_utils.normalize_rays(rays=rays_a)  # [R, 3]
  intersection_masks = []
  intersection_points = []
  intersection_zs = []
  # box_size = [box_length, box_width, box_height]
  for axis in range(3):
    plane_value = box_size[axis] / 2.0
    for _ in range(2):
      plane_value = -plane_value
      # Compute the scalar multiples of `rays_a` to apply in order to intersect
      # with the plane.
      t = ((plane_value - rays_start_point_in_box_frame[:, axis]) /
           rays_a[:, axis])  # [R,]

      # Compute the distances between ray origins and the plane.
      z = ((plane_value - rays_start_point_in_box_frame[:, axis]) /
           normalized_rays_a[:, axis])
      intersection_points_i = []

      # Initialize a mask which represents whether each ray intersects with the
      # current plane.
      intersection_masks_i = tf.cast(tf.ones_like(t, dtype=tf.int32),
                                     tf.bool)  # [R,]
      for axis2 in range(3):
        # Compute the point of intersection for the current axis.
        intersection_points_i_axis2 = (  # [R,]
            rays_start_point_in_box_frame[:, axis2] + t * rays_a[:, axis2])
        intersection_points_i.append(intersection_points_i_axis2)  # 3x [R,]

        # Update the intersection mask depending on whether the intersection
        # point is within bounds.
        intersection_masks_i = tf.logical_and(  # [R,]
            intersection_masks_i,
            tf.logical_and(
                intersection_points_i_axis2 <=
                (box_size[axis2] / 2.0 + epsilon),
                intersection_points_i_axis2 >=
                (-box_size[axis2] / 2.0 - epsilon)))
      intersection_points_i = tf.stack(intersection_points_i, axis=1)  # [R, 3]
      intersection_masks.append(intersection_masks_i)  # List of [R,]
      intersection_points.append(intersection_points_i)  # List of [R, 3]
      intersection_zs.append(z)  # List of [R,]
  intersection_masks = tf.stack(intersection_masks, axis=1)  # [R, 6]
  intersection_points = tf.stack(intersection_points, axis=1)  # [R, 6, 3]
  intersection_zs = tf.stack(intersection_zs, axis=1)  # [R, 6]

  # Compute a mask over rays with exactly two plane intersections out of the six
  # planes. More intersections are possible if the ray coincides with a box
  # edge or corner, but we'll ignore these cases for now.
  intersection_masks_any = tf.equal(  # [R,]
      tf.reduce_sum(tf.cast(intersection_masks, dtype=tf.int32), axis=1), 2)
  indices = tf.cast(  # [R,]
      tf.range(tf.shape(intersection_masks_any)[0]), dtype=tf.int32)
  # Apply the intersection masks over tensors.
  indices = tf.boolean_mask(indices, intersection_masks_any)  # [R',]
  intersection_masks = tf.boolean_mask(
      intersection_masks,  # [R', 6]
      intersection_masks_any)
  intersection_points = tf.boolean_mask(
      intersection_points,  # [R', 6, 3]
      intersection_masks_any)
  intersection_points = tf.reshape(  # [R', 2, 3]
      tf.boolean_mask(intersection_points, intersection_masks), [-1, 2, 3])
  intersection_zs = tf.boolean_mask(  # [R', 6]
      intersection_zs, intersection_masks_any)
  intersection_zs = tf.reshape(  # [R', 2]
      tf.boolean_mask(intersection_zs, intersection_masks), [-1, 2])
  return (rays_start_point_in_box_frame, intersection_masks_any,
          intersection_points, indices, intersection_zs)


def ray_box_intersection_pairwise(box_center,
                                  box_rotation_matrix,
                                  box_length,
                                  box_width,
                                  box_height,
                                  rays_start_point,
                                  rays_end_point,
                                  exclude_negative_t=True,
                                  exclude_enlarged_t=True,
                                  epsilon=0.000001):
  """Intersects a set of rays with a box.

  Note: The intersection points are returned in the box coordinate frame.
  Note: Make sure the start and end point of the rays are not the same.
  Note: Even though a start and end point is passed for each ray, rays are
    never ending and can intersect a box beyond their start / end points.

  Args:
    box_center: A tf.float32 tensor of size [3] or [r, 3]
    box_rotation_matrix: A tf.float32 tensor of size [3, 3] or [r, 3, 3].
    box_length: A scalar tf.float32 tensor or of size [r].
    box_width: A scalar tf.float32 tensor or of size [r].
    box_height: A scalar tf.float32 tensor or of size [r].
    rays_start_point: A tf.float32 tensor of size [r, 3] where r is the number
      of rays.
    rays_end_point: A tf.float32 tensor of size [r, 3] where r is the number of
      rays.
    exclude_negative_t: bool.
    exclude_enlarged_t: bool
    epsilon: A very small number.

  Returns:
    intersection_points_in_box_frame: A tf.float32 tensor of size [r', 2, 3]
      that contains intersection points in box coordinate frame.
    indices_of_intersecting_rays: A tf.int32 tensor of size [r'].
    intersection_ts: A tf.float32 tensor of size [r'].
  """
  r = tf.shape(rays_start_point)[0]
  box_length = tf.broadcast_to(box_length, [r])
  box_width = tf.broadcast_to(box_width, [r])
  box_height = tf.broadcast_to(box_height, [r])
  box_center = tf.broadcast_to(box_center, [r, 3])
  box_rotation_matrix = tf.broadcast_to(box_rotation_matrix, [r, 3, 3])
  rays_start_point_in_box_frame, rays_end_point_in_box_frame = (
      ray_to_box_coordinate_frame_pairwise(
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
      # Compute the scalar multiples of `rays_a` to apply in order to intersect
      # with the plane.
      t = ((plane_value - rays_start_point_in_box_frame[:, axis]) /  # [R,]
           rays_a[:, axis])
      # The current axis only intersects with plane if the ray is not parallel
      # with the plane. Note that this will result in `t` being +/- infinity.
      intersects_with_plane = tf.math.abs(rays_a[:, axis]) > epsilon
      if exclude_negative_t:
        t = tf.math.maximum(t, 0.0)  # [R,]
      if exclude_enlarged_t:
        t = tf.math.minimum(t, 1.0)  # [R,]
      intersection_points_i = []

      # Initialize a mask which represents whether each ray intersects with the
      # current plane.
      intersection_masks_i = tf.cast(tf.ones_like(t, dtype=tf.int32),
                                     tf.bool)  # [R,]
      for axis2 in range(3):
        # Compute the point of intersection for the current axis.
        intersection_points_i_axis2 = (  # [R,]
            rays_start_point_in_box_frame[:, axis2] + t * rays_a[:, axis2])
        intersection_points_i.append(intersection_points_i_axis2)  # 3x [R,]

        # Update the intersection mask depending on whether the intersection
        # point is within bounds.
        intersection_masks_i = tf.logical_and(  # [R,]
            tf.logical_and(intersection_masks_i, intersects_with_plane),
            tf.logical_and(
                intersection_points_i_axis2 <=
                (box_size[axis2] / 2.0 + epsilon),
                intersection_points_i_axis2 >=
                (-box_size[axis2] / 2.0 - epsilon)))
      intersection_points_i = tf.stack(intersection_points_i, axis=1)  # [R, 3]
      intersection_masks.append(intersection_masks_i)  # List of [R,]
      intersection_points.append(intersection_points_i)  # List of [R, 3]
  intersection_masks = tf.stack(intersection_masks, axis=1)  # [R, 6]
  intersection_points = tf.stack(intersection_points, axis=1)  # [R, 6, 3]

  # Compute a mask over rays with exactly two plane intersections out of the six
  # planes. More intersections are possible if the ray coincides with a box
  # edge or corner, but we'll ignore these cases for now.
  intersection_masks_any = tf.equal(  # [R,]
      tf.reduce_sum(tf.cast(intersection_masks, dtype=tf.int32), axis=1), 2)
  indices = tf.cast(  # [R,]
      tf.range(tf.shape(intersection_masks_any)[0]), dtype=tf.int32)
  # Apply the intersection masks over tensors.
  indices = tf.boolean_mask(indices, intersection_masks_any)  # [R',]
  intersection_masks = tf.boolean_mask(
      intersection_masks,  # [R', 6]
      intersection_masks_any)
  intersection_points = tf.boolean_mask(
      intersection_points,  # [R', 6, 3]
      intersection_masks_any)
  intersection_points = tf.reshape(  # [R', 2, 3]
      tf.boolean_mask(intersection_points, intersection_masks), [-1, 2, 3])
  return rays_start_point_in_box_frame, intersection_masks_any, intersection_points, indices


def compute_bounds_from_intersect_points(rays_o, intersect_indices,
                                         intersect_points):
  """Computes bounds from intersection points.

  Note: Make sure that inputs are in the same coordinate frame.

  Args:
    rays_o: [R, 3] tf.float32.
    intersect_indices: [R', 1] tf.float32.
    intersect_points: [R', 2, 3] tf.float32.

  Returns:
    intersect_bounds: [R', 2] tf.float32.

  where R is the number of rays and R' is the number of intersecting rays.
  """
  start = time.time()
  intersect_rays_o = tf.gather_nd(  # [R', 3]
      params=rays_o,  # [R, 3]
      indices=intersect_indices,  # [R', 1]
  )
  logging.info(
      '[compute_bounds_from_intersect_points] tf.gather_nd took %f seconds.',
      time.time() - start)
  intersect_rays_o = intersect_rays_o[:, None, :]  # [R', 1, 3]
  intersect_diff = intersect_points - intersect_rays_o  # [R', 2, 3]
  intersect_bounds = tf.linalg.norm(intersect_diff, axis=2)  # [R', 2]

  logging.info('[compute_bounds_from_intersect_points] rays_o: %s',
               rays_o.shape)
  logging.info('[compute_bounds_from_intersect_points] intersect_indices: %s',
               intersect_indices.shape)
  logging.info('[compute_bounds_from_intersect_points] intersect_rays_o: %s',
               intersect_rays_o.shape)
  logging.info('[compute_bounds_from_intersect_points] intersect_points: %s',
               intersect_points.shape)
  logging.info('[compute_bounds_from_intersect_points] intersect_diff: %s',
               intersect_diff.shape)
  logging.info('[compute_bounds_from_intersect_points] intersect_bounds: %s',
               intersect_bounds.shape)

  # Sort the bounds so that near comes before far for all rays.
  intersect_bounds = tf.sort(intersect_bounds, axis=1)  # [R', 2]
  logging.info('[compute_bounds_from_intersect_points] intersect_bounds: %s',
               intersect_bounds.shape)

  # For some reason the sort function returns [R', ?] instead of [R', 2], so we
  # will explicitly reshape it.
  intersect_bounds = tf.reshape(intersect_bounds, [-1, 2])  # [R', 2]
  logging.info('[compute_bounds_from_intersect_points] intersect_bounds: %s',
               intersect_bounds.shape)
  return intersect_bounds


def compute_ray_bbox_bounds(rays_o, rays_d, box_dims, box_center,
                            box_rotation, far_limit):
  """Computes near and far bounds for rays intersecting with bounding boxes.

  Args:
    rays_o: [R, 3] tf.float32. A set of ray origins.
    rays_d: [R, 3] tf.float32. A set of ray directions.
    box_dims: [3,] tf.float32. Bounding box dimensions.
    box_center: [3,] tf.float32. The center of the box.
    box_rotation: [3, 3] tf.float32. The rotation matrix of the box.
    far_limit: float. The maximum far value to use.

  Returns:
    bounds: [R, 2] tf.float32. The bounds per-ray.
    intersect_mask: [R,] tf.float32. The mask denoting intersections.
  """
  del far_limit

  # Compute ray destinations.
  normalized_rays_d = ray_utils.normalize_rays(rays=rays_d)
  rays_dst = rays_o + 1e10 * normalized_rays_d

  # Transform the rays from world to box coordinate frame.
  (rays_o_in_box_frame, intersect_mask, intersect_points_in_box_frame,
   intersect_indices, _) = (
       ray_box_intersection(
           box_center=box_center,
           box_rotation_matrix=box_rotation,
           box_size=box_dims,
           rays_start_point=rays_o,
           rays_end_point=rays_dst))
  intersect_indices = intersect_indices[:, None]  # [R', 1]

  intersect_bounds = compute_bounds_from_intersect_points(
      rays_o=rays_o_in_box_frame,
      intersect_indices=intersect_indices,
      intersect_points=intersect_points_in_box_frame)

  num_rays = tf.shape(rays_o)[0]
  bounds = tf.scatter_nd(  # [R, 2]
      indices=intersect_indices,  # [R', 1]
      updates=intersect_bounds,  # [R', 2]
      shape=[num_rays, 2])
  return bounds, intersect_mask


def compute_ray_bbox_bounds_pairwise(rays_o, rays_d, box_length,
                                     box_width, box_height, box_center,
                                     box_rotation, far_limit):
  """Computes near and far bounds for rays intersecting with bounding boxes.

  Args:
    rays_o: [R, 3] tf.float32. A set of ray origins.
    rays_d: [R, 3] tf.float32. A set of ray directions.
    box_length: scalar or [R,] tf.float32. Bounding box length.
    box_width: scalar or [R,] tf.float32. Bounding box width.
    box_height: scalar or [R,] tf.float32. Bounding box height.
    box_center: [3,] or [R, 3] tf.float32. The center of the box.
    box_rotation: [3, 3] or [R, 3, 3] tf.float32. The box rotation matrix.
    far_limit: float. The maximum far value to use.

  Returns:
    intersect_bounds: [R', 2] tf.float32. The bounds per-ray, sorted in
      ascending order.
    intersect_indices: [R', 1] tf.float32. The intersection indices.
    intersect_mask: [R,] tf.float32. The mask denoting intersections.
  """
  del far_limit

  # Compute ray destinations.
  normalized_rays_d = ray_utils.normalize_rays(rays=rays_d)
  rays_dst = rays_o + 1e10 * normalized_rays_d
  logging.info('[compute_ray_bbox_bounds_pairwise] rays_d: %s', rays_d.shape)
  logging.info('[compute_ray_bbox_bounds_pairwise] normalized_rays_d: %s',
               normalized_rays_d.shape)
  logging.info('[compute_ray_bbox_bounds_pairwise] rays_dst: %s',
               rays_dst.shape)

  # Transform the rays from world to box coordinate frame.
  start = time.time()
  (rays_o_in_box_frame, intersect_mask, intersect_points_in_box_frame,
   intersect_indices) = (
       ray_box_intersection_pairwise(
           box_center=box_center,
           box_rotation_matrix=box_rotation,
           box_length=box_length,
           box_width=box_width,
           box_height=box_height,
           rays_start_point=rays_o,
           rays_end_point=rays_dst))
  logging.info('ray_box_intersection_pairwise took %f seconds.',
               time.time() - start)
  intersect_indices = intersect_indices[:, None]  # [R', 1]
  intersect_bounds = compute_bounds_from_intersect_points(
      rays_o=rays_o_in_box_frame,
      intersect_indices=intersect_indices,
      intersect_points=intersect_points_in_box_frame)
  return intersect_bounds, intersect_indices, intersect_mask


def compute_ray_bbox_bounds_npy(rays_o, rays_d, box_dims, box_center,
                                box_rotation, far_limit):
  """Computes near and far bounds for rays intersecting with bounding boxes.

  Args:
    rays_o: [R, 3] np.float32. A set of ray origins.
    rays_d: [R, 3] np.float32. A set of ray directions.
    box_dims: [3,] np.float32. Bounding box dimensions.
    box_center: [3,] np.float32. The center of the box.
    box_rotation: [3, 3] np.float32. The rotation matrix of the box.
    far_limit: float. The maximum far value to use.

  Returns:
    bounds: [R, 2] np.float32. The set of bounds.
  """
  rays_o = tf.constant(rays_o, dtype=tf.float32)
  rays_d = tf.constant(rays_d, dtype=tf.float32)
  box_dims = tf.constant(box_dims, dtype=tf.float32)
  box_center = tf.constant(box_center, dtype=tf.float32)
  box_rotation = tf.constant(box_rotation, dtype=tf.float32)

  bounds, _ = compute_ray_bbox_bounds(
      rays_o=rays_o,
      rays_d=rays_d,
      box_dims=box_dims,
      box_center=box_center,
      box_rotation=box_rotation,
      far_limit=far_limit)
  bounds = bounds.numpy()
  return bounds


def compute_ray_bbox_bounds_pairwise_npy(rays_o, rays_d, box_length, box_width,
                                         box_height, box_center, box_rotation,
                                         far_limit):
  """Computes near and far bounds for rays intersecting with bounding boxes.

  Args:
    rays_o: [R, 3] np.float32. A set of ray origins.
    rays_d: [R, 3] np.float32. A set of ray directions.
    box_length: scalar or [R,] np.float32. Bounding box length.
    box_width: scalar or [R,] np.float32. Bounding box width.
    box_height: scalar or [R,] np.float32. Bounding box height.
    box_center: [3,] np.float32. The center of the box.
    box_rotation: [3, 3] np.float32. The rotation matrix of the box.
    far_limit: float. The maximum far value to use.

  Returns:
    bounds: [R, 2] np.float32. The set of bounds.
  """
  num_rays = len(rays_o)

  rays_o = tf.constant(rays_o, dtype=tf.float32)
  rays_d = tf.constant(rays_d, dtype=tf.float32)
  box_length = tf.constant(box_length, dtype=tf.float32)
  box_width = tf.constant(box_width, dtype=tf.float32)
  box_height = tf.constant(box_height, dtype=tf.float32)
  box_center = tf.constant(box_center, dtype=tf.float32)
  box_rotation = tf.constant(box_rotation, dtype=tf.float32)

  intersect_bounds, intersect_indices, _ = compute_ray_bbox_bounds_pairwise(
      rays_o=rays_o,
      rays_d=rays_d,
      box_length=box_length,
      box_width=box_width,
      box_height=box_height,
      box_center=box_center,
      box_rotation=box_rotation,
      far_limit=far_limit)
  intersect_indices = intersect_indices.numpy()[:, 0]
  intersect_bounds = intersect_bounds.numpy()

  # Scatter back to original set of rays.
  bounds = np.zeros((num_rays, 2), dtype=np.float32)
  bounds[intersect_indices] = intersect_bounds
  return bounds
