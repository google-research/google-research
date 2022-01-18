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

"""A library of projections between 2D and 3D."""

import tensorflow as tf


def to_camera_frame(world_frame_points,
                    rotate_world_to_camera,
                    translate_world_to_camera):
  """Transform from world frame to camera frame.

  Args:
    world_frame_points: A tf.Tensor of shape (N, 3) of 3D locations in world
      frame we want transformed into a tf.Tensor in camera frame.
    rotate_world_to_camera: The world to camera frame rotation as a (3, 3)
      tf.Tensor.
    translate_world_to_camera: The world to camera translation as a (3,)
      tf.Tensor

  Returns:
    camera_frame_points: A tf.Tensor of shape (N, 3), which is 3D locations
      in camera frame.
  """
  return tf.tensordot(world_frame_points, rotate_world_to_camera,
                      axes=(1, 1)) + translate_world_to_camera


def to_world_frame(camera_frame_points,
                   rotate_world_to_camera,
                   translate_world_to_camera):
  """Convert from camera frame to world frame.

  Args:
    camera_frame_points: A tf.Tensor of shape (N, 3) of 3D locations in
      camera frame we want transformed into a tf.Tensor in world frame.
    rotate_world_to_camera: The world to camera frame rotation as a (3, 3)
      tf.Tensor.
    translate_world_to_camera: The world to camera translation as a (3,)
      tf.Tensor

  Returns:
    world_frame_points: A tf.Tensor of shape (N, 3) which is 3D locations
      in world frame.
  """
  return tf.tensordot(camera_frame_points - translate_world_to_camera,
                      tf.transpose(rotate_world_to_camera), axes=(1, 1))


def to_image_frame(camera_frame_points,
                   image_height,
                   image_width,
                   camera_intrinsics,
                   forward_axis=2):
  """Project points from camera frame (3D) into the image (2D).

  Args:
    camera_frame_points: A tf.Tensor of shape (N, 3) of 3D locations
      we want projected into the image.
    image_height: The image height as an int.
    image_width: The image width as an int.
    camera_intrinsics: The camera intrinsics to project from camera
      frame to the image plane. Should be a (3, 3) tensor.
    forward_axis: Which axis point forwards from the camera.

  Returns:
    image_points: A tf.Tensor of shape (N, 2) with pixel locations of dtype
      tf.float32. Each row has the format of (x, y).
    within_image: A boolean tf.Tensor of shape (N,) indicating whether
      each point falls within the image or not.

  Raises:
    ValueError: If shape of camera_frame_points is not (N, 3).
  """
  if camera_frame_points.get_shape().as_list()[1] != 3:
    raise ValueError('Expecting camera frame points to be shape (N, 3)')

  within_image = camera_frame_points[:, forward_axis] >= 0
  image_frame_points = tf.tensordot(camera_frame_points, camera_intrinsics,
                                    axes=(1, 1))
  image_frame_points = image_frame_points[:, :2] / image_frame_points[:, 2:3]

  # Compute image bounds check
  rounded = tf.cast(tf.round(image_frame_points), dtype=tf.int32)
  within_image &= (rounded[:, 0] >= 0) & (rounded[:, 0] < image_width)
  within_image &= (rounded[:, 1] >= 0) & (rounded[:, 1] < image_height)

  return image_frame_points, within_image


def image_frame_to_camera_frame(image_frame, camera_intrinsics):
  """Un-project points that are in image frame to the camera fame.

  Given a camera intrinsics matrix K, and a set of 2d points in image frame,
  this functions returns the 3D points at unit depth.

  Currently this function only supports a linear camera model, where intrinsics
  is 3 x 3 matrix e.g
      [fx  s  cx]
  K = [0  fy  cy]
      [0   0   1]
  where fx, fy being focal length in pixels, s being skew, and cx, cy is
  principal point in pixels.

  Args:
    image_frame: A tensor of shape (N, 2) containing N 2D image projections.
    camera_intrinsics: A tensor of shape (3, 3) containing intrinsic matrix.

  Returns:
    A (N, 3) tensor containing N 3D point positions in camera frame at unit
      depth (z value).
  """
  num_points = tf.shape(image_frame)[0]
  camera_frame = tf.einsum(
      'ij,nj->ni', tf.linalg.inv(camera_intrinsics),
      tf.concat(
          [image_frame,
           tf.ones([num_points, 1], dtype=image_frame.dtype)],
          axis=1))
  camera_frame = camera_frame / tf.expand_dims(camera_frame[:, 2], axis=1)
  return camera_frame


def create_image_from_point_values_unbatched(
    pixel_locations,
    pixel_values,
    image_height,
    image_width,
    default_value=0,
    use_sparse_tensor=False):
  """Creates an image (like depth) from a list of pixel locations and values.

  Args:
    pixel_locations: A tf.int32 tensor of shape [N, 2] with u, v pixel
      locations.
    pixel_values: A tensor of shape [N, m] or [N,] with per pixel values.
    image_height: An int for the image height.
    image_width: An int for the image width.
    default_value: default fill value of the output image tensor for pixels
      other than pixel_locations.
    use_sparse_tensor: Whether to use the sparse tensor version of scatter_nd.

  Returns:
    image: An image where every pixel in pixel_location has a value
      according to pixel_values.

  Raises:
    ValueError: if pixel_locations or pixel_values ranks are incompatible.
    ValueError: if you try to have a non-zero default value without using
      use_sparse_tensor
  """
  if len(pixel_locations.shape) != 2:
    raise ValueError('pixel_locations should be rank 2.')
  if len(pixel_values.shape) not in [1, 2]:
    raise ValueError('pixel_values should have rank of 1 or 2')
  if len(pixel_values.shape) == 1:
    pixel_values = tf.expand_dims(pixel_values, axis=1)

  valid_locations_y = tf.logical_and(
      tf.greater_equal(pixel_locations[:, 0], 0),
      tf.less(pixel_locations[:, 0], image_height))
  valid_locations_x = tf.logical_and(
      tf.greater_equal(pixel_locations[:, 1], 0),
      tf.less(pixel_locations[:, 1], image_width))
  valid_locations = tf.logical_and(valid_locations_y, valid_locations_x)
  pixel_locations = tf.boolean_mask(pixel_locations, valid_locations)
  pixel_values = tf.boolean_mask(pixel_values, valid_locations)

  n = tf.shape(pixel_locations)[0]
  value_dim = pixel_values.get_shape().as_list()[1]
  # In: [N, 2] w/ i, j
  pixel_locations = tf.tile(
      tf.expand_dims(pixel_locations, axis=1), [1, value_dim, 1])
  # Out: [N, value_dim, 2]

  pixel_locations_addition = tf.tile(
      tf.reshape(tf.range(value_dim, dtype=tf.int32), [1, value_dim, 1]),
      [n, 1, 1])
  # Out: [N, value_dim, 1]
  pixel_locations = tf.concat([pixel_locations, pixel_locations_addition],
                              axis=2)
  # Out: [N, value_dim, 3] (y, x, c)
  pixel_locations_2d = tf.reshape(pixel_locations, [n * value_dim, 3])
  if use_sparse_tensor:
    image = tf.SparseTensor(
        indices=tf.cast(pixel_locations_2d, dtype=tf.int64),
        values=tf.reshape(pixel_values, [n * value_dim]),
        dense_shape=(image_height, image_width, value_dim))
    return tf.sparse.to_dense(
        sp_input=image, default_value=default_value, validate_indices=False)
  else:
    image = tf.scatter_nd(
        indices=tf.cast(pixel_locations_2d, dtype=tf.int64),
        updates=tf.reshape(pixel_values - default_value, [n * value_dim]),
        shape=(image_height, image_width, value_dim))
    image += default_value
    return image


def create_image_from_point_values(
    pixel_locations,
    pixel_values,
    num_valid_points,
    image_height,
    image_width,
    default_value=0,
    use_sparse_tensor=False):
  """Creates an image (like depth) from a list of pixel locations and values.

  Args:
    pixel_locations: A tf.int32 tensor of shape [batch_size, N, 2] with u, v
      pixel locations.
    pixel_values: A tensor of shape [batch_size, N, m] with per pixel values.
    num_valid_points: A tensor of shape [batch_size] containing the number of
      valid points in each example of the batch.
    image_height: An int for the image height.
    image_width: An int for the image width.
    default_value: default fill value of the output image tensor for pixels
      other than pixel_locations.
    use_sparse_tensor: Whether to use the sparse tensor version of scatter_nd.

  Returns:
    image: An image of size [batch_size, image_height, image_width, m] where
      every pixel in pixel_location has a value according to pixel_values.

  Raises:
    ValueError: if pixel_locations or pixel_values ranks are incompatible.
    ValueError: if you try to have a non-zero default value without using
      use_sparse_tensor.
  """
  def _create_image_from_point_values_i(i):
    num_valid_points_i = num_valid_points[i]
    pixel_locations_i = pixel_locations[i, :num_valid_points_i, :]
    pixel_values_i = pixel_values[i, :num_valid_points_i, :]
    return create_image_from_point_values_unbatched(
        pixel_locations=pixel_locations_i,
        pixel_values=pixel_values_i,
        image_height=image_height,
        image_width=image_width,
        default_value=default_value,
        use_sparse_tensor=use_sparse_tensor)

  batch_size = pixel_locations.get_shape().as_list()[0]
  if batch_size is None:
    raise ValueError('Batch size is unknown at graph construction time.')
  return tf.map_fn(_create_image_from_point_values_i, tf.range(batch_size),
                   tf.float32)


def move_image_values_to_points(
    image_values,
    image_point_indices,
    num_points,
    default_value=-1,
    use_sparse_tensor=True):
  """Transfers image values to the corresponding points.

  Args:
    image_values: A tensor of shape [height, width, num_values] that contains
      values in image that needs to be transferred to points.
    image_point_indices: A tf.int32 tensor of shape [height, width, 1] that
      contains index of the point corresponding to each pixel. If a pixel does
      not correspond to a point, it should have a value of -1.
    num_points: A tf.int32 scalar.
    default_value: default fill value of the output point tensor for points
      that do not have a corresponding pixel.
    use_sparse_tensor: Whether to use the sparse tensor version of scatter_nd.

  Returns:
    A tensor of size [num_points, num_values].

  Raises:
    ValueError: If image_values is not a rank 3 tensor.
  """
  if len(image_values.shape) != 3:
    raise ValueError('`image_values` should be rank 3 tensor.')
  value_dim = tf.shape(image_values)[2]
  image_point_indices = tf.reshape(image_point_indices, [-1, 1])
  image_values = tf.reshape(image_values, [-1, value_dim])
  valid_indices = tf.greater_equal(tf.reshape(image_point_indices, [-1]), 0)
  image_point_indices = tf.boolean_mask(image_point_indices, valid_indices)
  image_values = tf.boolean_mask(image_values, valid_indices)
  image_values = tf.reshape(image_values, [-1])
  num_indices = tf.shape(image_point_indices)[0]
  image_point_indices = tf.tile(image_point_indices, [1, value_dim])
  value_dim_tiled = tf.tile(
      tf.reshape(tf.range(value_dim), [1, -1]), [num_indices, 1])
  image_point_indices = tf.stack([image_point_indices, value_dim_tiled], axis=2)
  image_point_indices = tf.cast(
      tf.reshape(image_point_indices, [-1, 2]), dtype=tf.int64)
  if use_sparse_tensor:
    point_values = tf.SparseTensor(
        indices=image_point_indices,
        values=image_values,
        dense_shape=(num_points, value_dim))
    return tf.sparse.to_dense(
        sp_input=point_values,
        default_value=default_value,
        validate_indices=False)
  else:
    point_values = tf.scatter_nd(
        indices=image_point_indices,
        updates=image_values - default_value,
        shape=(num_points, value_dim))
    point_values += default_value
  return point_values


def update_pixel_locations_given_deformed_meshgrid(pixel_locations,
                                                   original_meshgrid,
                                                   deformed_meshgrid):
  """Updates the point pixel locations given a deformed meshgrid.

  Args:
    pixel_locations: A tf.int32 tensor of shape [N, 2] with y, x pixel
      locations.
    original_meshgrid: A tf.int32 tensor of size [height, width, 2] with y, x
      pixel locations. The assumption is that meshgrid values start from 1.
    deformed_meshgrid: A tf.int32 tensor of
      size [deformed_height, deformed_width, 2] with y, x pixel locations.
      Invalid positions have values less or equal to 0.

  Returns:
    update_pixel_locations: A tf.int32 tensor of shape [N, 2] with y, x pixel
      locations.
  """
  max_y = tf.reduce_max(original_meshgrid[:, :, 0]) + 1
  max_x = tf.reduce_max(original_meshgrid[:, :, 1]) + 1
  pixel_indices = (pixel_locations[:, 0] + 1) * max_x + (
      pixel_locations[:, 1] + 1)
  valid_pixel_locations_y = tf.logical_and(
      tf.greater_equal(pixel_locations[:, 0], 0),
      tf.less(pixel_locations[:, 0],
              tf.shape(original_meshgrid)[0]))
  valid_pixel_locations_x = tf.logical_and(
      tf.greater_equal(pixel_locations[:, 1], 0),
      tf.less(pixel_locations[:, 1],
              tf.shape(original_meshgrid)[1]))
  valid_pixel_locations = tf.logical_and(valid_pixel_locations_y,
                                         valid_pixel_locations_x)
  pixel_indices *= tf.cast(valid_pixel_locations, dtype=pixel_indices.dtype)
  valid_deformed_positions = tf.reduce_all(
      tf.greater(deformed_meshgrid, 0), axis=2)
  valid_deformed_positions = tf.reshape(valid_deformed_positions, [-1])
  x_deformed_meshgrid, y_deformed_meshgrid = tf.meshgrid(
      tf.range(tf.shape(deformed_meshgrid)[1]),
      tf.range(tf.shape(deformed_meshgrid)[0]))
  yx_deformed_meshgrid = tf.stack([y_deformed_meshgrid, x_deformed_meshgrid],
                                  axis=2)
  yx_deformed_meshgrid = tf.boolean_mask(
      tf.reshape(yx_deformed_meshgrid, [-1, 2]), valid_deformed_positions)
  deformed_indices = (
      deformed_meshgrid[:, :, 0] * max_x + deformed_meshgrid[:, :, 1])
  deformed_indices = tf.boolean_mask(
      tf.reshape(deformed_indices, [-1]), valid_deformed_positions)
  deformed_meshgrid = tf.boolean_mask(
      tf.reshape(deformed_meshgrid, [-1, 2]), valid_deformed_positions)
  scatter_nd_indices = tf.concat([
      tf.stack(
          [deformed_indices, tf.zeros_like(deformed_indices)], axis=1),
      tf.stack(
          [deformed_indices, tf.ones_like(deformed_indices)], axis=1)
  ],
                                 axis=0)
  scatter_nd_updates = (
      tf.concat([yx_deformed_meshgrid[:, 0], yx_deformed_meshgrid[:, 1]],
                axis=0) + 1)
  map_original_indices_to_deformed_yx = tf.scatter_nd(
      indices=tf.cast(scatter_nd_indices, dtype=tf.int64),
      updates=scatter_nd_updates,
      shape=[max_y * max_x, 2])
  map_original_indices_to_deformed_yx -= 1
  return tf.gather(map_original_indices_to_deformed_yx, pixel_indices)


def project_points_with_depth_visibility_check(point_positions,
                                               camera_intrinsics,
                                               camera_rotation_matrix,
                                               camera_translation,
                                               image_width,
                                               image_height,
                                               depth_image,
                                               depth_intrinsics=None,
                                               depth_threshold=0.1):
  """Project 3D points to image with depthmap based visibility check.

  Args:
    point_positions: A tf.float32 tensor of shape [N, 3] containing N 3D point
      positions.
    camera_intrinsics: A tf.float32 tensor of shape [3, 3] contains intrinsic
      matrix.
    camera_rotation_matrix: A tf.float32 tensor of size [3, 3].
    camera_translation: A tf.float32 tensor of size [3].
    image_width: Width of image.
    image_height: Height of image.
    depth_image: Depth image as 2D tensor.
    depth_intrinsics: A tf.float32 tensor of size [3, 3]. If None, it is set to
      be same as camera_intrinsics.
    depth_threshold: Threshold for depth checking.

  Returns:
    points_in_image_frame: A tf.int32 tensor of size [N, 2] containing the x, y
      location of point projections in image.
    visibility: A tf.bool tensor of size [N] which denotes if a point is visible
      from the image.
  """
  if depth_intrinsics is None:
    depth_intrinsics = camera_intrinsics

  image_height = tf.convert_to_tensor(image_height, dtype=tf.int32)
  image_width = tf.convert_to_tensor(image_width, dtype=tf.int32)
  depth_image_height = tf.shape(depth_image)[0]
  depth_image_width = tf.shape(depth_image)[1]

  # Points in camera frame
  points_in_camera_frame = tf.linalg.einsum('ij,nj->ni', camera_rotation_matrix,
                                            point_positions) + tf.expand_dims(
                                                camera_translation, axis=0)

  # Points in image frame.
  points_in_image_frame = tf.linalg.einsum('ij,nj->ni', camera_intrinsics,
                                           points_in_camera_frame)
  points_in_image_frame = tf.cast(
      points_in_image_frame[:, :2] / points_in_image_frame[:, 2:3],
      dtype=tf.int32)

  # Points in depth frame.
  points_in_depth_frame = tf.linalg.einsum('ij,nj->ni', depth_intrinsics,
                                           points_in_camera_frame)
  points_in_depth_frame = tf.cast(
      points_in_depth_frame[:, :2] / points_in_depth_frame[:, 2:3],
      dtype=tf.int32)

  # Check if point is in front of camera.
  visibility = tf.greater(points_in_camera_frame[:, 2], 0.0)

  # Check if within color image.
  visibility &= tf.math.reduce_all(
      tf.greater_equal(points_in_image_frame, 0), axis=1)
  visibility &= tf.math.reduce_all(
      tf.less(points_in_image_frame,
              tf.expand_dims(tf.stack([image_width, image_height]), axis=0)),
      axis=1)

  # Check if within depth image.
  visibility &= tf.math.reduce_all(
      tf.greater_equal(points_in_depth_frame, 0), axis=1)
  visibility &= tf.math.reduce_all(
      tf.less(
          points_in_depth_frame,
          tf.expand_dims(
              tf.stack([depth_image_width, depth_image_height]), axis=0)),
      axis=1)

  # Check if the depth of points is within some threshold of depth_image.
  points_in_depth_frame = tf.boolean_mask(points_in_depth_frame, visibility)
  points_in_depth_frame_y = points_in_depth_frame[:, 1]
  points_in_depth_frame_x = points_in_depth_frame[:, 0]
  indices = (
      points_in_depth_frame_y * depth_image_width + points_in_depth_frame_x)

  visible_points_in_camera_frame = tf.boolean_mask(points_in_camera_frame,
                                                   visibility)
  depth_of_visible_points_in_camera_frame = visible_points_in_camera_frame[:, 2]
  depth_of_visible_points_in_depth_frame = tf.gather(
      tf.reshape(depth_image, [-1]), indices)
  valid_depths_visible = tf.less_equal(
      tf.abs(depth_of_visible_points_in_camera_frame -
             depth_of_visible_points_in_depth_frame), depth_threshold)
  visibility_indices = tf.cast(tf.where(visibility), dtype=tf.int32)
  valid_depths = tf.scatter_nd(
      indices=visibility_indices,
      updates=tf.cast(valid_depths_visible, dtype=tf.int32),
      shape=tf.shape(visibility))
  visibility &= tf.cast(valid_depths, dtype=tf.bool)

  return points_in_image_frame, visibility
