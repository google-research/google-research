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

"""Preprocessing utility functions."""

import functools
import math
import tensorflow as tf
from tf3d import standard_fields
from tf3d.utils import box_utils
from tf3d.utils import instance_segmentation_utils
from tf3d.utils import projections
from tf3d.utils import rotation_matrix
from tf3d.utils import voxel_utils


def randomly_sample_points(mesh_inputs, view_indices_2d_inputs,
                           target_num_points):
  """Randomly subsamples points and their properties to target_num_points.

  Args:
    mesh_inputs: A dictionary containing input mesh (point) tensors.
    view_indices_2d_inputs: A dictionary containing input point to view
      correspondence tensors.
    target_num_points: Target number of points to sample.
  """
  if target_num_points is None:
    return

  points = mesh_inputs[standard_fields.InputDataFields.point_positions]
  num_points = tf.shape(points)[0]
  indices = tf.random.uniform(
      shape=[target_num_points], minval=0, maxval=num_points, dtype=tf.int32)
  for key in sorted(mesh_inputs):
    mesh_inputs[key] = tf.gather(mesh_inputs[key], indices)
  for key in sorted(view_indices_2d_inputs):
    view_indices_2d_inputs[key] = tf.transpose(
        tf.gather(
            tf.transpose(view_indices_2d_inputs[key], [1, 0, 2]), indices),
        [1, 0, 2])


def translate_points(points, delta_x, delta_y, delta_z):
  return points + tf.expand_dims(tf.stack([delta_x, delta_y, delta_z]), axis=0)


def rotation_matrix_from_rotation_around_axis(rotation_angle, axis):
  """Computes and returns a rotation matrix given rotation angle and axis.

  Args:
    rotation_angle: A float value. Angle in radians.
    axis: A value in [0, 1, 2] for rotating around x, y, z axis.

  Returns:
    A tf.float32 tensor of size [3, 3].

  Raises:
    ValueError: If axis value is not in [0, 1, 2].
  """
  if axis == 0:
    return rotation_matrix.from_rotation_around_x(angle=rotation_angle)
  elif axis == 1:
    return rotation_matrix.from_rotation_around_y(angle=rotation_angle)
  elif axis == 2:
    return rotation_matrix.from_rotation_around_z(angle=rotation_angle)
  else:
    raise ValueError(('Invalid axis value: %d' % axis))


def rotate_points_around_axis(points, rotation_angle, axis, rotation_center):
  """Rotates points around axis.

  Args:
    points: A tf.float32 tensor of size [N, 3] containing points.
    rotation_angle: A float value containing the rotation angle in radians.
    axis: A value in [0, 1, 2] for rotating around x, y, z axis.
    rotation_center: A tf.float32 tensor of size [3]. Rotation will take place
      around this point.

  Returns:
    rotated_points: A tf.float32 tensor of size [N, 3] containing points.
  """
  rotation_center = tf.convert_to_tensor(rotation_center, dtype=tf.float32)
  rotation_matrix_tensor = rotation_matrix_from_rotation_around_axis(
      rotation_angle=rotation_angle, axis=axis)
  points_wrt_center = points - tf.expand_dims(rotation_center, axis=0)
  rotated_points_wrt_center = tf.linalg.matvec(
      tf.expand_dims(rotation_matrix_tensor, axis=0), points_wrt_center)
  return rotated_points_wrt_center + tf.expand_dims(rotation_center, axis=0)


def rotate_objects_around_axis(object_centers, object_rotation_matrices,
                               object_rotations_axis, rotation_angle, axis,
                               rotation_center):
  """Rotates boxes around axis.

  Args:
    object_centers: A tf.float32 tensor of size [N, 3] containing points.
    object_rotation_matrices: A tf.float32 tensor of size [N, 3, 3] containing
      object rotation matrices.
    object_rotations_axis: A tf.float32 tensor of size [N, 1] containing object
      rotations around axis in radians.
    rotation_angle: A float value containing the rotation angle in radians.
    axis: A value in [0, 1, 2] for rotating around x, y, z axis.
    rotation_center: A tf.float32 tensor of size [3]. Rotation will take place
      around this point.

  Returns:
    A tf.float32 tensor of size [N, 3] containing rotated object center
      locations.
    A tf.float32 tensor of size [N, 3, 3] or None.
    A tf.float32 tensor of size [N, 1] or None.
  """
  rotation_center = tf.convert_to_tensor(rotation_center, dtype=tf.float32)
  object_centers = rotate_points_around_axis(
      points=object_centers,
      rotation_angle=rotation_angle,
      axis=axis,
      rotation_center=rotation_center)
  if object_rotation_matrices is not None:
    rotation_matrix_tensor = rotation_matrix_from_rotation_around_axis(
        rotation_angle=rotation_angle, axis=axis)
    object_rotation_matrices = tf.linalg.matmul(
        tf.expand_dims(rotation_matrix_tensor, axis=0),
        object_rotation_matrices)
  if object_rotations_axis is not None:
    object_rotations_axis += rotation_angle
  return object_centers, object_rotation_matrices, object_rotations_axis


def randomly_scale_points_and_objects(mesh_inputs, object_inputs,
                                      min_scale_ratio, max_scale_ratio):
  """Scale points and objects.

  Picks a random scale ratio between min and max scale ratio. Then scales
  the point locations, object centers and object sizes based on that.
  Note that this function does not return a value. Instead it modifies the
  values inside the inputs dictionary.

  Please note that this function only transfers the points, and the geometry
  that corresponds images and points are not taken into account. The assumption
  is that points and image pixels are related based on already precomputed
  correspondences.

  Args:
    mesh_inputs: A dictionary containing mesh input tensors.
    object_inputs: A dictionary containing object input tensors.
    min_scale_ratio: A float value. Minimum scale ratio.
    max_scale_ratio: A float value. Maximum scale ratio.
  """
  if min_scale_ratio is not None and max_scale_ratio is not None:
    scale_ratio = tf.random.uniform([],
                                    minval=min_scale_ratio,
                                    maxval=max_scale_ratio,
                                    dtype=tf.float32)
    for key in [
        standard_fields.InputDataFields.point_positions,
        standard_fields.InputDataFields.object_length_points,
        standard_fields.InputDataFields.object_center_points,
        standard_fields.InputDataFields.object_height_points,
        standard_fields.InputDataFields.object_width_points
    ]:
      if key in mesh_inputs:
        mesh_inputs[key] *= scale_ratio
    for key in [
        standard_fields.InputDataFields.objects_center,
        standard_fields.InputDataFields.objects_length,
        standard_fields.InputDataFields.objects_height,
        standard_fields.InputDataFields.objects_width,
    ]:
      if key in object_inputs:
        object_inputs[key] *= scale_ratio


def rotate_points_and_objects_around_axis(mesh_inputs, object_inputs,
                                          min_degree_rotation,
                                          max_degree_rotation, axis,
                                          rotation_center):
  """Rotates both points and objects around axis.

  Warning: For now only object rotation around y axis is supported.

  Args:
    mesh_inputs: A dictionary containing mesh input tensors.
    object_inputs: A dictionary containing object input tensors.
    min_degree_rotation: A float. Minimum degree of rotation around the axis.
    max_degree_rotation: A float. Maximum degree of rotation around the axis.
    axis: A value in [0, 1, 2] for rotating around x, y, z axis.
    rotation_center: A tf.float32 tensor of size [3]. Rotation will take place
      around this point.
  """
  if min_degree_rotation is None or max_degree_rotation is None:
    return

  rotation_center = tf.convert_to_tensor(rotation_center, dtype=tf.float32)
  min_radian_rotation = math.pi * float(min_degree_rotation) / 180.0
  max_radian_rotation = math.pi * float(max_degree_rotation) / 180.0
  rotation_angle = tf.random.uniform([],
                                     minval=min_radian_rotation,
                                     maxval=max_radian_rotation,
                                     dtype=tf.float32)
  mesh_inputs[standard_fields.InputDataFields
              .point_positions] = rotate_points_around_axis(
                  points=mesh_inputs[
                      standard_fields.InputDataFields.point_positions],
                  rotation_angle=rotation_angle,
                  axis=axis,
                  rotation_center=rotation_center)
  object_rotations_axis = None
  if (standard_fields.InputDataFields.objects_center in object_inputs) and (
      standard_fields.InputDataFields.objects_rotation_matrix in object_inputs):
    object_rotation_matrices = object_inputs[
        standard_fields.InputDataFields.objects_rotation_matrix]
    (object_inputs[standard_fields.InputDataFields.objects_center],
     object_rotation_matrices,
     object_rotations_axis) = rotate_objects_around_axis(
         object_centers=object_inputs[
             standard_fields.InputDataFields.objects_center],
         object_rotation_matrices=object_rotation_matrices,
         object_rotations_axis=object_rotations_axis,
         rotation_angle=rotation_angle,
         axis=axis,
         rotation_center=rotation_center)
    object_inputs[standard_fields.InputDataFields
                  .objects_rotation_matrix] = object_rotation_matrices


def rotate_randomly(mesh_inputs, object_inputs, x_min_degree_rotation,
                    x_max_degree_rotation, y_min_degree_rotation,
                    y_max_degree_rotation, z_min_degree_rotation,
                    z_max_degree_rotation, rotation_center):
  """Rotates points and normals randomly around y axis.

  The rotation is applied to the points in the camera coordinate frame, and
  to the objects in camera coordinate frame.

  Args:
    mesh_inputs: A dictionary containing mesh input tensors.
    object_inputs: A dictionary containing object input tensors.
    x_min_degree_rotation: A float value containing the minimum rotation angle
      in degrees around x axis.
    x_max_degree_rotation: A float value containing the maximum rotation angle
      in degrees around x axis.
    y_min_degree_rotation: A float value containing the minimum rotation angle
      in degrees around y axis.
    y_max_degree_rotation: A float value containing the maximum rotation angle
      in degrees around y axis.
    z_min_degree_rotation: A float value containing the minimum rotation angle
      in degrees around z axis.
    z_max_degree_rotation: A float value containing the maximum rotation angle
      in degrees around z axis.
    rotation_center: A tf.float32 tensor of size [3]. Rotation will take place
      around this point.
  """
  rotate_points_and_objects_around_axis(
      mesh_inputs=mesh_inputs,
      object_inputs=object_inputs,
      min_degree_rotation=y_min_degree_rotation,
      max_degree_rotation=y_max_degree_rotation,
      axis=1,
      rotation_center=rotation_center)
  rotate_points_and_objects_around_axis(
      mesh_inputs=mesh_inputs,
      object_inputs=object_inputs,
      min_degree_rotation=x_min_degree_rotation,
      max_degree_rotation=x_max_degree_rotation,
      axis=0,
      rotation_center=rotation_center)
  rotate_points_and_objects_around_axis(
      mesh_inputs=mesh_inputs,
      object_inputs=object_inputs,
      min_degree_rotation=z_min_degree_rotation,
      max_degree_rotation=z_max_degree_rotation,
      axis=2,
      rotation_center=rotation_center)


def translate_randomly(mesh_inputs, object_inputs, delta_x_min, delta_x_max,
                       delta_y_min, delta_y_max, delta_z_min, delta_z_max):
  """Randomly moves the point location at training time."""
  delta_x = tf.random.uniform([],
                              minval=delta_x_min,
                              maxval=delta_x_max,
                              dtype=tf.float32)
  delta_y = tf.random.uniform([],
                              minval=delta_y_min,
                              maxval=delta_y_max,
                              dtype=tf.float32)
  delta_z = tf.random.uniform([],
                              minval=delta_z_min,
                              maxval=delta_z_max,
                              dtype=tf.float32)
  mesh_inputs[
      standard_fields.InputDataFields.point_positions] = translate_points(
          points=mesh_inputs[standard_fields.InputDataFields.point_positions],
          delta_x=delta_x,
          delta_y=delta_y,
          delta_z=delta_z)
  if standard_fields.InputDataFields.objects_center in object_inputs:
    object_inputs[
        standard_fields.InputDataFields.objects_center] = translate_points(
            points=object_inputs[
                standard_fields.InputDataFields.objects_center],
            delta_x=delta_x,
            delta_y=delta_y,
            delta_z=delta_z)


def set_point_instance_ids(mesh_inputs, object_inputs,
                           points_within_box_margin):
  """Set point instance ids."""
  if standard_fields.InputDataFields.objects_center not in object_inputs:
    return

  # Transfer object properties to points
  point_box_ids = box_utils.map_points_to_boxes(
      points=mesh_inputs[standard_fields.InputDataFields.point_positions],
      boxes_length=object_inputs[
          standard_fields.InputDataFields.objects_length],
      boxes_height=object_inputs[
          standard_fields.InputDataFields.objects_height],
      boxes_width=object_inputs[standard_fields.InputDataFields.objects_width],
      boxes_rotation_matrix=object_inputs[
          standard_fields.InputDataFields.objects_rotation_matrix],
      boxes_center=object_inputs[
          standard_fields.InputDataFields.objects_center],
      box_margin=points_within_box_margin)

  # Each point is assigned to an object instance. A point that does not fall
  # inside a valid object is set to 0 (background).
  mesh_inputs[standard_fields.InputDataFields.object_instance_id_points] = (
      point_box_ids + 1)


def remove_objects_by_num_points(mesh_inputs, object_inputs,
                                 min_num_points_in_objects):
  """Removes objects that have less than a certain number of points in them."""
  if standard_fields.InputDataFields.objects_center not in object_inputs:
    return

  point_box_ids = box_utils.map_points_to_boxes(
      points=mesh_inputs[standard_fields.InputDataFields.point_positions],
      boxes_length=object_inputs[
          standard_fields.InputDataFields.objects_length],
      boxes_height=object_inputs[
          standard_fields.InputDataFields.objects_height],
      boxes_width=object_inputs[standard_fields.InputDataFields.objects_width],
      boxes_rotation_matrix=object_inputs[
          standard_fields.InputDataFields.objects_rotation_matrix],
      boxes_center=object_inputs[
          standard_fields.InputDataFields.objects_center],
      box_margin=0.0)
  num_objects = tf.shape(
      object_inputs[standard_fields.InputDataFields.objects_center])[0]
  box_ids_one_hot = tf.one_hot(
      indices=tf.reshape(point_box_ids, [-1]),
      depth=num_objects,
      dtype=tf.int32)
  num_points_in_boxes = tf.reduce_sum(box_ids_one_hot, axis=0)
  valid_object_mask = tf.greater_equal(num_points_in_boxes,
                                       min_num_points_in_objects)
  for key in standard_fields.get_input_object_fields():
    if key in object_inputs:
      object_inputs[key] = tf.boolean_mask(object_inputs[key],
                                           valid_object_mask)


def _random_uniformly_sample_a_seed_point(object_instance_id_points):
  """Randomly samples a seed point and returns its index.

  It randomly samples one of the object instances, and it randomly samples the
  seed point from the points within the randomly sampled object instance.
  Note that the seed point could be sampled from background too.

  Args:
    object_instance_id_points: A tensor of size [num_points] or [num_points, 1]
      containing the object instance id of each point.

  Returns:
    A tf.int32 scalar containing the index of the randomly sampled seed point.
  """
  instance_ids = instance_segmentation_utils.map_labels_to_0_to_n(
      tf.reshape(object_instance_id_points, [-1]))
  indices, _ = (
      instance_segmentation_utils.randomly_select_one_point_per_segment(
          labels=instance_ids))
  return indices[tf.random.uniform([],
                                   minval=0,
                                   maxval=tf.shape(indices)[0],
                                   dtype=tf.int32)]


def _randomly_sample_a_seed_point_by_voxelization(point_locations, voxel_size):
  """Randomly samples a seed point from a random voxel."""
  voxel_size = tf.convert_to_tensor(voxel_size, dtype=tf.float32)
  start_location = tf.reduce_min(point_locations, axis=0)
  voxel_xyz_indices = tf.cast(
      tf.math.floordiv(point_locations - start_location, voxel_size),
      dtype=tf.int32)
  _, pooled_single_number_indices = voxel_utils.compute_pooled_voxel_indices(
      voxel_xyz_indices=voxel_xyz_indices, pooling_size=(1, 1, 1))
  unique_voxel_indices, point_indices = tf.unique(
      tf.reshape(pooled_single_number_indices, [-1]))
  random_voxel_index = tf.random.uniform(
      [], minval=0, maxval=tf.shape(unique_voxel_indices)[0], dtype=tf.int32)
  potential_seed_points = tf.cast(
      tf.reshape(tf.where(tf.equal(point_indices, random_voxel_index)), [-1]),
      dtype=tf.int32)
  random_seed_point_index = tf.random.uniform(
      [], minval=0, maxval=tf.shape(potential_seed_points)[0], dtype=tf.int32)
  return potential_seed_points[random_seed_point_index]


def _get_closest_points_to_random_seed_point(point_locations,
                                             object_instance_id_points,
                                             num_closest_points,
                                             max_distance,
                                             sample_by_voxelization=True,
                                             max_distance_to_voxel_ratio=8.0):
  """Randomly samples a seed point and returns the closest points to it.

  It randomly samples one of the object instances, and it randomly samples the
  seed point from the points within the randomly sampled object instance.

  Args:
    point_locations: A tf.float32 tensor of size [num_points, 3].
    object_instance_id_points: A tf.int32 tensor of size [num_points] or
      [num_points, 1].
    num_closest_points: Number of closest points to seed point to return.
    max_distance: Maximum distance of the points to the seed point. If None, it
      won't be used.
    sample_by_voxelization: If True, samples the seed point by voxelization.
    max_distance_to_voxel_ratio: This ratio is used to compute the voxel size
      used for sampling using voxelization. Contains the ratio between
      max_distance argument and voxel size.

  Returns:
    cropped_point_indices: A tf.int32 tensor of size [N'] containing the indices
      of the closest points to the randomly picked seed point.
    cropped_point_distances: A tf.float32 tensor of size [N'] containing the
      distance of the closest points to the randomly picked seed point.
    all_point_distances: A tf.float32 tensor of size [num_points] containing the
      distance of the all points to the seed point.
  """
  if max_distance is not None and sample_by_voxelization:
    seed_point_index = _randomly_sample_a_seed_point_by_voxelization(
        point_locations=point_locations,
        voxel_size=[
            max_distance / max_distance_to_voxel_ratio,
            max_distance / max_distance_to_voxel_ratio,
            max_distance / max_distance_to_voxel_ratio
        ])
  else:
    seed_point_index = _random_uniformly_sample_a_seed_point(
        object_instance_id_points=object_instance_id_points)
  seed_point_location = point_locations[seed_point_index:seed_point_index +
                                        1, :]
  point_distances = tf.norm(point_locations - seed_point_location, axis=1)
  num_closest_points = tf.minimum(
      tf.shape(point_distances)[0], num_closest_points)
  cropped_point_distances, cropped_point_indices = tf.math.top_k(
      -point_distances, k=num_closest_points, sorted=True)
  cropped_point_distances = -cropped_point_distances
  if max_distance is not None:
    valid_point_mask = tf.less_equal(cropped_point_distances, max_distance)
    cropped_point_indices = tf.boolean_mask(cropped_point_indices,
                                            valid_point_mask)
    cropped_point_distances = tf.boolean_mask(cropped_point_distances,
                                              valid_point_mask)
  return cropped_point_indices, cropped_point_distances, point_distances


def _complete_partial_objects(cropped_point_indices, object_instance_id_points):
  """Completes the objects that are partially in cropped_point_indices.

  Args:
    cropped_point_indices: A tf.int32 tensor of size [N].
    object_instance_id_points: A tf.int32 tensor of size [N'] or [N', 1].

  Returns:
    A tf.int32 tensor of size [N"] containing completed partially cropped
      objects.
  """
  object_instance_id_points = tf.reshape(object_instance_id_points, [-1])
  cropped_instance_ids = tf.gather(object_instance_id_points,
                                   cropped_point_indices)
  unique_cropped_instance_ids, _ = tf.unique(cropped_instance_ids)
  unique_cropped_instance_ids = tf.boolean_mask(
      unique_cropped_instance_ids, tf.greater(unique_cropped_instance_ids, 0))
  num_unique_cropped_ids = tf.shape(unique_cropped_instance_ids)[0]
  complete_instance_id_mask = tf.cast(
      tf.zeros_like(object_instance_id_points), dtype=tf.bool)

  def body_fn(instance_mask, i):
    instance_id = unique_cropped_instance_ids[i]
    instance_mask = tf.logical_or(
        instance_mask, tf.equal(object_instance_id_points, instance_id))
    i += 1
    return instance_mask, i

  cond_fn = lambda instance_mask, i: tf.less(i, num_unique_cropped_ids)

  (complete_instance_id_mask, _) = tf.while_loop(
      cond_fn, body_fn,
      [complete_instance_id_mask,
       tf.constant(0, dtype=tf.int32)])

  complete_instance_id_indices = tf.cast(
      tf.reshape(tf.where(complete_instance_id_mask), [-1]),
      dtype=cropped_point_indices.dtype)
  cropped_point_indices = tf.concat(
      [cropped_point_indices, complete_instance_id_indices], axis=0)
  cropped_point_indices, _ = tf.unique(
      cropped_point_indices, out_idx=cropped_point_indices.dtype)
  return cropped_point_indices


def _add_closest_background_points(cropped_point_indices,
                                   object_instance_id_points, point_distances,
                                   target_num_background_points):
  """Adds the closest background points to the cropped points.

  Args:
    cropped_point_indices: A tf.int32 tensor of size [N].
    object_instance_id_points: A tf.int32 tensor of size [num_points] or
      [num_points, 1].
    point_distances: A tf.float32 tensor of size [num_points].
    target_num_background_points: Number of closest background points to add. If
      None, background points will not be added.

  Returns:
    A tf.int32 tensor of size [N'] containing the cropped point indices
      augmented by closest background point indices.
  """
  if target_num_background_points is not None:
    object_instance_id_points = tf.reshape(object_instance_id_points, [-1])
    num_points = tf.shape(object_instance_id_points)[0]
    background_mask = tf.equal(object_instance_id_points, 0)
    background_point_indices = tf.boolean_mask(
        tf.range(num_points), background_mask)
    background_point_distances = tf.boolean_mask(point_distances,
                                                 background_mask)
    num_closest_points = tf.minimum(target_num_background_points,
                                    tf.shape(background_point_distances)[0])
    _, indices = tf.math.top_k(-background_point_distances, num_closest_points)
    background_point_indices = tf.gather(background_point_indices, indices)
    cropped_point_indices = tf.concat(
        [cropped_point_indices, background_point_indices], axis=0)
    cropped_point_indices, _ = tf.unique(
        cropped_point_indices, out_idx=cropped_point_indices.dtype)
  return cropped_point_indices


def crop_points_around_random_seed_point(mesh_inputs, view_indices_2d_inputs,
                                         num_closest_points, max_distance,
                                         num_background_points):
  """Randomly samples a seed point and crops points closest to that seed point.

  It randomly samples one of the object instances, and it randomly samples the
  seed point from the points within the randomly sampled object instance.

  Args:
    mesh_inputs: A dictionary containing point properties.
    view_indices_2d_inputs: A dictionary containing view indices 2d.
    num_closest_points: Number of points to sample closest to the randomly
      picked seed point.
    max_distance: Maximum distance of the points to the seed point. If None, it
      won't be used.
    num_background_points: Number of closest background points to add. If
      None, background points will not be added.
  """
  cropped_point_indices, _, point_distances = (
      _get_closest_points_to_random_seed_point(
          point_locations=mesh_inputs[
              standard_fields.InputDataFields.point_positions],
          object_instance_id_points=mesh_inputs[
              standard_fields.InputDataFields.object_instance_id_points],
          num_closest_points=num_closest_points,
          max_distance=max_distance))
  cropped_point_indices = _complete_partial_objects(
      cropped_point_indices=cropped_point_indices,
      object_instance_id_points=mesh_inputs[
          standard_fields.InputDataFields.object_instance_id_points])
  cropped_point_indices = _add_closest_background_points(
      cropped_point_indices=cropped_point_indices,
      object_instance_id_points=mesh_inputs[
          standard_fields.InputDataFields.object_instance_id_points],
      point_distances=point_distances,
      target_num_background_points=num_background_points)
  cropped_point_indices = tf.random.shuffle(cropped_point_indices)
  for key in standard_fields.get_input_point_fields():
    if key != standard_fields.InputDataFields.num_valid_points:
      if key in mesh_inputs:
        mesh_inputs[key] = tf.gather(mesh_inputs[key], cropped_point_indices)
  mesh_inputs[standard_fields.InputDataFields.num_valid_points] = tf.shape(
      cropped_point_indices)[0]
  for key in sorted(view_indices_2d_inputs):
    view_indices_2d_inputs[key] = tf.transpose(
        tf.gather(
            tf.transpose(view_indices_2d_inputs[key], [1, 0, 2]),
            cropped_point_indices), [1, 0, 2])


def preprocess_one_image(image, image_preprocess_fn,
                         points_in_image_frame, is_training):
  """Preprocess one image and its corresponding tensors.

  Args:
    image: A tensor of size [height, width, num_channels].
    image_preprocess_fn: Image preprocessing function.
    points_in_image_frame: A tf.int32 tensor of size [num_points, 2] containing
      (y, x) location of points in image.
    is_training: Whether at training stage or not.

  Returns:
    processed_image: The preprocessed image.
    points_in_image_frame: A tf.int32 tensor of size [num_points, 2] containing
      the updated (y, x) location of points in image.
  """
  image_inputs = {}
  image_inputs['image'] = image
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]
  x, y = tf.meshgrid(tf.range(image_width), tf.range(image_height))
  original_meshgrid = tf.stack([y, x], axis=2) + 1
  image_inputs['yx_meshgrid'] = original_meshgrid
  processed_image_inputs = image_preprocess_fn(inputs=image_inputs,
                                               is_training=is_training)
  processed_yx_meshgrid = processed_image_inputs.pop('yx_meshgrid')
  processed_points_in_image_frame = (
      projections.update_pixel_locations_given_deformed_meshgrid(
          pixel_locations=points_in_image_frame,
          original_meshgrid=original_meshgrid,
          deformed_meshgrid=processed_yx_meshgrid))
  return processed_image_inputs['image'], processed_points_in_image_frame


def preprocess_images(view_image_inputs, view_indices_2d_inputs,
                      image_preprocess_fn_dic, is_training):
  """Pre-processes the images."""
  if view_image_inputs is None or image_preprocess_fn_dic is None:
    return

  def preprocess_one_image_fn(i, view_name, get_output_dtypes=False):
    """Processes one image."""
    image_i = view_image_inputs[view_name][i, :, :, :]
    points_in_image_frame_i = view_indices_2d_inputs[view_name][i, :, :]
    image_preprocess_fn = image_preprocess_fn_dic[view_name]
    processed_image_i, processed_points_in_image_frame_i = (
        preprocess_one_image(
            image=image_i,
            image_preprocess_fn=image_preprocess_fn,
            points_in_image_frame=points_in_image_frame_i,
            is_training=is_training))
    if get_output_dtypes:
      return processed_image_i.dtype, processed_points_in_image_frame_i.dtype
    else:
      return processed_image_i, processed_points_in_image_frame_i

  for view_name in sorted(image_preprocess_fn_dic):
    if image_preprocess_fn_dic[view_name] is None:
      continue
    output_dtypes = preprocess_one_image_fn(
        i=0, view_name=view_name, get_output_dtypes=True)
    num_images = tf.shape(view_image_inputs[view_name])[0]
    view_image_inputs[view_name], view_indices_2d_inputs[view_name] = tf.map_fn(
        fn=functools.partial(preprocess_one_image_fn, view_name=view_name),
        elems=tf.range(num_images),
        dtype=output_dtypes)


def make_objects_axis_aligned(object_inputs):
  """Makes objects axis aligned (identity rotation matrix).

  The input boxes could have any rotation. This function will turn the input
  boxes into boxes that are axis aligned such that the axis aligned box contains
  the original box in it.

  Args:
    object_inputs: A dictionary that contains object information.
  """
  box_corners = box_utils.get_box_corners_3d(
      boxes_length=object_inputs[
          standard_fields.InputDataFields.objects_length],
      boxes_height=object_inputs[
          standard_fields.InputDataFields.objects_height],
      boxes_width=object_inputs[standard_fields.InputDataFields.objects_width],
      boxes_center=object_inputs[
          standard_fields.InputDataFields.objects_center],
      boxes_rotation_matrix=object_inputs[
          standard_fields.InputDataFields.objects_rotation_matrix])
  corners_min = tf.reduce_min(box_corners, axis=1)
  corners_max = tf.reduce_max(box_corners, axis=1)
  objects_length, objects_width, objects_height = tf.split(
      corners_max - corners_min, num_or_size_splits=3, axis=1)
  objects_center = (corners_max + corners_min) / 2.0
  objects_rotation_matrix = tf.tile(
      tf.expand_dims(tf.eye(3, dtype=tf.float32), axis=0),
      [tf.shape(corners_min)[0], 1, 1])
  object_inputs[standard_fields.InputDataFields.objects_length] = objects_length
  object_inputs[standard_fields.InputDataFields.objects_height] = objects_height
  object_inputs[standard_fields.InputDataFields.objects_width] = objects_width
  object_inputs[standard_fields.InputDataFields.objects_center] = objects_center
  object_inputs[standard_fields.InputDataFields
                .objects_rotation_matrix] = objects_rotation_matrix


def fit_objects_to_instance_id_points(mesh_inputs,
                                      object_inputs,
                                      epsilon=0.001):
  """Fits objects to points based on their instance ids.

  Args:
    mesh_inputs: A dictionary that contains point properties.
    object_inputs: A dictionary that contains object information.
    epsilon: A very small value.
  """
  if (standard_fields.InputDataFields.object_instance_id_points not in
      mesh_inputs):
    raise ValueError('object_instance_id_points not in mesh_inputs.')
  if standard_fields.InputDataFields.object_class_points not in mesh_inputs:
    raise ValueError('object_class_points not in mesh_inputs.')
  if standard_fields.InputDataFields.point_positions not in mesh_inputs:
    raise ValueError('point_positions not in mesh_inputs.')
  unique_instance_ids, point_instance_idx = tf.unique(
      tf.reshape(
          mesh_inputs[
              standard_fields.InputDataFields.object_instance_id_points], [-1]),
      out_idx=tf.int32)

  unique_instance_ids = tf.cast(unique_instance_ids, dtype=tf.int32)
  num_objects = tf.shape(unique_instance_ids)[0]

  def get_object_i(i):
    """Function that fits a box to each set of instance points."""
    object_instance_id_i = unique_instance_ids[i]
    points_i_mask = tf.equal(point_instance_idx, i)
    point_positions_i = tf.boolean_mask(
        mesh_inputs[standard_fields.InputDataFields.point_positions],
        points_i_mask)
    point_classes_i = tf.boolean_mask(
        mesh_inputs[standard_fields.InputDataFields.object_class_points],
        points_i_mask)
    min_object_xyz_i = tf.reduce_min(point_positions_i, axis=0)
    max_object_xyz_i = tf.reduce_max(point_positions_i, axis=0)
    object_center_i = (min_object_xyz_i + max_object_xyz_i) / 2.0
    object_size_i = max_object_xyz_i - min_object_xyz_i
    object_length_i = tf.expand_dims(object_size_i[0], axis=0)
    object_width_i = tf.expand_dims(object_size_i[1], axis=0)
    object_height_i = tf.expand_dims(object_size_i[2], axis=0)
    object_class_i = point_classes_i[0, :]
    return (object_center_i, object_length_i, object_width_i, object_height_i,
            object_class_i, object_instance_id_i)

  (object_inputs[standard_fields.InputDataFields.objects_center],
   object_inputs[standard_fields.InputDataFields.objects_length],
   object_inputs[standard_fields.InputDataFields.objects_width],
   object_inputs[standard_fields.InputDataFields.objects_height],
   object_inputs[standard_fields.InputDataFields.objects_class],
   unique_instance_ids) = tf.map_fn(
       fn=get_object_i,
       elems=tf.range(num_objects),
       dtype=(tf.float32, tf.float32, tf.float32, tf.float32, mesh_inputs[
           standard_fields.InputDataFields.object_class_points].dtype,
              unique_instance_ids.dtype))
  object_inputs[standard_fields.InputDataFields.objects_length] = tf.maximum(
      object_inputs[standard_fields.InputDataFields.objects_length], epsilon)
  object_inputs[standard_fields.InputDataFields.objects_height] = tf.maximum(
      object_inputs[standard_fields.InputDataFields.objects_height], epsilon)
  object_inputs[standard_fields.InputDataFields.objects_width] = tf.maximum(
      object_inputs[standard_fields.InputDataFields.objects_width], epsilon)
  object_inputs[
      standard_fields.InputDataFields.objects_rotation_matrix] = tf.tile(
          tf.expand_dims(tf.eye(3, dtype=tf.float32), axis=0),
          [num_objects, 1, 1])

  foreground_object_mask = tf.greater(
      tf.reshape(object_inputs[standard_fields.InputDataFields.objects_class],
                 [-1]), 0)
  for key in object_inputs:
    object_inputs[key] = tf.boolean_mask(object_inputs[key],
                                         foreground_object_mask)
  unique_instance_ids = tf.boolean_mask(unique_instance_ids,
                                        foreground_object_mask)

  def add_instance_id_i(instance_id_points_i, i):
    """Adds the mapped values for the instance id i."""
    current_instance_id = unique_instance_ids[i]
    instance_id_addition_i = tf.cast(
        tf.equal(
            mesh_inputs[standard_fields.InputDataFields
                        .object_instance_id_points], current_instance_id),
        dtype=tf.int32) * (
            i + 1)
    instance_id_points_i = tf.maximum(instance_id_points_i,
                                      instance_id_addition_i)
    i += 1
    return instance_id_points_i, i

  num_objects = tf.shape(unique_instance_ids)[0]
  cond = lambda instance_id_points_i, i: tf.less(i, num_objects)
  init_inputs = [
      tf.zeros_like(mesh_inputs[
          standard_fields.InputDataFields.object_instance_id_points]),
      tf.constant(0, dtype=tf.int32)
  ]
  mesh_inputs[standard_fields.InputDataFields
              .object_instance_id_points], _ = tf.while_loop(
                  cond=cond, body=add_instance_id_i, loop_vars=init_inputs)


def add_point_offsets(inputs, voxel_grid_cell_size):
  """Adds points offsets features to the inputs.

  Args:
    inputs: A dictionary containing input tensors.
    voxel_grid_cell_size: A three dimensional tuple determining the voxel grid
      size.
  """
  if standard_fields.InputDataFields.point_offsets not in inputs:
    inputs[standard_fields.InputDataFields.point_offsets] = tf.squeeze(
        voxel_utils.points_offset_in_voxels(
            points=tf.expand_dims(
                inputs[standard_fields.InputDataFields.point_positions],
                axis=0),
            grid_cell_size=voxel_grid_cell_size),
        axis=0)


def add_point_offset_bins(inputs, voxel_grid_cell_size, num_bins_x, num_bins_y,
                          num_bins_z):
  """Adds points offsets features to the inputs.

  Args:
    inputs: A dictionary containing input tensors.
    voxel_grid_cell_size: A three dimensional tuple determining the voxel grid
      size.
    num_bins_x: Number of bins in x dimension.
    num_bins_y: Number of bins in y dimension.
    num_bins_z: Number of bins in z dimension.
  """
  add_point_offsets(inputs=inputs, voxel_grid_cell_size=voxel_grid_cell_size)
  num_bins_coef_float = tf.convert_to_tensor(
      [[num_bins_x, num_bins_y, num_bins_z]], dtype=tf.float32)
  num_bins_coef_int = tf.convert_to_tensor(
      [[num_bins_x, num_bins_y, num_bins_z]], dtype=tf.int32)
  scaled_offsets = (inputs[standard_fields.InputDataFields.point_offsets] +
                    0.5) * num_bins_coef_float
  scaled_offsets = tf.minimum(
      tf.maximum(tf.cast(scaled_offsets, dtype=tf.int32), 0), num_bins_coef_int)
  scaled_offsets = (
      scaled_offsets[:, 0] + scaled_offsets[:, 1] * num_bins_x +
      scaled_offsets[:, 2] * num_bins_x * num_bins_y)
  inputs[standard_fields.InputDataFields.point_offset_bins] = tf.one_hot(
      indices=scaled_offsets,
      depth=(num_bins_x * num_bins_y * num_bins_z),
      dtype=tf.float32)


def voxelize_point_features(inputs, voxels_pad_or_clip_size,
                            voxel_grid_cell_size, point_feature_keys,
                            point_to_voxel_segment_func,
                            num_frame_to_load=1):
  """Voxelizes the point features.

  Args:
    inputs: A dictionary containing input tensors.
    voxels_pad_or_clip_size: Number of target voxels to pad or clip to. If None,
      it will not perform the voxel padding.
    voxel_grid_cell_size: A three dimensional tuple determining the voxel grid
      size.
    point_feature_keys: The keys used to form the voxel features.
    point_to_voxel_segment_func: The function used to aggregate the features of
      the points that fall in the same voxel.
    num_frame_to_load: If greater than 1, offset multiframe point cloud point
      features according to point frame index.
  """
  if len(point_feature_keys) == 1:
    point_features = tf.cast(inputs[point_feature_keys[0]], dtype=tf.float32)
  else:
    point_features = tf.concat(
        [tf.cast(inputs[key], dtype=tf.float32) for key in point_feature_keys],
        axis=1)
  if num_frame_to_load > 1:
    # Offset feature from different frames to non-overlapping dimensions
    offset_feature_list = []
    for i in range(num_frame_to_load):
      offset_feature_list.append(
          tf.where_v2(
              tf.math.equal(
                  inputs[standard_fields.InputDataFields.point_frame_index], i),
              point_features, 0))
    point_features = tf.concat(offset_feature_list, axis=1)

  (voxel_features, voxel_xyz_indices, num_valid_voxels, points_to_voxel_mapping,
   voxel_start_locations) = voxel_utils.pointcloud_to_sparse_voxel_grid(
       points=tf.expand_dims(
           inputs[standard_fields.InputDataFields.point_positions], axis=0),
       features=tf.expand_dims(point_features, axis=0),
       num_valid_points=tf.expand_dims(
           inputs[standard_fields.InputDataFields.num_valid_points], axis=0),
       grid_cell_size=voxel_grid_cell_size,
       voxels_pad_or_clip_size=voxels_pad_or_clip_size,
       segment_func=point_to_voxel_segment_func)
  inputs[standard_fields.InputDataFields.voxel_features] = tf.squeeze(
      voxel_features, axis=0)
  inputs[standard_fields.InputDataFields.voxel_xyz_indices] = tf.squeeze(
      voxel_xyz_indices, axis=0)
  inputs[standard_fields.InputDataFields.num_valid_voxels] = tf.squeeze(
      num_valid_voxels, axis=0)
  inputs[standard_fields.InputDataFields.points_to_voxel_mapping] = tf.squeeze(
      points_to_voxel_mapping, axis=0)
  inputs[standard_fields.InputDataFields.voxel_start_locations] = tf.squeeze(
      voxel_start_locations, axis=0)
  inputs[standard_fields.InputDataFields.voxel_positions] = (tf.cast(
      inputs[standard_fields.InputDataFields.voxel_xyz_indices],
      dtype=tf.float32) + 0.5) * tf.expand_dims(
          tf.convert_to_tensor(voxel_grid_cell_size, dtype=tf.float32),
          axis=0) + tf.expand_dims(
              inputs[standard_fields.InputDataFields.voxel_start_locations],
              axis=0)


def voxelize_property_tensor(inputs, point_tensor_key,
                             corresponding_voxel_tensor_key,
                             voxels_pad_or_clip_size, voxel_grid_cell_size,
                             segment_func=tf.math.unsorted_segment_mean):
  """Voxelizes a property tensor.

  Args:
    inputs: A dictionary containing input tensors.
    point_tensor_key: A string identifying the label tensor.
    corresponding_voxel_tensor_key: A string identifying the corresponding voxel
      tensor key.
    voxels_pad_or_clip_size: Number of target voxels to pad or clip to. If None,
      it will not perform the voxel padding.
    voxel_grid_cell_size: A three dimensional tuple determining the voxel grid
      size.
    segment_func: A tensorflow function that operates on segments. Examples are
      one of tf.math.unsorted_segment_{min/max/mean/prod/sum}.
  """
  if point_tensor_key in inputs:
    (voxel_tensor, _, _, _, _) = voxel_utils.pointcloud_to_sparse_voxel_grid(
        points=tf.expand_dims(
            inputs[standard_fields.InputDataFields.point_positions], axis=0),
        features=tf.expand_dims(inputs[point_tensor_key], axis=0),
        num_valid_points=tf.expand_dims(
            inputs[standard_fields.InputDataFields.num_valid_points], axis=0),
        grid_cell_size=voxel_grid_cell_size,
        voxels_pad_or_clip_size=voxels_pad_or_clip_size,
        segment_func=segment_func)
    inputs[corresponding_voxel_tensor_key] = tf.squeeze(voxel_tensor, axis=0)


def voxelize_label_tensor(inputs, point_tensor_key,
                          corresponding_voxel_tensor_key,
                          voxels_pad_or_clip_size, voxel_grid_cell_size):
  """Voxelizes a label tensor.

  Args:
    inputs: A dictionary containing input tensors.
    point_tensor_key: A string identifying the label tensor.
    corresponding_voxel_tensor_key: A string identifying the corresponding voxel
      tensor key.
    voxels_pad_or_clip_size: Number of target voxels to pad or clip to. If None,
      it will not perform the voxel padding.
    voxel_grid_cell_size: A three dimensional tuple determining the voxel grid
      size.
  """
  if point_tensor_key in inputs:
    label_tensor = tf.reshape(inputs[point_tensor_key], [-1])
    max_label = tf.reduce_max(label_tensor)
    label_one_hot = tf.one_hot(
        label_tensor, depth=(max_label + 1), dtype=tf.float32)
    (voxel_label_one_hot, _, _, _,
     _) = voxel_utils.pointcloud_to_sparse_voxel_grid(
         points=tf.expand_dims(
             inputs[standard_fields.InputDataFields.point_positions], axis=0),
         features=tf.expand_dims(label_one_hot, axis=0),
         num_valid_points=tf.expand_dims(
             inputs[standard_fields.InputDataFields.num_valid_points], axis=0),
         grid_cell_size=voxel_grid_cell_size,
         voxels_pad_or_clip_size=voxels_pad_or_clip_size,
         segment_func=tf.math.unsorted_segment_mean)
    inputs[corresponding_voxel_tensor_key] = tf.squeeze(
        tf.expand_dims(
            tf.math.argmax(voxel_label_one_hot, axis=2, output_type=tf.int32),
            axis=2),
        axis=0)


def voxelize_point_to_view_correspondences(inputs, view_indices_2d_inputs,
                                           voxels_pad_or_clip_size,
                                           voxel_grid_cell_size):
  """Voxelizes the point to image correspondences.

  Args:
    inputs: A dictionary containing input tensors.
    view_indices_2d_inputs: Dictionary containing the name of the 2d views.
    voxels_pad_or_clip_size: Number of target voxels to pad or clip to. If None,
      it will not perform the voxel padding.
    voxel_grid_cell_size: A three dimensional tuple determining the voxel grid
      size.
  """
  for key in sorted(view_indices_2d_inputs):
    voxel_correspondence, _, _, _, _ = (
        voxel_utils.pointcloud_to_sparse_voxel_grid(
            points=tf.expand_dims(
                inputs[standard_fields.InputDataFields.point_positions],
                axis=0),
            features=tf.cast(view_indices_2d_inputs[key], dtype=tf.float32),
            num_valid_points=tf.expand_dims(
                inputs[standard_fields.InputDataFields.num_valid_points],
                axis=0),
            grid_cell_size=voxel_grid_cell_size,
            voxels_pad_or_clip_size=voxels_pad_or_clip_size,
            segment_func=tf.math.unsorted_segment_mean))
    inputs[('%s/voxel_indices_2d' % key)] = tf.cast(
        voxel_correspondence, dtype=tf.int32)


def voxelize_semantic_labels(inputs, voxels_pad_or_clip_size,
                             voxel_grid_cell_size):
  """Voxelizes the semantic labels.

  Args:
    inputs: A dictionary containing input tensors.
    voxels_pad_or_clip_size: Number of target voxels to pad or clip to. If None,
      it will not perform the voxel padding.
    voxel_grid_cell_size: A three dimensional tuple determining the voxel grid
      size.
  """
  voxelize_label_tensor(
      inputs=inputs,
      point_tensor_key=standard_fields.InputDataFields.object_class_points,
      corresponding_voxel_tensor_key=standard_fields.InputDataFields
      .object_class_voxels,
      voxels_pad_or_clip_size=voxels_pad_or_clip_size,
      voxel_grid_cell_size=voxel_grid_cell_size)


def voxelize_instance_labels(inputs, voxels_pad_or_clip_size,
                             voxel_grid_cell_size):
  """Voxelizes the instance labels.

  Args:
    inputs: A dictionary containing input tensors.
    voxels_pad_or_clip_size: Number of target voxels to pad or clip to. If None,
      it will not perform the voxel padding.
    voxel_grid_cell_size: A three dimensional tuple determining the voxel grid
      size.
  """
  voxelize_label_tensor(
      inputs=inputs,
      point_tensor_key=standard_fields.InputDataFields
      .object_instance_id_points,
      corresponding_voxel_tensor_key=standard_fields.InputDataFields
      .object_instance_id_voxels,
      voxels_pad_or_clip_size=voxels_pad_or_clip_size,
      voxel_grid_cell_size=voxel_grid_cell_size)


def voxelize_object_properties(inputs, voxels_pad_or_clip_size,
                               voxel_grid_cell_size):
  """Voxelizes the object properties.

  Args:
    inputs: A dictionary containing input tensors.
    voxels_pad_or_clip_size: Number of target voxels to pad or clip to. If None,
      it will not perform the voxel padding.
    voxel_grid_cell_size: A three dimensional tuple determining the voxel grid
      size.
  """
  point_to_voxel_field_mapping = (
      standard_fields.get_input_point_to_voxel_field_mapping())
  for key in point_to_voxel_field_mapping:
    if key in inputs and inputs[key].dtype == tf.float32:
      voxelize_property_tensor(
          inputs=inputs,
          point_tensor_key=key,
          corresponding_voxel_tensor_key=point_to_voxel_field_mapping[key],
          voxels_pad_or_clip_size=voxels_pad_or_clip_size,
          voxel_grid_cell_size=voxel_grid_cell_size)


def remove_pointcloud_noise(mesh_inputs, view_indices_2d_inputs,
                            voxel_grid_cell_size, voxel_density_threshold):
  """Removes the noise (points that are low density).

  Args:
    mesh_inputs: A dictionary containing point properties.
    view_indices_2d_inputs: A dictionary containing view indices 2d.
    voxel_grid_cell_size: A three dimensional tuple determining the voxel grid
      size.
    voxel_density_threshold: Voxel density threshold. Points that belong to a
      voxel with less density than this threshold are removed.
  """
  points_position = mesh_inputs[standard_fields.InputDataFields.point_positions]
  num_points = tf.shape(points_position)[0]
  points_density = tf.ones([num_points, 1], dtype=tf.float32)
  if standard_fields.InputDataFields.num_valid_points in mesh_inputs:
    num_valid_points = mesh_inputs[
        standard_fields.InputDataFields.num_valid_points]
  else:
    num_valid_points = num_points
  (voxels_density, _, _, segment_ids, _) = (
      voxel_utils.pointcloud_to_sparse_voxel_grid(
          points=tf.expand_dims(points_position, axis=0),
          features=tf.expand_dims(points_density, axis=0),
          num_valid_points=tf.expand_dims(num_valid_points, axis=0),
          grid_cell_size=voxel_grid_cell_size,
          voxels_pad_or_clip_size=None,
          segment_func=tf.math.unsorted_segment_sum))
  voxels_density = tf.cast(tf.squeeze(voxels_density, axis=0), dtype=tf.int32)
  segment_ids = tf.squeeze(segment_ids, axis=0)
  points_density = tf.gather(voxels_density, segment_ids)
  valid_points = tf.greater_equal(
      tf.reshape(points_density, [-1]), voxel_density_threshold)
  for key in standard_fields.get_input_point_fields():
    if key != standard_fields.InputDataFields.num_valid_points:
      if key in mesh_inputs:
        mesh_inputs[key] = tf.boolean_mask(mesh_inputs[key], valid_points)
  mesh_inputs[
      standard_fields.InputDataFields.num_valid_points] = tf.math.reduce_sum(
          tf.cast(valid_points, dtype=tf.int32))
  for key in sorted(view_indices_2d_inputs):
    view_indices_2d_inputs[key] = tf.transpose(
        tf.boolean_mask(
            tf.transpose(view_indices_2d_inputs[key], [1, 0, 2]), valid_points),
        [1, 0, 2])
