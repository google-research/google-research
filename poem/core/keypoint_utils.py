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

"""Keypoint utility functions."""

import math

import tensorflow as tf

from poem.core import data_utils
from poem.core import distance_utils


def get_points(points, indices):
  """Gets points as the centers of points at specified indices.

  Args:
    points: A tensor for points. Shape = [..., num_points, point_dim].
    indices: A list of integers for point indices.

  Returns:
    A tensor for (center) points. Shape = [..., 1, point_dim].

  Raises:
    ValueError: If `indices` is empty.
  """
  if not indices:
    raise ValueError('`Indices` must be non-empty.')
  points = tf.gather(points, indices=indices, axis=-2)
  if len(indices) == 1:
    return points
  return tf.math.reduce_mean(points, axis=-2, keepdims=True)


def swap_x_y(points):
  """Swaps the order of the first two dimension (x and y) coordinate.

  Args:
    points: A tensor for points. Shape = [..., point_dim].

  Returns:
    A tensor for points with swapped x and y.

  Raises:
    ValueError: If point dimension is less than 2.
  """
  point_dim = points.shape.as_list()[-1]
  if point_dim < 2:
    raise ValueError('Point dimension must be greater than 2: %d.' % point_dim)
  perm_indices = [1, 0] + list(range(2, point_dim))
  return tf.gather(points, indices=perm_indices, axis=-1)


def override_points(points, from_indices_list, to_indices):
  """Overrides points with other points.

  Points at `to_indices` will be overridden with centers of points from
  `from_indices_list`.

  For example:

    from_indices_list = [[0, 1], [2]]
    to_indices = [3, 4]
    updated_points = override_points(from_indices_list, to_indices)

  Will result in:
    updated_points[..., 3, :] ==
      ((points[..., 0, :] + points[..., 1, :]) / 2 + points[..., 2, :]) / 2
    updated_points[..., 4, :] ==
      ((points[..., 0, :] + points[..., 1, :]) / 2 + points[..., 2, :]) / 2

  Args:
    points: A tensor for points to override. Shape = [..., num_points,
      point_dim].
    from_indices_list: A list of integer lists for point indices to compute
      overriding points.
    to_indices: A list of integers for point indices to be overridden.

  Returns:
    A tensor for updated points.
  """
  overriding_points = [
      get_points(points, from_indices) for from_indices in from_indices_list
  ]
  overriding_points = tf.concat(overriding_points, axis=-2)
  overriding_points = tf.math.reduce_mean(
      overriding_points, axis=-2, keepdims=True)
  overriding_points = data_utils.tile_last_dims(
      overriding_points, last_dim_multiples=[len(to_indices), 1])
  return data_utils.update_sub_tensor(
      points,
      indices=to_indices,
      axis=-2,
      update_func=lambda _: overriding_points)


def naive_normalize_points(points, point_masks):
  """Naively normalizes points by shifting and scaling.

  Args:
    points: A tensor for points. Shape = [..., num_points, point_dim].
    point_masks: A tensor for point validities. Shape = [..., num_points].

  Returns:
    points: A tensor for normalized points. Shape = [..., num_points,
      point_dim].
  """
  point_masks = tf.cast(point_masks, dtype=tf.bool)
  point_dim = tf.shape(points)[-1]

  def compute_centers(points, point_masks):
    """Computes centers of valid points."""
    valid_points = tf.boolean_mask(points, point_masks)
    return tf.math.reduce_mean(valid_points, axis=-2, keepdims=True)

  def compute_max_spans(points, point_masks):
    """Computes maximum point set spans in any direction."""
    num_points = tf.shape(points)[-2]
    point_masks = data_utils.tile_last_dims(
        tf.expand_dims(tf.expand_dims(point_masks, axis=-1), axis=-1),
        last_dim_multiples=[num_points, point_dim])
    diffs = tf.math.abs(
        data_utils.tile_last_dims(
            tf.expand_dims(points, axis=-2), last_dim_multiples=[num_points, 1])
        - data_utils.tile_last_dims(
            tf.expand_dims(points, axis=-3),
            last_dim_multiples=[num_points, 1, 1]))
    diffs = tf.where(point_masks, diffs, tf.zeros_like(diffs))
    max_spans = tf.squeeze(
        tf.math.reduce_max(diffs, axis=[-3, -2, -1], keepdims=True), axis=[-2])
    return max_spans

  centers = compute_centers(points, point_masks)
  max_spans = compute_max_spans(points, point_masks)
  points = (points - centers) / tf.math.maximum(1e-12, max_spans)
  points = tf.where(
      data_utils.tile_last_dims(
          tf.expand_dims(point_masks, axis=-1), last_dim_multiples=[point_dim]),
      points, tf.zeros_like(points))
  return points


def normalize_points(points, offset_point_indices,
                     scale_distance_point_index_pairs,
                     scale_distance_reduction_fn, scale_unit):
  """Normalizes points by shifting and scaling.

  Args:
    points: A tensor for points. Shape = [..., num_points, point_dim].
    offset_point_indices: A list of integers for the indices of the center
      points. If a single index is specified, its corresponding points will be
      used as centers. If multiple indices are specified, the centers of their
      corresponding points will be used as centers.
    scale_distance_point_index_pairs: A list of integer list pairs for the point
      index pairs to compute scale distances to be used as unit distances. For
      example, use [([0], [1]), ([2, 3], [4])] to compute scale distance as the
      reduction of the distance between point_0 and point_1 and the distance
      between the center(point_2, point_3) and point_4.
    scale_distance_reduction_fn: A Tensorflow reduction function handle for
      distance, e.g., tf.math.reduce_sum.
    scale_unit: A scalar for the scale unit whose value the scale distance will
      be scaled to.

  Returns:
    normalized_points: A tensor for normalized points. Shape = [..., num_points,
      point_dim].
    offset_points: A tensor for offset points used for normalization. Shape =
      [..., 1, point_dim].
    scale_distances: A tensor for scale distances used for normalization. Shape
      = [..., 1, 1].
  """
  offset_points = get_points(points, offset_point_indices)

  def compute_scale_distances():
    sub_scale_distances_list = []
    for lhs_indices, rhs_indices in scale_distance_point_index_pairs:
      lhs_points = get_points(points, lhs_indices)
      rhs_points = get_points(points, rhs_indices)
      sub_scale_distances_list.append(
          distance_utils.compute_l2_distances(
              lhs_points, rhs_points, keepdims=True))
    sub_scale_distances = tf.concat(sub_scale_distances_list, axis=-1)
    return scale_distance_reduction_fn(
        sub_scale_distances, axis=-1, keepdims=True)

  scale_distances = tf.math.maximum(1e-12, compute_scale_distances())
  normalized_points = (points - offset_points) / scale_distances * scale_unit
  return normalized_points, offset_points, scale_distances


def centralize_masked_points(points, point_masks):
  """Sets masked out points to the centers of rest of the points.

  Args:
    points: A tensor for points. Shape = [..., num_points, point_dim].
    point_masks: A tensor for the masks. Shape = [..., num_points].

  Returns:
    A tensor for points with masked out points centralized.
  """
  point_masks = tf.expand_dims(point_masks, axis=-1)
  kept_centers = data_utils.reduce_weighted_mean(
      points, weights=point_masks, axis=-2, keepdims=True)
  return tf.where(tf.cast(point_masks, dtype=tf.bool), points, kept_centers)


def standardize_points(points):
  """Standardizes points by centering and scaling.

  Args:
    points: A tensor for input points. Shape = [..., num_points, point_dim].

  Returns:
    points: A tensor for standardized points. Shape = [..., num_points,
      point_dim].
    offsets: A tensor for the applied offsets. Shape = [..., 1, point_dim].
    scales: A tensor for the applied inverse scales. Shape = [..., 1,
      point_dim].
  """
  offsets = tf.math.reduce_mean(points, axis=-2, keepdims=True)
  points -= offsets
  scales = tf.sqrt(tf.math.reduce_sum(points**2, axis=[-2, -1], keepdims=True))
  points /= scales
  return points, offsets, scales


def compute_procrustes_alignment_params(target_points,
                                        source_points,
                                        point_masks=None):
  """Computes Procrustes alignment parameters.

  Args:
    target_points: A tensor for target points. Shape = [..., num_points,
      point_dim].
    source_points: A tensor for source points. Shape = [..., num_points,
      point_dim].
    point_masks: A tensor for the masks. Shape = [..., num_points]. Ignored if
      None.

  Returns:
    rotations: A tensor for rotations. Shape = [..., point_dim, point_dim].
    scales: A tensor for scales. Shape = [..., 1, 1].
    translations: A tensor for translations. Shape = [..., 1, point_dim].
  """
  if point_masks is not None:
    target_points = centralize_masked_points(target_points, point_masks)
    source_points = centralize_masked_points(source_points, point_masks)

  # standardized_target_points: Shape = [..., num_points, point_dim].
  # target_offsets: Shape = [..., 1, point_dim].
  # target_scales: Shape = [..., 1, 1].
  standardized_target_points, target_offsets, target_scales = (
      standardize_points(target_points))
  # standardized_source_points: Shape = [..., num_points, point_dim].
  # source_offsets: Shape = [..., 1, point_dim].
  # source_scales: Shape = [..., 1, point_dim].
  standardized_source_points, source_offsets, source_scales = (
      standardize_points(source_points))
  # Shape = [..., point_dim, point_dim].
  a = tf.linalg.matmul(
      standardized_target_points, standardized_source_points, transpose_a=True)
  # s: Shape = [..., point_dim].
  # u: Shape = [..., point_dim, point_dim].
  # v: Shape = [..., point_dim, point_dim].
  s, u, v = tf.linalg.svd(a)
  # Shape = [..., point_dim, point_dim].
  r = tf.linalg.matmul(v, u, transpose_b=True)
  # Shape = [...].
  det_r = tf.linalg.det(r)
  # Shape = [...].
  signs = tf.math.sign(det_r)
  # Shape = [..., 1].
  signs = tf.expand_dims(signs, axis=-1)
  # Shape = [..., point_dim - 1].
  point_dim = target_points.shape.as_list()[-1]
  ones = data_utils.tile_last_dims(
      tf.ones_like(signs), last_dim_multiples=[point_dim - 1])
  # Shape = [..., point_dim].
  signs = tf.concat([ones, signs], axis=-1)
  s *= signs
  # Shape = [..., 1, point_dim].
  signs = tf.expand_dims(signs, axis=-2)
  v *= signs
  # Shape = [..., point_dim, point_dim].
  rotations = tf.linalg.matmul(v, u, transpose_b=True)
  # Shape = [..., 1, 1].
  scales = (
      tf.expand_dims(tf.math.reduce_sum(s, axis=-1, keepdims=True), axis=-1) *
      target_scales / source_scales)
  # Shape = [..., 1, point_dim].
  translations = target_offsets - scales * tf.linalg.matmul(
      source_offsets, rotations)
  return rotations, scales, translations


def procrustes_align_points(target_points, source_points, point_masks=None):
  """Performs Procrustes alignment on source points to target points.

  Args:
    target_points: A tensor for target points. Shape = [..., num_points,
      point_dim].
    source_points: A tensor for source points. Shape = [..., num_points,
      point_dim].
    point_masks: A tensor for the masks. Shape = [..., num_points]. Ignored if
      None.

  Returns:
    A tensor for aligned source points. Shape = [..., num_points, point_dim].
  """
  rotations, scales, translations = compute_procrustes_alignment_params(
      target_points, source_points, point_masks=point_masks)
  return translations + scales * tf.linalg.matmul(source_points, rotations)


def compute_mpjpes(lhs_points, rhs_points, point_masks=None):
  """Computes the Mean Per-Joint Position Errors (MPJPEs).

  If `point_masks` is specified, computes MPJPEs weighted by `point_masks`.

  Args:
    lhs_points: A tensor for the LHS points. Shape = [..., num_points,
      point_dim].
    rhs_points: A tensor for the RHS points. Shape = [..., num_points,
      point_dim].
    point_masks: A tensor for the masks. Shape = [..., num_points]. Ignored if
      None.

  Returns:
    A tensor for MPJPEs. Shape = [...].
  """
  distances = distance_utils.compute_l2_distances(
      lhs_points, rhs_points, keepdims=False)
  return data_utils.reduce_weighted_mean(
      distances, weights=point_masks, axis=-1)


def compute_procrustes_aligned_mpjpes(target_points,
                                      source_points,
                                      point_masks=None):
  """Computes MPJPEs after Procrustes alignment.

  Args:
    target_points: A tensor for target points. Shape = [..., num_points,
      point_dim].
    source_points: A tensor for source points. Shape = [..., num_points,
      point_dim].
    point_masks: A tensor for the masks. Shape = [..., num_points]. Ignored if
      None.

  Returns:
    A tensor for MPJPEs. Shape = [...].
  """
  aligned_source_points = procrustes_align_points(
      target_points, source_points, point_masks=point_masks)
  return compute_mpjpes(
      aligned_source_points, target_points, point_masks=point_masks)


def normalize_points_by_image_size(points, image_sizes):
  """Normalizes point coordinates by image sizes.

  Args:
    points: A tensor for normalized points by image size. Shape = [...,
      num_points, point_dim].
    image_sizes: A tensor for image sizes. Shape = [..., point_dim].

  Returns:
    A tensor for denormalized points. Shape = [..., num_points, point_dim].
  """
  if len(image_sizes.shape.as_list()) != len(points.shape.as_list()) - 1:
    raise ValueError(
        'Rank of `image_size` must be that of `points` minus 1: %d vs. %d.' %
        (len(image_sizes.shape.as_list()), len(points.shape.as_list())))
  return points / tf.expand_dims(
      tf.cast(image_sizes, dtype=points.dtype), axis=-2)


def denormalize_points_by_image_size(points, image_sizes):
  """Denormalizes point coordinates by image sizes.

  Args:
    points: A tensor for normalized points by image size. Shape = [...,
      num_points, point_dim].
    image_sizes: A tensor for image sizes. Shape = [..., point_dim].

  Returns:
    A tensor for denormalized points. Shape = [..., num_points, point_dim].
  """
  if len(image_sizes.shape.as_list()) != len(points.shape.as_list()) - 1:
    raise ValueError(
        'Rank of `image_size` must be that of `points` minus 1: %d vs. %d.' %
        (len(image_sizes.shape.as_list()), len(points.shape.as_list())))
  return points * tf.expand_dims(
      tf.cast(image_sizes, dtype=points.dtype), axis=-2)


def create_rotation_matrices_3d(azimuths, elevations, rolls):
  """Creates rotation matrices given rotation angles.

  Note that the created rotations are to be applied on points with layout (y, x,
  z).

  Args:
    azimuths: A tensor for azimuths angles. Shape = [...].
    elevations: A tensor for elevation angles. Shape = [...].
    rolls: A tensor for roll angles. Shape = [...].

  Returns:
    A tensor for rotation matrices. Shape = [..., 3, 3].
  """
  azi_cos = tf.math.cos(azimuths)
  azi_sin = tf.math.sin(azimuths)
  ele_cos = tf.math.cos(elevations)
  ele_sin = tf.math.sin(elevations)
  rol_cos = tf.math.cos(rolls)
  rol_sin = tf.math.sin(rolls)
  rotations_00 = azi_cos * ele_cos
  rotations_01 = azi_cos * ele_sin * rol_sin - azi_sin * rol_cos
  rotations_02 = azi_cos * ele_sin * rol_cos + azi_sin * rol_sin
  rotations_10 = azi_sin * ele_cos
  rotations_11 = azi_sin * ele_sin * rol_sin + azi_cos * rol_cos
  rotations_12 = azi_sin * ele_sin * rol_cos - azi_cos * rol_sin
  rotations_20 = -ele_sin
  rotations_21 = ele_cos * rol_sin
  rotations_22 = ele_cos * rol_cos
  rotations_0 = tf.stack([rotations_00, rotations_10, rotations_20], axis=-1)
  rotations_1 = tf.stack([rotations_01, rotations_11, rotations_21], axis=-1)
  rotations_2 = tf.stack([rotations_02, rotations_12, rotations_22], axis=-1)
  return tf.stack([rotations_0, rotations_1, rotations_2], axis=-1)


def rotate_points(rotation_matrices, points):
  """Applies rotation matrices to points.

  Assumes input points are centralized.

  Args:
    rotation_matrices: A tensor for rotation matrices. Shape = [..., point_dim,
      point_dim].
    points: A tensor for points to rotate. Shape = [..., point_dim].

  Returns:
   A tensor for rotated points. Shape = [..., point_dim].
  """
  operator = tf.linalg.LinearOperatorFullMatrix(rotation_matrices)
  rotated_points = operator.matvec(points)
  return rotated_points


def random_rotate_and_project_3d_to_2d(keypoints_3d,
                                       azimuth_range=(-math.pi, math.pi),
                                       elevation_range=(-math.pi / 6.0,
                                                        math.pi / 6.0),
                                       roll_range=(0.0, 0.0),
                                       default_camera=True,
                                       default_camera_z=2.0,
                                       seed=None):
  """Randomly rotates and projects 3D keypoints to 2D.

  Args:
    keypoints_3d: A tensor for 3D keypoints. Shape = [..., num_keypoints, 3].
    azimuth_range: A tuple for minimum and maximum azimuth angles to randomly
      rotate 3D keypoints with.
    elevation_range: A tuple for minimum and maximum elevation angles to
      randomly rotate 3D keypoints with.
    roll_range: A tuple for minimum and maximum roll angles to randomly rotate
      3D keypoints with.
    default_camera: Whether we want to transform to default camera view.
    default_camera_z: A float for depth of default camera position.
    seed: An integer for random seed.

  Returns:
    keypoints_2d: A tensor for projected 2D keypoints from randomly rotated 3D
      keypoints.
  """
  azimuths = tf.random.uniform(
      tf.shape(keypoints_3d)[:-2],
      minval=azimuth_range[0],
      maxval=azimuth_range[1],
      seed=seed)
  elevations = tf.random.uniform(
      tf.shape(keypoints_3d)[:-2],
      minval=elevation_range[0],
      maxval=elevation_range[1],
      seed=seed)
  rolls = tf.random.uniform(
      tf.shape(keypoints_3d)[:-2],
      minval=roll_range[0],
      maxval=roll_range[1],
      seed=seed)
  rotation_matrices = create_rotation_matrices_3d(azimuths, elevations, rolls)
  # TODO(liuti): Reconcile this with `rotate_points`.
  keypoints_3d = tf.linalg.matrix_transpose(
      tf.matmul(rotation_matrices, keypoints_3d, transpose_b=True))
  if default_camera:

    def transform_to_default_camera(points):
      default_rotation_to_camera = tf.constant([
          [0.0, 0.0, -1.0],
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
      ])
      default_center = tf.constant([0.0, 0.0, default_camera_z])
      rotated_points = rotate_points(default_rotation_to_camera, points)
      transformed_points = rotated_points + default_center
      return transformed_points

    keypoints_3d = transform_to_default_camera(keypoints_3d)
  keypoints_2d = (
      keypoints_3d[Ellipsis, :-1] / tf.math.maximum(1e-12, keypoints_3d[Ellipsis, -1:]))
  return keypoints_2d


def select_keypoints_by_name(keypoints,
                             input_keypoint_names,
                             output_keypoint_names,
                             keypoint_masks=None):
  """Selects keypoints by name.

  Note that it is users' responsibility to make sure that the output keypoint
  name list is a subset of the input keypoint names.

  Args:
    keypoints: A tensor for input keypoints. Shape = [..., num_input_keypoints,
      point_dim].
    input_keypoint_names: A list of strings for input keypoint names.
    output_keypoint_names: A list of strings for output keypoint names.
    keypoint_masks: A tensor for input keypoint masks. Shape = [....
      num_input_keypoints]. Ignored if None.

  Returns:
    output_keypoints: A tensor for output keypoints. Shape = [...,
      num_output_keypoints, point_dim].
    output_keypoint_masks: A tensor for output keypoint masks. Shape = [....
      num_output_keypoints]. None if input mask tensor is None.
  """
  input_to_output_indices = [
      input_keypoint_names.index(keypoint_name)
      for keypoint_name in output_keypoint_names
  ]
  output_keypoints = tf.gather(keypoints, input_to_output_indices, axis=-2)

  output_keypoint_masks = None
  if keypoint_masks:
    output_keypoint_masks = tf.gather(
        keypoint_masks, input_to_output_indices, axis=-1)

  return output_keypoints, output_keypoint_masks


def random_project_and_select_keypoints(keypoints_3d,
                                        keypoint_profile_3d,
                                        output_keypoint_names,
                                        azimuth_range,
                                        elevation_range,
                                        roll_range,
                                        keypoint_masks_3d=None,
                                        default_camera_z=2.0,
                                        seed=None):
  """Generates 2D keypoints from random 3D keypoint projection.

  Note that the compatible 3D keypoint names (if specified during 2D keypoint
  profile initialization) will be used for the 2D keypoint profile.

  Args:
    keypoints_3d: A tensor for input 3D keypoints. Shape = [...,
      num_keypoints_3d, 3].
    keypoint_profile_3d: A KeypointProfile3D object for input keypoints.
    output_keypoint_names: A list of keypoint names to select 2D projection
      with. Must be a subset of the 3D keypoint names.
    azimuth_range: A tuple for minimum and maximum azimuth angles to randomly
      rotate 3D keypoints with.
    elevation_range: A tuple for minimum and maximum elevation angles to
      randomly rotate 3D keypoints with.
    roll_range: A tuple for minimum and maximum roll angles to randomly rotate
      3D keypoints with.
    keypoint_masks_3d: A tensor for input 3D keypoint masks. Shape = [...,
      num_keypoints_3d]. Ignored if None.
    default_camera_z: A float for depth of default camera position.
    seed: An integer for random seed.

  Returns:
    keypoints_2d: A tensor for output 2D keypoints. Shape = [...,
      num_keypoints_2d, 2].
    keypoint_masks_2d: A tensor for output 2D keypoint masks. Shape = [...,
      num_keypoints_2d]. None if input 3D mask is not specified.

  Raises:
    ValueError: If keypoint profile has unsupported dimensionality.
  """
  if keypoint_profile_3d.keypoint_dim != 3:
    raise ValueError('Unsupported input keypoint dimension: %d.' %
                     keypoint_profile_3d.keypoint_dim)

  # First project 3D keypoints to 2D, then select/convert them to 2D profile.
  # TODO(liuti): Figure out why selecting keypoints first is an issue.
  keypoints_3d, _, _ = (
      keypoint_profile_3d.normalize(keypoints_3d, keypoint_masks_3d))
  keypoints_2d = random_rotate_and_project_3d_to_2d(
      keypoints_3d,
      azimuth_range=azimuth_range,
      elevation_range=elevation_range,
      roll_range=roll_range,
      default_camera=True,
      default_camera_z=default_camera_z,
      seed=seed)
  keypoints_2d, keypoint_masks_2d = select_keypoints_by_name(
      keypoints_2d,
      input_keypoint_names=keypoint_profile_3d.keypoint_names,
      output_keypoint_names=output_keypoint_names,
      keypoint_masks=keypoint_masks_3d)
  return keypoints_2d, keypoint_masks_2d


def remove_at_indices(keypoints, indices):
  """Removes keypoints at indices.

  Out-of-range indices will be ignored.

  Args:
    keypoints: A tensor for keypoints. Shape = [..., num_keypoints,
      keypoint_dim].
    indices: A list of integers for keypoint indices to remove.

  Returns:
    A tensor for left keypoints. Shape = [..., num_output_keypoints,
      keypoint_dim].
  """
  keep_indices = []
  for i in range(keypoints.shape.as_list()[-2]):
    if i not in indices:
      keep_indices.append(i)
  return tf.gather(keypoints, indices=keep_indices, axis=-2)


def insert_at_indices(keypoints, indices, insert_keypoints=None):
  """Inserts keypoints at indices.

  For example:
    keypoints = [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]
    indices = [1, 3, 3]
    insert_keypoints = [[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]]
    insert_at_indices(keypoints, indices, insert_keypoints) = [
      [[1.0, 2.0], [7.0, 8.0], [3.0, 4.0],
       [5.0, 6.0], [9.0, 10.0], [11.0, 12.0]]]

  Args:
    keypoints: A tensor for keypoints. Shape = [..., num_keypoints,
      keypoint_dim].
    indices: A list of indices to insert
    insert_keypoints: A tensor for keypoints to insert. Shape = [...,
      len(indices), keypoint_dim]. If None, inserts zero keypoints by default.

  Returns:
    A tensor for keypoints with insertion. Shape = [..., num_keypoints +
      len(indices), keypoint_dim].

  Raises:
    ValueError: If `indices` and `insert_keypoints` (if not None) size mismatch.
  """
  incremented_indices = [i + index for i, index in enumerate(indices)]
  num_input_keypoints = keypoints.shape.as_list()[-2]
  num_insert_keypoints = len(incremented_indices)
  num_output_keypoints = num_input_keypoints + num_insert_keypoints
  if insert_keypoints is None:
    zeros = tf.gather(
        tf.zeros_like(keypoints),
        indices=list(range(num_insert_keypoints)),
        axis=-2)
    keypoints = tf.concat([keypoints, zeros], axis=-2)
  else:
    if insert_keypoints.shape.as_list()[-2] != num_insert_keypoints:
      raise ValueError(
          'Keypoint to be inserted and insertion indices size mismatch: %d vs. '
          '%d.' % (insert_keypoints.shape.as_list()[-2], num_insert_keypoints))
    keypoints = tf.concat(
        [keypoints, tf.reverse(insert_keypoints, axis=[-2])], axis=-2)

  perm_indices = []
  head_index, tail_index = 0, num_output_keypoints - 1
  for i in range(num_output_keypoints):
    if i in incremented_indices:
      perm_indices.append(tail_index)
      tail_index -= 1
    else:
      perm_indices.append(head_index)
      head_index += 1

  return tf.gather(keypoints, indices=perm_indices, axis=-2)


def transfer_keypoint_masks(input_keypoint_masks,
                            input_keypoint_profile,
                            output_keypoint_profile,
                            enforce_surjectivity=True):
  """Transfers keypoint masks according to a different profile.

  Args:
    input_keypoint_masks: A list of tensors for input keypoint masks.
    input_keypoint_profile: A KeypointProfile object for input keypoints.
    output_keypoint_profile: A KeypointProfile object for output keypoints.
    enforce_surjectivity: A boolean for whether to enforce all output keypoint
      masks are transferred from input keypoint masks. If True and any output
      keypoint mask does not come from some input keypoint mask, error will be
      raised. If False, uncorresponded output keypoints will have all-one masks.

  Returns:
    A tensor for output keypoint masks.

  Raises:
    ValueError: `Enforce_surjective` is True, but mapping from input keypoint to
      output keypoint is not surjective.
  """
  input_keypoint_masks = tf.split(
      input_keypoint_masks,
      num_or_size_splits=input_keypoint_profile.keypoint_num,
      axis=-1)
  output_keypoint_masks = [None] * output_keypoint_profile.keypoint_num
  for part_name in input_keypoint_profile.standard_part_names:
    input_keypoint_index = input_keypoint_profile.get_standard_part_index(
        part_name)
    output_keypoint_index = output_keypoint_profile.get_standard_part_index(
        part_name)
    if len(output_keypoint_index) != 1:
      continue
    if len(input_keypoint_index) == 1:
      output_keypoint_masks[output_keypoint_index[0]] = (
          input_keypoint_masks[input_keypoint_index[0]])
    else:
      input_keypoint_mask_subset = [
          input_keypoint_masks[i] for i in input_keypoint_index
      ]
      output_keypoint_masks[output_keypoint_index[0]] = tf.math.reduce_prod(
          tf.stack(input_keypoint_mask_subset, axis=-1),
          axis=-1,
          keepdims=False)

  for i, output_keypoint_mask in enumerate(output_keypoint_masks):
    if output_keypoint_mask is None:
      if enforce_surjectivity:
        raise ValueError('Uncorresponded output keypoints: index = %d.' % i)
      else:
        output_keypoint_masks[i] = tf.ones_like(input_keypoint_masks[0])

  return tf.concat(output_keypoint_masks, axis=-1)
