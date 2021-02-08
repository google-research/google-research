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

"""Contains code for preprocessing the 3d semantic segmentation data."""

import math
import gin
import gin.tf
import six
import tensorflow as tf
from tf3d import standard_fields
from tf3d.utils import preprocessor_utils
from tf3d.utils import shape_utils


def change_intensity_range(intensities,
                           threshold=2.5,
                           normalization_factor1=2500.0,
                           normalization_factor2=12.0):
  """Changes the range of intensity values.

  Args:
    intensities: A tensor containing intensity values. It is assumed it has a
      range of 0 to around 65000.
    threshold: A parameter used for re-ranging intensity values.
    normalization_factor1: A parameter used for re-ranging intensity values.
    normalization_factor2: A parameter used for re-ranging intensity values.

  Returns:
    Tensor with re-ranged intensity values.
  """
  intensities = tf.cast(intensities, dtype=tf.float32)
  intensities_large_mask = tf.cast(
      tf.greater(intensities, threshold), dtype=tf.float32)
  intensities_small = intensities * (1.0 - intensities_large_mask)
  intensities_large = ((threshold +
                        (intensities - threshold) / normalization_factor2) *
                       intensities_large_mask)
  return ((intensities_small + intensities_large) / normalization_factor1) - 1.0


def rotate_points_around_axis(points, rotation_angle, axis=2):
  """Rotates points around axis.

  Args:
    points: A tf.float32 tensor of size [N, 3] containing points.
    rotation_angle: A float value containing the rotation angle in radians.
    axis: A value in [0, 1, 2] for rotating around x, y, z axis.

  Returns:
    rotated_points: A tf.float32 tensor of size [N, 3] containing points.
  """
  if axis not in [0, 1, 2]:
    raise ValueError(('axis is out of bound: %d' % axis))
  c = tf.cos(rotation_angle)
  s = tf.sin(rotation_angle)
  new_points = [points[:, 0], points[:, 1], points[:, 2]]
  other_axis = list(set([0, 1, 2]) - set([axis]))
  new_points[other_axis[0]] = (
      points[:, other_axis[0]] * c - points[:, other_axis[1]] * s)
  new_points[other_axis[1]] = (
      points[:, other_axis[0]] * s + points[:, other_axis[1]] * c)
  return tf.stack(new_points, axis=1)


def rotate_points_and_normals_motions_around_axis(points,
                                                  normals,
                                                  motions,
                                                  rotation_angle,
                                                  axis=2):
  """Rotates points and normals around an axis.

  Args:
    points: A tf.float32 tensor of size [N, 3] containing points.
    normals: A tf.float32 tensor of size [N, 3] containing points or None.
    motions: A tf.float32 tensor of size [N, 3] containing motion vectors or
      None.
    rotation_angle: A float value containing the rotation angle in radians.
    axis: A value in [0, 1, 2] for rotating around x, y, z axis.

  Returns:
    points_rotated: A tf.float32 tensor of size [N, 3] containing points.
    normals_rotated: A tf.float32 tensor of size [N, 3] containing points. If
      normals are None, rotated_normals will be None too.
  """
  points_rotated = rotate_points_around_axis(
      points=points, rotation_angle=rotation_angle, axis=axis)

  def _rotate_vector(vectors):
    vector_end = points + vectors
    vector_end_flipped = rotate_points_around_axis(
        vector_end, rotation_angle=rotation_angle, axis=axis)
    return vector_end_flipped - points_rotated

  normals_rotated = _rotate_vector(normals) if normals is not None else None
  motions_rotated = _rotate_vector(motions) if motions is not None else None

  return points_rotated, normals_rotated, motions_rotated


def rotate_randomly(points, normals, motions, x_min_degree_rotation,
                    x_max_degree_rotation, y_min_degree_rotation,
                    y_max_degree_rotation, z_min_degree_rotation,
                    z_max_degree_rotation):
  """Rotates points and normals randomly around all axes.

  Args:
    points: A tf.float32 tensor of size [N, 3] containing points.
    normals: A tf.float32 tensor of size [N, 3] containing points or None.
    motions: A tf.float32 tensor of size [N, 3] containing motion vectors or
      None.
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

  Returns:
    rotated_points: A tf.float32 tensor of size [N, 3] containing points.
    rotated_normals: A tf.float32 tensor of size [N, 3] containing points. If
      the normals are None, the rotated_normals will be None too.
  """
  if x_min_degree_rotation and x_max_degree_rotation:
    x_min_radian_rotation = math.pi * float(x_min_degree_rotation) / 180.0
    x_max_radian_rotation = math.pi * float(x_max_degree_rotation) / 180.0
    rotation_angle = tf.random.uniform([],
                                       minval=x_min_radian_rotation,
                                       maxval=x_max_radian_rotation,
                                       dtype=tf.float32)
    points, normals, motions = rotate_points_and_normals_motions_around_axis(
        points=points,
        normals=normals,
        motions=motions,
        rotation_angle=rotation_angle,
        axis=0)
  if y_min_degree_rotation and y_max_degree_rotation:
    y_min_radian_rotation = math.pi * float(y_min_degree_rotation) / 180.0
    y_max_radian_rotation = math.pi * float(y_max_degree_rotation) / 180.0
    rotation_angle = tf.random.uniform([],
                                       minval=y_min_radian_rotation,
                                       maxval=y_max_radian_rotation,
                                       dtype=tf.float32)
    points, normals, motions = rotate_points_and_normals_motions_around_axis(
        points=points,
        normals=normals,
        motions=motions,
        rotation_angle=rotation_angle,
        axis=1)
  if z_min_degree_rotation and z_max_degree_rotation:
    z_min_radian_rotation = math.pi * float(z_min_degree_rotation) / 180.0
    z_max_radian_rotation = math.pi * float(z_max_degree_rotation) / 180.0
    rotation_angle = tf.random.uniform([],
                                       minval=z_min_radian_rotation,
                                       maxval=z_max_radian_rotation,
                                       dtype=tf.float32)
    points, normals, motions = rotate_points_and_normals_motions_around_axis(
        points=points,
        normals=normals,
        motions=motions,
        rotation_angle=rotation_angle,
        axis=2)
  return points, normals, motions


def flip_points(points, x_rotate, y_rotate):
  return points * tf.stack((x_rotate, y_rotate, 1), axis=0)


def flip_points_and_normals_motions(points, normals, motions, x_rotate,
                                    y_rotate):
  """Flip points and normals against x or/and y axis.

  Args:
    points: A tf.float32 tensor of size [N, 3] containing points.
    normals: A tf.float32 tensor of size [N, 3] containing points or None.
    motions: A tf.float32 tensor of size [N, 3] containing motion vectors or
      None.
    x_rotate: A tf.float32 scalar tensor of either 1.0 or -1.0. If -1.0 then
      points and normals will flip against x axis.
    y_rotate: A tf.float32 scalar tensor of either 1.0 or -1.0. If -1.0 then
      points and normals will flip against y axis.

  Returns:
    flipped_points: Flipped points. A tf.float32 tensor of size [N, 3].
    flipped_normals: Flipped normals. A tf.float32 tensor of size [N, 3]. It
      will be None if normals is None.
  """
  flipped_points = flip_points(
      points=points, x_rotate=x_rotate, y_rotate=y_rotate)

  def _flip_vector(vectors):
    vector_end = points + vectors
    vector_end_flipped = flip_points(
        vector_end, x_rotate=x_rotate, y_rotate=y_rotate)
    return vector_end_flipped - flipped_points

  flipped_normals = _flip_vector(normals) if normals is not None else None
  flipped_motions = _flip_vector(motions) if motions is not None else None

  return flipped_points, flipped_normals, flipped_motions


def flip_randomly_points_and_normals_motions(points, normals, motions,
                                             is_training):
  """Flip points and normals against x or/and y axis.

  Args:
    points: A tf.float32 tensor of size [N, 3] containing points.
    normals: A tf.float32 tensor of size [N, 3] containing points or None.
    motions: A tf.float32 tensor of size [N, 3] containing motion vectors or
      None.
    is_training: True if in training stage. Random flipping only takes place
      during training.

  Returns:
    flipped_points: Flipped points. A tf.float32 tensor of size [N, 3].
    flipped_normals: Flipped normals. A tf.float32 tensor of size [N, 3]. It
      will be None of the normals is None.
  """
  if is_training:
    x_cond = tf.greater(
        tf.random.uniform([], minval=0.0, maxval=1.0, dtype=tf.float32), 0.5)
    x_rotate = tf.cond(x_cond, lambda: tf.constant(1.0, dtype=tf.float32),
                       lambda: tf.constant(-1.0, dtype=tf.float32))
    y_cond = tf.greater(
        tf.random.uniform([], minval=0.0, maxval=1.0, dtype=tf.float32), 0.5)
    y_rotate = tf.cond(y_cond, lambda: tf.constant(1.0, dtype=tf.float32),
                       lambda: tf.constant(-1.0, dtype=tf.float32))
    (points, normals, motions) = flip_points_and_normals_motions(
        points=points,
        normals=normals,
        motions=motions,
        x_rotate=x_rotate,
        y_rotate=y_rotate)
  return points, normals, motions


def randomly_crop_points(mesh_inputs,
                         view_indices_2d_inputs,
                         x_random_crop_size,
                         y_random_crop_size,
                         epsilon=1e-5):
  """Randomly crops points.

  Args:
    mesh_inputs: A dictionary containing input mesh (point) tensors.
    view_indices_2d_inputs: A dictionary containing input point to view
      correspondence tensors.
    x_random_crop_size: Size of the random crop in x dimension. If None, random
      crop will not take place on x dimension.
    y_random_crop_size: Size of the random crop in y dimension. If None, random
      crop will not take place on y dimension.
    epsilon: Epsilon (a very small value) used to add as a small margin to
      thresholds.
  """
  if x_random_crop_size is None and y_random_crop_size is None:
    return

  points = mesh_inputs[standard_fields.InputDataFields.point_positions]
  num_points = tf.shape(points)[0]
  # Pick a random point
  if x_random_crop_size is not None or y_random_crop_size is not None:
    random_index = tf.random.uniform([],
                                     minval=0,
                                     maxval=num_points,
                                     dtype=tf.int32)
    center_x = points[random_index, 0]
    center_y = points[random_index, 1]

  points_x = points[:, 0]
  points_y = points[:, 1]
  min_x = tf.reduce_min(points_x) - epsilon
  max_x = tf.reduce_max(points_x) + epsilon
  min_y = tf.reduce_min(points_y) - epsilon
  max_y = tf.reduce_max(points_y) + epsilon

  if x_random_crop_size is not None:
    min_x = center_x - x_random_crop_size / 2.0 - epsilon
    max_x = center_x + x_random_crop_size / 2.0 + epsilon

  if y_random_crop_size is not None:
    min_y = center_y - y_random_crop_size / 2.0 - epsilon
    max_y = center_y + y_random_crop_size / 2.0 + epsilon

  x_mask = tf.logical_and(tf.greater(points_x, min_x), tf.less(points_x, max_x))
  y_mask = tf.logical_and(tf.greater(points_y, min_y), tf.less(points_y, max_y))
  points_mask = tf.logical_and(x_mask, y_mask)

  for key in sorted(mesh_inputs):
    mesh_inputs[key] = tf.boolean_mask(mesh_inputs[key], points_mask)

  for key in sorted(view_indices_2d_inputs):
    view_indices_2d_inputs[key] = tf.transpose(
        tf.boolean_mask(
            tf.transpose(view_indices_2d_inputs[key], [1, 0, 2]), points_mask),
        [1, 0, 2])


def pick_labeled_image(mesh_inputs, view_image_inputs, view_indices_2d_inputs,
                       view_name):
  """Pick the image with most number of labeled points projecting to it."""
  if view_name not in view_image_inputs:
    return
  if view_name not in view_indices_2d_inputs:
    return
  if standard_fields.InputDataFields.point_loss_weights not in mesh_inputs:
    raise ValueError('The key `weights` is missing from mesh_inputs.')
  height = tf.shape(view_image_inputs[view_name])[1]
  width = tf.shape(view_image_inputs[view_name])[2]
  valid_points_y = tf.logical_and(
      tf.greater_equal(view_indices_2d_inputs[view_name][:, :, 0], 0),
      tf.less(view_indices_2d_inputs[view_name][:, :, 0], height))
  valid_points_x = tf.logical_and(
      tf.greater_equal(view_indices_2d_inputs[view_name][:, :, 1], 0),
      tf.less(view_indices_2d_inputs[view_name][:, :, 1], width))
  valid_points = tf.logical_and(valid_points_y, valid_points_x)
  image_total_weights = tf.reduce_sum(
      tf.cast(valid_points, dtype=tf.float32) * tf.squeeze(
          mesh_inputs[standard_fields.InputDataFields.point_loss_weights],
          axis=1),
      axis=1)
  image_total_weights = tf.cond(
      tf.equal(tf.reduce_sum(image_total_weights), 0),
      lambda: tf.reduce_sum(tf.cast(valid_points, dtype=tf.float32), axis=1),
      lambda: image_total_weights)
  best_image = tf.math.argmax(image_total_weights)
  view_image_inputs[view_name] = view_image_inputs[view_name][
      best_image:best_image + 1, :, :, :]
  view_indices_2d_inputs[view_name] = view_indices_2d_inputs[view_name][
      best_image:best_image + 1, :, :]


def _remove_second_return_lidar_points(mesh_inputs, view_indices_2d_inputs):
  """removes the points that are not lidar first-return ."""
  if standard_fields.InputDataFields.point_spin_coordinates not in mesh_inputs:
    raise ValueError('spin_coordinates not in mesh_inputs.')
  first_return_mask = tf.equal(
      tf.cast(
          mesh_inputs[standard_fields.InputDataFields.point_spin_coordinates]
          [:, 2],
          dtype=tf.int32), 0)
  for key in sorted(mesh_inputs):
    mesh_inputs[key] = tf.boolean_mask(mesh_inputs[key], first_return_mask)
  for key in sorted(view_indices_2d_inputs):
    view_indices_2d_inputs[key] = tf.transpose(
        tf.boolean_mask(
            tf.transpose(view_indices_2d_inputs[key], [1, 0, 2]),
            first_return_mask), [1, 0, 2])


def pad_or_clip(mesh_inputs, view_indices_2d_inputs, pad_or_clip_size):
  """Pads and clips the points and correspondences."""
  if standard_fields.InputDataFields.point_positions not in mesh_inputs:
    return
  num_valid_points = tf.shape(
      mesh_inputs[standard_fields.InputDataFields.point_positions])[0]
  if pad_or_clip_size:
    num_valid_points = tf.minimum(num_valid_points, pad_or_clip_size)
    for key in sorted(mesh_inputs.keys()):
      num_channels = mesh_inputs[key].get_shape().as_list()[1]
      mesh_inputs[key] = shape_utils.pad_or_clip_nd(
          tensor=mesh_inputs[key],
          output_shape=[pad_or_clip_size, num_channels])
    for key in sorted(view_indices_2d_inputs):
      num_images = view_indices_2d_inputs[key].get_shape().as_list()[0]
      if num_images is None:
        num_images = tf.shape(view_indices_2d_inputs[key])[0]
      view_indices_2d_inputs[key] = shape_utils.pad_or_clip_nd(
          tensor=(view_indices_2d_inputs[key] + 1),
          output_shape=[num_images, pad_or_clip_size, 2]) - 1
  mesh_inputs[
      standard_fields.InputDataFields.num_valid_points] = num_valid_points


@gin.configurable(
    'semantic_pointcloud_preprocess',
    denylist=['inputs', 'output_keys', 'is_training'])
def preprocess(inputs,
               output_keys=None,
               is_training=False,
               using_sequence_dataset=False,
               num_frame_to_load=1,
               transform_points_fn=None,
               image_preprocess_fn_dic=None,
               images_points_correspondence_fn=None,
               compute_semantic_labels_fn=None,
               compute_motion_labels_fn=None,
               view_names=(),
               points_key='points',
               colors_key='colors',
               normals_key='normals',
               intensities_key='intensities',
               elongations_key='elongations',
               semantic_labels_key='semantic_labels',
               motion_labels_key='motion_labels',
               spin_coords_key=None,
               points_in_image_frame_key=None,
               num_points_to_randomly_sample=None,
               x_min_degree_rotation=None,
               x_max_degree_rotation=None,
               y_min_degree_rotation=None,
               y_max_degree_rotation=None,
               z_min_degree_rotation=None,
               z_max_degree_rotation=None,
               points_pad_or_clip_size=None,
               voxels_pad_or_clip_size=None,
               voxel_grid_cell_size=(0.1, 0.1, 0.1),
               num_offset_bins_x=4,
               num_offset_bins_y=4,
               num_offset_bins_z=4,
               point_feature_keys=('point_offsets',),
               point_to_voxel_segment_func=tf.math.unsorted_segment_mean,
               x_random_crop_size=None,
               y_random_crop_size=None,
               min_scale_ratio=None,
               max_scale_ratio=None,
               semantic_labels_offset=0,
               ignore_labels=(),
               remove_unlabeled_images_and_points=False,
               labeled_view_name=None,
               only_keep_first_return_lidar_points=False):
  """Preprocesses a dictionary of `Tensor` inputs.

  If is_training=True, it will randomly rotate the points around the z axis,
  and will randomly flip the points with respect to x and/or y axis.

  Note that the preprocessor function does not correct normal vectors if they
  exist in the inputs.
  Note that the preprocessing effects all values of `inputs` that are `Tensors`.

  Args:
    inputs: A dictionary of inputs. Each value must be a `Tensor`.
    output_keys: Either None, or a list of strings containing the keys in the
      dictionary that is returned by the preprocess function.
    is_training: Whether we're training or testing.
    using_sequence_dataset: if true, the inputs will contain scene and multiple
      frames data.
    num_frame_to_load: If greater than 1, load multiframe point cloud point
      positions and its correspondence.
    transform_points_fn: Fn to transform other frames to a specific frame's
      coordinate.
    image_preprocess_fn_dic: Image preprocessing function. Maps view names to
      their image preprocessing functions. Set it to None, if there are no
      images to preprocess or you are not interested in preprocesing images.
    images_points_correspondence_fn: The function that computes correspondence
      between images and points.
    compute_semantic_labels_fn: If not None, semantic labels will be computed
      using this function.
    compute_motion_labels_fn: If not None, motion labels will be computed using
      this function.
    view_names: Names corresponding to 2d views of the scene.
    points_key: The key used for `points` in the inputs.
    colors_key: The key used for `colors` in the inputs.
    normals_key: The key used for 'normals' in the inputs.
    intensities_key: The key used for 'intensities' in the inputs.
    elongations_key: The key used for 'elongations' in the inputs.
    semantic_labels_key: The key used for 'semantic_labels' in the inputs.
    motion_labels_key: The key used for 'motion_labels' in the inputs.
    spin_coords_key: The key used for 'spin_coords' in the inputs. In Waymo
      data, spin_coords is a [num_points, 3] tensor that contains scan_index,
      shot_index, return_index. In Waymo data, return_index of the first return
      points is 0.
    points_in_image_frame_key: A string that identifies the tensor that contains
      the points_in_image_frame tensor. If None, it won't be used.
    num_points_to_randomly_sample: Number of points to randomly sample. If None,
      it will keep the original points and does not perform sampling.
    x_min_degree_rotation: Min degree of rotation around the x axis.
    x_max_degree_rotation: Max degree of ratation around the x axis.
    y_min_degree_rotation: Min degree of rotation around the y axis.
    y_max_degree_rotation: Max degree of ratation around the y axis.
    z_min_degree_rotation: Min degree of rotation around the z axis.
    z_max_degree_rotation: Max degree of ratation around the z axis.
    points_pad_or_clip_size: Number of target points to pad or clip to. If None,
      it will not perform the point padding.
    voxels_pad_or_clip_size: Number of target voxels to pad or clip to. If None,
      it will not perform the voxel padding.
    voxel_grid_cell_size: A three dimensional tuple determining the voxel grid
      size.
    num_offset_bins_x: Number of bins for point offsets in x direction.
    num_offset_bins_y: Number of bins for point offsets in y direction.
    num_offset_bins_z: Number of bins for point offsets in z direction.
    point_feature_keys: The keys used to form the voxel features.
    point_to_voxel_segment_func: The function used to aggregate the features
      of the points that fall in the same voxel.
    x_random_crop_size: Size of the random crop in x dimension. If None, random
      crop will not take place on x dimension.
    y_random_crop_size: Size of the random crop in y dimension. If None, random
      crop will not take place on y dimension.
    min_scale_ratio: Minimum scale ratio. Used for scaling point cloud.
    max_scale_ratio: Maximum scale ratio. Used for scaling point cloud.
    semantic_labels_offset: An integer offset that will be added to labels.
    ignore_labels: A tuple containing labels that should be ignored when
      computing the loss and metrics.
    remove_unlabeled_images_and_points: If True, removes the images that are not
      labeled and also removes the points that are associated with those images.
    labeled_view_name: The name of the view that is labeled, otherwise None.
    only_keep_first_return_lidar_points: If True, we only keep the first return
      lidar points.

  Returns:
    The mean subtracted points with an optional rotation applied.

  Raises:
    ValueError: if `inputs` doesn't contain the points_key.
    ValueError: if `points_in_image_frame` does not have rank 3.
  """
  inputs = dict(inputs)

  if using_sequence_dataset:
    all_frame_inputs = inputs
    scene = all_frame_inputs['scene']
    frame1 = all_frame_inputs['frame1']
    frame_start_index = all_frame_inputs['frame_start_index']
    inputs = dict(all_frame_inputs['frame0']
                 )  # so that the following processing code can be unchanged.

  # Initializing empty dictionary for mesh, image, indices_2d and non tensor
  # inputs.
  non_tensor_inputs = {}
  view_image_inputs = {}
  view_indices_2d_inputs = {}
  mesh_inputs = {}

  if image_preprocess_fn_dic is None:
    image_preprocess_fn_dic = {}

  # Convert all float64 to float32 and all int64 to int32.
  for key in sorted(inputs):
    if isinstance(inputs[key], tf.Tensor):
      if inputs[key].dtype == tf.float64:
        inputs[key] = tf.cast(inputs[key], dtype=tf.float32)
      if inputs[key].dtype == tf.int64:
        inputs[key] = tf.cast(inputs[key], dtype=tf.int32)

  if points_key in inputs:
    inputs[standard_fields.InputDataFields.point_positions] = inputs[points_key]
  if colors_key is not None and colors_key in inputs:
    inputs[standard_fields.InputDataFields.point_colors] = inputs[colors_key]
  if normals_key is not None and normals_key in inputs:
    inputs[standard_fields.InputDataFields.point_normals] = inputs[normals_key]
  if intensities_key is not None and intensities_key in inputs:
    inputs[standard_fields.InputDataFields
           .point_intensities] = inputs[intensities_key]
  if elongations_key is not None and elongations_key in inputs:
    inputs[standard_fields.InputDataFields
           .point_elongations] = inputs[elongations_key]
  if semantic_labels_key is not None and semantic_labels_key in inputs:
    inputs[standard_fields.InputDataFields
           .object_class_points] = inputs[semantic_labels_key]
  if motion_labels_key is not None and motion_labels_key in inputs:
    inputs[standard_fields.InputDataFields
           .object_flow_points] = inputs[motion_labels_key]
  if spin_coords_key is not None and spin_coords_key in inputs:
    inputs[standard_fields.InputDataFields
           .point_spin_coordinates] = inputs[spin_coords_key]

  # Acquire point / image correspondences.
  if images_points_correspondence_fn is not None:
    fn_outputs = images_points_correspondence_fn(inputs)
    if 'points_position' in fn_outputs:
      inputs[standard_fields.InputDataFields
             .point_positions] = fn_outputs['points_position']
    if 'points_intensity' in fn_outputs and intensities_key is not None:
      inputs[standard_fields.InputDataFields
             .point_intensities] = fn_outputs['points_intensity']
    if 'points_elongation' in fn_outputs and elongations_key is not None:
      inputs[standard_fields.InputDataFields
             .point_elongations] = fn_outputs['points_elongation']
    if 'points_label' in fn_outputs and semantic_labels_key is not None:
      inputs[standard_fields.InputDataFields
             .object_class_points] = fn_outputs['points_label']
    if 'view_images' in fn_outputs:
      for key in sorted(fn_outputs['view_images']):
        if len(fn_outputs['view_images'][key].shape) != 4:
          raise ValueError(('%s image should have rank 4.' % key))
      view_image_inputs = fn_outputs['view_images']
    if 'view_indices_2d' in fn_outputs:
      for key in sorted(fn_outputs['view_indices_2d']):
        if len(fn_outputs['view_indices_2d'][key].shape) != 3:
          raise ValueError(('%s indices_2d should have rank 3.' % key))
      view_indices_2d_inputs = fn_outputs['view_indices_2d']
  else:
    if points_in_image_frame_key is not None:
      inputs['rgb_view/features'] = inputs['image']
      inputs['rgb_view/indices_2d'] = inputs[points_in_image_frame_key]
      if len(inputs['rgb_view/indices_2d'].shape) != 3:
        raise ValueError('`points_in_image_frame` should have rank 3.')

  frame0 = inputs.copy()
  if num_frame_to_load > 1:
    point_positions_list = [
        frame0[standard_fields.InputDataFields.point_positions]
    ]
    if view_indices_2d_inputs:
      view_indices_2d_list = [view_indices_2d_inputs[view_names[0]]]
    frame_source_list = [
        tf.zeros([
            tf.shape(frame0[standard_fields.InputDataFields.point_positions])[0]
        ], tf.int32)
    ]
    for i in range(1, num_frame_to_load):
      target_frame_key = 'frame' + str(i)
      if images_points_correspondence_fn is not None:
        frame_i = images_points_correspondence_fn(
            all_frame_inputs[target_frame_key])
      else:
        raise ValueError(
            'images_points_correspondence_fn is needed for loading multi-frame pointclouds.'
        )
      transformed_point_positions = transform_points_fn(
          scene, frame_i['points_position'], frame_start_index,
          i + frame_start_index)
      point_positions_list.append(transformed_point_positions)
      if view_indices_2d_inputs:
        view_indices_2d_list.append(frame_i['view_indices_2d'][view_names[0]])
      frame_source_list.append(
          tf.ones([tf.shape(transformed_point_positions)[0]], tf.int32) * i)

    # add multi-frame info to override inputs and view_indices_2d_inputs
    inputs[standard_fields.InputDataFields.point_frame_index] = tf.expand_dims(
        tf.concat(frame_source_list, axis=0), axis=1)
    inputs[standard_fields.InputDataFields.point_positions] = tf.concat(
        point_positions_list, axis=0)
    if view_indices_2d_inputs:
      view_indices_2d_inputs[view_names[0]] = tf.concat(
          view_indices_2d_list, axis=1)

  # Validate inputs.
  if standard_fields.InputDataFields.point_positions not in inputs:
    raise ValueError('`inputs` must contain a point_positions')
  if inputs[standard_fields.InputDataFields.point_positions].shape.ndims != 2:
    raise ValueError('points must be of rank 2.')
  if inputs[standard_fields.InputDataFields.point_positions].shape[1] != 3:
    raise ValueError('point should be 3 dimensional.')

  # Remove normal nans.
  if standard_fields.InputDataFields.point_normals in inputs:
    inputs[standard_fields.InputDataFields.point_normals] = tf.where(
        tf.math.is_nan(inputs[standard_fields.InputDataFields.point_normals]),
        tf.zeros_like(inputs[standard_fields.InputDataFields.point_normals]),
        inputs[standard_fields.InputDataFields.point_normals])

  # Compute semantic labels if compute_semantic_labels_fn is not None
  # An example is when the ground-truth contains 3d object boxes and not per
  # point labels. This would be a function that infers point labels from boxes.
  if compute_semantic_labels_fn is not None:
    inputs[standard_fields.InputDataFields
           .object_class_points] = compute_semantic_labels_fn(
               inputs=frame0,
               points_key=standard_fields.InputDataFields.point_positions)
  if compute_motion_labels_fn is not None:
    inputs[standard_fields.InputDataFields
           .object_flow_points] = compute_motion_labels_fn(
               scene=scene,
               frame0=frame0,
               frame1=frame1,
               frame_start_index=frame_start_index,
               points_key=standard_fields.InputDataFields.point_positions)

  # Splitting inputs to {view_image_inputs,
  #                      view_indices_2d_inputs,
  #                      mesh_inputs,
  #                      non_tensor_inputs}
  mesh_keys = []
  for key in [
      standard_fields.InputDataFields.point_positions,
      standard_fields.InputDataFields.point_colors,
      standard_fields.InputDataFields.point_normals,
      standard_fields.InputDataFields.point_intensities,
      standard_fields.InputDataFields.point_elongations,
      standard_fields.InputDataFields.object_class_points,
      standard_fields.InputDataFields.point_spin_coordinates,
      standard_fields.InputDataFields.object_flow_points,
      standard_fields.InputDataFields.point_frame_index,
  ]:
    if key is not None and key in inputs:
      mesh_keys.append(key)
  view_image_names = [('%s/features' % key) for key in view_names]
  view_indices_2d_names = [('%s/indices_2d' % key) for key in view_names]

  # Additional key collecting
  for k, v in six.iteritems(inputs):
    if k in view_image_names:
      view_image_inputs[k] = v
    elif k in view_indices_2d_names:
      view_indices_2d_inputs[k] = v
    elif k in mesh_keys:
      if num_frame_to_load > 1:
        pad_size = tf.shape(inputs[standard_fields.InputDataFields
                                   .point_positions])[0] - tf.shape(v)[0]
        if k == standard_fields.InputDataFields.object_class_points:
          pad_value = -1
        else:
          pad_value = 0
        v = tf.pad(v, [[0, pad_size], [0, 0]], constant_values=pad_value)
      mesh_inputs[k] = v
    else:
      non_tensor_inputs[k] = v

  # Remove points that are not in the lidar first return (optional)
  if only_keep_first_return_lidar_points:
    _remove_second_return_lidar_points(
        mesh_inputs=mesh_inputs,
        view_indices_2d_inputs=view_indices_2d_inputs)

  # Randomly sample points
  preprocessor_utils.randomly_sample_points(
      mesh_inputs=mesh_inputs,
      view_indices_2d_inputs=view_indices_2d_inputs,
      target_num_points=num_points_to_randomly_sample)

  # Add weights if it does not exist in inputs. The weight of the points with
  # label in `ignore_labels` is set to 0. This helps the loss and metrics to
  # ignore those labels.
  use_weights = (
      standard_fields.InputDataFields.object_class_points in mesh_inputs or
      standard_fields.InputDataFields.object_flow_points in mesh_inputs)
  if use_weights:
    if num_frame_to_load > 1:
      num_valid_points_frame0 = tf.shape(
          frame0[standard_fields.InputDataFields.point_positions])[0]
      num_additional_frame_points = tf.shape(
          mesh_inputs[standard_fields.InputDataFields
                      .object_class_points])[0] - num_valid_points_frame0
      weights = tf.concat([
          tf.ones([num_valid_points_frame0, 1], tf.float32),
          tf.zeros([num_additional_frame_points, 1], tf.float32)
      ],
                          axis=0)
    else:
      weights = tf.ones_like(
          mesh_inputs[standard_fields.InputDataFields.object_class_points],
          dtype=tf.float32)

  if standard_fields.InputDataFields.object_class_points in mesh_inputs:
    mesh_inputs[standard_fields.InputDataFields.object_class_points] = tf.cast(
        mesh_inputs[standard_fields.InputDataFields.object_class_points],
        dtype=tf.int32)
    for ignore_label in ignore_labels:
      weights *= tf.cast(
          tf.not_equal(
              mesh_inputs[standard_fields.InputDataFields.object_class_points],
              ignore_label),
          dtype=tf.float32)
    mesh_inputs[standard_fields.InputDataFields.point_loss_weights] = weights
    mesh_inputs[standard_fields.InputDataFields
                .object_class_points] += semantic_labels_offset

  # We normalize the intensities and elongations to be in a smaller range.
  if standard_fields.InputDataFields.point_intensities in mesh_inputs:
    mesh_inputs[standard_fields.InputDataFields
                .point_intensities] = change_intensity_range(
                    intensities=mesh_inputs[
                        standard_fields.InputDataFields.point_intensities])
  if standard_fields.InputDataFields.point_elongations in mesh_inputs:
    mesh_inputs[standard_fields.InputDataFields.point_elongations] = (tf.cast(
        mesh_inputs[standard_fields.InputDataFields.point_elongations],
        dtype=tf.float32) * 2.0 / 255.0) - 1.0

  # Random scale the points.
  if min_scale_ratio is not None and max_scale_ratio is not None:
    scale_ratio = tf.random.uniform([],
                                    minval=min_scale_ratio,
                                    maxval=max_scale_ratio,
                                    dtype=tf.float32)
    mesh_inputs[standard_fields.InputDataFields.point_positions] *= scale_ratio
    if standard_fields.InputDataFields.object_flow_points in mesh_inputs:
      mesh_inputs[
          standard_fields.InputDataFields.object_flow_points] *= scale_ratio

  # Random crop the points.
  randomly_crop_points(
      mesh_inputs=mesh_inputs,
      view_indices_2d_inputs=view_indices_2d_inputs,
      x_random_crop_size=x_random_crop_size,
      y_random_crop_size=y_random_crop_size)

  # If training, pick the best labeled image and points that project to it.
  # In many datasets, only one image is labeled anyways.
  if remove_unlabeled_images_and_points:
    pick_labeled_image(
        mesh_inputs=mesh_inputs,
        view_image_inputs=view_image_inputs,
        view_indices_2d_inputs=view_indices_2d_inputs,
        view_name=labeled_view_name)

  # Process images.
  preprocessor_utils.preprocess_images(
      view_image_inputs=view_image_inputs,
      view_indices_2d_inputs=view_indices_2d_inputs,
      image_preprocess_fn_dic=image_preprocess_fn_dic,
      is_training=is_training)

  # Record the original points.
  original_points = mesh_inputs[standard_fields.InputDataFields.point_positions]
  if standard_fields.InputDataFields.point_colors in mesh_inputs:
    original_colors = mesh_inputs[standard_fields.InputDataFields.point_colors]
  if standard_fields.InputDataFields.point_normals in mesh_inputs:
    original_normals = mesh_inputs[
        standard_fields.InputDataFields.point_normals]

  # Update feature visibility count.
  if 'feature_visibility_count' in mesh_inputs:
    mesh_inputs['feature_visibility_count'] = tf.maximum(
        mesh_inputs['feature_visibility_count'], 1)
    mesh_inputs['features'] /= tf.cast(
        mesh_inputs['feature_visibility_count'], dtype=tf.float32)

  # Subtract mean from points.
  mean_points = tf.reduce_mean(
      mesh_inputs[standard_fields.InputDataFields.point_positions], axis=0)
  mesh_inputs[
      standard_fields.InputDataFields.point_positions] -= tf.expand_dims(
          mean_points, axis=0)

  # Rotate points randomly.
  if standard_fields.InputDataFields.point_normals in mesh_inputs:
    normals = mesh_inputs[standard_fields.InputDataFields.point_normals]
  else:
    normals = None

  if standard_fields.InputDataFields.object_flow_points in mesh_inputs:
    motions = mesh_inputs[standard_fields.InputDataFields.object_flow_points]
  else:
    motions = None

  (mesh_inputs[standard_fields.InputDataFields.point_positions],
   rotated_normals, rotated_motions) = rotate_randomly(
       points=mesh_inputs[standard_fields.InputDataFields.point_positions],
       normals=normals,
       motions=motions,
       x_min_degree_rotation=x_min_degree_rotation,
       x_max_degree_rotation=x_max_degree_rotation,
       y_min_degree_rotation=y_min_degree_rotation,
       y_max_degree_rotation=y_max_degree_rotation,
       z_min_degree_rotation=z_min_degree_rotation,
       z_max_degree_rotation=z_max_degree_rotation)

  # Random flipping in x and y directions.
  (mesh_inputs[standard_fields.InputDataFields.point_positions],
   flipped_normals, flipped_motions) = flip_randomly_points_and_normals_motions(
       points=mesh_inputs[standard_fields.InputDataFields.point_positions],
       normals=rotated_normals,
       motions=rotated_motions,
       is_training=is_training)
  if standard_fields.InputDataFields.point_normals in mesh_inputs:
    mesh_inputs[standard_fields.InputDataFields.point_normals] = flipped_normals
  if standard_fields.InputDataFields.object_flow_points in mesh_inputs:
    mesh_inputs[
        standard_fields.InputDataFields.object_flow_points] = flipped_motions
  # Normalize RGB to [-1.0, 1.0].
  if standard_fields.InputDataFields.point_colors in mesh_inputs:
    mesh_inputs[standard_fields.InputDataFields.point_colors] = tf.cast(
        mesh_inputs[standard_fields.InputDataFields.point_colors],
        dtype=tf.float32)
    mesh_inputs[standard_fields.InputDataFields.point_colors] *= (2.0 / 255.0)
    mesh_inputs[standard_fields.InputDataFields.point_colors] -= 1.0

  # Add original points to mesh inputs.
  mesh_inputs[standard_fields.InputDataFields
              .point_positions_original] = original_points
  if standard_fields.InputDataFields.point_colors in mesh_inputs:
    mesh_inputs[
        standard_fields.InputDataFields.point_colors_original] = original_colors
  if standard_fields.InputDataFields.point_normals in mesh_inputs:
    mesh_inputs[standard_fields.InputDataFields
                .point_normals_original] = original_normals

  # Pad or clip the point tensors.
  pad_or_clip(
      mesh_inputs=mesh_inputs,
      view_indices_2d_inputs=view_indices_2d_inputs,
      pad_or_clip_size=points_pad_or_clip_size)
  if num_frame_to_load > 1:
    # Note: num_valid_points is the sum of 'num_points_per_fram' for now.
    # num_points_per_frame is each frame's valid num of points.
    # TODO(huangrui): if random sampling is called earlier, the count here
    # is not guaranteed to be in order. need sorting.
    if num_points_to_randomly_sample is not None:
      raise ValueError(
          'randomly sample is not compatible with padding multi frame point clouds yet!'
      )
    _, _, mesh_inputs[
        standard_fields.InputDataFields
        .num_valid_points_per_frame] = tf.unique_with_counts(
            tf.reshape(
                mesh_inputs[standard_fields.InputDataFields.point_frame_index],
                [-1]))
    if points_pad_or_clip_size is not None:
      padded_points = tf.where_v2(
          tf.greater(
              points_pad_or_clip_size,
              mesh_inputs[standard_fields.InputDataFields.num_valid_points]),
          points_pad_or_clip_size -
          mesh_inputs[standard_fields.InputDataFields.num_valid_points], 0)

      # Correct the potential unique count error from optionally padded 0s point
      # frame index.
      mesh_inputs[
          standard_fields.InputDataFields.num_valid_points_per_frame] -= tf.pad(
              tf.expand_dims(padded_points, 0), [[
                  0,
                  tf.shape(mesh_inputs[standard_fields.InputDataFields
                                       .num_valid_points_per_frame])[0] - 1
              ]])

  # Putting back the dictionaries together
  processed_inputs = mesh_inputs.copy()
  processed_inputs.update(non_tensor_inputs)
  for key in sorted(view_image_inputs):
    processed_inputs[('%s/features' % key)] = view_image_inputs[key]
  for key in sorted(view_indices_2d_inputs):
    processed_inputs[('%s/indices_2d' % key)] = view_indices_2d_inputs[key]

  # Create features that do not exist
  if 'point_offsets' in point_feature_keys:
    preprocessor_utils.add_point_offsets(
        inputs=processed_inputs, voxel_grid_cell_size=voxel_grid_cell_size)
  if 'point_offset_bins' in point_feature_keys:
    preprocessor_utils.add_point_offset_bins(
        inputs=processed_inputs,
        voxel_grid_cell_size=voxel_grid_cell_size,
        num_bins_x=num_offset_bins_x,
        num_bins_y=num_offset_bins_y,
        num_bins_z=num_offset_bins_z)

  # Voxelize point features
  preprocessor_utils.voxelize_point_features(
      inputs=processed_inputs,
      voxels_pad_or_clip_size=voxels_pad_or_clip_size,
      voxel_grid_cell_size=voxel_grid_cell_size,
      point_feature_keys=point_feature_keys,
      point_to_voxel_segment_func=point_to_voxel_segment_func,
      num_frame_to_load=num_frame_to_load)

  # Voxelize point / image correspondence indices
  preprocessor_utils.voxelize_point_to_view_correspondences(
      inputs=processed_inputs,
      view_indices_2d_inputs=view_indices_2d_inputs,
      voxels_pad_or_clip_size=voxels_pad_or_clip_size,
      voxel_grid_cell_size=voxel_grid_cell_size)

  # Voxelizing the semantic labels
  preprocessor_utils.voxelize_semantic_labels(
      inputs=processed_inputs,
      voxels_pad_or_clip_size=voxels_pad_or_clip_size,
      voxel_grid_cell_size=voxel_grid_cell_size)

  # Voxelizing the loss weights
  preprocessor_utils.voxelize_property_tensor(
      inputs=processed_inputs,
      point_tensor_key=standard_fields.InputDataFields.point_loss_weights,
      corresponding_voxel_tensor_key=standard_fields.InputDataFields
      .voxel_loss_weights,
      voxels_pad_or_clip_size=voxels_pad_or_clip_size,
      voxel_grid_cell_size=voxel_grid_cell_size,
      segment_func=tf.math.unsorted_segment_max)

  # Voxelizing the object flow
  if standard_fields.InputDataFields.object_flow_points in processed_inputs:
    preprocessor_utils.voxelize_property_tensor(
        inputs=processed_inputs,
        point_tensor_key=standard_fields.InputDataFields.object_flow_points,
        corresponding_voxel_tensor_key='object_flow_voxels_max',
        voxels_pad_or_clip_size=voxels_pad_or_clip_size,
        voxel_grid_cell_size=voxel_grid_cell_size,
        segment_func=tf.math.unsorted_segment_max)
    preprocessor_utils.voxelize_property_tensor(
        inputs=processed_inputs,
        point_tensor_key=standard_fields.InputDataFields.object_flow_points,
        corresponding_voxel_tensor_key='object_flow_voxels_min',
        voxels_pad_or_clip_size=voxels_pad_or_clip_size,
        voxel_grid_cell_size=voxel_grid_cell_size,
        segment_func=tf.math.unsorted_segment_min)
    processed_inputs[
        standard_fields.InputDataFields.object_flow_voxels] = processed_inputs[
            'object_flow_voxels_max'] + processed_inputs[
                'object_flow_voxels_min']

  if num_frame_to_load > 1:
    mesh_inputs[standard_fields.InputDataFields.num_valid_points] = mesh_inputs[
        standard_fields.InputDataFields.num_valid_points_per_frame][0]

  # Filter preprocessed_inputs by output_keys if it is not None.
  if output_keys is not None:
    processed_inputs = {
        k: v for k, v in six.iteritems(processed_inputs) if k in output_keys
    }
  return processed_inputs
