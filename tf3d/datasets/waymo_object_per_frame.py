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

"""Waymo open dataset."""

import os
import gin
import gin.tf
import tensorflow as tf
import tensorflow_datasets as tfds

from tf3d.datasets.specs import waymo_frames
from tf3d.datasets.utils import example_parser
from tf3d.utils import box_utils


_FILE_PATTERN = '%s*.sst'
_FILE_PATTERN_TFRECORD = '%s*.tfrecords'

DATASET_DIR = None


DATASET_FORMAT = 'sstable'


def _get_feature_label_keys():
  """Extracts and returns the dataset feature and label keys."""
  feature_spec = waymo_frames.FRAME_FEATURE_SPEC.get_serialized_info()
  feature_dict = tfds.core.utils.flatten_nest_dict(feature_spec)
  feature_keys = []
  label_keys = []
  for key in sorted(feature_dict):
    if 'objects' in key:
      label_keys.append(key)
    else:
      feature_keys.append(key)
  return feature_keys, label_keys


def get_feature_keys():
  return _get_feature_label_keys()[0]


def get_label_keys():
  return _get_feature_label_keys()[1]


def get_file_pattern(split_name,
                     dataset_dir=DATASET_DIR,
                     dataset_format=DATASET_FORMAT):
  if dataset_format == DATASET_FORMAT:
    return os.path.join(dataset_dir, _FILE_PATTERN % split_name)
  elif dataset_format == 'tfrecord':
    return os.path.join(dataset_dir, _FILE_PATTERN_TFRECORD % split_name)


def get_decode_fn():
  """Returns a tfds decoder.

  Returns:
    A tf.data decoder.
  """

  def decode_fn(value):
    tensors = example_parser.decode_serialized_example(
        serialized_example=value, features=waymo_frames.FRAME_FEATURE_SPEC)
    tensor_dict = tfds.core.utils.flatten_nest_dict(tensors)
    return tensor_dict

  return decode_fn


def _prepare_lidar_points(inputs, lidar_names):
  """Integrates and returns the lidar points in vehicle coordinate frame."""
  points_position = []
  points_intensity = []
  points_elongation = []
  points_normal = []
  points_in_image_frame_xy = []
  points_in_image_frame_id = []
  for lidar_name in lidar_names:
    lidar_location = tf.reshape(inputs[('lidars/%s/extrinsics/t') % lidar_name],
                                [-1, 3])
    inside_no_label_zone = tf.reshape(
        inputs[('lidars/%s/pointcloud/inside_nlz' % lidar_name)], [-1])
    valid_points_mask = tf.math.logical_not(inside_no_label_zone)
    points_position_current_lidar = tf.boolean_mask(
        inputs[('lidars/%s/pointcloud/positions' % lidar_name)],
        valid_points_mask)
    points_position.append(points_position_current_lidar)
    points_intensity.append(
        tf.boolean_mask(inputs[('lidars/%s/pointcloud/intensity' % lidar_name)],
                        valid_points_mask))
    points_elongation.append(
        tf.boolean_mask(
            inputs[('lidars/%s/pointcloud/elongation' % lidar_name)],
            valid_points_mask))
    points_to_lidar_vectors = lidar_location - points_position_current_lidar
    points_normal_direction = points_to_lidar_vectors / tf.expand_dims(
        tf.norm(points_to_lidar_vectors, axis=1), axis=1)
    points_normal.append(points_normal_direction)
    points_in_image_frame_xy.append(
        tf.boolean_mask(
            inputs['lidars/%s/camera_projections/positions' % lidar_name],
            valid_points_mask))
    points_in_image_frame_id.append(
        tf.boolean_mask(inputs['lidars/%s/camera_projections/ids' % lidar_name],
                        valid_points_mask))
  points_position = tf.concat(points_position, axis=0)
  points_intensity = tf.concat(points_intensity, axis=0)
  points_elongation = tf.concat(points_elongation, axis=0)
  points_normal = tf.concat(points_normal, axis=0)
  points_in_image_frame_xy = tf.concat(points_in_image_frame_xy, axis=0)
  points_in_image_frame_id = tf.cast(
      tf.concat(points_in_image_frame_id, axis=0), dtype=tf.int32)
  points_in_image_frame_yx = tf.cast(
      tf.reverse(points_in_image_frame_xy, axis=[-1]), dtype=tf.int32)

  return (points_position, points_intensity, points_elongation, points_normal,
          points_in_image_frame_yx, points_in_image_frame_id)


@gin.configurable(
    'waymo_prepare_lidar_images_and_correspondences', denylist=['inputs'])
def prepare_lidar_images_and_correspondences(
    inputs,
    resized_image_height,
    resized_image_width,
    camera_names=('front', 'front_left', 'front_right', 'side_left',
                  'side_right'),
    lidar_names=('top', 'front', 'side_left', 'side_right', 'rear')):
  """Integrates and returns the lidars, cameras and their correspondences.

  Args:
    inputs: A dictionary containing the images and point / pixel
      correspondences.
    resized_image_height: Target height of the images.
    resized_image_width: Target width of the images.
    camera_names: List of cameras to include images from.
    lidar_names: List of lidars to include point clouds from.

  Returns:
    A tf.float32 tensor of size [num_points, 3] containing point positions.
    A tf.float32 tensor of size [num_points, 1] containing point intensities.
    A tf.float32 tensor of size [num_points, 1] containing point elongations.
    A tf.float32 tensor of size [num_points, 3] containing point normals.
    A tf.float32 tensor of size [num_images, resized_image_height,
      resized_image_width, 3].
    A tf.int32 tensor of size [num_images, num_points, 2].

  Raises:
    ValueError: If camera_names or lidar_names are empty lists.
  """
  if not camera_names:
    raise ValueError('camera_names should contain at least one name.')
  if not lidar_names:
    raise ValueError('lidar_names should contain at least one name.')

  (points_position, points_intensity, points_elongation, points_normal,
   points_in_image_frame_yx, points_in_image_frame_id) = _prepare_lidar_points(
       inputs=inputs, lidar_names=lidar_names)

  images = []
  points_in_image_frame = []

  for camera_name in camera_names:
    image_key = ('cameras/%s/image' % camera_name)
    image_height = tf.shape(inputs[image_key])[0]
    image_width = tf.shape(inputs[image_key])[1]
    height_ratio = tf.cast(
        resized_image_height, dtype=tf.float32) / tf.cast(
            image_height, dtype=tf.float32)
    width_ratio = tf.cast(
        resized_image_width, dtype=tf.float32) / tf.cast(
            image_width, dtype=tf.float32)
    if tf.executing_eagerly():
      resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    else:
      resize_method = tf.image.ResizeMethod.BILINEAR
      if inputs[image_key].dtype in [
          tf.int8, tf.uint8, tf.int16, tf.uint16, tf.int32, tf.int64
      ]:
        resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    images.append(
        tf.image.resize(
            images=inputs[image_key],
            size=[resized_image_height, resized_image_width],
            method=resize_method,
            antialias=True))
    camera_id = tf.cast(inputs[('cameras/%s/id' % camera_name)], dtype=tf.int32)
    valid_points = tf.equal(points_in_image_frame_id, camera_id)
    valid_points = tf.tile(valid_points, [1, 2])
    point_coords = tf.cast(
        tf.cast(points_in_image_frame_yx, dtype=tf.float32) *
        tf.stack([height_ratio, width_ratio]),
        dtype=tf.int32)
    points_in_image_frame_camera = tf.where(
        valid_points, point_coords, -tf.ones_like(valid_points, dtype=tf.int32))
    points_in_image_frame.append(points_in_image_frame_camera)
  num_images = len(images)
  images = tf.stack(images, axis=0)
  images.set_shape([num_images, resized_image_height, resized_image_width, 3])
  points_in_image_frame = tf.stack(points_in_image_frame, axis=0)
  return {
      'points_position': points_position,
      'points_intensity': points_intensity,
      'points_elongation': points_elongation,
      'points_normal': points_normal,
      'view_images': {'rgb_view': images},
      'view_indices_2d': {'rgb_view': points_in_image_frame}
  }


@gin.configurable(
    'waymo_object_per_frame_compute_semantic_labels', denylist=['inputs'])
def compute_semantic_labels(inputs, points_key, box_margin=0.1):
  """Computes ground-truth semantic labels of the points.

  If a point falls inside an object box, assigns it to the label of that box.
  Otherwise the point is assigned to background (unknown) which is label 0.

  Args:
    inputs: A dictionary containing points and objects.
    points_key: A string corresponding to the tensor of point positions in
      inputs.
    box_margin: A margin by which object boxes are grown. Useful to make sure
      points on the object box boundary fall inside the object.

  Returns:
    A tf.int32 tensor of size [num_points, 1] containing point semantic labels.

  Raises:
    ValueError: If the required object or point keys are not in inputs.
  """
  if points_key not in inputs:
    raise ValueError(('points_key: %s not in inputs.' % points_key))
  if 'objects/shape/dimension' not in inputs:
    raise ValueError('`objects/shape/dimension` not in inputs.')
  if 'objects/pose/R' not in inputs:
    raise ValueError('`objects/pose/R` not in inputs.')
  if 'objects/pose/t' not in inputs:
    raise ValueError('`objects/pose/t` not in inputs.')
  if 'objects/category/label' not in inputs:
    raise ValueError('`objects/category/label` not in inputs.')
  point_positions = inputs[points_key]
  boxes_length = inputs['objects/shape/dimension'][:, 0:1]
  boxes_width = inputs['objects/shape/dimension'][:, 1:2]
  boxes_height = inputs['objects/shape/dimension'][:, 2:3]
  boxes_rotation_matrix = inputs['objects/pose/R']
  boxes_center = inputs['objects/pose/t']
  boxes_label = tf.expand_dims(inputs['objects/category/label'], axis=1)
  boxes_label = tf.pad(boxes_label, paddings=[[1, 0], [0, 0]])
  points_box_index = box_utils.map_points_to_boxes(
      points=point_positions,
      boxes_length=boxes_length,
      boxes_height=boxes_height,
      boxes_width=boxes_width,
      boxes_rotation_matrix=boxes_rotation_matrix,
      boxes_center=boxes_center,
      box_margin=box_margin)
  return tf.gather(boxes_label, points_box_index + 1)


@gin.configurable(
    'waymo_object_per_frame_compute_motion_labels',
    denylist=['scene', 'frame0', 'frame1', 'frame_start_index',])
def compute_motion_labels(scene,
                          frame0,
                          frame1,
                          frame_start_index,
                          points_key,
                          box_margin=0.1):
  """Compute motion label for each point.

  Args:
    scene: dict of tensor containing scene.
    frame0: dict of tensor containing points and objects.
    frame1: dict of tensor containing points and objects.
    frame_start_index: starting frame index.
    points_key:  A string corresponding to the tensor of point positions in
      inputs.
    box_margin: A margin value to enlarge box, so that surrounding points are
      included.

  Returns:
    A motion tensor of [N, 3] shape.

  """
  point_positions = frame0[points_key]
  frame0_object_names = frame0['objects/name']
  frame1_object_names = frame1['objects/name']
  bool_matrix = tf.math.equal(
      tf.expand_dims(frame0_object_names, axis=1),
      tf.expand_dims(frame1_object_names, axis=0))
  match_indices = tf.where(bool_matrix)

  # object box level
  box_dimension = tf.gather(
      frame0['objects/shape/dimension'], match_indices[:, 0], axis=0)
  boxes_length = box_dimension[:, 0:1]
  boxes_width = box_dimension[:, 1:2]
  boxes_height = box_dimension[:, 2:3]
  boxes_rotation_matrix = tf.gather(
      frame0['objects/pose/R'], match_indices[:, 0], axis=0)
  boxes_center = tf.gather(
      frame0['objects/pose/t'], match_indices[:, 0], axis=0)
  frame1_box_rotation_matrix = tf.gather(
      frame1['objects/pose/R'], match_indices[:, 1], axis=0)
  frame1_box_center = tf.gather(
      frame1['objects/pose/t'], match_indices[:, 1], axis=0)

  # frame level
  frame0_rotation = scene['frames/pose/R'][frame_start_index]
  frame1_rotation = scene['frames/pose/R'][frame_start_index + 1]
  frame0_translation = scene['frames/pose/t'][frame_start_index]
  frame1_translation = scene['frames/pose/t'][frame_start_index + 1]

  frame1_box_center_global = tf.tensordot(
      frame1_box_center, frame1_rotation, axes=(1, 1)) + frame1_translation
  frame1_box_center_in_frame0 = tf.tensordot(
      frame1_box_center_global - frame0_translation,
      frame0_rotation,
      axes=(1, 0))

  # only find index on boxes that are matched between two frames
  points_box_index = box_utils.map_points_to_boxes(
      points=point_positions,
      boxes_length=boxes_length,
      boxes_height=boxes_height,
      boxes_width=boxes_width,
      boxes_rotation_matrix=boxes_rotation_matrix,
      boxes_center=boxes_center,
      box_margin=box_margin)

  # TODO(huangrui): disappered object box have 0 motion.
  # Probably consider set to nan or ignore_label.

  # 1. gather points in surviving matched box only,
  #    and replicate rotation/t to same length;
  # 2. get points in box frame, apply new rotation/t per point;
  # 3. new location minus old location -> motion vector;
  # 4. scatter it to a larger motion_vector with 0 for
  #    points ouside of matched boxes.

  # Need to limit boxes to those matched boxes.
  # otherwise the points_box_index will contain useless box.

  # index in all point array, of points that are inside the box.
  points_inside_box_index = tf.where(points_box_index + 1)[:, 0]
  box_index = tf.gather(points_box_index, points_inside_box_index)
  points_inside_box = tf.gather(point_positions, points_inside_box_index)
  box_rotation_per_point = tf.gather(boxes_rotation_matrix, box_index)
  box_center_per_point = tf.gather(boxes_center, box_index)
  # Tensor [N, 3, 3] and [N, 3]. note we are transform points reversely.
  points_in_box_frame = tf.einsum('ikj,ik->ij', box_rotation_per_point,
                                  points_inside_box - box_center_per_point)

  # Transform rotation of box from frame1 coordinate to frame0 coordinate
  # note, transpose is implemented via changing summation axis
  frame1_box_rotation_matrix_global = tf.transpose(
      tf.tensordot(frame1_rotation, frame1_box_rotation_matrix, axes=(1, 1)),
      perm=(1, 0, 2))
  frame1_box_rotation_matrix_in_frame0 = tf.transpose(
      tf.tensordot(
          frame0_rotation, frame1_box_rotation_matrix_global, axes=(0, 1)),
      perm=(1, 0, 2))

  # this is the points_position_after_following_frame1_box's motion.
  frame1_box_rotation_in_frame0_per_point = tf.gather(
      frame1_box_rotation_matrix_in_frame0, box_index)
  frame1_box_center_in_frame0_per_point = tf.gather(frame1_box_center_in_frame0,
                                                    box_index)

  points_in_box_frame1 = tf.einsum(
      'ijk,ik->ij', frame1_box_rotation_in_frame0_per_point,
      points_in_box_frame) + frame1_box_center_in_frame0_per_point
  motion_vector = points_in_box_frame1 - points_inside_box

  scattered_vector = tf.scatter_nd(
      indices=tf.expand_dims(points_inside_box_index, axis=1),
      updates=motion_vector,
      shape=tf.shape(point_positions, out_type=tf.dtypes.int64))

  return scattered_vector


@gin.configurable
def transform_pointcloud_to_another_frame(scene, point_positions,
                                          frame_source_index,
                                          frame_target_index):
  """transform each point in target_frame to source_frame coordinate based on their relative transformation.

  Args:
    scene: dict of tensor containing scene.
    point_positions: the tensor of point positions in inputs.
    frame_source_index: source frame index.
    frame_target_index: target frame index.

  Returns:
    A point cloud from frame_target warpped, of [N, 3] shape.

  """
  world_rotation_frame0 = scene['frames/pose/R'][frame_source_index]
  world_rotation_frame1 = scene['frames/pose/R'][frame_target_index]
  world_translation_frame0 = scene['frames/pose/t'][frame_source_index]
  world_translation_frame1 = scene['frames/pose/t'][frame_target_index]

  point_positions_world = tf.tensordot(
      point_positions, world_rotation_frame1,
      axes=(1, 1)) + world_translation_frame1
  point_positions_frame0 = tf.tensordot(
      point_positions_world - world_translation_frame0,
      world_rotation_frame0,
      axes=(1, 0))
  return point_positions_frame0
