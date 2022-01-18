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

"""Functions for preparing data to be compatible with object detection pipeline.

Functions to prepare Waymo, scannet and kitti datasets.
"""
import enum
import gin
import gin.tf
import tensorflow as tf
import tensorflow_datasets as tfds
from tf3d import standard_fields
# TODO(alirezafathi): Remove internal mark when dataset files are moved to tf3d.
from tf3d.datasets.specs import waymo_frames
from tf3d.utils import projections


class ObjectDifficulty(enum.IntEnum):
  SUPER_HARD = 0
  HARD = 1
  MODERATE = 2
  EASY = 3


def _random_string_generator(num_numbers=5, max_number_value=100000):
  string_tensors = []
  for _ in range(num_numbers):
    random_number = tf.random.uniform([],
                                      minval=0,
                                      maxval=max_number_value,
                                      dtype=tf.int32)
    string_tensors.append(tf.strings.as_string(random_number))
  return tf.strings.join(string_tensors)


@gin.configurable
def prepare_scannet_scene_dataset(inputs, valid_object_classes=None):
  """Maps the fields from loaded input to standard fields.

  Args:
    inputs: A dictionary of input tensors.
    valid_object_classes: List of valid object classes. if None, it is ignored.

  Returns:
    A dictionary of input tensors with standard field names.
  """
  prepared_inputs = {}
  if 'mesh/vertices/positions' in inputs:
    prepared_inputs[standard_fields.InputDataFields
                    .point_positions] = inputs['mesh/vertices/positions']
  if 'mesh/vertices/normals' in inputs:
    prepared_inputs[standard_fields.InputDataFields
                    .point_normals] = inputs['mesh/vertices/normals']
    prepared_inputs[standard_fields.InputDataFields.point_normals] = tf.where(
        tf.math.is_nan(
            prepared_inputs[standard_fields.InputDataFields.point_normals]),
        tf.zeros_like(
            prepared_inputs[standard_fields.InputDataFields.point_normals]),
        prepared_inputs[standard_fields.InputDataFields.point_normals])
  if 'mesh/vertices/colors' in inputs:
    prepared_inputs[standard_fields.InputDataFields
                    .point_colors] = inputs['mesh/vertices/colors'][:, 0:3]
    prepared_inputs[standard_fields.InputDataFields.point_colors] = tf.cast(
        prepared_inputs[standard_fields.InputDataFields.point_colors],
        dtype=tf.float32)
    prepared_inputs[standard_fields.InputDataFields.point_colors] *= (2.0 /
                                                                      255.0)
    prepared_inputs[standard_fields.InputDataFields.point_colors] -= 1.0
  if 'scene_name' in inputs:
    prepared_inputs[standard_fields.InputDataFields
                    .camera_image_name] = inputs['scene_name']
  if 'mesh/vertices/semantic_labels' in inputs:
    prepared_inputs[
        standard_fields.InputDataFields
        .object_class_points] = inputs['mesh/vertices/semantic_labels']
  if 'mesh/vertices/instance_labels' in inputs:
    prepared_inputs[
        standard_fields.InputDataFields.object_instance_id_points] = tf.reshape(
            inputs['mesh/vertices/instance_labels'], [-1])

  if valid_object_classes is not None:
    valid_objects_mask = tf.cast(
        tf.zeros_like(
            prepared_inputs[
                standard_fields.InputDataFields.object_class_points],
            dtype=tf.int32),
        dtype=tf.bool)
    for object_class in valid_object_classes:
      valid_objects_mask = tf.logical_or(
          valid_objects_mask,
          tf.equal(
              prepared_inputs[
                  standard_fields.InputDataFields.object_class_points],
              object_class))
    valid_objects_mask = tf.cast(
        valid_objects_mask,
        dtype=prepared_inputs[
            standard_fields.InputDataFields.object_class_points].dtype)
    prepared_inputs[standard_fields.InputDataFields
                    .object_class_points] *= valid_objects_mask
  return prepared_inputs


@gin.configurable
def prepare_scannet_frame_dataset(inputs,
                                  min_pixel_depth=0.3,
                                  max_pixel_depth=6.0,
                                  valid_object_classes=None):
  """Maps the fields from loaded input to standard fields.

  Args:
    inputs: A dictionary of input tensors.
    min_pixel_depth: Pixels with depth values less than this are pruned.
    max_pixel_depth: Pixels with depth values more than this are pruned.
    valid_object_classes: List of valid object classes. if None, it is ignored.

  Returns:
    A dictionary of input tensors with standard field names.
  """
  prepared_inputs = {}
  if 'cameras/rgbd_camera/intrinsics/K' not in inputs:
    raise ValueError('Intrinsic matrix is missing.')
  if 'cameras/rgbd_camera/extrinsics/R' not in inputs:
    raise ValueError('Extrinsic rotation matrix is missing.')
  if 'cameras/rgbd_camera/extrinsics/t' not in inputs:
    raise ValueError('Extrinsics translation is missing.')
  if 'cameras/rgbd_camera/depth_image' not in inputs:
    raise ValueError('Depth image is missing.')
  if 'cameras/rgbd_camera/color_image' not in inputs:
    raise ValueError('Color image is missing.')
  if 'frame_name' in inputs:
    prepared_inputs[standard_fields.InputDataFields
                    .camera_image_name] = inputs['frame_name']
  camera_intrinsics = inputs['cameras/rgbd_camera/intrinsics/K']
  depth_image = inputs['cameras/rgbd_camera/depth_image']
  image_height = tf.shape(depth_image)[0]
  image_width = tf.shape(depth_image)[1]
  x, y = tf.meshgrid(
      tf.range(image_width), tf.range(image_height), indexing='xy')
  x = tf.reshape(tf.cast(x, dtype=tf.float32) + 0.5, [-1, 1])
  y = tf.reshape(tf.cast(y, dtype=tf.float32) + 0.5, [-1, 1])
  point_positions = projections.image_frame_to_camera_frame(
      image_frame=tf.concat([x, y], axis=1),
      camera_intrinsics=camera_intrinsics)
  rotate_world_to_camera = inputs['cameras/rgbd_camera/extrinsics/R']
  translate_world_to_camera = inputs['cameras/rgbd_camera/extrinsics/t']
  point_positions = projections.to_world_frame(
      camera_frame_points=point_positions,
      rotate_world_to_camera=rotate_world_to_camera,
      translate_world_to_camera=translate_world_to_camera)
  prepared_inputs[standard_fields.InputDataFields
                  .point_positions] = point_positions * tf.reshape(
                      depth_image, [-1, 1])
  depth_values = tf.reshape(depth_image, [-1])
  valid_depth_mask = tf.logical_and(
      tf.greater_equal(depth_values, min_pixel_depth),
      tf.less_equal(depth_values, max_pixel_depth))
  prepared_inputs[standard_fields.InputDataFields.point_colors] = tf.reshape(
      tf.cast(inputs['cameras/rgbd_camera/color_image'], dtype=tf.float32),
      [-1, 3])
  prepared_inputs[standard_fields.InputDataFields.point_colors] *= (2.0 / 255.0)
  prepared_inputs[standard_fields.InputDataFields.point_colors] -= 1.0
  prepared_inputs[
      standard_fields.InputDataFields.point_positions] = tf.boolean_mask(
          prepared_inputs[standard_fields.InputDataFields.point_positions],
          valid_depth_mask)
  prepared_inputs[
      standard_fields.InputDataFields.point_colors] = tf.boolean_mask(
          prepared_inputs[standard_fields.InputDataFields.point_colors],
          valid_depth_mask)
  if 'cameras/rgbd_camera/semantic_image' in inputs:
    prepared_inputs[
        standard_fields.InputDataFields.object_class_points] = tf.cast(
            tf.reshape(inputs['cameras/rgbd_camera/semantic_image'], [-1, 1]),
            dtype=tf.int32)
    prepared_inputs[
        standard_fields.InputDataFields.object_class_points] = tf.boolean_mask(
            prepared_inputs[
                standard_fields.InputDataFields.object_class_points],
            valid_depth_mask)
  if 'cameras/rgbd_camera/instance_image' in inputs:
    prepared_inputs[
        standard_fields.InputDataFields.object_instance_id_points] = tf.cast(
            tf.reshape(inputs['cameras/rgbd_camera/instance_image'], [-1]),
            dtype=tf.int32)
    prepared_inputs[standard_fields.InputDataFields
                    .object_instance_id_points] = tf.boolean_mask(
                        prepared_inputs[standard_fields.InputDataFields
                                        .object_instance_id_points],
                        valid_depth_mask)

  if valid_object_classes is not None:
    valid_objects_mask = tf.cast(
        tf.zeros_like(
            prepared_inputs[
                standard_fields.InputDataFields.object_class_points],
            dtype=tf.int32),
        dtype=tf.bool)
    for object_class in valid_object_classes:
      valid_objects_mask = tf.logical_or(
          valid_objects_mask,
          tf.equal(
              prepared_inputs[
                  standard_fields.InputDataFields.object_class_points],
              object_class))
    valid_objects_mask = tf.cast(
        valid_objects_mask,
        dtype=prepared_inputs[
            standard_fields.InputDataFields.object_class_points].dtype)
    prepared_inputs[standard_fields.InputDataFields
                    .object_class_points] *= valid_objects_mask
  return prepared_inputs


@gin.configurable
def prepare_waymo_open_dataset(inputs,
                               valid_object_classes=None,
                               max_object_distance_from_source=74.88):
  """Maps the fields from loaded input to standard fields.

  Args:
    inputs: A dictionary of input tensors.
    valid_object_classes: List of valid object classes. if None, it is ignored.
    max_object_distance_from_source: Maximum distance of objects from source. It
      will be ignored if None.

  Returns:
    A dictionary of input tensors with standard field names.
  """
  prepared_inputs = {}
  if standard_fields.InputDataFields.point_positions in inputs:
    prepared_inputs[standard_fields.InputDataFields.point_positions] = inputs[
        standard_fields.InputDataFields.point_positions]
  if standard_fields.InputDataFields.point_intensities in inputs:
    prepared_inputs[standard_fields.InputDataFields.point_intensities] = inputs[
        standard_fields.InputDataFields.point_intensities]
  if standard_fields.InputDataFields.point_elongations in inputs:
    prepared_inputs[standard_fields.InputDataFields.point_elongations] = inputs[
        standard_fields.InputDataFields.point_elongations]
  if standard_fields.InputDataFields.point_normals in inputs:
    prepared_inputs[standard_fields.InputDataFields.point_normals] = inputs[
        standard_fields.InputDataFields.point_normals]
  if 'cameras/front/intrinsics/K' in inputs:
    prepared_inputs[standard_fields.InputDataFields
                    .camera_intrinsics] = inputs['cameras/front/intrinsics/K']
  if 'cameras/front/extrinsics/R' in inputs:
    prepared_inputs[
        standard_fields.InputDataFields
        .camera_rotation_matrix] = inputs['cameras/front/extrinsics/R']
  if 'cameras/front/extrinsics/t' in inputs:
    prepared_inputs[standard_fields.InputDataFields
                    .camera_translation] = inputs['cameras/front/extrinsics/t']
  if 'cameras/front/image' in inputs:
    prepared_inputs[standard_fields.InputDataFields
                    .camera_image] = inputs['cameras/front/image']
    prepared_inputs[standard_fields.InputDataFields
                    .camera_raw_image] = inputs['cameras/front/image']
    prepared_inputs[standard_fields.InputDataFields
                    .camera_original_image] = inputs['cameras/front/image']
  if 'scene_name' in inputs and 'frame_name' in inputs:
    prepared_inputs[
        standard_fields.InputDataFields.camera_image_name] = tf.strings.join(
            [inputs['scene_name'], inputs['frame_name']], separator='_')
  if 'objects/pose/R' in inputs:
    prepared_inputs[standard_fields.InputDataFields
                    .objects_rotation_matrix] = inputs['objects/pose/R']
  if 'objects/pose/t' in inputs:
    prepared_inputs[standard_fields.InputDataFields
                    .objects_center] = inputs['objects/pose/t']
  if 'objects/shape/dimension' in inputs:
    prepared_inputs[
        standard_fields.InputDataFields.objects_length] = tf.reshape(
            inputs['objects/shape/dimension'][:, 0], [-1, 1])
    prepared_inputs[standard_fields.InputDataFields.objects_width] = tf.reshape(
        inputs['objects/shape/dimension'][:, 1], [-1, 1])
    prepared_inputs[
        standard_fields.InputDataFields.objects_height] = tf.reshape(
            inputs['objects/shape/dimension'][:, 2], [-1, 1])
  if 'objects/category/label' in inputs:
    prepared_inputs[standard_fields.InputDataFields.objects_class] = tf.reshape(
        inputs['objects/category/label'], [-1, 1])
  if valid_object_classes is not None:
    valid_objects_mask = tf.cast(
        tf.zeros_like(
            prepared_inputs[standard_fields.InputDataFields.objects_class],
            dtype=tf.int32),
        dtype=tf.bool)
    for object_class in valid_object_classes:
      valid_objects_mask = tf.logical_or(
          valid_objects_mask,
          tf.equal(
              prepared_inputs[standard_fields.InputDataFields.objects_class],
              object_class))
    valid_objects_mask = tf.reshape(valid_objects_mask, [-1])
    for key in standard_fields.get_input_object_fields():
      if key in prepared_inputs:
        prepared_inputs[key] = tf.boolean_mask(prepared_inputs[key],
                                               valid_objects_mask)

  if max_object_distance_from_source is not None:
    if standard_fields.InputDataFields.objects_center in prepared_inputs:
      object_distances = tf.norm(
          prepared_inputs[standard_fields.InputDataFields.objects_center][:,
                                                                          0:2],
          axis=1)
      valid_mask = tf.less(object_distances, max_object_distance_from_source)
      for key in standard_fields.get_input_object_fields():
        if key in prepared_inputs:
          prepared_inputs[key] = tf.boolean_mask(prepared_inputs[key],
                                                 valid_mask)

  return prepared_inputs


@gin.configurable
def prepare_kitti_dataset(inputs, valid_object_classes=None):
  """Maps the fields from loaded input to standard fields.

  Args:
    inputs: A dictionary of input tensors.
    valid_object_classes: List of valid object classes. if None, it is ignored.

  Returns:
    A dictionary of input tensors with standard field names.
  """
  prepared_inputs = {}
  prepared_inputs[standard_fields.InputDataFields.point_positions] = inputs[
      standard_fields.InputDataFields.point_positions]
  prepared_inputs[standard_fields.InputDataFields.point_intensities] = inputs[
      standard_fields.InputDataFields.point_intensities]
  prepared_inputs[standard_fields.InputDataFields
                  .camera_intrinsics] = inputs['cameras/cam02/intrinsics/K']
  prepared_inputs[standard_fields.InputDataFields.
                  camera_rotation_matrix] = inputs['cameras/cam02/extrinsics/R']
  prepared_inputs[standard_fields.InputDataFields
                  .camera_translation] = inputs['cameras/cam02/extrinsics/t']
  prepared_inputs[standard_fields.InputDataFields
                  .camera_image] = inputs['cameras/cam02/image']
  prepared_inputs[standard_fields.InputDataFields
                  .camera_raw_image] = inputs['cameras/cam02/image']
  prepared_inputs[standard_fields.InputDataFields
                  .camera_original_image] = inputs['cameras/cam02/image']
  if 'scene_name' in inputs and 'frame_name' in inputs:
    prepared_inputs[
        standard_fields.InputDataFields.camera_image_name] = tf.strings.join(
            [inputs['scene_name'], inputs['frame_name']], separator='_')
  if 'objects/pose/R' in inputs:
    prepared_inputs[standard_fields.InputDataFields
                    .objects_rotation_matrix] = inputs['objects/pose/R']
  if 'objects/pose/t' in inputs:
    prepared_inputs[standard_fields.InputDataFields
                    .objects_center] = inputs['objects/pose/t']
  if 'objects/shape/dimension' in inputs:
    prepared_inputs[
        standard_fields.InputDataFields.objects_length] = tf.reshape(
            inputs['objects/shape/dimension'][:, 0], [-1, 1])
    prepared_inputs[standard_fields.InputDataFields.objects_width] = tf.reshape(
        inputs['objects/shape/dimension'][:, 1], [-1, 1])
    prepared_inputs[
        standard_fields.InputDataFields.objects_height] = tf.reshape(
            inputs['objects/shape/dimension'][:, 2], [-1, 1])
  if 'objects/category/label' in inputs:
    prepared_inputs[standard_fields.InputDataFields.objects_class] = tf.reshape(
        inputs['objects/category/label'], [-1, 1])
  if valid_object_classes is not None:
    valid_objects_mask = tf.cast(
        tf.zeros_like(
            prepared_inputs[standard_fields.InputDataFields.objects_class],
            dtype=tf.int32),
        dtype=tf.bool)
    for object_class in valid_object_classes:
      valid_objects_mask = tf.logical_or(
          valid_objects_mask,
          tf.equal(
              prepared_inputs[standard_fields.InputDataFields.objects_class],
              object_class))
    valid_objects_mask = tf.reshape(valid_objects_mask, [-1])
    for key in standard_fields.get_input_object_fields():
      if key in prepared_inputs:
        prepared_inputs[key] = tf.boolean_mask(prepared_inputs[key],
                                               valid_objects_mask)

  return prepared_inputs


@gin.configurable
def prepare_proxy_dataset(inputs):
  """Maps the fields from loaded input to standard fields.

  Args:
    inputs: A dictionary of input tensors.

  Returns:
    A dictionary of input tensors with standard field names.
  """
  prepared_inputs = {}

  # Points
  prepared_inputs[standard_fields.InputDataFields.point_positions] = inputs[
      standard_fields.InputDataFields.point_positions]
  prepared_inputs[standard_fields.InputDataFields.point_intensities] = inputs[
      standard_fields.InputDataFields.point_intensities]

  # Camera
  prepared_inputs[
      standard_fields.InputDataFields.camera_intrinsics] = tf.reshape(
          inputs['camera_intrinsics'], [3, 3])
  prepared_inputs[
      standard_fields.InputDataFields.camera_rotation_matrix] = tf.reshape(
          inputs['camera_rotation_matrix'], [3, 3])
  prepared_inputs[
      standard_fields.InputDataFields.camera_translation] = tf.reshape(
          inputs['camera_translation'], [3])
  prepared_inputs[
      standard_fields.InputDataFields.camera_image] = inputs['image']
  prepared_inputs[
      standard_fields.InputDataFields.camera_raw_image] = inputs['image']
  prepared_inputs[
      standard_fields.InputDataFields.camera_original_image] = inputs['image']
  prepared_inputs[standard_fields.InputDataFields
                  .camera_image_name] = _random_string_generator()

  # objects pose
  prepared_inputs[
      standard_fields.InputDataFields.objects_rotation_matrix] = tf.reshape(
          inputs['objects_rotation'], [-1, 3, 3])
  prepared_inputs[standard_fields.InputDataFields.objects_center] = tf.reshape(
      inputs['objects_center'], [-1, 3])

  # objects size
  prepared_inputs[standard_fields.InputDataFields.objects_length] = tf.reshape(
      inputs['objects_length'], [-1, 1])
  prepared_inputs[standard_fields.InputDataFields.objects_width] = tf.reshape(
      inputs['objects_width'], [-1, 1])
  prepared_inputs[standard_fields.InputDataFields.objects_height] = tf.reshape(
      inputs['objects_height'], [-1, 1])

  # labels
  prepared_inputs[standard_fields.InputDataFields.objects_class] = tf.reshape(
      inputs['objects_class'], [-1, 1])

  return prepared_inputs


def compute_kitti_difficulty(boxes, occlusions, truncations, image_height):
  """Computes box difficulty as Hard(1), Moderate(2), Easy(3) or 0 (Super hard).

  Easy: height >=40 Px, occlusion <= 0, truncation <= 0.15
  Moderate: height >=25 Px, occlusion <= 1, truncation <= 0.30
  Hard: height >=25 Px, occlusion <= 2, truncation <= 0.50

  Note that 'Hard' box is also 'Moderate' and 'Easy'.

  Returns a (N, 1) tensor containing object difficulty with following labelmap:
    0: SuperHard
    1: Hard
    2: Moderate
    3: Easy

  TODO(abhijitkundu): Since difficulty level is very specific to kitti, this
  function should be in kitti evaluation rather than detection preprocessor.

  Args:
    boxes: (N, 4) tensor of 2d boxes with [ymin, xmin, ymax, xmax] each row.
    occlusions: (N, 1) tensor containing box occlusion level
    truncations: (N, 1) tensor containing box truncation level
    image_height: Image height.

  Returns:
  A (N, 1) int32 tensor containing per box difficulty labels with 0 (SuperHard),
  1 (Hard), 2 (Moderate) and 3 (Easy).
  """
  # box heights in pixels
  heights = tf.reshape((boxes[:, 2] - boxes[:, 0]), [-1, 1]) * tf.cast(
      image_height, dtype=tf.float32)

  # compute binary masks for each difficulty level
  is_easy = (heights >= 40.0) & (occlusions <= 0) & (truncations <= 0.15)
  is_moderate = (heights >= 25.0) & (occlusions <= 1) & (truncations <= 0.30)
  is_hard = (heights >= 25.0) & (occlusions <= 2) & (truncations <= 0.50)

  # set difficulty map
  difficulty = tf.maximum(
      tf.maximum(
          tf.cast(is_hard, dtype=tf.int32) * ObjectDifficulty.HARD,
          tf.cast(is_moderate, dtype=tf.int32) * ObjectDifficulty.MODERATE),
      tf.cast(is_easy, dtype=tf.int32) * ObjectDifficulty.EASY)
  return difficulty


def get_waymo_per_frame_with_prediction_feature_spec(
    num_object_classes,
    encoded_features_dimension,
    include_encoded_features=True):
  """Returns a tfds feature spec with regular per frame entries and predictions.

  Args:
    num_object_classes: Number of object classes.
    encoded_features_dimension: Encoded features dimension.
    include_encoded_features: If True, it will include encoded features.
      Otherwise, it will not include them.

  Returns:
    A tfds feature spec.
  """
  prediction_feature_dict = {
      standard_fields.DetectionResultFields.object_rotation_matrix_points:
          tfds.features.Tensor(shape=(None, 3, 3), dtype=tf.float32),
      standard_fields.DetectionResultFields.object_length_points:
          tfds.features.Tensor(shape=(None, 1), dtype=tf.float32),
      standard_fields.DetectionResultFields.object_height_points:
          tfds.features.Tensor(shape=(None, 1), dtype=tf.float32),
      standard_fields.DetectionResultFields.object_width_points:
          tfds.features.Tensor(shape=(None, 1), dtype=tf.float32),
      standard_fields.DetectionResultFields.object_center_points:
          tfds.features.Tensor(shape=(None, 3), dtype=tf.float32),
      standard_fields.DetectionResultFields.object_semantic_points:
          tfds.features.Tensor(
              shape=(None, num_object_classes), dtype=tf.float32),
      standard_fields.DetectionResultFields.objects_rotation_matrix:
          tfds.features.Tensor(shape=(None, 3, 3), dtype=tf.float32),
      standard_fields.DetectionResultFields.objects_length:
          tfds.features.Tensor(shape=(None, 1), dtype=tf.float32),
      standard_fields.DetectionResultFields.objects_height:
          tfds.features.Tensor(shape=(None, 1), dtype=tf.float32),
      standard_fields.DetectionResultFields.objects_width:
          tfds.features.Tensor(shape=(None, 1), dtype=tf.float32),
      standard_fields.DetectionResultFields.objects_center:
          tfds.features.Tensor(shape=(None, 3), dtype=tf.float32),
      standard_fields.DetectionResultFields.objects_class:
          tfds.features.Tensor(shape=(None, 1), dtype=tf.float32),
      standard_fields.DetectionResultFields.objects_score:
          tfds.features.Tensor(shape=(None, 1), dtype=tf.float32),
  }
  if include_encoded_features:
    prediction_feature_dict[standard_fields.DetectionResultFields
                            .encoded_features_points] = tfds.features.Tensor(
                                shape=(None, encoded_features_dimension),
                                dtype=tf.float32)
    prediction_feature_dict[standard_fields.DetectionResultFields
                            .objects_encoded_features] = tfds.features.Tensor(
                                shape=(None, encoded_features_dimension),
                                dtype=tf.float32)
  prediction_feature_spec = tfds.features.FeaturesDict(prediction_feature_dict)
  output_feature_spec_dict = {
      k: v for k, v in waymo_frames.FRAME_FEATURE_SPEC.items()
  }
  output_feature_spec_dict['predictions'] = prediction_feature_spec
  return tfds.features.FeaturesDict(output_feature_spec_dict)
