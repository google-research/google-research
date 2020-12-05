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

"""3D Object Detection preprocessing functions."""

import logging
import gin
import gin.tf
import tensorflow as tf
from tf3d import standard_fields
from tf3d.utils import preprocessor_utils
from object_detection.utils import shape_utils


_OBJECT_KEYS = [
    standard_fields.InputDataFields.objects_length,
    standard_fields.InputDataFields.objects_height,
    standard_fields.InputDataFields.objects_width,
    standard_fields.InputDataFields.objects_center,
    standard_fields.InputDataFields.objects_rotation_matrix,
    standard_fields.InputDataFields.objects_class,
    standard_fields.InputDataFields.objects_difficulty,
    standard_fields.InputDataFields.objects_has_3d_info,
]

_POINTS_WITHIN_IMAGE_BY_MARGIN = 'points_within_image_by_margin'


def _filter_valid_objects(inputs):
  """Removes the objects that do not contain 3d info.

  Args:
    inputs: A dictionary containing input tensors.
  """
  if standard_fields.InputDataFields.objects_class not in inputs:
    return

  valid_objects_mask = tf.reshape(
      tf.greater(inputs[standard_fields.InputDataFields.objects_class], 0),
      [-1])
  if standard_fields.InputDataFields.objects_has_3d_info in inputs:
    objects_with_3d_info = tf.reshape(
        tf.cast(
            inputs[standard_fields.InputDataFields.objects_has_3d_info],
            dtype=tf.bool), [-1])
    valid_objects_mask = tf.logical_and(objects_with_3d_info,
                                        valid_objects_mask)
  if standard_fields.InputDataFields.objects_difficulty in inputs:
    valid_objects_mask = tf.logical_and(
        valid_objects_mask,
        tf.greater(
            tf.reshape(
                inputs[standard_fields.InputDataFields.objects_difficulty],
                [-1]), 0))
  for key in _OBJECT_KEYS:
    if key in inputs:
      inputs[key] = tf.boolean_mask(inputs[key], valid_objects_mask)


def _cast_objects_class(inputs):
  """Set the object rotation and semantic properties.

  Args:
    inputs: A dictionary containing input tensors.
  """
  if standard_fields.InputDataFields.objects_class in inputs:
    inputs[standard_fields.InputDataFields.objects_class] = tf.cast(
        inputs[standard_fields.InputDataFields.objects_class], dtype=tf.int32)


def _randomly_transform_points_boxes(
    mesh_inputs, object_inputs, x_min_degree_rotation, x_max_degree_rotation,
    y_min_degree_rotation, y_max_degree_rotation, z_min_degree_rotation,
    z_max_degree_rotation, rotation_center, min_scale_ratio, max_scale_ratio,
    translation_range):
  """Randomly rotate and translate points and boxes.

  Args:
    mesh_inputs: A dictionary containing mesh input tensors.
    object_inputs: A dictionary containing object input tensors.
    x_min_degree_rotation: Min degree of rotation around the x axis.
    x_max_degree_rotation: Max degree of ratation around the x axis.
    y_min_degree_rotation: Min degree of rotation around the y axis.
    y_max_degree_rotation: Max degree of ratation around the y axis.
    z_min_degree_rotation: Min degree of rotation around the z axis.
    z_max_degree_rotation: Max degree of ratation around the z axis.
    rotation_center: A 3d point that points are rotated around that.
    min_scale_ratio: Minimum scale ratio.
    max_scale_ratio: Maximum scale ratio.
    translation_range: A float value corresponding to the magnitude of
      translation in x, y, and z directions. If None, there will not be a
      translation.
  """
  # Random rotation of points in camera frame
  preprocessor_utils.rotate_randomly(
      mesh_inputs=mesh_inputs,
      object_inputs=object_inputs,
      x_min_degree_rotation=x_min_degree_rotation,
      x_max_degree_rotation=x_max_degree_rotation,
      y_min_degree_rotation=y_min_degree_rotation,
      y_max_degree_rotation=y_max_degree_rotation,
      z_min_degree_rotation=z_min_degree_rotation,
      z_max_degree_rotation=z_max_degree_rotation,
      rotation_center=rotation_center)
  # Random scaling
  preprocessor_utils.randomly_scale_points_and_objects(
      mesh_inputs=mesh_inputs,
      object_inputs=object_inputs,
      min_scale_ratio=min_scale_ratio,
      max_scale_ratio=max_scale_ratio)
  # Random translation
  if translation_range is not None:
    if translation_range < 0:
      raise ValueError('Translation range should be positive')
    preprocessor_utils.translate_randomly(
        mesh_inputs=mesh_inputs,
        object_inputs=object_inputs,
        delta_x_min=-translation_range,
        delta_x_max=translation_range,
        delta_y_min=-translation_range,
        delta_y_max=translation_range,
        delta_z_min=-translation_range,
        delta_z_max=translation_range)


def _transfer_object_properties_to_points(inputs):
  """Sets the object properties for the points that fall inside objects.

  Args:
    inputs: A dictionary containing input tensors.
  """
  dic = {}
  if standard_fields.InputDataFields.objects_class in inputs:
    dic[standard_fields.InputDataFields.object_class_points] = inputs[
        standard_fields.InputDataFields.objects_class]
  if standard_fields.InputDataFields.objects_center in inputs:
    dic[standard_fields.InputDataFields.object_center_points] = inputs[
        standard_fields.InputDataFields.objects_center]
  if standard_fields.InputDataFields.objects_length in inputs:
    dic[standard_fields.InputDataFields.object_length_points] = inputs[
        standard_fields.InputDataFields.objects_length]
  if standard_fields.InputDataFields.objects_height in inputs:
    dic[standard_fields.InputDataFields.object_height_points] = inputs[
        standard_fields.InputDataFields.objects_height]
  if standard_fields.InputDataFields.objects_width in inputs:
    dic[standard_fields.InputDataFields.object_width_points] = inputs[
        standard_fields.InputDataFields.objects_width]
  if standard_fields.InputDataFields.objects_rotation_matrix in inputs:
    dic[standard_fields.InputDataFields.object_rotation_matrix_points] = inputs[
        standard_fields.InputDataFields.objects_rotation_matrix]

  for key, value in dic.items():
    if len(value.get_shape().as_list()) == 1:
      paddings = [[1, 0]]
    elif len(value.get_shape().as_list()) == 2:
      paddings = [[1, 0], [0, 0]]
    elif len(value.get_shape().as_list()) == 3:
      paddings = [[1, 0], [0, 0], [0, 0]]
    else:
      raise ValueError(('Invalid shape for %s' % key))
    temp_tensor = tf.pad(value, paddings=paddings)
    id_mapping = tf.reshape(
        inputs[standard_fields.InputDataFields.object_instance_id_points], [-1])
    inputs[key] = tf.gather(temp_tensor, id_mapping)


def _pad_or_clip_point_properties(inputs, pad_or_clip_size):
  """Pads or clips the inputs point properties.

  If pad_or_clip_size is None, it won't perform any action.

  Args:
    inputs: A dictionary containing input tensors.
    pad_or_clip_size: Number of target points to pad or clip to. If None, it
      will not perform the padding.
  """
  inputs[standard_fields.InputDataFields.num_valid_points] = tf.shape(
      inputs[standard_fields.InputDataFields.point_positions])[0]
  if pad_or_clip_size is not None:
    inputs[standard_fields.InputDataFields.num_valid_points] = tf.minimum(
        inputs[standard_fields.InputDataFields.num_valid_points],
        pad_or_clip_size)
    for key in sorted(standard_fields.get_input_point_fields()):
      if key == standard_fields.InputDataFields.num_valid_points:
        continue
      if key in inputs:
        tensor_rank = len(inputs[key].get_shape().as_list())
        padding_shape = [pad_or_clip_size]
        for i in range(1, tensor_rank):
          padding_shape.append(inputs[key].get_shape().as_list()[i])
        inputs[key] = shape_utils.pad_or_clip_nd(
            tensor=inputs[key], output_shape=padding_shape)


def split_inputs(inputs,
                 input_field_mapping_fn,
                 image_preprocess_fn_dic,
                 images_points_correspondence_fn):
  """Splits inputs to view_image_inputs, view_indices_2d_inputs, mesh_inputs.

  Args:
    inputs: Input dictionary.
    input_field_mapping_fn: A function that maps the input fields to the
      fields expected by object detection pipeline.
    image_preprocess_fn_dic: A dictionary of image preprocessing functions.
    images_points_correspondence_fn: A function that returns image and points
      correspondences.

  Returns:
    view_image_inputs: A dictionary containing image inputs.
    view_indices_2d_inputs: A dictionary containing indices 2d inputs.
    mesh_inputs: A dictionary containing mesh inputs.
    object_inputs: A dictionary containing object inputs.
    non_tensor_inputs: Other inputs.
  """
  # Initializing empty dictionary for mesh, image, indices_2d and non tensor
  # inputs.
  non_tensor_inputs = {}
  view_image_inputs = {}
  view_indices_2d_inputs = {}
  mesh_inputs = {}
  object_inputs = {}
  if image_preprocess_fn_dic is None:
    image_preprocess_fn_dic = {}
  # Acquire point / image correspondences.
  if images_points_correspondence_fn is not None:
    fn_outputs = images_points_correspondence_fn(inputs)
    if 'points_position' in fn_outputs:
      inputs[standard_fields.InputDataFields
             .point_positions] = fn_outputs['points_position']
    if 'points_intensity' in fn_outputs:
      inputs[standard_fields.InputDataFields
             .point_intensities] = fn_outputs['points_intensity']
    if 'points_elongation' in fn_outputs:
      inputs[standard_fields.InputDataFields
             .point_elongations] = fn_outputs['points_elongation']
    if 'points_normal' in fn_outputs:
      inputs[standard_fields.InputDataFields
             .point_normals] = fn_outputs['points_normal']
    if 'points_color' in fn_outputs:
      inputs[standard_fields.InputDataFields
             .point_colors] = fn_outputs['points_color']
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

  if input_field_mapping_fn is not None:
    inputs = input_field_mapping_fn(inputs)

  # Setting mesh inputs
  mesh_keys = []
  for key in standard_fields.get_input_point_fields():
    if key in inputs:
      mesh_keys.append(key)
  object_keys = []
  for key in standard_fields.get_input_object_fields():
    if key in inputs:
      object_keys.append(key)
  for k, v in inputs.items():
    if k in mesh_keys:
      mesh_inputs[k] = v
    elif k in object_keys:
      object_inputs[k] = v
    else:
      non_tensor_inputs[k] = v
  logging.info('view image inputs')
  logging.info(view_image_inputs)
  logging.info('view indices 2d inputs')
  logging.info(view_indices_2d_inputs)
  logging.info('mesh inputs')
  logging.info(mesh_inputs)
  logging.info('object inputs')
  logging.info(object_inputs)
  logging.info('non_tensor_inputs')
  logging.info(non_tensor_inputs)
  return (view_image_inputs, view_indices_2d_inputs, mesh_inputs, object_inputs,
          non_tensor_inputs)


@gin.configurable(
    'object_detection_preprocess',
    blacklist=['inputs', 'output_keys', 'is_training'])
def preprocess(inputs,
               output_keys=None,
               is_training=False,
               input_field_mapping_fn=None,
               image_preprocess_fn_dic=None,
               images_points_correspondence_fn=None,
               points_pad_or_clip_size=None,
               voxels_pad_or_clip_size=None,
               voxel_grid_cell_size=(0.1, 0.1, 0.1),
               num_offset_bins_x=4,
               num_offset_bins_y=4,
               num_offset_bins_z=4,
               point_feature_keys=('point_offset_bins',),
               point_to_voxel_segment_func=tf.math.unsorted_segment_mean,
               x_min_degree_rotation=None,
               x_max_degree_rotation=None,
               y_min_degree_rotation=None,
               y_max_degree_rotation=None,
               z_min_degree_rotation=None,
               z_max_degree_rotation=None,
               rotation_center=(0.0, 0.0, 0.0),
               min_scale_ratio=None,
               max_scale_ratio=None,
               translation_range=None,
               points_within_box_margin=0.0,
               num_points_to_randomly_sample=None,
               crop_points_around_random_seed_point=False,
               crop_num_points=None,
               crop_radius=None,
               crop_num_background_points=None,
               make_objects_axis_aligned=False,
               min_num_points_in_objects=0,
               fit_objects_to_instance_id_points=False,
               voxel_density_threshold=None,
               voxel_density_grid_cell_size=None):
  """Preprocesses data before running 3D object detection.

  Args:
    inputs: A dictionary of inputs. Each value must be a `Tensor`.
    output_keys: Either None, or a list of strings containing the keys in the
      dictionary that is returned by the preprocess function.
    is_training: Whether at training stage or not.
    input_field_mapping_fn: A function that maps the input fields to the
      fields expected by object detection pipeline.
    image_preprocess_fn_dic: Image preprocessing function. Maps view names to
      their image preprocessing functions. Set it to None, if there are no
      images to preprocess or you are not interested in preprocesing images.
    images_points_correspondence_fn: The function that computes correspondence
      between images and points.
    points_pad_or_clip_size: Number of target points to pad or clip to. If None,
      it will not perform the padding.
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
    x_min_degree_rotation: Min degree of rotation around the x axis.
    x_max_degree_rotation: Max degree of ratation around the x axis.
    y_min_degree_rotation: Min degree of rotation around the y axis.
    y_max_degree_rotation: Max degree of ratation around the y axis.
    z_min_degree_rotation: Min degree of rotation around the z axis.
    z_max_degree_rotation: Max degree of ratation around the z axis.
    rotation_center: Center of rotation.
    min_scale_ratio: Minimum scale ratio.
    max_scale_ratio: Maximum scale ratio.
    translation_range: A float value corresponding to the range of random
      translation in x, y, z directions. If None, no translation would happen.
    points_within_box_margin: A margin to add to box radius when deciding which
      points fall inside each box.
    num_points_to_randomly_sample: Number of points to randomly sample. If None,
      it will keep the original points and does not perform sampling.
    crop_points_around_random_seed_point: If True, randomly samples a seed
      point and crops the closest `points_pad_or_clip_size` points to the seed
      point. The random seed point selection is based on the following
      procedure. First an object box is randomly selected. Then a random point
      from the random box is selected. Note that the random seed point could be
      sampled from background as well.
    crop_num_points: Number of points to crop.
    crop_radius: The maximum distance of the cropped points from the randomly
      sampled point. If None, it won't be used.
    crop_num_background_points: Minimum number of background points in crop. If
      None, it won't get applied.
    make_objects_axis_aligned: If True, the objects will become axis aligned,
      meaning that they will have identity rotation matrix.
    min_num_points_in_objects: Remove objects that have less number of points
      in them than this value.
    fit_objects_to_instance_id_points: If True, it will fit objects to points
      based on their instance ids.
    voxel_density_threshold: Points that belong to a voxel with a density lower
      than this will be removed.
    voxel_density_grid_cell_size: Voxel grid size for removing noise based on
      voxel density threshold.

  Returns:
    inputs: The inputs processed according to our configuration.

  Raises:
    ValueError: If input dictionary is missing any of the required keys.
  """
  inputs = dict(inputs)

  # Convert all float64 to float32 and all int64 to int32.
  for key in sorted(inputs):
    if isinstance(inputs[key], tf.Tensor):
      if inputs[key].dtype == tf.float64:
        inputs[key] = tf.cast(inputs[key], dtype=tf.float32)
      if inputs[key].dtype == tf.int64:
        if key == 'timestamp':
          continue
        else:
          inputs[key] = tf.cast(inputs[key], dtype=tf.int32)

  (view_image_inputs, view_indices_2d_inputs, mesh_inputs, object_inputs,
   non_tensor_inputs) = split_inputs(
       inputs=inputs,
       input_field_mapping_fn=input_field_mapping_fn,
       image_preprocess_fn_dic=image_preprocess_fn_dic,
       images_points_correspondence_fn=images_points_correspondence_fn)

  if standard_fields.InputDataFields.point_positions not in mesh_inputs:
    raise ValueError('Key %s is missing' %
                     standard_fields.InputDataFields.point_positions)

  # Randomly sample points (optional)
  preprocessor_utils.randomly_sample_points(
      mesh_inputs=mesh_inputs,
      view_indices_2d_inputs=view_indices_2d_inputs,
      target_num_points=num_points_to_randomly_sample)

  # Remove low density points
  if voxel_density_threshold is not None:
    preprocessor_utils.remove_pointcloud_noise(
        mesh_inputs=mesh_inputs,
        view_indices_2d_inputs=view_indices_2d_inputs,
        voxel_grid_cell_size=voxel_density_grid_cell_size,
        voxel_density_threshold=voxel_density_threshold)

  rotation_center = tf.convert_to_tensor(rotation_center, dtype=tf.float32)

  # Remove objects that do not have 3d info.
  _filter_valid_objects(inputs=object_inputs)

  # Cast the objects_class to tf.int32.
  _cast_objects_class(inputs=object_inputs)

  # Remove objects that have less than a certain number of poitns
  if min_num_points_in_objects > 0:
    preprocessor_utils.remove_objects_by_num_points(
        mesh_inputs=mesh_inputs,
        object_inputs=object_inputs,
        min_num_points_in_objects=min_num_points_in_objects)

  # Set point box ids.
  preprocessor_utils.set_point_instance_ids(
      mesh_inputs=mesh_inputs,
      object_inputs=object_inputs,
      points_within_box_margin=points_within_box_margin)

  # Process images.
  preprocessor_utils.preprocess_images(
      view_image_inputs=view_image_inputs,
      view_indices_2d_inputs=view_indices_2d_inputs,
      image_preprocess_fn_dic=image_preprocess_fn_dic,
      is_training=is_training)

  # Randomly transform points and boxes.
  _randomly_transform_points_boxes(
      mesh_inputs=mesh_inputs,
      object_inputs=object_inputs,
      x_min_degree_rotation=x_min_degree_rotation,
      x_max_degree_rotation=x_max_degree_rotation,
      y_min_degree_rotation=y_min_degree_rotation,
      y_max_degree_rotation=y_max_degree_rotation,
      z_min_degree_rotation=z_min_degree_rotation,
      z_max_degree_rotation=z_max_degree_rotation,
      rotation_center=rotation_center,
      min_scale_ratio=min_scale_ratio,
      max_scale_ratio=max_scale_ratio,
      translation_range=translation_range)

  # Randomly crop points around a random seed point.
  if crop_points_around_random_seed_point:
    preprocessor_utils.crop_points_around_random_seed_point(
        mesh_inputs=mesh_inputs,
        view_indices_2d_inputs=view_indices_2d_inputs,
        num_closest_points=crop_num_points,
        max_distance=crop_radius,
        num_background_points=crop_num_background_points)

  if fit_objects_to_instance_id_points:
    preprocessor_utils.fit_objects_to_instance_id_points(
        mesh_inputs=mesh_inputs, object_inputs=object_inputs)

  if make_objects_axis_aligned:
    preprocessor_utils.make_objects_axis_aligned(object_inputs=object_inputs)

  # Putting back the dictionaries together
  inputs = mesh_inputs.copy()
  inputs.update(object_inputs)
  inputs.update(non_tensor_inputs)
  for key in sorted(view_image_inputs):
    inputs[('%s/features' % key)] = view_image_inputs[key]
  for key in sorted(view_indices_2d_inputs):
    inputs[('%s/indices_2d' % key)] = view_indices_2d_inputs[key]

  # Transfer object properties to points, and randomly rotate the points around
  # y axis at training time.
  _transfer_object_properties_to_points(inputs=inputs)

  # Pad or clip points and their properties.
  _pad_or_clip_point_properties(
      inputs=inputs, pad_or_clip_size=points_pad_or_clip_size)

  # Create features that do not exist
  preprocessor_utils.add_point_offsets(
      inputs=inputs, voxel_grid_cell_size=voxel_grid_cell_size)
  preprocessor_utils.add_point_offset_bins(
      inputs=inputs,
      voxel_grid_cell_size=voxel_grid_cell_size,
      num_bins_x=num_offset_bins_x,
      num_bins_y=num_offset_bins_y,
      num_bins_z=num_offset_bins_z)

  # Voxelize point features
  preprocessor_utils.voxelize_point_features(
      inputs=inputs,
      voxels_pad_or_clip_size=voxels_pad_or_clip_size,
      voxel_grid_cell_size=voxel_grid_cell_size,
      point_feature_keys=point_feature_keys,
      point_to_voxel_segment_func=point_to_voxel_segment_func)

  # Voxelizing the semantic labels
  preprocessor_utils.voxelize_semantic_labels(
      inputs=inputs,
      voxels_pad_or_clip_size=voxels_pad_or_clip_size,
      voxel_grid_cell_size=voxel_grid_cell_size)

  # Voxelizing the instance labels
  preprocessor_utils.voxelize_instance_labels(
      inputs=inputs,
      voxels_pad_or_clip_size=voxels_pad_or_clip_size,
      voxel_grid_cell_size=voxel_grid_cell_size)

  # Voxelize the object properties
  preprocessor_utils.voxelize_object_properties(
      inputs=inputs,
      voxels_pad_or_clip_size=voxels_pad_or_clip_size,
      voxel_grid_cell_size=voxel_grid_cell_size)

  # Filter preinputs by output_keys if it is not None.
  if output_keys is not None:
    for key in list(inputs):
      if key not in output_keys:
        inputs.pop(key, None)

  return inputs
