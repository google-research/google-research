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

"""Contains classes specifying naming conventions used for 3d object detection.


Specifies:
  InputDataFields: Standard fields used by reader/preprocessor/batcher.
  DetectionResultFields: Standard fields returned by object detector.
"""


class InputDataFields(object):
  """Names for the input tensors.

  Holds the standard data field names to use for identifying input tensors. This
  should be used by the decoder to identify keys for the returned tensor_dict
  containing input tensors. And it should be used by the model to identify the
  tensors it needs.

  Attributes:
    *** CAMERA ***
    camera_intrinsics: Camera intrinsics (calibration).
    camera_rotation_matrix: Camera rotation matrix.
    camera_translation: Camera translation.
    camera_image: Preprocessed image that is used as the input to the network.
    camera_image_name: String containing the image name.
    camera_raw_image: Image without that did go through some preprocessing but
      always kept its original uint8 pixel values.
    camera_original_image: Original image with uint8 values.

    *** OBJECT FIELDS ***
    objects_length: A tf.float32 tensor of size [N, 1] containing the
      object lengths.
    objects_height: A tf.float32 tensor of size [N, 1] containing the
      object heights.
    objects_width: A tf.float32 tensor of size [N, 1] containing the
      object widths.
    objects_rotation_matrix: A tf.float32 tensor of size [N, 3, 3] containing
      the object rotation matrices in camera coordinate frame.
    objects_center: A tf.float32 tensor of size [N, 3] containing the
      object center translation in camera coordinate frame.
    objects_class: A tf.int32 tensor of size [N, 1] containing
      the ground-truth class label of the objects.
    objects_is_difficult: A tf.int32 tensor of size [N, 1] containing 0-1 values
      determining if an object is difficult to detect.
    objects_instance_id: A tf.int32 tensor of size [N, 1] containing instance id
      of the ground-truth boxes. Id of 0 means background.
    objects_flow: A tf.float32 tensor of size [N, 3] containing the flow of the
      ground-truth boxes.

    *** POINT FIELDS ***
    point_positions: A tf.float32 tensor of size [N, 3] containing the 3d
      location of the points.
    num_valid_points: A tf.int32 scalar containing the number of points.
    object_class_points: A tf.int32 tensor of size [N, 1] containing the
      semantic category of the object that each point belongs to.
    object_rotation_matrix_points: A tf.float32 tensor of size [N, 3, 3]
      containing the rotation matrix of the object that each point belongs to.
    object_length_points: A tf.float32 tensor of size [N, 1] containing the
      length of the object that each point belongs to.
    object_height_points: A tf.float32 tensor of size [N, 1] containing the
      height of the object that each point belongs to.
    object_width_points: A tf.float32 tensor of size [N, 1] containing the
      width of the object that each point belongs to.
    object_center_points: A tf.float32 tensor of size [N, 3] containing the
      center of the object that each point belongs to.
    object_instance_id_points: A tf.int32 tensor of size [N, 1] containing the
      object instance id that each point belongs to.
    object_flow_points: A tf.float32 tensor of size [N, 3] containing the
      flow of the object that each point belongs to.
    shape_gt_sdf = A tf.float32 tensor of size [num_objects, num_queries]
      containing the ground-truth SDF for the shape queries.

    *** SHAPE FIELDS ***
    shape_valid_queries_mask: A tf.bool tensor of
      size [num_objects, num_queries] representing valid shape queries after
      padding.
    shape_sdf_queries: A tf.float32 tensor of size [num_objects, num_queries, 3]
      containing the query points for computing sdf.
    shape_sampled_object_points_inds: A tf.int32 tensor of size [num_objects,
      num_points_per_object] representing the ids of points sampled for
      each of the objects.

    *** INSTANCE SEGMENTATION FIELDS ***
    instance_segments_mask: A tf.float32 tensor of
      size [batch_size, num_instances, num_points].

    *** MISC FIELDS ***
    camera_intrinsics: A tensor containing the camera intrinsics matrix.
    category_index: Set from label map. Contains the mapping from indices to
      categories.
  """
  # Camera fields
  camera_intrinsics = 'camera_intrinsics'
  camera_rotation_matrix = 'camera_rotation_matrix'
  camera_translation = 'camera_translation'
  camera_image = 'camera_image'
  camera_image_name = 'camera_image_name'
  camera_raw_image = 'camera_raw_image'
  camera_original_image = 'camera_original_image'

  # Object fields
  objects_length = 'objects_length'
  objects_height = 'objects_height'
  objects_width = 'objects_width'
  objects_rotation_matrix = 'objects_rotation_matrix'
  objects_center = 'objects_center'
  objects_class = 'objects_class'
  objects_difficulty = 'objects_difficulty'
  objects_instance_id = 'objects_instance_id'
  objects_has_3d_info = 'objects_has_3d_info'
  objects_flow = 'objects_flow'

  # Point fields
  point_positions_original = 'point_positions_original'
  point_normals_original = 'point_normals_original'
  point_colors_original = 'point_colors_original'
  point_positions = 'point_positions'
  point_intensities = 'point_intensities'
  point_elongations = 'point_elongations'
  point_normals = 'point_normals'
  point_colors = 'point_colors'
  point_spin_coordinates = 'point_spin_coordinates'
  point_offsets = 'point_offsets'
  point_offset_bins = 'point_offset_bins'
  point_loss_weights = 'point_loss_weights'
  num_valid_points = 'num_valid_points'
  num_valid_points_per_frame = 'num_valid_points_per_frame'
  points_encoded_features = 'points_encoded_features'
  object_class_points = 'object_class_points'
  object_rotation_matrix_points = 'object_rotation_matrix_points'
  object_length_points = 'object_length_points'
  object_height_points = 'object_height_points'
  object_width_points = 'object_width_points'
  object_center_points = 'object_center_points'
  object_instance_id_points = 'object_instance_id_points'
  object_flow_points = 'object_flow_points'

  # Voxel fields
  voxel_xyz_indices = 'voxel_xyz_indices'
  voxel_features = 'voxel_features'
  voxel_start_locations = 'voxel_start_locations'
  voxel_positions = 'voxel_positions'
  voxel_intensities = 'voxel_intensities'
  voxel_elongations = 'voxel_elongations'
  voxel_normals = 'voxel_normals'
  voxel_colors = 'voxel_colors'
  voxel_loss_weights = 'voxel_loss_weights'
  num_valid_voxels = 'num_valid_voxels'
  points_to_voxel_mapping = 'points_to_voxel_mapping'
  point_frame_index = 'point_frame_index'
  voxels_encoded_features = 'voxels_encoded_features'
  object_class_voxels = 'object_class_voxels'
  object_rotation_matrix_voxels = 'object_rotation_matrix_voxels'
  object_length_voxels = 'object_length_voxels'
  object_height_voxels = 'object_height_voxels'
  object_width_voxels = 'object_width_voxels'
  object_center_voxels = 'object_center_voxels'
  object_instance_id_voxels = 'object_instance_id_voxels'
  object_flow_voxels = 'object_flow_voxels'

  # Shape fields
  shape_gt_sdf = 'shape_gt_sdf'
  shape_sdf_queries = 'shape_sdf_queries'
  shape_sampled_object_points_inds = 'shape_sampled_object_points_inds'
  shape_valid_queries_mask = 'shape_valid_queries_mask'

  # Instance segment fields
  instance_segments_voxel_mask = 'instance_segments_voxel_mask'
  instance_segments_point_mask = 'instance_segments_point_mask'

  # Misc
  category_index = 'category_index'


class DetectionResultFields(object):
  """Naming conventions for storing the output of the detector.

  Attributes:

    *** OBJECT FIELDS ***
    objects_class: A tf.int32 tensor of shape [N, 1] containing the class of the
      detected objects.
    objects_score: A tf.float32 tensor of shape [N, 1] or [N, num_classes]
      containing the score of the detected objects.
    objects_length: A tf.float32 tensor of shape [N, 1] containing the length of
      the detected objects.
    objects_height: A tf.float32 tensor of shape [N, 1] containing the height of
      the detected objects.
    objects_width: A tf.float32 tensor of shape [N, 1] containing the width of
      the detected objects.
    objects_rotation_y: A tf.float32 tensor of shape [N, 1] containing the
      rotation of the detected objects around y axis.
    objects_rotation_y_cos: A tf.float32 tensor of shape [N, 1] containing the
      cos of the rotation of the detected objects around y axis.
    objects_rotation_y_sin: A tf.float32 tensor of shape [N, 1] containing the
      sin of the rotation of the detected objects around y axis.
    objects_center: A tf.float32 tensor of shape [N, 3] containing the
      center of the detected objects.
    objects_shape_embedding: A tf.float32 of size [N_objects,
      embedding_dim] containing the embedding vector prediceted for each object.

    *** ANCHOR FIELDS ***
    anchors_length: A tf.float32 tensor of shape [N, 1] containing the length of
      the object anchors.
    anchors_height: A tf.float32 tensor of shape [N, 1] containing the height of
      the object anchors.
    anchors_width: A tf.float32 tensor of shape [N, 1] containing the width of
      the object anchors.
    anchors_rotation_y: A tf.float32 tensor of shape [N, 1] containing the
      rotation of the object anchors around y axis.
    anchors_rotation_y_cos: A tf.float32 tensor of shape [N, 1] containing the
      cos of the rotation of the object anchors around y axis.
    anchors_rotation_y_sin: A tf.float32 tensor of shape [N, 1] containing the
      sin of the rotation of the object anchors around y axis.

    *** POINTS FIELDS ***
    object_semantic_points: A tf.float32 tensor of size [N, num_classes]
      that contains predicted semantic logits.
    object_rotation_y_cos_points: A tf.float32 tensor of size [N, 1] containing
      the predicted rotation_y_cos of the object each point belongs to.
    object_rotation_y_sin_points: A tf.float32 tensor of size [N, 1] containing
      the predicted rotation_y_sin of the object each point belongs to.
    object_length_points: A tf.float32 tensor of size [N, 1] containing the
      predicted length of the object each point belongs to.
    object_height_points: A tf.float32 tensor of size [N, 1] containing the
      predicted height of the object each point belongs to.
    object_width_points: A tf.float32 tensor of size [N, 1] containing the
      predicted width of the object each point belongs to.
    object_center_points: A tf.float32 tensor of size [N, 3] containing the
      predicted center of the object each point belongs to.
    shape_embedding_points: A tf.float32 tensor of size [N, embedding_dim]
      containing the predicted D-dimensional shape embedding by the network.
    instance_embedding_points: A tf.float32 tensor of size [N, embedding_dim]
      containing the predicted D-dimensional instance embedding by the network.

    *** INSTANCE SEGMENTATION FIELDS ***
    instance_segments_voxel_mask: A tf.float32 tensor of
      size [batch_size, num_instances, num_voxels].
    instance_segments_point_mask: A tf.float32 tensor of
      size [batch_size, num_instances, num_points].

    shape_sdf_points = A tf.float32 tensor of size [num_objects, num_queries]
      containing the predicted sdf by the sdf decoder.
  """
  # Object fields
  objects_class = 'detected_objects_class'
  objects_score = 'detected_objects_score'
  objects_length = 'detected_objects_length'
  objects_height = 'detected_objects_height'
  objects_width = 'detected_objects_width'
  objects_rotation_x_cos = 'detected_objects_rotation_x_cos'
  objects_rotation_x_sin = 'detected_objects_rotation_x_sin'
  objects_rotation_y_cos = 'detected_objects_rotation_y_cos'
  objects_rotation_y_sin = 'detected_objects_rotation_y_sin'
  objects_rotation_z_cos = 'detected_objects_rotation_z_cos'
  objects_rotation_z_sin = 'detected_objects_rotation_z_sin'
  objects_rotation_matrix = 'detected_objects_rotation_matrix'
  objects_center = 'detected_objects_center'
  objects_encoded_features = 'detected_objects_encoded_features'
  objects_anchor_location = 'detected_objects_anchor_location'
  objects_flow = 'detected_objects_flow'
  objects_shape_embedding = 'predicted_objects_shape_embedding'

  # Anchor fields
  anchors_length = 'anchors_length'
  anchors_height = 'anchors_height'
  anchors_width = 'anchors_width'
  anchors_center = 'anchors_center'
  anchors_rotation_y = 'anchors_rotation_y'
  anchors_rotation_y_cos = 'anchors_rotation_y_cos'
  anchors_rotation_y_sin = 'anchors_rotation_y_sin'

  # Point fields
  object_semantic_points = 'predicted_object_semantic_points'
  object_rotation_x_cos_points = 'predicted_object_rotation_x_cos_points'
  object_rotation_x_sin_points = 'predicted_object_rotation_x_sin_points'
  object_rotation_y_cos_points = 'predicted_object_rotation_y_cos_points'
  object_rotation_y_sin_points = 'predicted_object_rotation_y_sin_points'
  object_rotation_z_cos_points = 'predicted_object_rotation_z_cos_points'
  object_rotation_z_sin_points = 'predicted_object_rotation_z_sin_points'
  object_rotation_matrix_points = 'predicted_object_rotation_matrix_points'
  object_length_points = 'predicted_object_length_points'
  object_height_points = 'predicted_object_height_points'
  object_width_points = 'predicted_object_width_points'
  object_center_points = 'predicted_object_center_points'
  object_weight_points = 'predicted_object_weight_points'
  object_anchor_location_points = 'predicted_object_anchor_location_points'
  object_flow_points = 'predicted_object_flow_points'
  shape_embedding_points = 'predicted_shape_embedding_points'
  shape_embedding_sampled_points = 'predicted_shape_embedding_sampled_points'
  shape_sdf_points = 'predicted_shape_sdf_points'
  instance_embedding_points = 'predicted_instance_embedding_points'
  encoded_features_points = 'encoded_features_points'

  # Voxel fields
  object_semantic_voxels = 'predicted_object_semantic_voxels'
  object_rotation_x_cos_voxels = 'predicted_object_rotation_x_cos_voxels'
  object_rotation_x_sin_voxels = 'predicted_object_rotation_x_sin_voxels'
  object_rotation_y_cos_voxels = 'predicted_object_rotation_y_cos_voxels'
  object_rotation_y_sin_voxels = 'predicted_object_rotation_y_sin_voxels'
  object_rotation_z_cos_voxels = 'predicted_object_rotation_z_cos_voxels'
  object_rotation_z_sin_voxels = 'predicted_object_rotation_z_sin_voxels'
  object_rotation_matrix_voxels = 'predicted_object_rotation_matrix_voxels'
  object_length_voxels = 'predicted_object_length_voxels'
  object_height_voxels = 'predicted_object_height_voxels'
  object_width_voxels = 'predicted_object_width_voxels'
  object_center_voxels = 'predicted_object_center_voxels'
  object_weight_voxels = 'predicted_object_weight_voxels'
  object_anchor_location_voxels = 'predicted_object_anchor_location_voxels'
  object_flow_voxels = 'predicted_object_flow_voxels'
  instance_embedding_voxels = 'predicted_instance_embedding_voxels'
  shape_embedding_voxels = 'predicted_shape_embedding_voxels'
  shape_embedding_sampled_voxels = 'predicted_shape_embedding_sampled_voxels'
  shape_sdf_voxels = 'predicted_shape_sdf_voxels'
  encoded_features_voxels = 'encoded_features_voxels'

  # Intermediate point fields
  intermediate_object_semantic_points = 'intermediate_predicted_object_semantic_points'
  intermediate_object_rotation_x_cos_points = 'intermediate_predicted_object_rotation_x_cos_points'
  intermediate_object_rotation_x_sin_points = 'intermediate_predicted_object_rotation_x_sin_points'
  intermediate_object_rotation_y_cos_points = 'intermediate_predicted_object_rotation_y_cos_points'
  intermediate_object_rotation_y_sin_points = 'intermediate_predicted_object_rotation_y_sin_points'
  intermediate_object_rotation_z_cos_points = 'intermediate_predicted_object_rotation_z_cos_points'
  intermediate_object_rotation_z_sin_points = 'intermediate_predicted_object_rotation_z_sin_points'
  intermediate_object_rotation_matrix_points = 'intermediate_predicted_object_rotation_matrix_points'
  intermediate_object_length_points = 'intermediate_predicted_object_length_points'
  intermediate_object_height_points = 'intermediate_predicted_object_height_points'
  intermediate_object_width_points = 'intermediate_predicted_object_width_points'
  intermediate_object_center_points = 'intermediate_predicted_object_center_points'
  intermediate_object_flow_points = 'intermediate_predicted_object_flow_points'
  intermediate_instance_embedding_points = 'intermediate_instance_embedding_points'

  # Intermediate voxel fields
  intermediate_object_semantic_voxels = 'intermediate_predicted_object_semantic_voxels'
  intermediate_object_rotation_x_cos_voxels = 'intermediate_predicted_object_rotation_x_cos_voxels'
  intermediate_object_rotation_x_sin_voxels = 'intermediate_predicted_object_rotation_x_sin_voxels'
  intermediate_object_rotation_y_cos_voxels = 'intermediate_predicted_object_rotation_y_cos_voxels'
  intermediate_object_rotation_y_sin_voxels = 'intermediate_predicted_object_rotation_y_sin_voxels'
  intermediate_object_rotation_z_cos_voxels = 'intermediate_predicted_object_rotation_z_cos_voxels'
  intermediate_object_rotation_z_sin_voxels = 'intermediate_predicted_object_rotation_z_sin_voxels'
  intermediate_object_rotation_matrix_voxels = 'intermediate_predicted_object_rotation_matrix_voxels'
  intermediate_object_length_voxels = 'intermediate_predicted_object_length_voxels'
  intermediate_object_height_voxels = 'intermediate_predicted_object_height_voxels'
  intermediate_object_width_voxels = 'intermediate_predicted_object_width_voxels'
  intermediate_object_center_voxels = 'intermediate_predicted_object_center_voxels'
  intermediate_object_flow_voxels = 'intermediate_predicted_object_flow_voxels'
  intermediate_instance_embedding_voxels = 'intermediate_instance_embedding_voxels'

  # Instance segment fields
  instance_segments_voxel_mask = 'predicted_instance_segments_voxel_mask'
  instance_segments_point_mask = 'predicted_instance_segments_point_mask'


def get_input_object_fields():
  return [
      InputDataFields.objects_length,
      InputDataFields.objects_height,
      InputDataFields.objects_width,
      InputDataFields.objects_center,
      InputDataFields.objects_rotation_matrix,
      InputDataFields.objects_class,
      InputDataFields.objects_instance_id,
      InputDataFields.objects_difficulty,
      InputDataFields.objects_has_3d_info,
      InputDataFields.objects_flow,
      InputDataFields.instance_segments_voxel_mask,
      InputDataFields.instance_segments_point_mask,
  ]


def get_input_point_fields():
  return [
      InputDataFields.point_positions,
      InputDataFields.point_intensities,
      InputDataFields.point_elongations,
      InputDataFields.point_normals,
      InputDataFields.point_colors,
      InputDataFields.num_valid_points,
      InputDataFields.object_length_points,
      InputDataFields.object_height_points,
      InputDataFields.object_width_points,
      InputDataFields.object_center_points,
      InputDataFields.object_rotation_matrix_points,
      InputDataFields.object_class_points,
      InputDataFields.object_instance_id_points,
      InputDataFields.object_flow_points,
  ]


def get_input_voxel_fields():
  return [
      InputDataFields.voxel_positions,
      InputDataFields.voxel_intensities,
      InputDataFields.voxel_elongations,
      InputDataFields.voxel_normals,
      InputDataFields.num_valid_voxels,
      InputDataFields.object_length_voxels,
      InputDataFields.object_height_voxels,
      InputDataFields.object_width_voxels,
      InputDataFields.object_center_voxels,
      InputDataFields.object_rotation_matrix_voxels,
      InputDataFields.object_class_voxels,
      InputDataFields.object_instance_id_voxels,
      InputDataFields.object_flow_voxels,
  ]


def get_output_object_fields():
  return [
      DetectionResultFields.objects_length,
      DetectionResultFields.objects_height,
      DetectionResultFields.objects_width,
      DetectionResultFields.objects_center,
      DetectionResultFields.objects_rotation_x_cos,
      DetectionResultFields.objects_rotation_x_sin,
      DetectionResultFields.objects_rotation_y_cos,
      DetectionResultFields.objects_rotation_y_sin,
      DetectionResultFields.objects_rotation_z_cos,
      DetectionResultFields.objects_rotation_z_sin,
      DetectionResultFields.objects_rotation_matrix,
      DetectionResultFields.objects_class,
      DetectionResultFields.objects_score,
      DetectionResultFields.objects_encoded_features,
      DetectionResultFields.objects_flow,
      DetectionResultFields.instance_segments_voxel_mask,
      DetectionResultFields.instance_segments_point_mask,
      DetectionResultFields.objects_anchor_location,
  ]


def get_output_anchor_fields():
  return [
      DetectionResultFields.anchors_length,
      DetectionResultFields.anchors_height,
      DetectionResultFields.anchors_width,
      DetectionResultFields.anchors_center,
      DetectionResultFields.anchors_rotation_y,
      DetectionResultFields.anchors_rotation_y_cos,
      DetectionResultFields.anchors_rotation_y_sin,
  ]


def get_output_point_fields():
  return [
      DetectionResultFields.object_rotation_x_cos_points,
      DetectionResultFields.object_rotation_x_sin_points,
      DetectionResultFields.object_rotation_y_cos_points,
      DetectionResultFields.object_rotation_y_sin_points,
      DetectionResultFields.object_rotation_z_cos_points,
      DetectionResultFields.object_rotation_z_sin_points,
      DetectionResultFields.object_rotation_matrix_points,
      DetectionResultFields.object_length_points,
      DetectionResultFields.object_height_points,
      DetectionResultFields.object_width_points,
      DetectionResultFields.object_center_points,
      DetectionResultFields.object_semantic_points,
      DetectionResultFields.object_weight_points,
      DetectionResultFields.object_flow_points,
      DetectionResultFields.encoded_features_points,
      DetectionResultFields.intermediate_object_rotation_x_cos_points,
      DetectionResultFields.intermediate_object_rotation_x_sin_points,
      DetectionResultFields.intermediate_object_rotation_y_cos_points,
      DetectionResultFields.intermediate_object_rotation_y_sin_points,
      DetectionResultFields.intermediate_object_rotation_z_cos_points,
      DetectionResultFields.intermediate_object_rotation_z_sin_points,
      DetectionResultFields.intermediate_object_rotation_matrix_points,
      DetectionResultFields.intermediate_object_length_points,
      DetectionResultFields.intermediate_object_height_points,
      DetectionResultFields.intermediate_object_width_points,
      DetectionResultFields.intermediate_object_center_points,
      DetectionResultFields.intermediate_object_semantic_points,
      DetectionResultFields.intermediate_object_flow_points,
      DetectionResultFields.shape_embedding_points,
      DetectionResultFields.instance_embedding_points,
      DetectionResultFields.object_anchor_location_points,
  ]


def get_output_voxel_fields():
  return [
      DetectionResultFields.object_rotation_x_cos_voxels,
      DetectionResultFields.object_rotation_x_sin_voxels,
      DetectionResultFields.object_rotation_y_cos_voxels,
      DetectionResultFields.object_rotation_y_sin_voxels,
      DetectionResultFields.object_rotation_z_cos_voxels,
      DetectionResultFields.object_rotation_z_sin_voxels,
      DetectionResultFields.object_rotation_matrix_voxels,
      DetectionResultFields.object_length_voxels,
      DetectionResultFields.object_height_voxels,
      DetectionResultFields.object_width_voxels,
      DetectionResultFields.object_center_voxels,
      DetectionResultFields.object_semantic_voxels,
      DetectionResultFields.object_weight_voxels,
      DetectionResultFields.object_flow_voxels,
      DetectionResultFields.encoded_features_voxels,
      DetectionResultFields.instance_embedding_voxels,
      DetectionResultFields.intermediate_object_rotation_x_cos_voxels,
      DetectionResultFields.intermediate_object_rotation_x_sin_voxels,
      DetectionResultFields.intermediate_object_rotation_y_cos_voxels,
      DetectionResultFields.intermediate_object_rotation_y_sin_voxels,
      DetectionResultFields.intermediate_object_rotation_z_cos_voxels,
      DetectionResultFields.intermediate_object_rotation_z_sin_voxels,
      DetectionResultFields.intermediate_object_rotation_matrix_voxels,
      DetectionResultFields.intermediate_object_length_voxels,
      DetectionResultFields.intermediate_object_height_voxels,
      DetectionResultFields.intermediate_object_width_voxels,
      DetectionResultFields.intermediate_object_center_voxels,
      DetectionResultFields.intermediate_object_semantic_voxels,
      DetectionResultFields.intermediate_object_flow_voxels,
      DetectionResultFields.intermediate_instance_embedding_voxels,
  ]


def get_output_point_to_intermediate_field_mapping():
  return {
      DetectionResultFields.object_rotation_x_cos_points:
          DetectionResultFields.intermediate_object_rotation_x_cos_points,
      DetectionResultFields.object_rotation_x_sin_points:
          DetectionResultFields.intermediate_object_rotation_x_sin_points,
      DetectionResultFields.object_rotation_y_cos_points:
          DetectionResultFields.intermediate_object_rotation_y_cos_points,
      DetectionResultFields.object_rotation_y_sin_points:
          DetectionResultFields.intermediate_object_rotation_y_sin_points,
      DetectionResultFields.object_rotation_z_cos_points:
          DetectionResultFields.intermediate_object_rotation_z_cos_points,
      DetectionResultFields.object_rotation_z_sin_points:
          DetectionResultFields.intermediate_object_rotation_z_sin_points,
      DetectionResultFields.object_rotation_matrix_points:
          DetectionResultFields.intermediate_object_rotation_matrix_points,
      DetectionResultFields.object_length_points:
          DetectionResultFields.intermediate_object_length_points,
      DetectionResultFields.object_height_points:
          DetectionResultFields.intermediate_object_height_points,
      DetectionResultFields.object_width_points:
          DetectionResultFields.intermediate_object_width_points,
      DetectionResultFields.object_center_points:
          DetectionResultFields.intermediate_object_center_points,
      DetectionResultFields.object_semantic_points:
          DetectionResultFields.intermediate_object_semantic_points,
      DetectionResultFields.object_flow_points:
          DetectionResultFields.intermediate_object_flow_points,
      DetectionResultFields.instance_embedding_points:
          DetectionResultFields.intermediate_instance_embedding_points,
  }


def get_output_voxel_to_intermediate_field_mapping():
  return {
      DetectionResultFields.object_rotation_x_cos_voxels:
          DetectionResultFields.intermediate_object_rotation_x_cos_voxels,
      DetectionResultFields.object_rotation_x_sin_voxels:
          DetectionResultFields.intermediate_object_rotation_x_sin_voxels,
      DetectionResultFields.object_rotation_y_cos_voxels:
          DetectionResultFields.intermediate_object_rotation_y_cos_voxels,
      DetectionResultFields.object_rotation_y_sin_voxels:
          DetectionResultFields.intermediate_object_rotation_y_sin_voxels,
      DetectionResultFields.object_rotation_z_cos_voxels:
          DetectionResultFields.intermediate_object_rotation_z_cos_voxels,
      DetectionResultFields.object_rotation_z_sin_voxels:
          DetectionResultFields.intermediate_object_rotation_z_sin_voxels,
      DetectionResultFields.object_rotation_matrix_voxels:
          DetectionResultFields.intermediate_object_rotation_matrix_voxels,
      DetectionResultFields.object_length_voxels:
          DetectionResultFields.intermediate_object_length_voxels,
      DetectionResultFields.object_height_voxels:
          DetectionResultFields.intermediate_object_height_voxels,
      DetectionResultFields.object_width_voxels:
          DetectionResultFields.intermediate_object_width_voxels,
      DetectionResultFields.object_center_voxels:
          DetectionResultFields.intermediate_object_center_voxels,
      DetectionResultFields.object_semantic_voxels:
          DetectionResultFields.intermediate_object_semantic_voxels,
      DetectionResultFields.object_flow_voxels:
          DetectionResultFields.intermediate_object_flow_voxels,
      DetectionResultFields.instance_embedding_voxels:
          DetectionResultFields.intermediate_instance_embedding_voxels,
  }


def get_output_point_to_object_field_mapping():
  return {
      DetectionResultFields.object_rotation_x_cos_points:
          DetectionResultFields.objects_rotation_x_cos,
      DetectionResultFields.object_rotation_x_sin_points:
          DetectionResultFields.objects_rotation_x_sin,
      DetectionResultFields.object_rotation_y_cos_points:
          DetectionResultFields.objects_rotation_y_cos,
      DetectionResultFields.object_rotation_y_sin_points:
          DetectionResultFields.objects_rotation_y_sin,
      DetectionResultFields.object_rotation_z_cos_points:
          DetectionResultFields.objects_rotation_z_cos,
      DetectionResultFields.object_rotation_z_sin_points:
          DetectionResultFields.objects_rotation_z_sin,
      DetectionResultFields.object_rotation_matrix_points:
          DetectionResultFields.objects_rotation_matrix,
      DetectionResultFields.object_length_points:
          DetectionResultFields.objects_length,
      DetectionResultFields.object_height_points:
          DetectionResultFields.objects_height,
      DetectionResultFields.object_width_points:
          DetectionResultFields.objects_width,
      DetectionResultFields.object_center_points:
          DetectionResultFields.objects_center,
      DetectionResultFields.object_semantic_points:
          DetectionResultFields.objects_score,
      DetectionResultFields.encoded_features_points:
          DetectionResultFields.objects_encoded_features,
      DetectionResultFields.object_flow_points:
          DetectionResultFields.objects_flow,
      DetectionResultFields.object_anchor_location_points:
          DetectionResultFields.objects_anchor_location,
  }


def get_output_voxel_to_object_field_mapping():
  return {
      DetectionResultFields.object_rotation_x_cos_voxels:
          DetectionResultFields.objects_rotation_x_cos,
      DetectionResultFields.object_rotation_x_sin_voxels:
          DetectionResultFields.objects_rotation_x_sin,
      DetectionResultFields.object_rotation_y_cos_voxels:
          DetectionResultFields.objects_rotation_y_cos,
      DetectionResultFields.object_rotation_y_sin_voxels:
          DetectionResultFields.objects_rotation_y_sin,
      DetectionResultFields.object_rotation_z_cos_voxels:
          DetectionResultFields.objects_rotation_z_cos,
      DetectionResultFields.object_rotation_z_sin_voxels:
          DetectionResultFields.objects_rotation_z_sin,
      DetectionResultFields.object_rotation_matrix_voxels:
          DetectionResultFields.objects_rotation_matrix,
      DetectionResultFields.object_length_voxels:
          DetectionResultFields.objects_length,
      DetectionResultFields.object_height_voxels:
          DetectionResultFields.objects_height,
      DetectionResultFields.object_width_voxels:
          DetectionResultFields.objects_width,
      DetectionResultFields.object_center_voxels:
          DetectionResultFields.objects_center,
      DetectionResultFields.object_semantic_voxels:
          DetectionResultFields.objects_score,
      DetectionResultFields.encoded_features_voxels:
          DetectionResultFields.objects_encoded_features,
      DetectionResultFields.object_flow_voxels:
          DetectionResultFields.objects_flow,
  }


def get_output_voxel_to_point_field_mapping():
  return {
      DetectionResultFields.object_rotation_x_cos_voxels:
          DetectionResultFields.object_rotation_x_cos_points,
      DetectionResultFields.object_rotation_x_sin_voxels:
          DetectionResultFields.object_rotation_x_sin_points,
      DetectionResultFields.object_rotation_y_cos_voxels:
          DetectionResultFields.object_rotation_y_cos_points,
      DetectionResultFields.object_rotation_y_sin_voxels:
          DetectionResultFields.object_rotation_y_sin_points,
      DetectionResultFields.object_rotation_z_cos_voxels:
          DetectionResultFields.object_rotation_z_cos_points,
      DetectionResultFields.object_rotation_z_sin_voxels:
          DetectionResultFields.object_rotation_z_sin_points,
      DetectionResultFields.object_rotation_matrix_voxels:
          DetectionResultFields.object_rotation_matrix_points,
      DetectionResultFields.object_length_voxels:
          DetectionResultFields.object_length_points,
      DetectionResultFields.object_height_voxels:
          DetectionResultFields.object_height_points,
      DetectionResultFields.object_width_voxels:
          DetectionResultFields.object_width_points,
      DetectionResultFields.object_center_voxels:
          DetectionResultFields.object_center_points,
      DetectionResultFields.object_semantic_voxels:
          DetectionResultFields.object_semantic_points,
      DetectionResultFields.encoded_features_voxels:
          DetectionResultFields.encoded_features_points,
      DetectionResultFields.object_flow_voxels:
          DetectionResultFields.object_flow_points,
      DetectionResultFields.shape_embedding_voxels:
          DetectionResultFields.shape_embedding_points,
      DetectionResultFields.shape_embedding_sampled_voxels:
          DetectionResultFields.shape_embedding_sampled_points,
      DetectionResultFields.shape_sdf_voxels:
          DetectionResultFields.shape_sdf_points,
      DetectionResultFields.instance_embedding_voxels:
          DetectionResultFields.instance_embedding_points,
  }


def get_input_point_to_object_field_mapping():
  return {
      InputDataFields.object_rotation_matrix_points:
          InputDataFields.objects_rotation_matrix,
      InputDataFields.object_length_points:
          InputDataFields.objects_length,
      InputDataFields.object_height_points:
          InputDataFields.objects_height,
      InputDataFields.object_width_points:
          InputDataFields.objects_width,
      InputDataFields.object_center_points:
          InputDataFields.objects_center,
      InputDataFields.object_class_points:
          InputDataFields.objects_class,
      InputDataFields.object_instance_id_points:
          InputDataFields.objects_instance_id,
      InputDataFields.object_flow_points:
          InputDataFields.objects_flow,
  }


def get_input_voxel_to_object_field_mapping():
  return {
      InputDataFields.object_rotation_matrix_voxels:
          InputDataFields.objects_rotation_matrix,
      InputDataFields.object_length_voxels:
          InputDataFields.objects_length,
      InputDataFields.object_height_voxels:
          InputDataFields.objects_height,
      InputDataFields.object_width_voxels:
          InputDataFields.objects_width,
      InputDataFields.object_center_voxels:
          InputDataFields.objects_center,
      InputDataFields.object_class_voxels:
          InputDataFields.objects_class,
      InputDataFields.object_instance_id_voxels:
          InputDataFields.objects_instance_id,
      InputDataFields.object_flow_voxels:
          InputDataFields.objects_flow,
  }


def get_input_point_to_voxel_field_mapping():
  return {
      InputDataFields.object_class_points:
          InputDataFields.object_class_voxels,
      InputDataFields.object_rotation_matrix_points:
          InputDataFields.object_rotation_matrix_voxels,
      InputDataFields.object_length_points:
          InputDataFields.object_length_voxels,
      InputDataFields.object_height_points:
          InputDataFields.object_height_voxels,
      InputDataFields.object_width_points:
          InputDataFields.object_width_voxels,
      InputDataFields.object_center_points:
          InputDataFields.object_center_voxels,
      InputDataFields.object_instance_id_points:
          InputDataFields.object_instance_id_voxels,
      InputDataFields.object_flow_points:
          InputDataFields.object_flow_voxels,
  }


def check_input_point_fields(inputs):
  for field in get_input_point_fields():
    if field in inputs:
      if field == InputDataFields.num_valid_points:
        if len(inputs[field].shape) != 1:
          raise ValueError('Num points should be a rank 1 tensor.')
      elif len(inputs[field].shape) not in (2, 3, 4):
        raise ValueError(('Input %s should be rank 2, 3 or 4.' % field))


def check_output_point_fields(outputs):
  for field in get_output_point_fields():
    if field in outputs and outputs[field] is not None:
      if len(outputs[field].shape) not in (2, 3, 4):
        raise ValueError(('Output %s should be rank 2, 3 or 4.' % field))


def check_input_voxel_fields(inputs):
  for field in get_input_voxel_fields():
    if field in inputs:
      if field == InputDataFields.num_valid_voxels:
        if len(inputs[field].shape) != 1:
          raise ValueError('Num voxels should be a rank 1 tensor.')
      elif len(inputs[field].shape) not in (2, 3, 4):
        raise ValueError(('Input %s should be rank 2, 3 or 4.' % field))


def check_output_voxel_fields(outputs):
  for field in get_output_voxel_fields():
    if field in outputs and outputs[field] is not None:
      if len(outputs[field].shape) not in (2, 3, 4):
        raise ValueError(('Output %s should be rank 2, 3 or 4.' % field))

