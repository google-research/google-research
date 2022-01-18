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

"""Common constants and utility functions."""

# Input tf.Example field keys.
TFE_KEY_IMAGE_HEIGHT = 'image/height'
TFE_KEY_IMAGE_WIDTH = 'image/width'
TFE_KEY_PREFIX_KEYPOINT_2D = 'image/object/part/'
TFE_KEY_SUFFIX_KEYPOINT_2D = ['/center/y', '/center/x']
TFE_KEY_PREFIX_KEYPOINT_3D = 'image/object/part_3d/'
TFE_KEY_SUFFIX_KEYPOINT_3D = ['/center/y', '/center/x', '/center/z']
TFE_KEY_SUFFIX_KEYPOINT_SCORE = '/score'
TFE_KEY_FEATURE = 'feature/data'
TFE_KEY_CLASS_LABEL_ID = 'image/class/label'
TFE_KEY_CLASS_LABEL_CONFIDENCE = 'image/class/confidence'

# Input tf.SequenceExample context feature field keys.
TFSE_KEY_IMAGE_HEIGHT = 'image/height'
TFSE_KEY_IMAGE_WIDTH = 'image/width'
TFSE_KEY_SUFFIX_KEYPOINT_2D = ['/region/point/y', '/region/point/x']
TFSE_KEY_SUFFIX_KEYPOINT_2D_SCORE = '/region/point/score'
TFSE_KEY_SUFFIX_KEYPOINT_3D = [
    '/region/3d_point/y', '/region/3d_point/x', '/region/3d_point/z'
]
TFSE_KEY_SUFFIX_KEYPOINT_3D_SCORE = '/region/3d_point/score'

# Input keys.
KEY_IMAGE_SIZES = 'image_sizes'
KEY_KEYPOINTS_2D = 'keypoints_2d'
KEY_KEYPOINT_SCORES_2D = 'keypoint_scores_2d'
KEY_KEYPOINT_MASKS_2D = 'keypoint_masks_2d'
KEY_PREPROCESSED_KEYPOINTS_2D = 'preprocessed_keypoints_2d'
KEY_PREPROCESSED_KEYPOINT_MASKS_2D = 'preprocessed_keypoint_masks_2d'
KEY_OFFSET_POINTS_2D = 'offset_points_2d'
KEY_SCALE_DISTANCES_2D = 'scale_distances_2d'
KEY_KEYPOINTS_3D = 'keypoints_3d'
KEY_PREPROCESSED_KEYPOINTS_3D = 'preprocessed_keypoints_3d'
KEY_OFFSET_POINTS_3D = 'offset_points_3d'
KEY_SCALE_DISTANCES_3D = 'scale_distances_3d'
KEY_EMBEDDING_MEANS = 'unnormalized_embeddings'
KEY_EMBEDDING_STDDEVS = 'embedding_stddevs'
KEY_EMBEDDING_SAMPLES = 'unnormalized_embedding_samples'
KEY_PREDICTED_KEYPOINTS_3D = 'predicted_keypoints_3d'
KEY_ALL_PREDICTED_KEYPOINTS_3D = 'all_predicted_keypoints_3d'
KEY_FEATURES = 'features'
KEY_CLASS_TARGETS = 'class_targets'
KEY_CLASS_WEIGHTS = 'class_weights'

# Model input keypoint types.
# 2D keypoints from input tables.
MODEL_INPUT_KEYPOINT_TYPE_2D_INPUT = '2D_INPUT'
# 2D projections of 3D keypoints.
MODEL_INPUT_KEYPOINT_TYPE_3D_PROJECTION = '3D_PROJECTION'
# Both 2D keypoints from input tables and 2D projections if 3D keypoints.
MODEL_INPUT_KEYPOINT_TYPE_2D_INPUT_AND_3D_PROJECTION = (
    '2D_INPUT_AND_3D_PROJECTION')
# Supported model input keypoint types for training.
SUPPORTED_TRAINING_MODEL_INPUT_KEYPOINT_TYPES = [
    MODEL_INPUT_KEYPOINT_TYPE_2D_INPUT,
    MODEL_INPUT_KEYPOINT_TYPE_3D_PROJECTION,
    MODEL_INPUT_KEYPOINT_TYPE_2D_INPUT_AND_3D_PROJECTION,
]
# Supported model input keypoint types for inference.
SUPPORTED_INFERENCE_MODEL_INPUT_KEYPOINT_TYPES = [
    MODEL_INPUT_KEYPOINT_TYPE_2D_INPUT,
]

# Model input keypoint mask types.
# No use.
MODEL_INPUT_KEYPOINT_MASK_TYPE_NO_USE = 'NO_USE'
# Masks 2D keypoint coordinates (to 0).
MODEL_INPUT_KEYPOINT_MASK_TYPE_MASK_KEYPOINTS = 'MASK_KEYPOINTS'
# Concatenates with 2D keypoint coordinates as input.
MODEL_INPUT_KEYPOINT_MASK_TYPE_AS_INPUT = 'AS_INPUT'
# Masks and concatenates with 2D keypoint coordinates as input.
MODEL_INPUT_KEYPOINT_MASK_TYPE_MASK_KEYPOINTS_AND_AS_INPUT = (
    'MASK_KEYPOINTS_AND_AS_INPUT')
# Supported model input keypoint mask types.
SUPPORTED_MODEL_INPUT_KEYPOINT_MASK_TYPES = [
    MODEL_INPUT_KEYPOINT_MASK_TYPE_NO_USE,
    MODEL_INPUT_KEYPOINT_MASK_TYPE_MASK_KEYPOINTS,
    MODEL_INPUT_KEYPOINT_MASK_TYPE_AS_INPUT,
    MODEL_INPUT_KEYPOINT_MASK_TYPE_MASK_KEYPOINTS_AND_AS_INPUT,
]

# Base model types.
# Simple Baseline architecutre: Martinez, et al. A simple yet effective baseline
# for 3d human pose estimation. ICCV 2017.
BASE_MODEL_TYPE_SIMPLE = 'SIMPLE'
# Temporal Simple Baseline.
BASE_MODEL_TYPE_TEMPORAL_SIMPLE = 'TEMPORAL_SIMPLE'
# Temporal Simple Baseline late fusion.
BASE_MODEL_TYPE_TEMPORAL_SIMPLE_LATE_FUSE = 'TEMPORAL_SIMPLE_LATE_FUSE'
# Supported base model types.
SUPPORTED_BASE_MODEL_TYPES = [
    BASE_MODEL_TYPE_SIMPLE,
    BASE_MODEL_TYPE_TEMPORAL_SIMPLE,
    BASE_MODEL_TYPE_TEMPORAL_SIMPLE_LATE_FUSE,
]

# Embedding types.
# Point embedding.
EMBEDDING_TYPE_POINT = 'POINT'
# Gaussian embedding with diagonal covariance matrix.
EMBEDDING_TYPE_GAUSSIAN = 'GAUSSIAN'
# Gaussian embedding with scalar variance.
EMBEDDING_TYPE_GAUSSIAN_SCALAR_VAR = 'GAUSSIAN_SCALAR_VAR'
# Supported embedding types.
SUPPORTED_EMBEDDING_TYPES = [
    EMBEDDING_TYPE_POINT,
    EMBEDDING_TYPE_GAUSSIAN,
    EMBEDDING_TYPE_GAUSSIAN_SCALAR_VAR,
]

# Embedding distance types.
# Distance computed using embedding centers.
DISTANCE_TYPE_CENTER = 'CENTER'
# Distance computed using embedding samples.
DISTANCE_TYPE_SAMPLE = 'SAMPLE'
# Distance computed using both embedding centers and samples.
DISTANCE_TYPE_CENTER_AND_SAMPLE = 'CENTER_AND_SAMPLE'
# Supported distance types.
SUPPORTED_DISTANCE_TYPES = [
    DISTANCE_TYPE_CENTER,
    DISTANCE_TYPE_SAMPLE,
    DISTANCE_TYPE_CENTER_AND_SAMPLE,
]

# Embedding distance pair types.
# Reduces distances between all pairs between two lists of samples.
DISTANCE_PAIR_TYPE_ALL_PAIRS = 'ALL_PAIRS'
# Reduces distances only between corrresponding pairs between two lists of
# samples.
DISTANCE_PAIR_TYPE_CORRESPONDING_PAIRS = 'CORRESPONDING_PAIRS'
# Supported distance pair types.
SUPPORTED_DISTANCE_PAIR_TYPES = [
    DISTANCE_PAIR_TYPE_ALL_PAIRS,
    DISTANCE_PAIR_TYPE_CORRESPONDING_PAIRS,
]

# Embedding distance kernels.
# Squared L2 distance.
DISTANCE_KERNEL_SQUARED_L2 = 'SQUARED_L2'
# L2-based sigmoid matching probability.
DISTANCE_KERNEL_L2_SIGMOID_MATCHING_PROB = 'L2_SIGMOID_MATCHING_PROB'
# Squared L2-based sigmoid matching probability.
DISTANCE_KERNEL_SQUARED_L2_SIGMOID_MATCHING_PROB = (
    'SQUARED_L2_SIGMOID_MATCHING_PROB')
# Expected likelihood.
DISTANCE_KERNEL_EXPECTED_LIKELIHOOD = 'EXPECTED_LIKELIHOOD'
# Supported distance kernels.
SUPPORTED_DISTANCE_KERNELS = [
    DISTANCE_KERNEL_SQUARED_L2,
    DISTANCE_KERNEL_L2_SIGMOID_MATCHING_PROB,
    DISTANCE_KERNEL_SQUARED_L2_SIGMOID_MATCHING_PROB,
    DISTANCE_KERNEL_EXPECTED_LIKELIHOOD,
]

# Embedding distance reductions.
# Mean of all distances.
DISTANCE_REDUCTION_MEAN = 'MEAN'
# Mean of distances not larger than the median of all distances.
DISTANCE_REDUCTION_LOWER_HALF_MEAN = 'LOWER_HALF_MEAN'
# Negative logarithm of the mean of all distances.
DISTANCE_REDUCTION_NEG_LOG_MEAN = 'NEG_LOG_MEAN'
# Negative logarithm of mean of distances no larger than the distance median.
DISTANCE_REDUCTION_LOWER_HALF_NEG_LOG_MEAN = 'LOWER_HALF_NEG_LOG_MEAN'
# One minus the mean of all distances.
DISTANCE_REDUCTION_ONE_MINUS_MEAN = 'ONE_MINUS_MEAN'
# Supported embedding distance reductions.
SUPPORTED_PAIRWISE_DISTANCE_REDUCTIONS = [
    DISTANCE_REDUCTION_MEAN,
    DISTANCE_REDUCTION_LOWER_HALF_MEAN,
    DISTANCE_REDUCTION_NEG_LOG_MEAN,
    DISTANCE_REDUCTION_LOWER_HALF_NEG_LOG_MEAN,
    DISTANCE_REDUCTION_ONE_MINUS_MEAN,
]
SUPPORTED_COMPONENTWISE_DISTANCE_REDUCTIONS = [DISTANCE_REDUCTION_MEAN]

# 3D keypoint distance measurement type.
# Normalized/Procrustes-aligned MPJPE.
KEYPOINT_DISTANCE_TYPE_MPJPE = 'MPJPE'
# Supported 3D keypoint distance measurement type.
SUPPORTED_KEYPOINT_DISTANCE_TYPES = [KEYPOINT_DISTANCE_TYPE_MPJPE]

# Activation function names.
ACTIVATION_FN_NONE = 'NONE'
ACTIVATION_FN_RELU = 'RELU'


def validate(value, supported_values):
  """Validates if value is supported.

  Args:
    value: A Python variable.
    supported_values: A list of supported variable values.

  Raises:
    ValueError: If `value` is not in `supported_values`.
  """
  if value not in supported_values:
    raise ValueError('Unsupported value for `%s`: `%s`.' % (value.name, value))
