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

"""A model for training depth egomotion prediction."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import tensorflow.compat.v1 as tf

from depth_and_motion_learning import depth_prediction_nets
from depth_and_motion_learning import intrinsics_utils
from depth_and_motion_learning import maybe_summary
from depth_and_motion_learning import object_motion_nets
from depth_and_motion_learning import parameter_container
from depth_and_motion_learning import transform_utils
from depth_and_motion_learning.dataset import data_processing
from depth_and_motion_learning.dataset import reader_cityscapes
from depth_and_motion_learning.losses import loss_aggregator

DEFAULT_PARAMS = {
    'batch_size': None,
    'input': {
        'data_path': '',

        # If the average L1 distance between two image is less than this
        # threshold, they will be assumed to be near duplicates - a situation
        # that happens often in robot footage, when the camera and the scene is
        # static.
        'duplicates_filter_threshold': 0.01,

        # Size of shuffling queue. Larger - better shuffling. Smaller - faster
        # and less host memory usage.
        'shuffle_queue_size': 1024,

        # Used in tf.data.Dataset.prefetch.
        'prefetch_size': 32,

        # Allows arbitrary parameters to be passed to the reader.
        'reader': {},
    },
    'image_preprocessing': {
        'data_augmentation': True,
        # Size into which images will be resized, after random cropping.
        'image_height': 128,
        'image_width': 416,
    },
    'loss_weights': {
        'rgb_consistency': 1.0,
        'ssim': 3.0,
        'depth_consistency': 0.0,
        'depth_smoothing': 0.001,
        'depth_supervision': 0.0,
        'rotation_cycle_consistency': 1e-3,
        'translation_cycle_consistency': 5e-2,
        'depth_variance': 0.0,
        'motion_smoothing': 1.0,
        'motion_drift': 0.2,
    },
    'loss_params': {
        # Stops gradient on the target depth when computing the depth
        # consistency loss.
        'target_depth_stop_gradient': True,
        # Normalize the scale by the mean depth.
        'scale_normalization': False,
    },
    'depth_predictor_params': {
        'layer_norm_noise_rampup_steps': 10000,
        'weight_decay': 0.0,
        'learn_scale': False,
        'reflect_padding': False,
    },
    'motion_prediction_params': {
        'weight_reg': 0.0,
        'align_corners': True,
        'auto_mask': True,
    },
    'learn_intrinsics': {
        'enabled': False,

        # If True, learn the same set of intrinsic params will be assigned to a
        # given video_id (works with the YouTube format in /dataset).
        'per_video': False,

        # If per_video is true, this is the maximal number of video ids for
        # which the hash table that keeps track of the intrsinsics.
        'max_number_of_videos': 1000,
    },

    # True to feed depth predictions into the motion field network.
    'cascade': True,
    # True to use a pretrained mask network to confine moving objects.
    'use_mask': False,
    'learn_egomotion': True,

    # Number of pixels ro dilate the foreground mask by (0 to not dilate).
    'foreground_dilation': 8,

    # If nonzero, motion fields will be unfrozen after motion_field_burnin_steps
    # steps. Over the first half of the motion_field_burnin_steps steps, the
    # motion fields will be zero. Then the ramp up is linear.
    'motion_field_burnin_steps': 20000,

    # TPUEstimator keys, to allow strict ParameterContainer usage.
    'context': None,
    'use_tpu': None,
}


def loss_fn(features, mode, params):
  """Computes the training loss for depth and egomotion training.

  This function is written with TPU-friendlines in mind.

  Args:
    features: A dictionary mapping strings to tuples of (tf.Tensor, tf.Tensor),
      representing pairs of frames. The loss will be calculated from these
      tensors. The expected endpoints are 'rgb', 'depth', 'intrinsics_mat'
      and 'intrinsics_mat_inv'.
    mode: One of tf.estimator.ModeKeys: TRAIN, PREDICT or EVAL.
    params: A dictionary with hyperparameters that optionally override
      DEFAULT_PARAMS above.

  Returns:
    A dictionary mapping each loss name (see DEFAULT_PARAMS['loss_weights']'s
    keys) to a scalar tf.Tensor representing the respective loss. The total
    training loss.

  Raises:
    ValueError: `features` endpoints that don't conform with their expected
       structure.
  """
  params = parameter_container.ParameterContainer.from_defaults_and_overrides(
      DEFAULT_PARAMS, params, is_strict=True, strictness_depth=2)

  if len(features['rgb']) != 2 or 'depth' in features and len(
      features['depth']) != 2:
    raise ValueError('RGB and depth endpoints are expected to be a tuple of two'
                     ' tensors. Rather, they are %s.' % str(features))

  # On tpu we strive to stack tensors together and perform ops once on the
  # entire stack, to save time HBM memory. We thus stack the batch-of-first-
  # frames and the batch-of-second frames, for both depth and RGB. The batch
  # dimension of rgb_stack and gt_depth_stack are thus twice the original batch
  # size.
  rgb_stack = tf.concat(features['rgb'], axis=0)

  depth_predictor = depth_prediction_nets.ResNet18DepthPredictor(
      mode, params.depth_predictor_params.as_dict())
  predicted_depth = depth_predictor.predict_depth(rgb_stack)
  maybe_summary.histogram('PredictedDepth', predicted_depth)

  endpoints = {}
  endpoints['predicted_depth'] = tf.split(predicted_depth, 2, axis=0)
  endpoints['rgb'] = features['rgb']

  # We make the heuristic that depths that are less than 0.2 meters are not
  # accurate. This is a rough placeholder for a confidence map that we're going
  # to have in future.
  if 'depth' in features:
    endpoints['groundtruth_depth'] = features['depth']

  if params.cascade:
    motion_features = [
        tf.concat([features['rgb'][0], endpoints['predicted_depth'][0]],
                  axis=-1),
        tf.concat([features['rgb'][1], endpoints['predicted_depth'][1]],
                  axis=-1)
    ]
  else:
    motion_features = features['rgb']

  motion_features_stack = tf.concat(motion_features, axis=0)
  flipped_motion_features_stack = tf.concat(motion_features[::-1], axis=0)
  # Unlike `rgb_stack`, here we stacked the frames in reverse order along the
  # Batch dimension. By concatenating the two stacks below along the channel
  # axis, we create the following tensor:
  #
  #         Channel dimension (3)
  #   _                                 _
  #  |  Frame1-s batch | Frame2-s batch  |____Batch
  #  |_ Frame2-s batch | Frame1-s batch _|    dimension (0)
  #
  # When we send this tensor to the motion prediction network, the first and
  # second halves of the result represent the camera motion from Frame1 to
  # Frame2 and from Frame2 to Frame1 respectively. Further below we impose a
  # loss that drives these two to be the inverses of one another
  # (cycle-consistency).
  pairs = tf.concat([motion_features_stack, flipped_motion_features_stack],
                    axis=-1)

  rot, trans, residual_translation, intrinsics_mat = (
      object_motion_nets.motion_field_net(
          images=pairs,
          weight_reg=params.motion_prediction_params.weight_reg,
          align_corners=params.motion_prediction_params.align_corners,
          auto_mask=params.motion_prediction_params.auto_mask))

  if params.motion_field_burnin_steps > 0.0:
    step = tf.to_float(tf.train.get_or_create_global_step())
    burnin_steps = tf.to_float(params.motion_field_burnin_steps)
    residual_translation *= tf.clip_by_value(2 * step / burnin_steps - 1, 0.0,
                                             1.0)

  # If using grouth truth egomotion
  if not params.learn_egomotion:
    egomotion_mat = tf.concat(features['egomotion_mat'], axis=0)
    rot = transform_utils.angles_from_matrix(egomotion_mat[:, :3, :3])
    trans = egomotion_mat[:, :3, 3]
    trans = tf.expand_dims(trans, 1)
    trans = tf.expand_dims(trans, 1)

  if params.use_mask:
    mask = tf.to_float(tf.concat(features['mask'], axis=0) > 0)
    if params.foreground_dilation > 0:
      pool_size = params.foreground_dilation * 2 + 1
      mask = tf.nn.max_pool(mask, [1, pool_size, pool_size, 1], [1] * 4, 'SAME')
    residual_translation *= mask

  maybe_summary.histogram('ResidualTranslation', residual_translation)
  maybe_summary.histogram('BackgroundTranslation', trans)
  maybe_summary.histogram('Rotation', rot)
  endpoints['residual_translation'] = tf.split(residual_translation, 2, axis=0)
  endpoints['background_translation'] = tf.split(trans, 2, axis=0)
  endpoints['rotation'] = tf.split(rot, 2, axis=0)

  if not params.learn_intrinsics.enabled:
    endpoints['intrinsics_mat'] = features['intrinsics_mat']
    endpoints['intrinsics_mat_inv'] = features['intrinsics_mat_inv']
  elif params.learn_intrinsics.per_video:
    int_mat = intrinsics_utils.create_and_fetch_intrinsics_per_video_index(
        features['video_index'][0],
        params.image_preprocessing.image_height,
        params.image_preprocessing.image_width,
        max_video_index=params.learn_intrinsics.max_number_of_videos)
    endpoints['intrinsics_mat'] = tf.concat([int_mat] * 2, axis=0)
    endpoints['intrinsics_mat_inv'] = intrinsics_utils.invert_intrinsics_matrix(
        int_mat)
  else:
    # The intrinsic matrix should be the same, no matter the order of
    # images (mat = inv_mat). It's probably a good idea to enforce this
    # by a loss, but for now we just take their average as a prediction for the
    # intrinsic matrix.
    intrinsics_mat = 0.5 * sum(tf.split(intrinsics_mat, 2, axis=0))
    endpoints['intrinsics_mat'] = [intrinsics_mat] * 2
    endpoints['intrinsics_mat_inv'] = [
        intrinsics_utils.invert_intrinsics_matrix(intrinsics_mat)] * 2

  aggregator = loss_aggregator.DepthMotionFieldLossAggregator(
      endpoints, params.loss_weights.as_dict(), params.loss_params.as_dict())

  # Add some more summaries.
  maybe_summary.image('rgb0', features['rgb'][0])
  maybe_summary.image('rgb1', features['rgb'][1])
  disp0, disp1 = tf.split(aggregator.output_endpoints['disparity'], 2, axis=0)
  maybe_summary.image('disparity0/grayscale', disp0)
  maybe_summary.image_with_colormap('disparity0/plasma',
                                    tf.squeeze(disp0, axis=3), 'plasma', 0.0)
  maybe_summary.image('disparity1/grayscale', disp1)
  maybe_summary.image_with_colormap('disparity1/plasma',
                                    tf.squeeze(disp1, axis=3), 'plasma', 0.0)
  if maybe_summary.summaries_enabled():
    if 'depth' in features:
      gt_disp0 = 1.0 / tf.maximum(features['depth'][0], 0.5)
      gt_disp1 = 1.0 / tf.maximum(features['depth'][1], 0.5)
      maybe_summary.image('disparity_gt0', gt_disp0)
      maybe_summary.image('disparity_gt1', gt_disp1)

    depth_proximity_weight0, depth_proximity_weight1 = tf.split(
        aggregator.output_endpoints['depth_proximity_weight'], 2, axis=0)
    maybe_summary.image('consistency_weight0',
                        tf.expand_dims(depth_proximity_weight0, -1))
    maybe_summary.image('consistency_weight1',
                        tf.expand_dims(depth_proximity_weight1, -1))
    maybe_summary.image('trans', aggregator.output_endpoints['trans'])
    maybe_summary.image('trans_inv', aggregator.output_endpoints['inv_trans'])
    maybe_summary.image('trans_res', endpoints['residual_translation'][0])
    maybe_summary.image('trans_res_inv', endpoints['residual_translation'][1])

  return aggregator.losses


def input_fn(params):
  """An Estimator's input_fn for reading and preprocessing training data.

  Reads pairs of RGBD frames from sstables, filters out near duplicates and
  performs data augmentation.

  Args:
    params: A dictionary with hyperparameters.

  Returns:
    A tf.data.Dataset object.
  """

  params = parameter_container.ParameterContainer.from_defaults_and_overrides(
      DEFAULT_PARAMS, params, is_strict=True, strictness_depth=2)
  dataset = reader_cityscapes.read_frame_pairs_from_data_path(
      params.input.data_path, params.input.reader)

  if params.learn_intrinsics.enabled and params.learn_intrinsics.per_video:
    intrinsics_ht = intrinsics_utils.HashTableIndexer(
        params.learn_intrinsics.max_number_of_videos)

  def key_to_index(input_endpoints):
    video_id = input_endpoints.pop('video_id', None)
    if (video_id is not None and params.learn_intrinsics.enabled and
        params.learn_intrinsics.per_video):
      index = intrinsics_ht.get_or_create_index(video_id[0])
      input_endpoints['video_index'] = index
      input_endpoints['video_index'] = tf.stack([index] * 2)
    return input_endpoints

  dataset = dataset.map(key_to_index)

  def is_duplicate(endpoints):
    """Implements a simple duplicate filter, based on L1 difference in RGB."""
    return tf.greater(
        tf.reduce_mean(tf.abs(endpoints['rgb'][1] - endpoints['rgb'][0])),
        params.input.duplicates_filter_threshold)

  if params.input.duplicates_filter_threshold > 0.0:
    dataset = dataset.filter(is_duplicate)

  # Add data augmentation
  if params.image_preprocessing.data_augmentation:
    if params.learn_intrinsics.per_video:
      raise ('Data augemnation together with learn_intrinsics.per_video is not '
             'yet supported.')

    def random_crop_and_resize_fn(endpoints):
      return data_processing.random_crop_and_resize_pipeline(
          endpoints, params.image_preprocessing.image_height,
          params.image_preprocessing.image_width)

    augmentation_fn = random_crop_and_resize_fn
  else:

    def resize_fn(endpoints):
      return data_processing.resize_pipeline(
          endpoints, params.image_preprocessing.image_height,
          params.image_preprocessing.image_width)

    augmentation_fn = resize_fn

  dataset = dataset.map(augmentation_fn)
  dataset = dataset.shuffle(params.input.shuffle_queue_size)
  dataset = dataset.batch(params.batch_size, drop_remainder=True)
  return dataset.prefetch(params.input.prefetch_size)


def get_vars_to_restore_fn(initialization):
  """Returns a vars_to_restore_fn for various types of `initialization`.

  Args:
    initialization: A string, the type of the initialization. Currently only
      'imagenet' is supported.

  Raises:
    ValueError: `initialization` is not supported
  """

  if initialization == 'imagenet':

    def is_blacklisted(name):
      for key in ['Adam', 'iconv', 'depth_scale', 'upconv', 'disp']:
        if key in name:
          return True
      return False

    def vars_to_restore_fn():
      """Returns a dictionary mapping checkpoint variable names to variables."""
      vars_to_restore = {}
      for v in tf.global_variables():
        if is_blacklisted(v.op.name):
          print(v.op.name, 'is blacklisted')
          continue
        if v.op.name.startswith('depth_prediction'):
          name = v.op.name.replace('moving_mean', 'mu')
          name = name.replace('moving_variance', 'sigma')
          vars_to_restore[name[len('depth_prediction') + 1:]] = v
      return vars_to_restore

    return vars_to_restore_fn

  else:
    raise ValueError('Unknown initialization %s' % initialization)


def preprocess_masks(endpoints):

  def create_mobile_mask(input_mask):
    return tf.reduce_all(tf.not_equal(0, input_mask), axis=2, keepdims=True)

  output = dict(endpoints)
  output['mask'] = tuple([create_mobile_mask(m) for m in endpoints['mask']])
  return output


def infer_depth(rgb_image, params):
  """Runs depth inference given an RGB frame.

  Args:
    rgb_image: A tf.Tensor or shape [B, H, W, 3] containing RGB images.
    params: A dictionary of parameters contraining overrides for
      DEFAULT_PARAMS.

  Returns:
    A tf.Tensor of shape [B, H, W, 1] containing the inferred depths.
  """
  if rgb_image.shape.rank != 4:
    raise ValueError('rgb_image should have rank 4, not %d.' %
                     rgb_image.shape.rank)
  params = parameter_container.ParameterContainer.from_defaults_and_overrides(
      DEFAULT_PARAMS, params, is_strict=True, strictness_depth=2)

  depth_predictor = depth_prediction_nets.ResNet18DepthPredictor(
      tf.estimator.ModeKeys.PREDICT, params.depth_predictor_params.as_dict())
  return depth_predictor.predict_depth(rgb_image)


def infer_egomotion(rgb_image1, rgb_image2, params):
  """Runs egomotion inference given two RGB frames.

  Args:
    rgb_image1: A tf.Tensor or shape [B, H, W, 3] containing RGB images, the
      first frame.
    rgb_image2: A tf.Tensor or shape [B, H, W, 3] containing RGB images, the
      second frame.
    params: A dictionary of parameters contraining overrides for DEFAULT_PARAMS.

  Returns:
    A tuple of two tf.Tensors of shape [B, 3] containing the inferred rotation
    angles and translation vector components.
  """
  params = parameter_container.ParameterContainer.from_defaults_and_overrides(
      DEFAULT_PARAMS, params, is_strict=True, strictness_depth=2)
  if rgb_image1.shape.rank != 4 or rgb_image2.shape.rank != 4:
    raise ValueError('rgb_image1 and rgb_image1 should have rank 4, not '
                     '%d and %d.' %
                     (rgb_image1.shape.rank, rgb_image2.shape.rank))
  rgb_stack = tf.concat([rgb_image1, rgb_image2], axis=0)
  flipped_rgb_stack = tf.concat([rgb_image2, rgb_image1], axis=0)

  rot, trans, _ = object_motion_nets.motion_vector_net(tf.concat(
      [rgb_stack, flipped_rgb_stack], axis=3), 0.0, False)

  rot12, rot21 = tf.split(rot, 2, axis=0)
  trans12, trans21 = tf.split(trans, 2, axis=0)

  # rot12 and rot21 should be the inverses on of the other, but in reality they
  # not exactly are. Averaging rot12 and inv(rot21) gives a better estimator for
  # the rotation. Similarly, trans12 and rot12*trans21 should be the negatives
  # one of the other, so we average rot12*trans21 and trans12
  # to get a better estimator. TODO(gariel): Check if there's an estimator
  # with less variance.
  avg_rot = 0.5 * (tf.linalg.inv(rot21) + rot12)
  avg_trans = 0.5 * (-tf.squeeze(
      tf.matmul(rot12, tf.expand_dims(trans21, -1)), axis=-1) + trans12)

  return avg_rot, avg_trans
