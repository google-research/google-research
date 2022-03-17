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

"""Pose embedding model training base code."""

import functools
import math

from absl import flags
import tensorflow.compat.v1 as tf
import tf_slim

from poem.core import data_utils
from poem.core import input_generator
from poem.core import keypoint_utils
from poem.core import loss_utils
from poem.core import pipeline_utils
from poem.core import visualization_utils

FLAGS = flags.FLAGS

flags.DEFINE_list('input_table', None,
                  'CSV of input tf.Example training table patterns.')
flags.mark_flag_as_required('input_table')

flags.DEFINE_string('train_log_dir', None,
                    'Directory to save checkpoints and summary logs.')
flags.mark_flag_as_required('train_log_dir')

flags.DEFINE_string('input_keypoint_profile_name_3d', 'LEGACY_3DH36M17',
                    'Profile name for input 3D keypoints.')

flags.DEFINE_string(
    'input_keypoint_profile_name_2d', 'LEGACY_2DCOCO13',
    'Profile name for 2D keypoints from input sources. Use None to ignore input'
    ' 2D keypoints.')

# See `common_module.SUPPORTED_TRAINING_MODEL_INPUT_KEYPOINT_TYPES`.
flags.DEFINE_string('model_input_keypoint_type', '2D_INPUT_AND_3D_PROJECTION',
                    'Type of model input keypoints.')

flags.DEFINE_float(
    'uniform_keypoint_jittering_max_offset_2d', 0.0,
    'Maximum 2D keypoint jittering offset. Random jittering offset within '
    '[-uniform_keypoint_jittering_max_offset_2d, '
    'uniform_keypoint_jittering_max_offset_2d] is to be added to each keypoint '
    '2D. Note that the jittering happens after the 2D normalization. Ignored if'
    ' non-positive.')

flags.DEFINE_float(
    'gaussian_keypoint_jittering_offset_stddev_2d', 0.0,
    'Standard deviation of Gaussian 2D keypoint jittering offset. Random '
    'jittering offset sampled from N(0, '
    'gaussian_keypoint_jittering_offset_stddev_2d) is to be added to each '
    'keypoints. Note that the jittering happens after the 2D normalization. '
    'Ignored if non-positive.')

flags.DEFINE_string('model_input_keypoint_mask_type', 'NO_USE',
                    'Usage type of model input keypoint masks.')

flags.DEFINE_float(
    'min_input_keypoint_score_2d', -1.0,
    'Minimum threshold for input keypoint score binarization. Use negative '
    'value to ignore. Only used if 2D keypoint masks are used.')

flags.DEFINE_list(
    'keypoint_dropout_probs', ['0.0', '0.0'],
    'CSV of 2-tuple probability (probability_to_apply, probability_to_drop) for'
    ' performing stratified keypoint dropout.')

flags.DEFINE_bool('set_on_mask_for_non_anchors', False,
                  'Whether to always use on (1) masks for non-anchor samples.')

flags.DEFINE_bool(
    'mix_mask_sub_batches', False,
    'Whether to apply sub-batch mixing to processed masks and all-one masks.')

flags.DEFINE_list(
    'forced_mask_on_part_names', None,
    'CSV of standard names of parts of which the masks are forced on (setting '
    'value to 1.0) during training. See '
    '`KeypointProfile.get_standard_part_index` for standard part names.')

flags.DEFINE_list(
    'forced_mask_off_part_names', None,
    'CSV of standard names of parts of which the masks are forced off (setting '
    'value to 0.0) during training. See '
    '`KeypointProfile.get_standard_part_index` for standard part names.')

# See `common_module.SUPPORTED_EMBEDDING_TYPES`.
flags.DEFINE_string('embedding_type', 'GAUSSIAN', 'Type of embeddings.')

flags.DEFINE_integer('embedding_size', 16, 'Size of predicted embeddings.')

flags.DEFINE_integer(
    'num_embedding_components', 1,
    'Number of embedding components, e.g., the number of Gaussians in mixture.')

flags.DEFINE_integer('num_embedding_samples', 20,
                     'Number of samples from embedding distributions.')

# See `common_module.SUPPORTED_BASE_MODEL_TYPES`.
flags.DEFINE_string('base_model_type', 'SIMPLE', 'Type of base model.')

flags.DEFINE_integer('num_fc_blocks', 2, 'Number of fully connected blocks.')

flags.DEFINE_integer('num_fcs_per_block', 2,
                     'Number of fully connected layers per block.')

flags.DEFINE_integer('num_hidden_nodes', 1024,
                     'Number of nodes in each hidden fully connected layer.')

flags.DEFINE_integer(
    'num_bottleneck_nodes', 0,
    'Number of nodes in the bottleneck layer before the output layer(s). '
    'Ignored if non-positive.')

flags.DEFINE_float(
    'weight_max_norm', 0.0,
    'Maximum norm of fully connected layer weights. Only used if positive.')

flags.DEFINE_float('dropout_rate', 0.3,
                   'Dropout rate for the fully-connected layers.')

# See `common_module.SUPPORTED_KEYPOINT_DISTANCE_TYPES`.
flags.DEFINE_string(
    'keypoint_distance_type', 'MPJPE',
    'Type of 3D keypoint distance to determine negative match.')

flags.DEFINE_float(
    'min_negative_keypoint_mpjpe', 0.1,
    'Minimum MPJPE gap for two poses to be considered as negative match. Only'
    ' used if `keypoint_distance_type` is `MPJPE`. If negative, uses all other '
    'samples as negative matches.')

# See `common_module.SUPPORTED_DISTANCE_TYPES`.
flags.DEFINE_string('triplet_distance_type', 'SAMPLE',
                    'Type of distance to use in triplet loss computation.')

# See `common_module.SUPPORTED_DISTANCE_KERNELS`.
flags.DEFINE_string('triplet_distance_kernel', 'L2_SIGMOID_MATCHING_PROB',
                    'Distance kernel to use in triplet loss computation.')

# See `common_module.SUPPORTED_DISTANCE_REDUCTIONS`.
flags.DEFINE_string(
    'triplet_pairwise_reduction', 'NEG_LOG_MEAN',
    'Pairwise reducer to use in triplet loss computation. Use default value if '
    '`triplet_distance_type` is `CENTER`.')

# See `common_module.SUPPORTED_COMPONENTWISE_DISTANCE_REDUCTIONS`.
flags.DEFINE_string(
    'triplet_componentwise_reduction', 'MEAN',
    'Component-wise reducer to use in triplet loss computation.')

flags.DEFINE_float('triplet_loss_margin', 0.69314718056, 'Triplet loss margin.')

flags.DEFINE_bool(
    'use_inferred_keypoint_masks_for_triplet_label', False,
    'Whether to infer 3D keypoint masks from input 2D keypoint masks and apply'
    'them to compute triplet labels. If True, surjective mapping is required '
    'from input 2D keypoint profile to 3D keypoint profile.')

flags.DEFINE_bool(
    'use_normalized_embeddings_for_triplet_loss', False,
    'Whether to use normalized embeddings for triplet loss computation.')

flags.DEFINE_bool(
    'use_normalized_embeddings_for_triplet_mining', None,
    'Whether to use normalized embeddings for mining triplets. Use None for the'
    ' same value as `use_normalized_embeddings_for_triplet_loss`.')

flags.DEFINE_bool(
    'use_semi_hard_triplet_negatives', True,
    'Whether to mine semi-hard triplets instead of hard triplets.')

flags.DEFINE_bool('exclude_inactive_triplet_loss', True,
                  'Whether to exclude inactive triplets in loss computation.')

flags.DEFINE_float(
    'kl_regularization_loss_weight', 0.001,
    'Weight for KL regularization loss. Use 0 to ignore this loss.')

flags.DEFINE_float(
    'kl_regularization_prior_stddev', 1.0,
    'Standard deviation of prior Gaussian distributions to compute KL '
    'regularization loss')

flags.DEFINE_float(
    'positive_pairwise_loss_weight', 0.005,
    'Weight for positive pairwise loss. Use 0 to ignore this loss.')

# See `common_module.SUPPORTED_DISTANCE_TYPES`.
flags.DEFINE_string(
    'positive_pairwise_distance_type', None,
    'Type of distance to use in positive_pairwise loss computation. Use None '
    'for the same value as `triplet_distance_type`.')

# See `common_module.SUPPORTED_DISTANCE_KERNELS`.
flags.DEFINE_string(
    'positive_pairwise_distance_kernel', None,
    'Distance kernel to use in positive pairwise loss computation. Use None for'
    ' the same value as `triplet_distance_kernel`.')

# See `common_module.SUPPORTED_PAIRWISE_DISTANCE_REDUCTIONS`.
flags.DEFINE_string(
    'positive_pairwise_pairwise_reduction', None,
    'Pairwise reducer to use in positive_pairwise loss computation. Use default'
    ' value if `triplet_distance_type` is `CENTER`. Use None for the same value'
    ' as `triplet_pairwise_reduction`.')

# See `common_module.SUPPORTED_COMPONENTWISE_DISTANCE_REDUCTIONS`.
flags.DEFINE_string(
    'positive_pairwise_componentwise_reduction', None,
    'Component-wise reducer to use in positive_pairwise loss computation. Use '
    'None for the same value as `triplet_componentwise_reduction`.')

flags.DEFINE_bool(
    'use_normalized_embeddings_for_positive_pairwise_loss', None,
    'Whether to use normalized embeddings for positive pairwise loss '
    'computation. Use None for the same value as '
    '`use_normalized_embeddings_for_triplet_loss`.')

flags.DEFINE_list(
    'batch_size', ['256'],
    'CSV of training batch sizes to use from each input table. Must have the '
    'same length as `input_table`.')

flags.DEFINE_enum('optimizer', 'ADAGRAD', ['ADAGRAD', 'ADAM', 'ADAMW'],
                  'Optimizer to use.')

flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')

flags.DEFINE_enum('learning_rate_schedule', '',
                  ['', 'EXP_DECAY', 'LINEAR_DECAY'],
                  'Learning rate schedule to use. Use empty to skip.')

flags.DEFINE_integer(
    'num_steps', 5000000,
    'Number of training steps. Use None to train indefinitely.')

flags.DEFINE_integer(
    'num_warmup_steps', None,
    'Number of linear warmup training steps. Use None to skip warmup.')

flags.DEFINE_string('init_model_checkpoint', None,
                    'Path to checkpoint to initialize from.')

flags.DEFINE_float('gradient_clip_norm', 0.0,
                   'Norm gradients are clipped to. Only used if positive.')

flags.DEFINE_float(
    'gradient_clip_global_norm', 0.0,
    'Global norm gradients are clipped to. Only used if positive.')

flags.DEFINE_bool('use_moving_average', True,
                  'Whether to use exponential moving average.')

flags.DEFINE_float(
    'moving_average_decay', 0.9999,
    'Exponential moving average decay. Only used if `use_moving_average` is '
    'True.')

flags.DEFINE_integer(
    'input_shuffle_buffer_size', 2097152,
    'Input shuffle buffer size. A large number beneifts shuffling quality.')

flags.DEFINE_float(
    'sigmoid_raw_a_initial', -0.65,
    'Initial value of sigmoid `raw_a` parameter. We initialize the sigmoid '
    'parameters to a constant to avoid model being stuck in a `dead zone` at '
    'the beginning of training. The actual value of `a` will be ELU(raw_a) + '
    '1.')

flags.DEFINE_float(
    'sigmoid_b_initial', -0.5,
    'Initial value of sigmoid `b` parameter. We initialize the sigmoid '
    'parameters to a constant to avoid model being stuck in a `dead zone` at '
    'the beginning of training.')

flags.DEFINE_float(
    'sigmoid_a_max', -1.0,
    'Maximum value of sigmoid `a` parameter. Ignored if None or non-positive.')

flags.DEFINE_list(
    'random_projection_azimuth_range', ['-180.0', '180.0'],
    'CSV of 2-tuple rotation angle limit (lower_limit, upper_limit) for '
    'performing random azimuth rotations on 3D poses before projection for '
    'camera augmentation. For sequential inputs, also supports CSV of 4-tuple '
    'for (starting_lower_limit, starting_upper_limit, delta_lower_limit, '
    'delta_upper_limit).')

flags.DEFINE_list(
    'random_projection_elevation_range', ['-30.0', '30.0'],
    'CSV of 2-tuple rotation angle limit (lower_limit, upper_limit) for '
    'performing random elevation rotations on 3D poses before projection for '
    'camera augmentation. For sequential inputs, also supports CSV of 4-tuple '
    'for (starting_lower_limit, starting_upper_limit, delta_lower_limit, '
    'delta_upper_limit).')

flags.DEFINE_list(
    'random_projection_roll_range', ['-30.0', '30.0'],
    'CSV of 2-tuple rotation angle limit (lower_limit, upper_limit) for '
    'performing random roll rotations on 3D poses before projection for camera '
    'augmentation. For sequential inputs, also supports CSV of 4-tuple '
    'for (starting_lower_limit, starting_upper_limit, delta_lower_limit, '
    'delta_upper_limit).')

flags.DEFINE_list(
    'random_projection_camera_depth_range', [],
    'CSV of 2-tuple depth limit (lower_limit, upper_limit) for performing '
    'random camera positioning from 3D poses before projection for camera '
    'augmentation.')

flags.DEFINE_bool('profile_only', False,
                  'Whether to profile the training graph and exit.')

flags.DEFINE_integer('startup_delay_steps', 15, 'Startup step delay.')

flags.DEFINE_integer(
    'save_summaries_secs', 120,
    'Time interval in seconds between which summaries are saved.')

flags.DEFINE_integer(
    'save_interval_secs', 300,
    'Time interval in seconds between which a model checkpoint is saved.')

flags.DEFINE_float('keep_checkpoint_every_n_hours', 0.5,
                   'How often in hours to keep checkpoints.')

flags.DEFINE_integer('log_every_n_steps', 100,
                     'Number of steps between which summaries are logged.')

flags.DEFINE_bool('summarize_gradients', False,
                  'Whether to summarize gradients.')

flags.DEFINE_bool('summarize_inputs', False,
                  'Whether to visualize input 2D poses.')

flags.DEFINE_bool(
    'summarize_percentiles', True,
    'Whether to summarize percentiles of certain variables, e.g., embedding '
    'distances in triplet loss. Consider turning this off in case '
    'tensorflow_probability percentile computation causes failures at random '
    'due to empty tensor.')

flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'Number of parameter servers. If the value is 0, the parameters are '
    'handled locally by the worker.')

flags.DEFINE_integer('task', 0, 'Task replica identifier for training.')


def _validate_and_setup(common_module, keypoint_profiles_module, models_module,
                        keypoint_distance_config_override,
                        create_model_input_fn_kwargs, embedder_fn_kwargs):
  """Validates and sets up training configurations."""
  # Set default values for unspecified flags.
  if FLAGS.use_normalized_embeddings_for_triplet_mining is None:
    FLAGS.use_normalized_embeddings_for_triplet_mining = (
        FLAGS.use_normalized_embeddings_for_triplet_loss)
  if FLAGS.use_normalized_embeddings_for_positive_pairwise_loss is None:
    FLAGS.use_normalized_embeddings_for_positive_pairwise_loss = (
        FLAGS.use_normalized_embeddings_for_triplet_loss)
  if FLAGS.positive_pairwise_distance_type is None:
    FLAGS.positive_pairwise_distance_type = FLAGS.triplet_distance_type
  if FLAGS.positive_pairwise_distance_kernel is None:
    FLAGS.positive_pairwise_distance_kernel = FLAGS.triplet_distance_kernel
  if FLAGS.positive_pairwise_pairwise_reduction is None:
    FLAGS.positive_pairwise_pairwise_reduction = (
        FLAGS.triplet_pairwise_reduction)
  if FLAGS.positive_pairwise_componentwise_reduction is None:
    FLAGS.positive_pairwise_componentwise_reduction = (
        FLAGS.triplet_componentwise_reduction)

  # Validate flags.
  validate_flag = common_module.validate
  validate_flag(FLAGS.model_input_keypoint_type,
                common_module.SUPPORTED_TRAINING_MODEL_INPUT_KEYPOINT_TYPES)
  validate_flag(FLAGS.model_input_keypoint_mask_type,
                common_module.SUPPORTED_MODEL_INPUT_KEYPOINT_MASK_TYPES)
  validate_flag(FLAGS.embedding_type, common_module.SUPPORTED_EMBEDDING_TYPES)
  validate_flag(FLAGS.base_model_type, common_module.SUPPORTED_BASE_MODEL_TYPES)
  validate_flag(FLAGS.keypoint_distance_type,
                common_module.SUPPORTED_KEYPOINT_DISTANCE_TYPES)
  validate_flag(FLAGS.triplet_distance_type,
                common_module.SUPPORTED_DISTANCE_TYPES)
  validate_flag(FLAGS.triplet_distance_kernel,
                common_module.SUPPORTED_DISTANCE_KERNELS)
  validate_flag(FLAGS.triplet_pairwise_reduction,
                common_module.SUPPORTED_PAIRWISE_DISTANCE_REDUCTIONS)
  validate_flag(FLAGS.triplet_componentwise_reduction,
                common_module.SUPPORTED_COMPONENTWISE_DISTANCE_REDUCTIONS)
  validate_flag(FLAGS.positive_pairwise_distance_type,
                common_module.SUPPORTED_DISTANCE_TYPES)
  validate_flag(FLAGS.positive_pairwise_distance_kernel,
                common_module.SUPPORTED_DISTANCE_KERNELS)
  validate_flag(FLAGS.positive_pairwise_pairwise_reduction,
                common_module.SUPPORTED_PAIRWISE_DISTANCE_REDUCTIONS)
  validate_flag(FLAGS.positive_pairwise_componentwise_reduction,
                common_module.SUPPORTED_COMPONENTWISE_DISTANCE_REDUCTIONS)

  if FLAGS.embedding_type == common_module.EMBEDDING_TYPE_POINT:
    if FLAGS.triplet_distance_type != common_module.DISTANCE_TYPE_CENTER:
      raise ValueError(
          'No support for triplet distance type `%s` for embedding type `%s`.' %
          (FLAGS.triplet_distance_type, FLAGS.embedding_type))
    if FLAGS.kl_regularization_loss_weight > 0.0:
      raise ValueError(
          'No support for KL regularization loss for embedding type `%s`.' %
          FLAGS.embedding_type)

  if ((FLAGS.triplet_distance_type in [
      common_module.DISTANCE_TYPE_SAMPLE,
  ] or FLAGS.positive_pairwise_distance_type in [
      common_module.DISTANCE_TYPE_SAMPLE,
  ]) and FLAGS.num_embedding_samples <= 0):
    raise ValueError(
        'Must specify positive `num_embedding_samples` to use `%s` '
        'triplet/positive pairwise distance type.' %
        FLAGS.triplet_distance_type)

  if (((FLAGS.triplet_distance_kernel in [
      common_module.DISTANCE_KERNEL_L2_SIGMOID_MATCHING_PROB,
      common_module.DISTANCE_KERNEL_SQUARED_L2_SIGMOID_MATCHING_PROB,
      common_module.DISTANCE_KERNEL_EXPECTED_LIKELIHOOD,
  ]) != (FLAGS.triplet_pairwise_reduction in [
      common_module.DISTANCE_REDUCTION_NEG_LOG_MEAN,
      common_module.DISTANCE_REDUCTION_LOWER_HALF_NEG_LOG_MEAN,
      common_module.DISTANCE_REDUCTION_ONE_MINUS_MEAN
  ])) or ((FLAGS.positive_pairwise_distance_kernel in [
      common_module.DISTANCE_KERNEL_L2_SIGMOID_MATCHING_PROB,
      common_module.DISTANCE_KERNEL_SQUARED_L2_SIGMOID_MATCHING_PROB,
      common_module.DISTANCE_KERNEL_EXPECTED_LIKELIHOOD,
  ]) != (FLAGS.positive_pairwise_pairwise_reduction in [
      common_module.DISTANCE_REDUCTION_NEG_LOG_MEAN,
      common_module.DISTANCE_REDUCTION_LOWER_HALF_NEG_LOG_MEAN,
      common_module.DISTANCE_REDUCTION_ONE_MINUS_MEAN
  ]))):
    raise ValueError(
        'Must use `L2_SIGMOID_MATCHING_PROB` or `EXPECTED_LIKELIHOOD` distance '
        'kernel and `NEG_LOG_MEAN` or `LOWER_HALF_NEG_LOG_MEAN` parwise reducer'
        ' in pairs.')

  keypoint_profile_2d = keypoint_profiles_module.create_keypoint_profile_or_die(
      FLAGS.input_keypoint_profile_name_2d)

  # Set up configurations.
  configs = {
      'keypoint_profile_3d':
          keypoint_profiles_module.create_keypoint_profile_or_die(
              FLAGS.input_keypoint_profile_name_3d),
      'keypoint_profile_2d':
          keypoint_profile_2d,
      'create_model_input_fn':
          functools.partial(
              input_generator.create_model_input,
              model_input_keypoint_mask_type=(
                  FLAGS.model_input_keypoint_mask_type),
              uniform_keypoint_jittering_max_offset_2d=(
                  FLAGS.uniform_keypoint_jittering_max_offset_2d),
              gaussian_keypoint_jittering_offset_stddev_2d=(
                  FLAGS.gaussian_keypoint_jittering_offset_stddev_2d),
              keypoint_dropout_probs=[
                  float(x) for x in FLAGS.keypoint_dropout_probs
              ],
              set_on_mask_for_non_anchors=FLAGS.set_on_mask_for_non_anchors,
              mix_mask_sub_batches=FLAGS.mix_mask_sub_batches,
              forced_mask_on_part_names=FLAGS.forced_mask_on_part_names,
              forced_mask_off_part_names=FLAGS.forced_mask_off_part_names,
              **create_model_input_fn_kwargs),
      'embedder_fn':
          models_module.get_embedder(
              base_model_type=FLAGS.base_model_type,
              embedding_type=FLAGS.embedding_type,
              num_embedding_components=FLAGS.num_embedding_components,
              embedding_size=FLAGS.embedding_size,
              num_embedding_samples=FLAGS.num_embedding_samples,
              is_training=True,
              num_fc_blocks=FLAGS.num_fc_blocks,
              num_fcs_per_block=FLAGS.num_fcs_per_block,
              num_hidden_nodes=FLAGS.num_hidden_nodes,
              num_bottleneck_nodes=FLAGS.num_bottleneck_nodes,
              weight_max_norm=FLAGS.weight_max_norm,
              dropout_rate=FLAGS.dropout_rate,
              **embedder_fn_kwargs),
      'triplet_embedding_keys':
          pipeline_utils.get_embedding_keys(
              FLAGS.triplet_distance_type, common_module=common_module),
      'triplet_mining_embedding_keys':
          pipeline_utils.get_embedding_keys(
              FLAGS.triplet_distance_type,
              replace_samples_with_means=True,
              common_module=common_module),
      'positive_pairwise_embedding_keys':
          pipeline_utils.get_embedding_keys(
              FLAGS.positive_pairwise_distance_type,
              common_module=common_module),
      'summarize_matching_sigmoid_vars':
          FLAGS.triplet_distance_kernel in [
              common_module.DISTANCE_KERNEL_L2_SIGMOID_MATCHING_PROB,
              common_module.DISTANCE_KERNEL_SQUARED_L2_SIGMOID_MATCHING_PROB
          ],
      'random_projection_azimuth_range': [
          float(x) / 180.0 * math.pi
          for x in FLAGS.random_projection_azimuth_range
      ],
      'random_projection_elevation_range': [
          float(x) / 180.0 * math.pi
          for x in FLAGS.random_projection_elevation_range
      ],
      'random_projection_roll_range': [
          float(x) / 180.0 * math.pi for x in FLAGS.random_projection_roll_range
      ],
      'random_projection_camera_depth_range': [
          float(x) for x in FLAGS.random_projection_camera_depth_range
      ],
  }

  embedding_sample_distance_fn_kwargs = {
      'EXPECTED_LIKELIHOOD_min_stddev': 0.1,
      'EXPECTED_LIKELIHOOD_max_squared_mahalanobis_distance': 100.0,
  }
  if FLAGS.triplet_distance_kernel in [
      common_module.DISTANCE_KERNEL_L2_SIGMOID_MATCHING_PROB,
      common_module.DISTANCE_KERNEL_SQUARED_L2_SIGMOID_MATCHING_PROB
  ] or FLAGS.positive_pairwise_distance_kernel in [
      common_module.DISTANCE_KERNEL_L2_SIGMOID_MATCHING_PROB,
      common_module.DISTANCE_KERNEL_SQUARED_L2_SIGMOID_MATCHING_PROB
  ]:
    # We only need sigmoid parameters when a related distance kernel is used.
    sigmoid_raw_a, sigmoid_a, sigmoid_b = pipeline_utils.get_sigmoid_parameters(
        name='MatchingSigmoid',
        raw_a_initial_value=FLAGS.sigmoid_raw_a_initial,
        b_initial_value=FLAGS.sigmoid_b_initial,
        a_range=(None, FLAGS.sigmoid_a_max))
    embedding_sample_distance_fn_kwargs.update({
        'L2_SIGMOID_MATCHING_PROB_a': sigmoid_a,
        'L2_SIGMOID_MATCHING_PROB_b': sigmoid_b,
        'SQUARED_L2_SIGMOID_MATCHING_PROB_a': sigmoid_a,
        'SQUARED_L2_SIGMOID_MATCHING_PROB_b': sigmoid_b,
    })
    configs.update({
        'sigmoid_raw_a': sigmoid_raw_a,
        'sigmoid_a': sigmoid_a,
        'sigmoid_b': sigmoid_b,
    })

  configs.update({
      'triplet_embedding_sample_distance_fn':
          loss_utils.create_sample_distance_fn(
              pair_type=common_module.DISTANCE_PAIR_TYPE_ALL_PAIRS,
              distance_kernel=FLAGS.triplet_distance_kernel,
              pairwise_reduction=FLAGS.triplet_pairwise_reduction,
              componentwise_reduction=FLAGS.triplet_componentwise_reduction,
              **embedding_sample_distance_fn_kwargs),
      'positive_pairwise_embedding_sample_distance_fn':
          loss_utils.create_sample_distance_fn(
              pair_type=common_module.DISTANCE_PAIR_TYPE_ALL_PAIRS,
              distance_kernel=FLAGS.positive_pairwise_distance_kernel,
              pairwise_reduction=FLAGS.positive_pairwise_pairwise_reduction,
              componentwise_reduction=(
                  FLAGS.positive_pairwise_componentwise_reduction),
              **embedding_sample_distance_fn_kwargs),
  })

  if FLAGS.keypoint_distance_type == common_module.KEYPOINT_DISTANCE_TYPE_MPJPE:
    configs.update({
        'keypoint_distance_fn':
            keypoint_utils.compute_procrustes_aligned_mpjpes,
        'min_negative_keypoint_distance':
            FLAGS.min_negative_keypoint_mpjpe
    })
  # We use the following assignments to get around pytype check failures.
  # TODO(liuti): Figure out a better workaround.
  if 'keypoint_distance_fn' in keypoint_distance_config_override:
    configs['keypoint_distance_fn'] = (
        keypoint_distance_config_override['keypoint_distance_fn'])
  if 'min_negative_keypoint_distance' in keypoint_distance_config_override:
    configs['min_negative_keypoint_distance'] = (
        keypoint_distance_config_override['min_negative_keypoint_distance'])
  if ('keypoint_distance_fn' not in configs or
      'min_negative_keypoint_distance' not in configs):
    raise ValueError('Invalid keypoint distance config: %s.' % str(configs))

  if FLAGS.task == 0 and not FLAGS.profile_only:
    # Save all key flags.
    pipeline_utils.create_dir_and_save_flags(flags, FLAGS.train_log_dir,
                                             'all_flags.train.json')

  return configs


def run(master, input_dataset_class, common_module, keypoint_profiles_module,
        models_module, input_example_parser_creator, keypoint_preprocessor_3d,
        keypoint_distance_config_override, create_model_input_fn_kwargs,
        embedder_fn_kwargs):
  """Runs training pipeline.

  Args:
    master: BNS name of the TensorFlow master to use.
    input_dataset_class: An input dataset class that matches input table type.
    common_module: A Python module that defines common flags and constants.
    keypoint_profiles_module: A Python module that defines keypoint profiles.
    models_module: A Python module that defines base model architectures.
    input_example_parser_creator: A function handle for creating data parser
      function. If None, uses the default parser creator.
    keypoint_preprocessor_3d: A function handle for preprocessing raw 3D
      keypoints.
    keypoint_distance_config_override: A dictionary for keypoint distance
      configuration to override the defaults. Ignored if empty.
    create_model_input_fn_kwargs: A dictionary of addition kwargs for create the
      model input creator function.
    embedder_fn_kwargs: A dictionary of additional kwargs for creating the
      embedder function.
  """
  g = tf.Graph()
  with g.as_default():
    with tf.device(tf.train.replica_device_setter(FLAGS.num_ps_tasks)):
      configs = _validate_and_setup(
          common_module=common_module,
          keypoint_profiles_module=keypoint_profiles_module,
          models_module=models_module,
          keypoint_distance_config_override=keypoint_distance_config_override,
          create_model_input_fn_kwargs=create_model_input_fn_kwargs,
          embedder_fn_kwargs=embedder_fn_kwargs)

      def create_inputs():
        """Creates pipeline and model inputs."""
        inputs = pipeline_utils.read_batch_from_dataset_tables(
            FLAGS.input_table,
            batch_sizes=[int(x) for x in FLAGS.batch_size],
            num_instances_per_record=2,
            shuffle=True,
            num_epochs=None,
            keypoint_names_3d=configs['keypoint_profile_3d'].keypoint_names,
            keypoint_names_2d=configs['keypoint_profile_2d'].keypoint_names,
            min_keypoint_score_2d=FLAGS.min_input_keypoint_score_2d,
            shuffle_buffer_size=FLAGS.input_shuffle_buffer_size,
            common_module=common_module,
            dataset_class=input_dataset_class,
            input_example_parser_creator=input_example_parser_creator)

        (inputs[common_module.KEY_KEYPOINTS_3D],
         keypoint_preprocessor_side_outputs_3d) = keypoint_preprocessor_3d(
             inputs[common_module.KEY_KEYPOINTS_3D],
             keypoint_profile_3d=configs['keypoint_profile_3d'],
             normalize_keypoints_3d=True)
        inputs.update(keypoint_preprocessor_side_outputs_3d)

        inputs['model_inputs'], side_inputs = configs['create_model_input_fn'](
            inputs[common_module.KEY_KEYPOINTS_2D],
            inputs[common_module.KEY_KEYPOINT_MASKS_2D],
            inputs[common_module.KEY_PREPROCESSED_KEYPOINTS_3D],
            model_input_keypoint_type=FLAGS.model_input_keypoint_type,
            normalize_keypoints_2d=True,
            keypoint_profile_2d=configs['keypoint_profile_2d'],
            keypoint_profile_3d=configs['keypoint_profile_3d'],
            azimuth_range=configs['random_projection_azimuth_range'],
            elevation_range=configs['random_projection_elevation_range'],
            roll_range=configs['random_projection_roll_range'],
            normalized_camera_depth_range=(
                configs['random_projection_camera_depth_range']))
        data_utils.merge_dict(side_inputs, inputs)
        return inputs

      inputs = create_inputs()
      outputs, _ = configs['embedder_fn'](inputs['model_inputs'])
      summaries = {
          'train/batch_size':
              tf.shape(outputs[common_module.KEY_EMBEDDING_MEANS])[0]
      }

      def add_triplet_loss():
        """Adds triplet loss."""
        anchor_keypoints_3d, positive_keypoints_3d = tf.unstack(
            inputs[common_module.KEY_KEYPOINTS_3D], num=2, axis=1)

        anchor_keypoint_masks_3d, positive_keypoint_masks_3d = None, None
        if FLAGS.use_inferred_keypoint_masks_for_triplet_label:
          anchor_keypoint_masks_2d, positive_keypoint_masks_2d = tf.unstack(
              inputs[common_module.KEY_PREPROCESSED_KEYPOINT_MASKS_2D],
              num=2,
              axis=1)
          anchor_keypoint_masks_3d = keypoint_utils.transfer_keypoint_masks(
              anchor_keypoint_masks_2d,
              input_keypoint_profile=configs['keypoint_profile_2d'],
              output_keypoint_profile=configs['keypoint_profile_3d'],
              enforce_surjectivity=True)
          positive_keypoint_masks_3d = keypoint_utils.transfer_keypoint_masks(
              positive_keypoint_masks_2d,
              input_keypoint_profile=configs['keypoint_profile_2d'],
              output_keypoint_profile=configs['keypoint_profile_3d'],
              enforce_surjectivity=True)

        triplet_anchor_embeddings, triplet_positive_embeddings = tf.unstack(
            pipeline_utils.stack_embeddings(outputs,
                                            configs['triplet_embedding_keys']),
            axis=1)
        if FLAGS.use_normalized_embeddings_for_triplet_loss:
          triplet_anchor_embeddings = tf.math.l2_normalize(
              triplet_anchor_embeddings, axis=-1)
          triplet_positive_embeddings = tf.math.l2_normalize(
              triplet_positive_embeddings, axis=-1)

        triplet_anchor_mining_embeddings, triplet_positive_mining_embeddings = (
            tf.unstack(
                pipeline_utils.stack_embeddings(
                    outputs, configs['triplet_mining_embedding_keys']),
                axis=1))
        if FLAGS.use_normalized_embeddings_for_triplet_mining:
          triplet_anchor_mining_embeddings = tf.math.l2_normalize(
              triplet_anchor_mining_embeddings, axis=-1)
          triplet_positive_mining_embeddings = tf.math.l2_normalize(
              triplet_positive_mining_embeddings, axis=-1)

        triplet_loss, triplet_loss_summaries = (
            loss_utils.compute_keypoint_triplet_losses(
                anchor_embeddings=triplet_anchor_embeddings,
                positive_embeddings=triplet_positive_embeddings,
                match_embeddings=triplet_positive_embeddings,
                anchor_keypoints=anchor_keypoints_3d,
                match_keypoints=positive_keypoints_3d,
                margin=FLAGS.triplet_loss_margin,
                min_negative_keypoint_distance=(
                    configs['min_negative_keypoint_distance']),
                use_semi_hard=FLAGS.use_semi_hard_triplet_negatives,
                exclude_inactive_triplet_loss=(
                    FLAGS.exclude_inactive_triplet_loss),
                anchor_keypoint_masks=anchor_keypoint_masks_3d,
                match_keypoint_masks=positive_keypoint_masks_3d,
                embedding_sample_distance_fn=(
                    configs['triplet_embedding_sample_distance_fn']),
                keypoint_distance_fn=configs['keypoint_distance_fn'],
                anchor_mining_embeddings=triplet_anchor_mining_embeddings,
                positive_mining_embeddings=triplet_positive_mining_embeddings,
                match_mining_embeddings=triplet_positive_mining_embeddings,
                summarize_percentiles=FLAGS.summarize_percentiles))
        tf.losses.add_loss(triplet_loss, loss_collection=tf.GraphKeys.LOSSES)
        summaries.update(triplet_loss_summaries)
        summaries['train/triplet_loss'] = triplet_loss

      def add_kl_regularization_loss():
        """Adds KL regularization loss."""
        kl_regularization_loss, kl_regularization_loss_summaries = (
            loss_utils.compute_kl_regularization_loss(
                outputs[common_module.KEY_EMBEDDING_MEANS],
                stddevs=outputs[common_module.KEY_EMBEDDING_STDDEVS],
                prior_stddev=FLAGS.kl_regularization_prior_stddev,
                loss_weight=FLAGS.kl_regularization_loss_weight))
        tf.losses.add_loss(
            kl_regularization_loss,
            loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
        summaries.update(kl_regularization_loss_summaries)
        summaries['train/kl_regularization_loss'] = kl_regularization_loss

      def add_positive_pairwise_loss():
        """Adds positive pairwise loss."""
        (positive_pairwise_anchor_embeddings,
         positive_pairwise_positive_embeddings) = tf.unstack(
             pipeline_utils.stack_embeddings(
                 outputs,
                 configs['positive_pairwise_embedding_keys'],
                 common_module=common_module),
             axis=1)
        if FLAGS.use_normalized_embeddings_for_positive_pairwise_loss:
          positive_pairwise_anchor_embeddings = tf.math.l2_normalize(
              positive_pairwise_anchor_embeddings, axis=-1)
          positive_pairwise_positive_embeddings = tf.math.l2_normalize(
              positive_pairwise_positive_embeddings, axis=-1)
        positive_pairwise_loss, positive_pairwise_loss_summaries = (
            loss_utils.compute_positive_pairwise_loss(
                positive_pairwise_anchor_embeddings,
                positive_pairwise_positive_embeddings,
                loss_weight=FLAGS.positive_pairwise_loss_weight,
                distance_fn=configs[
                    'positive_pairwise_embedding_sample_distance_fn']))
        tf.losses.add_loss(
            positive_pairwise_loss, loss_collection=tf.GraphKeys.LOSSES)
        summaries.update(positive_pairwise_loss_summaries)
        summaries['train/positive_pairwise_loss'] = positive_pairwise_loss

      add_triplet_loss()
      if FLAGS.kl_regularization_loss_weight > 0.0:
        add_kl_regularization_loss()
      if FLAGS.positive_pairwise_loss_weight > 0.0:
        add_positive_pairwise_loss()
      total_loss = tf.losses.get_total_loss()
      summaries['train/total_loss'] = total_loss

      if configs['summarize_matching_sigmoid_vars']:
        # Summarize variables used in matching sigmoid.
        # TODO(liuti): Currently the variable for `raw_a` is named `a` in
        # checkpoints, and true `a` may be referred to as `a_plus` for historic
        # reasons. Consolidate the naming.
        summaries.update({
            'train/MatchingSigmoid/a': configs['sigmoid_raw_a'],
            'train/MatchingSigmoid/a_plus': configs['sigmoid_a'],
            'train/MatchingSigmoid/b': configs['sigmoid_b'],
        })

      if FLAGS.use_moving_average:
        pipeline_utils.add_moving_average(FLAGS.moving_average_decay)

      learning_rate = pipeline_utils.get_learning_rate(
          FLAGS.learning_rate_schedule,
          FLAGS.learning_rate,
          decay_steps=FLAGS.num_steps,
          num_warmup_steps=FLAGS.num_warmup_steps)
      optimizer = pipeline_utils.get_optimizer(
          FLAGS.optimizer.upper(), learning_rate=learning_rate)
      init_fn = pipeline_utils.get_init_fn(
          train_dir=FLAGS.train_log_dir,
          model_checkpoint=FLAGS.init_model_checkpoint)
      train_op = tf_slim.training.create_train_op(
          total_loss,
          optimizer,
          transform_grads_fn=pipeline_utils.get_clip_grads_fn(
              max_norm=FLAGS.gradient_clip_norm,
              max_global_norm=FLAGS.gradient_clip_global_norm),
          summarize_gradients=FLAGS.summarize_gradients)
      saver = tf.train.Saver(
          keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
          pad_step_number=True)
      summaries['train/learning_rate'] = learning_rate

      image_summary = {}
      if FLAGS.summarize_inputs:
        image_summary.update({
            'poses_2d/AnchorPositivePair':
                visualization_utils.tf_draw_poses_2d(
                    data_utils.flatten_first_dims(
                        inputs[common_module.KEY_PREPROCESSED_KEYPOINTS_2D],
                        num_last_dims_to_keep=2),
                    keypoint_profile_2d=configs['keypoint_profile_2d'],
                    num_cols=2),
        })
      pipeline_utils.add_summary(
          scalars_to_summarize=summaries, images_to_summarize=image_summary)

      if FLAGS.profile_only:
        pipeline_utils.profile()
        return

      tf_slim.learning.train(
          train_op,
          logdir=FLAGS.train_log_dir,
          log_every_n_steps=FLAGS.log_every_n_steps,
          master=master,
          is_chief=FLAGS.task == 0,
          number_of_steps=FLAGS.num_steps,
          init_fn=init_fn,
          save_summaries_secs=FLAGS.save_summaries_secs,
          startup_delay_steps=FLAGS.startup_delay_steps * FLAGS.task,
          saver=saver,
          save_interval_secs=FLAGS.save_interval_secs,
          session_config=tf.ConfigProto(
              allow_soft_placement=True, log_device_placement=False))
