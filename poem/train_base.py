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

"""Pose embedding model training base code."""

import math

from absl import flags
import tensorflow.compat.v1 as tf
import tf_slim

from poem.core import data_utils
from poem.core import keypoint_utils
from poem.core import loss_utils
from poem.core import pipeline_utils

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
    'min_input_keypoint_score_2d', -1.0,
    'Minimum threshold for input keypoint score binarization. Use negative '
    'value to ignore. Only used if 2D keypoint masks are used.')

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
    ' used if `keypoint_distance_type` is `MPJPE`.')

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

flags.DEFINE_enum('optimizer', 'ADAGRAD', ['ADAGRAD'], 'Optimizer to use.')

flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')

flags.DEFINE_integer(
    'num_steps', 5000000,
    'Number of training steps. Use None to train indefinitely.')

flags.DEFINE_string('init_model_checkpoint', None,
                    'Path to checkpoint to initialize from.')

flags.DEFINE_float('gradient_clip_norm', 0.0,
                   'Norm gradients are clipped to. Only used if positive.')

flags.DEFINE_bool('use_moving_average', True,
                  'Whether to use exponential moving average.')

flags.DEFINE_float(
    'moving_average_decay', 0.9999,
    'Exponential moving average decay. Only used if `use_moving_average` is '
    'True.')

flags.DEFINE_integer(
    'input_shuffle_buffer_size', 2097152,
    'Input shuffle buffer size. A large number beneifts shuffling quality.')

flags.DEFINE_list(
    'random_projection_azimuth_range', ['-180.0', '180.0'],
    'CSV of 2-tuple rotation angle limit (lower_limit, upper_limit) for '
    'performing random azimuth rotations on 3D poses before projection for '
    'keypoint augmentation.')

flags.DEFINE_list(
    'random_projection_elevation_range', ['-30.0', '30.0'],
    'CSV of 2-tuple rotation angle limit (lower_limit, upper_limit) for '
    'performing random elevation rotations on 3D poses before projection for '
    'keypoint augmentation.')

flags.DEFINE_list(
    'random_projection_roll_range', ['-30.0', '30.0'],
    'CSV of 2-tuple rotation angle limit (lower_limit, upper_limit) for '
    'performing random roll rotations on 3D poses before projection for '
    'keypoint augmentation.')

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
                        keypoint_distance_config_override):
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
    if FLAGS.triplet_distance_type in [
        common_module.DISTANCE_TYPE_SAMPLE,
        common_module.DISTANCE_TYPE_CENTER_AND_SAMPLE
    ]:
      raise ValueError(
          'No support for triplet distance type `%s` for embedding type `%s`.' %
          (FLAGS.triplet_distance_type, FLAGS.embedding_type))
    if FLAGS.kl_regularization_loss_weight > 0.0:
      raise ValueError(
          'No support for KL regularization loss for embedding type `%s`.' %
          FLAGS.embedding_type)

  if ((FLAGS.triplet_distance_type in [
      common_module.DISTANCE_TYPE_SAMPLE,
      common_module.DISTANCE_TYPE_CENTER_AND_SAMPLE
  ] or FLAGS.positive_pairwise_distance_type in [
      common_module.DISTANCE_TYPE_SAMPLE,
      common_module.DISTANCE_TYPE_CENTER_AND_SAMPLE
  ]) and FLAGS.num_embedding_samples <= 0):
    raise ValueError(
        'Must specify positive `num_embedding_samples` to use `%s` '
        'triplet/positive pairwise distance type.' %
        FLAGS.triplet_distance_type)

  if (((FLAGS.triplet_distance_kernel in [
      common_module.DISTANCE_KERNEL_L2_SIGMOID_MATCHING_PROB,
      common_module.DISTANCE_KERNEL_EXPECTED_LIKELIHOOD
  ]) != (FLAGS.triplet_pairwise_reduction in [
      common_module.DISTANCE_REDUCTION_NEG_LOG_MEAN,
      common_module.DISTANCE_REDUCTION_LOWER_HALF_NEG_LOG_MEAN,
      common_module.DISTANCE_REDUCTION_ONE_MINUS_MEAN
  ])) or ((FLAGS.positive_pairwise_distance_kernel in [
      common_module.DISTANCE_KERNEL_L2_SIGMOID_MATCHING_PROB,
      common_module.DISTANCE_KERNEL_EXPECTED_LIKELIHOOD
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
              affinity_matrix=keypoint_profile_2d.keypoint_affinity_matrix),
      'triplet_embedding_keys':
          pipeline_utils.get_embedding_keys(
              FLAGS.triplet_distance_type, common_module=common_module),
      'triplet_mining_embedding_keys':
          pipeline_utils.get_embedding_keys(
              FLAGS.triplet_distance_type,
              replace_samples_with_means=True,
              common_module=common_module),
      'triplet_embedding_sample_distance_fn':
          loss_utils.create_sample_distance_fn(
              pair_type=common_module.DISTANCE_PAIR_TYPE_ALL_PAIRS,
              distance_kernel=FLAGS.triplet_distance_kernel,
              pairwise_reduction=FLAGS.triplet_pairwise_reduction,
              componentwise_reduction=FLAGS.triplet_componentwise_reduction,
              # We initialize the sigmoid parameters to avoid model being stuck
              # in a `dead zone` at the beginning of training.
              L2_SIGMOID_MATCHING_PROB_a_initializer=(
                  tf.initializers.constant(-0.65)),
              L2_SIGMOID_MATCHING_PROB_b_initializer=(
                  tf.initializers.constant(-0.5)),
              EXPECTED_LIKELIHOOD_min_stddev=0.1,
              EXPECTED_LIKELIHOOD_max_squared_mahalanobis_distance=100.0),
      'positive_pairwise_embedding_keys':
          pipeline_utils.get_embedding_keys(
              FLAGS.positive_pairwise_distance_type,
              common_module=common_module),
      'positive_pairwise_embedding_sample_distance_fn':
          loss_utils.create_sample_distance_fn(
              pair_type=common_module.DISTANCE_PAIR_TYPE_ALL_PAIRS,
              distance_kernel=FLAGS.positive_pairwise_distance_kernel,
              pairwise_reduction=FLAGS.positive_pairwise_pairwise_reduction,
              componentwise_reduction=(
                  FLAGS.positive_pairwise_componentwise_reduction),
              # We initialize the sigmoid parameters to avoid model being stuck
              # in a `dead zone` at the beginning of training.
              L2_SIGMOID_MATCHING_PROB_a_initializer=(
                  tf.initializers.constant(-0.65)),
              L2_SIGMOID_MATCHING_PROB_b_initializer=(
                  tf.initializers.constant(-0.5)),
              EXPECTED_LIKELIHOOD_min_stddev=0.1,
              EXPECTED_LIKELIHOOD_max_squared_mahalanobis_distance=100.0),
      'summarize_matching_sigmoid_vars':
          FLAGS.triplet_distance_kernel in
          [common_module.DISTANCE_KERNEL_L2_SIGMOID_MATCHING_PROB],
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
  }

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
        create_model_input_fn, keypoint_distance_config_override):
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
    create_model_input_fn: A function handle for creating model inputs.
    keypoint_distance_config_override: A dictionary for keypoint distance
      configuration to override the defaults. Ignored if empty.
  """
  configs = _validate_and_setup(
      common_module=common_module,
      keypoint_profiles_module=keypoint_profiles_module,
      models_module=models_module,
      keypoint_distance_config_override=keypoint_distance_config_override)

  g = tf.Graph()
  with g.as_default():
    with tf.device(tf.train.replica_device_setter(FLAGS.num_ps_tasks)):

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

        inputs['model_inputs'], side_inputs = create_model_input_fn(
            inputs[common_module.KEY_KEYPOINTS_2D],
            inputs[common_module.KEY_KEYPOINT_MASKS_2D],
            inputs[common_module.KEY_KEYPOINTS_3D],
            model_input_keypoint_type=FLAGS.model_input_keypoint_type,
            normalize_keypoints_2d=True,
            keypoint_profile_2d=configs['keypoint_profile_2d'],
            keypoint_profile_3d=configs['keypoint_profile_3d'],
            azimuth_range=configs['random_projection_azimuth_range'],
            elevation_range=configs['random_projection_elevation_range'],
            roll_range=configs['random_projection_roll_range'])
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
        with tf.variable_scope('MatchingSigmoid', reuse=True):
          summaries['train/MatchingSigmoid/a'] = tf.get_variable('a')
          summaries['train/MatchingSigmoid/a_plus'] = (
              tf.nn.elu(tf.get_variable('a')) + 1.0)
          summaries['train/MatchingSigmoid/b'] = tf.get_variable('b')

      if FLAGS.use_moving_average:
        pipeline_utils.add_moving_average(FLAGS.moving_average_decay)

      learning_rate = FLAGS.learning_rate
      optimizer = pipeline_utils.get_optimizer(
          FLAGS.optimizer.upper(), learning_rate=learning_rate)
      init_fn = pipeline_utils.get_init_fn(
          train_dir=FLAGS.train_log_dir,
          model_checkpoint=FLAGS.init_model_checkpoint)
      train_op = tf_slim.learning.create_train_op(
          total_loss,
          optimizer,
          clip_gradient_norm=FLAGS.gradient_clip_norm,
          summarize_gradients=FLAGS.summarize_gradients)
      saver = tf.train.Saver(
          keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
          pad_step_number=True)
      summaries['train/learning_rate'] = learning_rate

      pipeline_utils.add_summary(scalars_to_summarize=summaries)

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
