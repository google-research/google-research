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

"""Pose representation training base code."""

import os
import time

from absl import flags
from absl import logging
import tensorflow as tf

from poem.core import data_utils
from poem.core import pipeline_utils
from poem.cv_mim import algorithms
from poem.cv_mim import pipelines
from poem.cv_mim import utils

FLAGS = flags.FLAGS

flags.DEFINE_string('log_dir_path', None, 'Path to save checkpoints and logs.')
flags.mark_flag_as_required('log_dir_path')

flags.DEFINE_string('input_table', None,
                    'CSV of input tf.Example table pattern.')
flags.mark_flag_as_required('input_table')

flags.DEFINE_string('keypoint_profile_name_2d', 'LEGACY_2DCOCO13',
                    'Profile name for input 2D keypoints.')

flags.DEFINE_string(
    'keypoint_profile_name_3d', 'LEGACY_3DH36M17',
    'Profile name for input 3D keypoints '
    '(H36M: LEGACY_3DH36M17; NTU-RGBD: 3DSTD13).')

# See `common_module.SUPPORTED_TRAINING_MODEL_INPUT_KEYPOINT_TYPES`.
flags.DEFINE_string(
    'model_input_keypoint_type', '2D_INPUT_AND_3D_PROJECTION',
    'Type of model input keypoints (H36M: 2D_INPUT_AND_3D_PROJECTION; NTU-RGBD:'
    ' 2D_INPUT).')

flags.DEFINE_enum('algorithm_type', 'DISENTANGLE',
                  algorithms.SUPPORTED_ALGORITHM_TYPES,
                  'Type of the algorithm used for training.')

flags.DEFINE_enum('fusion_op_type', 'MOE', algorithms.SUPPORTED_FUSION_OP_TYPES,
                  'Type of the fusion operation for encoder.')

flags.DEFINE_integer('pose_embedding_dim', 32,
                     'Dimension of the pose embedding.')

flags.DEFINE_integer('view_embedding_dim', 32,
                     'Dimension of the view embedding.')

flags.DEFINE_enum('embedder_type', 'POINT', ['POINT', 'GAUSSIAN'],
                  'Type of the embedder.')

flags.DEFINE_float('view_loss_weight', 5.0, 'Weight of view loss.')

flags.DEFINE_float('regularization_loss_weight', 1.0,
                   'Weight of (KL-divergence) regularization loss.')

flags.DEFINE_float('disentangle_loss_weight', 0.5,
                   'Weight of disentanglement loss.')

flags.DEFINE_integer(
    'shuffle_buffer_size', 2097152,
    'Input shuffle buffer size. A large number beneifts shuffling quality.')

flags.DEFINE_float('learning_rate', 2e-2, 'Initial learning rate.')

flags.DEFINE_integer('batch_size', 256, 'Batch size in terms of trainig.')

flags.DEFINE_integer('num_iterations', 5000000,
                     'Num of iterations in terms of trainig.')

flags.DEFINE_boolean('compile', True,
                     'Compiles functions for faster tf training.')

logging.set_verbosity('info')
logging.set_stderrthreshold('info')


def _validate(common_module):
  """Validates training configurations."""
  # Validate flags.
  validate_flag = common_module.validate
  validate_flag(FLAGS.model_input_keypoint_type,
                common_module.SUPPORTED_TRAINING_MODEL_INPUT_KEYPOINT_TYPES)


def run(input_dataset_class, common_module, keypoint_profiles_module,
        input_example_parser_creator, keypoint_preprocessor_3d):
  """Runs training pipeline.

  Args:
    input_dataset_class: An input dataset class that matches input table type.
    common_module: A Python module that defines common flags and constants.
    keypoint_profiles_module: A Python module that defines keypoint profiles.
    input_example_parser_creator: A function handle for creating data parser
      function. If None, uses the default parser creator.
    keypoint_preprocessor_3d: A function handle for preprocessing raw 3D
      keypoints.
  """
  _validate(common_module)

  log_dir_path = FLAGS.log_dir_path
  pipeline_utils.create_dir_and_save_flags(flags, log_dir_path,
                                           'all_flags.train.json')

  # Setup summary writer.
  summary_writer = tf.summary.create_file_writer(
      os.path.join(log_dir_path, 'train_logs'), flush_millis=10000)

  # Setup configuration.
  keypoint_profile_2d = keypoint_profiles_module.create_keypoint_profile_or_die(
      FLAGS.keypoint_profile_name_2d)
  keypoint_profile_3d = keypoint_profiles_module.create_keypoint_profile_or_die(
      FLAGS.keypoint_profile_name_3d)

  model = algorithms.get_algorithm(
      algorithm_type=FLAGS.algorithm_type,
      pose_embedding_dim=FLAGS.pose_embedding_dim,
      view_embedding_dim=FLAGS.view_embedding_dim,
      fusion_op_type=FLAGS.fusion_op_type,
      view_loss_weight=FLAGS.view_loss_weight,
      regularization_loss_weight=FLAGS.regularization_loss_weight,
      disentangle_loss_weight=FLAGS.disentangle_loss_weight,
      embedder_type=FLAGS.embedder_type)
  optimizers = algorithms.get_optimizers(
      algorithm_type=FLAGS.algorithm_type, learning_rate=FLAGS.learning_rate)
  global_step = optimizers['encoder_optimizer'].iterations
  ckpt_manager, _, _ = utils.create_checkpoint(
      log_dir_path, **optimizers, model=model, global_step=global_step)

  # Setup the training dataset.
  dataset = pipelines.create_dataset_from_tables(
      [FLAGS.input_table], [FLAGS.batch_size],
      num_instances_per_record=2,
      shuffle=True,
      num_epochs=None,
      drop_remainder=True,
      keypoint_names_2d=keypoint_profile_2d.keypoint_names,
      keypoint_names_3d=keypoint_profile_3d.keypoint_names,
      shuffle_buffer_size=FLAGS.shuffle_buffer_size,
      dataset_class=input_dataset_class,
      input_example_parser_creator=input_example_parser_creator)

  def train_one_iteration(inputs):
    """Trains the model for one iteration.

    Args:
      inputs: A dictionary for training inputs.

    Returns:
      The training loss for this iteration.
    """
    _, side_outputs = pipelines.create_model_input(
        inputs, FLAGS.model_input_keypoint_type, keypoint_profile_2d,
        keypoint_profile_3d)

    keypoints_2d = side_outputs[common_module.KEY_PREPROCESSED_KEYPOINTS_2D]
    keypoints_3d, _ = keypoint_preprocessor_3d(
        inputs[common_module.KEY_KEYPOINTS_3D],
        keypoint_profile_3d,
        normalize_keypoints_3d=True)
    keypoints_2d, keypoints_3d = data_utils.shuffle_batches(
        [keypoints_2d, keypoints_3d])

    return model.train((keypoints_2d, keypoints_3d), **optimizers)

  if FLAGS.compile:
    train_one_iteration = tf.function(train_one_iteration)

  record_every_n_steps = min(100, FLAGS.num_iterations)
  save_ckpt_every_n_steps = min(10000, FLAGS.num_iterations)

  with summary_writer.as_default():
    with tf.summary.record_if(global_step % record_every_n_steps == 0):
      start = time.time()
      for inputs in dataset:
        if global_step >= FLAGS.num_iterations:
          break

        model_losses = train_one_iteration(inputs)
        duration = time.time() - start
        start = time.time()

        for tag, losses in model_losses.items():
          for name, loss in losses.items():
            tf.summary.scalar(
                'train/{}/{}'.format(tag, name), loss, step=global_step)

        for tag, optimizer in optimizers.items():
          tf.summary.scalar(
              'train/{}_learning_rate'.format(tag),
              optimizer.lr,
              step=global_step)

        tf.summary.scalar('train/batch_time', duration, step=global_step)
        tf.summary.scalar('global_step/sec', 1 / duration, step=global_step)

        if global_step % record_every_n_steps == 0:
          logging.info('Iter[{}/{}], {:.6f}s/iter, loss: {:.4f}'.format(
              global_step.numpy(), FLAGS.num_iterations, duration,
              model_losses['encoder']['total_loss'].numpy()))

        # Save checkpoint.
        if global_step % save_ckpt_every_n_steps == 0:
          ckpt_manager.save(checkpoint_number=global_step)
          logging.info('Checkpoint saved at step %d.', global_step.numpy())
