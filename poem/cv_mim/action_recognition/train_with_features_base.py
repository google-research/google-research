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

"""Pre-computed feature-based action recognition training base code."""

import math
import os
import time

from absl import flags
from absl import logging
import tensorflow as tf
from tensorflow_addons import optimizers as tfa_optimizers

from poem.core import pipeline_utils
from poem.cv_mim import pipelines
from poem.cv_mim import utils
from poem.cv_mim.action_recognition import models

FLAGS = flags.FLAGS

flags.DEFINE_string('log_dir_path', None, 'Path to save checkpoints and logs.')
flags.mark_flag_as_required('log_dir_path')

flags.DEFINE_list('input_tables', None,
                  'A list of input tf.Example table pattern.')
flags.mark_flag_as_required('input_tables')

flags.DEFINE_list('batch_sizes', None,
                  'A list of batch size for each input table.')
flags.mark_flag_as_required('batch_sizes')

flags.DEFINE_integer('input_features_dim', 0, 'Dimension of input features.'
                     'Use zero or negative for using 2D keypoints as inputs.')

flags.DEFINE_string('keypoint_profile_name_2d', 'LEGACY_2DCOCO13',
                    'Profile name for input 2D keypoints.')

flags.DEFINE_boolean('compile', True,
                     'Compiles functions for faster tf training.')

flags.DEFINE_integer(
    'shuffle_buffer_size', 1157,
    'Input shuffle buffer size (PennAction: 1157; NTU-RGBD: 32726).')

flags.DEFINE_float('learning_rate', 5e-3, 'Initial learning rate.')

flags.DEFINE_enum('classifier_type', 'CONVNET',
                  models.SUPPORTED_CLASSIFIER_TYPES, 'Type of the classifier.')

flags.DEFINE_integer(
    'downsample_rate', 2,
    'Downsample rate of input videos (PennAction: 2; NTU-RGBD: 1).')

flags.DEFINE_integer(
    'num_classes', 14,
    'Number of action classes (PennAction: 14; NTU-RGBD: 49).')

flags.DEFINE_integer(
    'num_frames', 663,
    'Number of frames in each video (PennAction: 663; NTU-RGBD: 300).')

flags.DEFINE_integer('num_iterations', 1000000,
                     'Num of iterations in terms of trainig.')

logging.set_verbosity('info')
logging.set_stderrthreshold('info')


def run(input_dataset_class, common_module, keypoint_profiles_module,
        input_example_parser_creator):
  """Runs training pipeline.

  Args:
    input_dataset_class: An input dataset class that matches input table type.
    common_module: A Python module that defines common flags and constants.
    keypoint_profiles_module: A Python module that defines keypoint profiles.
    input_example_parser_creator: A function handle for creating data parser
      function. If None, uses the default parser creator.
  """
  log_dir_path = FLAGS.log_dir_path
  pipeline_utils.create_dir_and_save_flags(
      flags, log_dir_path, 'all_flags.train_with_features.json')

  # Setup summary writer.
  summary_writer = tf.summary.create_file_writer(
      os.path.join(log_dir_path, 'train_logs'), flush_millis=10000)

  # Setup configuration.
  keypoint_profile_2d = keypoint_profiles_module.create_keypoint_profile_or_die(
      FLAGS.keypoint_profile_name_2d)

  # Setup model.
  input_length = math.ceil(FLAGS.num_frames / FLAGS.downsample_rate)
  if FLAGS.input_features_dim > 0:
    feature_dim = FLAGS.input_features_dim
    input_shape = (input_length, feature_dim)
  else:
    feature_dim = None
    input_shape = (input_length, 13 * 2)
  classifier = models.get_temporal_classifier(
      FLAGS.classifier_type,
      input_shape=input_shape,
      num_classes=FLAGS.num_classes)
  ema_classifier = tf.keras.models.clone_model(classifier)
  optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
  optimizer = tfa_optimizers.MovingAverage(optimizer)
  global_step = optimizer.iterations
  ckpt_manager, _, _ = utils.create_checkpoint(
      log_dir_path,
      optimizer=optimizer,
      model=classifier,
      ema_model=ema_classifier,
      global_step=global_step)

  # Setup the training dataset.
  dataset = pipelines.create_dataset_from_tables(
      FLAGS.input_tables, [int(x) for x in FLAGS.batch_sizes],
      num_instances_per_record=1,
      shuffle=True,
      drop_remainder=True,
      num_epochs=None,
      keypoint_names_2d=keypoint_profile_2d.keypoint_names,
      feature_dim=feature_dim,
      num_classes=FLAGS.num_classes,
      num_frames=FLAGS.num_frames,
      shuffle_buffer_size=FLAGS.shuffle_buffer_size,
      common_module=common_module,
      dataset_class=input_dataset_class,
      input_example_parser_creator=input_example_parser_creator)
  loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

  def train_one_iteration(inputs):
    """Trains the model for one iteration.

    Args:
      inputs: A dictionary for training inputs.

    Returns:
      loss: The training loss for this iteration.
    """
    if FLAGS.input_features_dim > 0:
      features = inputs[common_module.KEY_FEATURES]
    else:
      features, _ = pipelines.create_model_input(
          inputs, common_module.MODEL_INPUT_KEYPOINT_TYPE_2D_INPUT,
          keypoint_profile_2d)

    features = tf.squeeze(features, axis=1)
    features = features[:, ::FLAGS.downsample_rate, :]
    labels = inputs[common_module.KEY_CLASS_TARGETS]
    labels = tf.squeeze(labels, axis=1)

    with tf.GradientTape() as tape:
      outputs = classifier(features, training=True)
      regularization_loss = sum(classifier.losses)
      crossentropy_loss = loss_object(labels, outputs)
      total_loss = crossentropy_loss + regularization_loss

    trainable_variables = classifier.trainable_variables
    grads = tape.gradient(total_loss, trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))

    for grad, trainable_variable in zip(grads, trainable_variables):
      tf.summary.scalar(
          'summarize_grads/' + trainable_variable.name,
          tf.linalg.norm(grad),
          step=global_step)

    return dict(
        total_loss=total_loss,
        crossentropy_loss=crossentropy_loss,
        regularization_loss=regularization_loss)

  if FLAGS.compile:
    train_one_iteration = tf.function(train_one_iteration)

  record_every_n_steps = min(5, FLAGS.num_iterations)
  save_ckpt_every_n_steps = min(500, FLAGS.num_iterations)

  with summary_writer.as_default():
    with tf.summary.record_if(global_step % record_every_n_steps == 0):
      start = time.time()
      for inputs in dataset:
        if global_step >= FLAGS.num_iterations:
          break

        model_losses = train_one_iteration(inputs)
        duration = time.time() - start
        start = time.time()

        for name, loss in model_losses.items():
          tf.summary.scalar('train/' + name, loss, step=global_step)

        tf.summary.scalar('train/learning_rate', optimizer.lr, step=global_step)
        tf.summary.scalar('train/batch_time', duration, step=global_step)
        tf.summary.scalar('global_step/sec', 1 / duration, step=global_step)

        if global_step % record_every_n_steps == 0:
          logging.info('Iter[{}/{}], {:.6f}s/iter, loss: {:.4f}'.format(
              global_step.numpy(), FLAGS.num_iterations, duration,
              model_losses['total_loss'].numpy()))

        # Save checkpoint.
        if global_step % save_ckpt_every_n_steps == 0:
          utils.assign_moving_average_vars(classifier, ema_classifier,
                                           optimizer)
          ckpt_manager.save(checkpoint_number=global_step)
          logging.info('Checkpoint saved at step %d.', global_step.numpy())
