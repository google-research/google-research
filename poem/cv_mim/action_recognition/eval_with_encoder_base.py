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

"""Encoder-based action recognition evaluation with SSTable inputs."""

import math
import os

from absl import flags
from absl import logging
import tensorflow as tf

from poem.core import pipeline_utils
from poem.cv_mim import algorithms
from poem.cv_mim import pipelines
from poem.cv_mim import utils
from poem.cv_mim.action_recognition import models

FLAGS = flags.FLAGS

flags.DEFINE_string('eval_name', None, 'The name of this evaluation task.')
flags.mark_flag_as_required('eval_name')

flags.DEFINE_string('log_dir_path', None, 'Path to save checkpoints and logs.')
flags.mark_flag_as_required('log_dir_path')

flags.DEFINE_string('encoder_checkpoint_path', None,
                    'Path to load the encoder checkpoint.')
flags.mark_flag_as_required('encoder_checkpoint_path')

flags.DEFINE_enum('encoder_algorithm_type', 'DISENTANGLE',
                  algorithms.SUPPORTED_ALGORITHM_TYPES,
                  'Type of the algorithm used for training the encoder.')

flags.DEFINE_integer('encoder_pose_embedding_dim', 32,
                     'Dimension of the pose embedding.')

flags.DEFINE_integer(
    'encoder_view_embedding_dim', 32,
    'Dimension of the view embedding if encoder_algorithm_type is DISENTANGLE.')

flags.DEFINE_enum('encoder_embedder_type', 'POINT', ['POINT', 'GAUSSIAN'],
                  'Type of the encoder embedder.')

flags.DEFINE_string(
    'encoder_output_activation', 'embedder',
    'Activation name of the encoder output to be used as the input.')

flags.DEFINE_integer('encoder_output_dim', 32,
                     'Dimension of the encoder features.')

flags.DEFINE_list('input_tables', None,
                  'A list of input tf.Example table pattern.')
flags.mark_flag_as_required('input_tables')

flags.DEFINE_list('batch_sizes', None,
                  'A list of batch size for each input table.')
flags.mark_flag_as_required('batch_sizes')

flags.DEFINE_string('keypoint_profile_name_2d', 'LEGACY_2DCOCO13',
                    'Profile name for input 2D keypoints.')

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

flags.DEFINE_boolean('use_moving_average', False,
                     'Whether to use exponential moving average.')

flags.DEFINE_boolean('compile', True,
                     'Compiles functions for faster tf training.')

flags.DEFINE_boolean('continuous_eval', True, 'Evaluates continuously.')

flags.DEFINE_integer('max_iteration', 1000000, 'Maximum iteration of trainig.')

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
  if not tf.io.gfile.exists(log_dir_path):
    raise ValueError(
        'The directory {} does not exist. Please provide a new log_dir_path.'
        .format(log_dir_path))
  eval_log_dir = os.path.join(log_dir_path, FLAGS.eval_name)
  pipeline_utils.create_dir_and_save_flags(flags, eval_log_dir,
                                           'all_flags.eval_with_encoder.json')

  # Setup summary writer.
  summary_writer = tf.summary.create_file_writer(
      eval_log_dir, flush_millis=10000)

  # Setup configuration.
  keypoint_profile_2d = keypoint_profiles_module.create_keypoint_profile_or_die(
      FLAGS.keypoint_profile_name_2d)

  # Setup model.
  model = algorithms.get_algorithm(
      algorithm_type=FLAGS.encoder_algorithm_type,
      pose_embedding_dim=FLAGS.encoder_pose_embedding_dim,
      view_embedding_dim=FLAGS.encoder_view_embedding_dim,
      embedder_type=FLAGS.encoder_embedder_type)
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint.restore(FLAGS.encoder_checkpoint_path).expect_partial()
  encoder = model.encoder

  classifier = models.get_temporal_classifier(
      FLAGS.classifier_type,
      input_shape=(math.ceil(FLAGS.num_frames / FLAGS.downsample_rate),
                   FLAGS.encoder_output_dim),
      num_classes=FLAGS.num_classes)
  ema_classifier = tf.keras.models.clone_model(classifier)
  global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
  dataset = pipelines.create_dataset_from_tables(
      FLAGS.input_tables, [int(x) for x in FLAGS.batch_sizes],
      num_instances_per_record=1,
      shuffle=False,
      num_epochs=1,
      drop_remainder=False,
      keypoint_names_2d=keypoint_profile_2d.keypoint_names,
      num_classes=FLAGS.num_classes,
      num_frames=FLAGS.num_frames,
      common_module=common_module,
      dataset_class=input_dataset_class,
      input_example_parser_creator=input_example_parser_creator)

  if FLAGS.compile:
    encoder.call = tf.function(encoder.call)
    classifier.call = tf.function(classifier.call)
    ema_classifier.call = tf.function(ema_classifier.call)

  top_1_best_accuracy = None
  top_5_best_accuracy = None
  evaluated_last_ckpt = False

  def timeout_fn():
    """Timeout function to stop the evaluation."""
    return evaluated_last_ckpt

  def evaluate_once():
    """Evaluates the model for one time."""
    _, status, _ = utils.create_checkpoint(
        log_dir_path,
        model=classifier,
        ema_model=ema_classifier,
        global_step=global_step)
    status.expect_partial()
    logging.info('Last checkpoint [iteration: %d] restored at %s.',
                 global_step.numpy(), log_dir_path)

    if global_step.numpy() >= FLAGS.max_iteration:
      nonlocal evaluated_last_ckpt
      evaluated_last_ckpt = True

    top_1_accuracy = tf.keras.metrics.CategoricalAccuracy()
    top_5_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
    for inputs in dataset:
      _, side_outputs = pipelines.create_model_input(
          inputs, common_module.MODEL_INPUT_KEYPOINT_TYPE_2D_INPUT,
          keypoint_profile_2d)

      keypoints_2d = side_outputs[common_module.KEY_PREPROCESSED_KEYPOINTS_2D]
      keypoints_2d = tf.squeeze(keypoints_2d, axis=1)
      features = keypoints_2d[:, ::FLAGS.downsample_rate, Ellipsis]
      labels = inputs[common_module.KEY_CLASS_TARGETS]
      labels = tf.squeeze(labels, axis=1)

      batch_size, num_frames, num_joints, feature_dim = features.shape
      features = tf.reshape(features, (-1, num_joints, feature_dim))
      _, features = encoder(features, training=False)
      features = features[FLAGS.encoder_output_activation]
      features = tf.reshape(features, (batch_size, num_frames, -1))
      if (FLAGS.encoder_output_activation == 'embedder') and (
          FLAGS.encoder_algorithm_type != algorithms.TYPE_ALGORITHM_ALIGN
      ):
        features, _ = tf.split(
            features,
            num_or_size_splits=[
                FLAGS.encoder_pose_embedding_dim,
                FLAGS.encoder_view_embedding_dim
            ],
            axis=-1)

      if FLAGS.use_moving_average:
        outputs = ema_classifier(features, training=False)
      else:
        outputs = classifier(features, training=False)
      top_1_accuracy.update_state(y_true=labels, y_pred=outputs)
      top_5_accuracy.update_state(y_true=labels, y_pred=outputs)

    nonlocal top_1_best_accuracy
    if (top_1_best_accuracy is None or
        top_1_accuracy.result().numpy() > top_1_best_accuracy):
      top_1_best_accuracy = top_1_accuracy.result().numpy()

    nonlocal top_5_best_accuracy
    if (top_5_best_accuracy is None or
        top_5_accuracy.result().numpy() > top_5_best_accuracy):
      top_5_best_accuracy = top_5_accuracy.result().numpy()

    tf.summary.scalar(
        'eval/Basic/Top1_Accuracy',
        top_1_accuracy.result(),
        step=global_step.numpy())
    tf.summary.scalar(
        'eval/Best/Top1_Accuracy',
        top_1_best_accuracy,
        step=global_step.numpy())
    tf.summary.scalar(
        'eval/Basic/Top5_Accuracy',
        top_5_accuracy.result(),
        step=global_step.numpy())
    tf.summary.scalar(
        'eval/Best/Top5_Accuracy',
        top_5_best_accuracy,
        step=global_step.numpy())
    logging.info('Accuracy: {:.2f}'.format(top_1_accuracy.result().numpy()))

  with summary_writer.as_default():
    with tf.summary.record_if(True):
      if FLAGS.continuous_eval:
        for _ in tf.train.checkpoints_iterator(log_dir_path, timeout=1,
                                               timeout_fn=timeout_fn):
          evaluate_once()
      else:
        evaluate_once()
