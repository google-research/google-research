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

"""Generic training script for training InceptionV3FCN from a checkpoint."""

import tensorflow.compat.v1 as tf
import tf_slim as slim

from nopad_inception_v3_fcn import inception_v3_fcn
from nopad_inception_v3_fcn import network_params
from tensorflow_models.slim.datasets import dataset_factory
from tensorflow_models.slim.preprocessing import preprocessing_factory


tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer(
    'receptive_field_size', 911,
    'The receptive field of the InceptionV3FCN. Should be 911 or 129.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 5,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'batch_size', 16, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_number_of_steps', None, 'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

FLAGS = tf.app.flags.FLAGS

# Number of parallel dataset readers.
DATASET_READERS = 4
# Number of threads used for preprocessing input data into batches.
PREPROCESSING_THREADS = 4


def _get_init_fn():
  """Returns a function to initialize model from a checkpoint."""
  if FLAGS.checkpoint_path is None:
    return None

  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  if tf.train.latest_checkpoint(FLAGS.train_dir):
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s',
        FLAGS.train_dir)
    return None

  exclusions = []
  if FLAGS.checkpoint_exclude_scopes:
    exclusions = [scope.strip()
                  for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

  variables_to_restore = []
  for var in slim.get_model_variables():
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        break
    else:
      variables_to_restore.append(var)

  if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
  else:
    checkpoint_path = FLAGS.checkpoint_path

  tf.logging.info('Fine-tuning from %s', checkpoint_path)

  return slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)


def _get_variables_to_train():
  """Returns a list of variables to train."""
  if FLAGS.trainable_scopes is None:
    return tf.trainable_variables()
  scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]
  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train


def main(_):
  tf.disable_eager_execution()

  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    #########################
    # Configure the network #
    #########################
    inception_params = network_params.InceptionV3FCNParams(
        receptive_field_size=FLAGS.receptive_field_size,
        prelogit_dropout_keep_prob=0.8,
        depth_multiplier=0.1,
        min_depth=16,
        inception_fcn_stride=16,
    )
    conv_params = network_params.ConvScopeParams(
        dropout=False,
        dropout_keep_prob=0.8,
        batch_norm=True,
        batch_norm_decay=0.99,
        l2_weight_decay=4e-05,
    )
    network_fn = inception_v3_fcn.get_inception_v3_fcn_network_fn(
        inception_params,
        conv_params,
        num_classes=dataset.num_classes,
        is_training=True,
    )

    #####################################
    # Select the preprocessing function #
    #####################################
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        'inception_v3', is_training=True)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=DATASET_READERS,
        common_queue_capacity=20 * FLAGS.batch_size,
        common_queue_min=10 * FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])
    train_image_size = FLAGS.receptive_field_size
    image = image_preprocessing_fn(image, train_image_size, train_image_size)
    images, labels = tf.train.batch([image, label],
                                    batch_size=FLAGS.batch_size,
                                    num_threads=PREPROCESSING_THREADS,
                                    capacity=5 * FLAGS.batch_size)
    labels = slim.one_hot_encoding(labels, dataset.num_classes)

    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(images)

    slim.losses.softmax_cross_entropy(logits, labels)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('losses/Total_Loss', total_loss)

    optimizer = tf.train.RMSPropOptimizer(0.01)

    train_op = slim.learning.create_train_op(
        total_loss,
        optimizer,
        variables_to_train=_get_variables_to_train())

    ###########################
    # Kicks off the training. #
    ###########################
    slim.learning.train(
        train_op,
        logdir=FLAGS.train_dir,
        init_fn=_get_init_fn(),
        number_of_steps=FLAGS.max_number_of_steps,
        log_every_n_steps=FLAGS.log_every_n_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        session_config=tf.ConfigProto(allow_soft_placement=True))


if __name__ == '__main__':
  tf.app.run()
