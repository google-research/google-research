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

# Lint as: python2, python3
"""Tensorflow code for training and evaluating deep homography models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags
import six
from six.moves import range
import tensorflow.compat.v1 as tf
from deep_homography import hmg_util
from deep_homography import models
from tensorflow.contrib import slim as contrib_slim

slim = contrib_slim

flags.DEFINE_string('master', 'local', 'Master of the training')
flags.DEFINE_integer('ps_tasks', 0, 'Number of paramater servers')
flags.DEFINE_enum('mode', 'train', ['train', 'eval'], 'Mode of this run')
flags.DEFINE_integer('task', 0, 'Task id')
flags.DEFINE_string('train_dir', '/tmp/train',
                    'Where to write the checkpoints for training')
flags.DEFINE_string('eval_dir', '',
                    'Where to write the checkpoints for eval')
flags.DEFINE_string('model_path', '',
                    'Where to find the checkpoints for eval')
flags.DEFINE_string('vgg_model_path', '',
                    'Where to find the vgg network checkpoint')
flags.DEFINE_string('data_pattern', '', 'Glob pattern of input data')
flags.DEFINE_enum('data_type', 'ava', ['coco', 'ava', 'ava_seq'],
                  'training data type')

flags.DEFINE_integer('num_frames_per_sample', 9,
                     'Number of frames in one sample')
flags.DEFINE_integer('batch_size', 3, 'Batch size')
flags.DEFINE_integer('queue_size', 100, 'Batch queue size')
flags.DEFINE_integer('num_threads', 3, 'The number of threads in the queue')
flags.DEFINE_integer('train_height', 128, 'Height of training images')
flags.DEFINE_integer('train_width', 128, 'Width of training images')
flags.DEFINE_float('max_shift', 16,
                   'Maximum random shift when creating training samples')
flags.DEFINE_boolean('mix', False,
                     'Whether to randomly scale random shift sizes')
flags.DEFINE_boolean('screen', False,
                     'Whether to remove highly distorted homography')
flags.DEFINE_integer('frame_gap', 0, 'Temporal gap between two selected frames')
flags.DEFINE_integer('max_frame_gap', 5, 'Maximal frame gap')

flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_integer('lr_decay_steps', 100000, 'Decay steps for learning rate')
flags.DEFINE_float('lr_decay_rate', 0.8, 'Decay rate for learning rate')
flags.DEFINE_float('weight_decay', 0.00004, 'weight decay coefficient')
flags.DEFINE_float('dropout_keep', 0.8, 'probability that an element is kept')
flags.DEFINE_integer('num_eval_steps', 10, 'Number of eval steps per cycle')
flags.DEFINE_integer('max_step', 100000, 'the maximal number of global steps')

flags.DEFINE_enum('loss', 'l2', ['l2', 'hier_l2', 'hier_ld'], 'loss function')
flags.DEFINE_boolean('random_flip', False,
                     'Whether randomly flip training examples left or right')
flags.DEFINE_boolean('random_reverse', False,
                     'Whether randomly reverse the video sequence')
flags.DEFINE_float('pixel_noise', 2, 'Amount of random noise added to a pixel')
flags.DEFINE_integer('num_level', 2, 'Number of hierarchical levels')
flags.DEFINE_integer('num_layer', 6,
                     'Number of layers in the motion feature network')
flags.DEFINE_integer('level_wise', 1,
                     'Whether to train networks level by level')

flags.DEFINE_enum('mask_method', 'f4', ['f4', 'f5', 'f6'], 'Masking method')
flags.DEFINE_enum('network_id', 'hier', ['hier', 'fmask_sem'],
                  'Type of network')
flags.DEFINE_boolean('block_prop', False,
                     'Whether block back propagation between different levels')

FLAGS = flags.FLAGS


def predict_homography(inputs, network_id='cvgghmg', reuse=None,
                       is_training=True, scope='hier_hmg'):
  """Estimates homography using a selected deep neural network.

  Args:
    inputs: batch of input image pairs of data type float32 and of shape
      [batch_size, height, width, None]
    network_id: deep neural network method
    reuse: whether to reuse this network weights
    is_training: whether used for training or testing
    scope: the scope of variables in this function
  Raises:
    ValueError: The nework_id was not good.
  Returns:
    a list of homographies at each level and a list of images warped by
    the list of corresponding homographies
  """
  with slim.arg_scope(models.homography_arg_scope(
      weight_decay=FLAGS.weight_decay)):
    if network_id == 'hier':
      return models.hier_homography_estimator(
          inputs, num_param=8, num_layer=FLAGS.num_layer,
          num_level=FLAGS.num_level,
          dropout_keep_prob=FLAGS.dropout_keep,
          is_training=is_training, reuse=reuse, scope=scope)
    elif network_id == 'fmask_sem':
      return models.hier_homography_fmask_estimator(
          inputs, num_param=8, num_layer=FLAGS.num_layer,
          num_level=FLAGS.num_level,
          dropout_keep_prob=FLAGS.dropout_keep,
          is_training=is_training, reuse=reuse, scope=scope)
    else:
      raise ValueError('Unknown network_id: %s' % network_id)


def get_samples(to_gray, mode):
  """Get training or testing samples.

  Args:
    to_gray: whether prepare color or gray scale training images
    mode: 'train' or 'eval', specifying whether preparing images for training or
      testing
  Raises:
    ValueError: The data_type was not good.
  Returns:
    a batch of training images and the corresponding ground-truth homographies
  """
  if FLAGS.data_type == 'coco':
    batch_frames, batch_labels = hmg_util.get_batchpairs_coco(
        FLAGS.data_pattern, FLAGS.max_shift, batch_size=FLAGS.batch_size,
        queue_size=FLAGS.queue_size, num_threads=FLAGS.num_threads,
        train_height=FLAGS.train_height, train_width=FLAGS.train_width,
        pixel_noise=FLAGS.pixel_noise, mix=FLAGS.mix, screen=FLAGS.screen,
        to_gray=to_gray, mode=mode)
  elif FLAGS.data_type == 'ava':
    batch_frames, batch_labels = hmg_util.get_batchpairs_ava(
        FLAGS.data_pattern, FLAGS.max_shift, batch_size=FLAGS.batch_size,
        queue_size=FLAGS.queue_size, num_threads=FLAGS.num_threads,
        train_height=FLAGS.train_height, train_width=FLAGS.train_width,
        pixel_noise=FLAGS.pixel_noise, mix=FLAGS.mix, screen=FLAGS.screen,
        to_gray=to_gray, mode=mode)
  elif FLAGS.data_type == 'ava_seq':
    batch_frames, batch_labels = hmg_util.get_batchseqs_ava(
        FLAGS.data_pattern, FLAGS.num_frames_per_sample, FLAGS.max_shift,
        batch_size=FLAGS.batch_size, queue_size=FLAGS.queue_size,
        num_threads=FLAGS.num_threads,
        train_height=FLAGS.train_height, train_width=FLAGS.train_width,
        pixel_noise=FLAGS.pixel_noise, mix=FLAGS.mix, screen=FLAGS.screen,
        to_gray=to_gray, mode=mode)
  else:
    raise ValueError('Unknown data_type: %s' % FLAGS.data_type)
  return batch_frames, batch_labels


def run_train(scope):
  """Trains a network.

  Args:
    scope: the scope of variables in this function
  """
  with tf.Graph().as_default():
    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
      to_gray = True
      if 'sem' in FLAGS.network_id:
        to_gray = False
      batch_frames, batch_labels = get_samples(to_gray, 'train')
      batch_hmg_prediction, _ = predict_homography(
          batch_frames, network_id=FLAGS.network_id, is_training=True,
          scope=scope)

      if FLAGS.loss == 'hier_l2':
        for level in range(FLAGS.num_level):
          delta_level = FLAGS.num_level - level -1
          scale = 2 ** delta_level
          l2 = tf.losses.mean_squared_error(batch_labels / scale,
                                            batch_hmg_prediction[level])
          slim.summaries.add_scalar_summary(l2, 'l2%d' % delta_level, 'losses')
      elif FLAGS.loss == 'hier_ld':
        for level in range(FLAGS.num_level):
          delta_level = FLAGS.num_level - level -1
          scale = 2 ** delta_level
          diff = tf.reshape(batch_labels / scale - batch_hmg_prediction[level],
                            [FLAGS.batch_size, 4, 2])
          l2d = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(diff), 2)))
          tf.losses.add_loss(l2d)
          slim.summaries.add_scalar_summary(l2d, 'l2%d' % delta_level, 'losses')
      else:
        l2 = tf.losses.mean_squared_error(
            batch_labels, batch_hmg_prediction[FLAGS.num_level - 1])
      slim.summaries.add_scalar_summary(slim.losses.get_total_loss(),
                                        'loss', 'losses')

      global_step = slim.get_or_create_global_step()
      learning_rate_decay = tf.train.exponential_decay(
          learning_rate=FLAGS.learning_rate,
          global_step=global_step,
          decay_steps=FLAGS.lr_decay_steps,
          decay_rate=FLAGS.lr_decay_rate,
          staircase=True)
      optimizer = tf.train.AdamOptimizer(learning_rate_decay)

      is_chief = (FLAGS.task == 0)
      train_op = slim.learning.create_train_op(slim.losses.get_total_loss(),
                                               optimizer=optimizer)
      saver = tf.train.Saver(max_to_keep=20)
      if FLAGS.level_wise == 0:
        variables_to_restore = []
        for i in range(0, FLAGS.num_level - 1):
          variables = slim.get_variables(scope='%s/level%d' % (scope, i))
          variables_to_restore = variables_to_restore + variables
        init_fn = slim.assign_from_checkpoint_fn(FLAGS.model_path,
                                                 variables_to_restore)
      elif 'sem' in FLAGS.network_id:
        variables_to_restore = slim.get_variables(scope='vgg_16')
        init_fn = slim.assign_from_checkpoint_fn(FLAGS.vgg_model_path,
                                                 variables_to_restore)
      else:
        init_fn = None
      slim.learning.train(
          train_op=train_op,
          logdir=FLAGS.train_dir,
          save_summaries_secs=60,
          save_interval_secs=600,
          saver=saver,
          number_of_steps=FLAGS.max_step,
          master=FLAGS.master,
          is_chief=is_chief,
          init_fn=init_fn)


def run_eval(scope):
  """Evaluates a network.

  Args:
    scope: the scope of variables in this function
  """
  to_gray = True
  if 'sem' in FLAGS.network_id:
    to_gray = False
  batch_frames, batch_labels = get_samples(to_gray, 'eval')
  batch_hmg_prediction, _ = predict_homography(
      batch_frames, network_id=FLAGS.network_id, is_training=False, scope=scope)

  loss_dict = {}
  if 'hier' in FLAGS.network_id or 'mask' in FLAGS.network_id:
    for level in range(0, FLAGS.num_level):
      delta_level = FLAGS.num_level - level -1
      scale = 2 ** delta_level
      if FLAGS.loss == 'hier_ld':
        diff = tf.reshape(batch_labels / scale - batch_hmg_prediction[level],
                          [FLAGS.batch_size, 4, 2])
        sqrt_diff = tf.sqrt(tf.reduce_sum(tf.square(diff), 2))
        loss_dict['l2%d' % delta_level] = tf.metrics.mean(sqrt_diff)
      else:
        loss_dict['l2%d' % delta_level] = slim.metrics.mean_squared_error(
            batch_labels / scale, batch_hmg_prediction[level])
  else:
    loss_dict['loss'] = slim.metrics.mean_squared_error(
        batch_labels, batch_hmg_prediction[FLAGS.num_level - 1])

  names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(
      loss_dict)
  for name, value in six.iteritems(names_to_values):
    slim.summaries.add_scalar_summary(value, name, 'losses', print_summary=True)

  slim.evaluation.evaluation_loop(
      master=FLAGS.master,
      eval_interval_secs=60,
      checkpoint_dir=FLAGS.train_dir,
      logdir=FLAGS.eval_dir,
      eval_op=list(names_to_updates.values()),
      num_evals=FLAGS.num_eval_steps,
  )


def main(_):
  if FLAGS.mode == 'train':
    run_train('hier_hmg')
  elif FLAGS.mode == 'eval':
    run_eval('hier_hmg')
  else:
    raise ValueError('Unknown mode: %s' % FLAGS.mode)

if __name__ == '__main__':
  flags.mark_flag_as_required('train_dir')
  app.run(main)
