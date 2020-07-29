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

# Copyright 2020 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains MentorMix models.

See the README.md file for compilation and running instructions.
"""

import os
import time
import cifar_data_provider
import numpy as np
import resnet_model
import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils

flags = tf.app.flags

flags.DEFINE_integer('batch_size', 128, 'The number of images in each batch.')

flags.DEFINE_string('master', None, 'BNS name of the TensorFlow master to use.')

flags.DEFINE_string('data_dir', '', 'Data dir')

flags.DEFINE_string('train_log_dir', '', 'Directory to the save trained model.')

flags.DEFINE_string('dataset_name', 'cifar100', 'cifar10 or cifar100')

flags.DEFINE_string('studentnet', 'resnet32', 'network backbone.')

flags.DEFINE_float('learning_rate', 0.1, 'The learning rate')
flags.DEFINE_float('learning_rate_decay_factor', 0.9,
                   'learning rate decay factor')

flags.DEFINE_float('num_epochs_per_decay', 3,
                   'Number of epochs after which learning rate decays.')

flags.DEFINE_integer(
    'save_summaries_secs', 120,
    'The frequency with which summaries are saved, in seconds.')

flags.DEFINE_integer(
    'save_interval_secs', 1200,
    'The frequency with which the model is saved, in seconds.')

flags.DEFINE_integer('max_number_of_steps', 100000,
                     'The maximum number of gradient steps.')

flags.DEFINE_integer(
    'ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')

flags.DEFINE_string('device_id', '0', 'GPU device ID to run the job.')

# Learned MentorNet location
flags.DEFINE_string('trained_mentornet_dir', '',
                    'Directory where to find the trained MentorNet model.')

flags.DEFINE_list('example_dropout_rates', '0.0, 100',
                  'Comma-separated list indicating the example drop-out rate.'
                  'This has little impact to the performance.')

# Hyper-parameters for MentorMix to tune
flags.DEFINE_integer('burn_in_epoch', 0, 'Number of first epochs to perform'
                     'burn-in. In the burn-in period, every sample has a'
                     'fixed 1.0 weight.')

flags.DEFINE_float('loss_p_percentile', 0.7, 'p-percentile used to compute'
                   'the loss moving average.')

flags.DEFINE_float('mixup_alpha', 8.0, 'Alpha parameter for the beta'
                   'distribution to sample during mixup.')

flags.DEFINE_bool('second_reweight', True, 'Whether to weight the mixed up'
                  'examples again with mentornet')
FLAGS = flags.FLAGS

# Turn this on if there are no log outputs
tf.logging.set_verbosity(tf.logging.INFO)


def resnet_train_step(sess, train_op, global_step, train_step_kwargs):
  """Function that takes a gradient step and specifies whether to stop.

  Args:
    sess: The current session.
    train_op: An `Operation` that evaluates the gradients and returns the
      total loss.
    global_step: A `Tensor` representing the global training step.
    train_step_kwargs: A dictionary of keyword arguments.

  Returns:
    The total loss and a boolean indicating whether or not to stop training.

  Raises:
    ValueError: if 'should_trace' is in `train_step_kwargs` but `logdir` is not.
  """
  start_time = time.time()

  total_loss = tf.get_collection('total_loss')[0]

  _, np_global_step, total_loss_val = sess.run(
      [train_op, global_step, total_loss])

  time_elapsed = time.time() - start_time

  if 'should_log' in train_step_kwargs:
    if sess.run(train_step_kwargs['should_log']):
      tf.logging.info('global step %d: loss = %.4f (%.3f sec/step)',
                      np_global_step, total_loss_val, time_elapsed)

  if 'should_stop' in train_step_kwargs:
    should_stop = sess.run(train_step_kwargs['should_stop'])
  else:
    should_stop = False
  return total_loss, should_stop


def train_resnet_mentormix(max_step_run):
  """Trains the mentornet with the student resnet model.

  Args:
    max_step_run: The maximum number of gradient steps.
  """
  if not os.path.exists(FLAGS.train_log_dir):
    os.makedirs(FLAGS.train_log_dir)
  g = tf.Graph()

  with g.as_default():
    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
      tf_global_step = tf.train.get_or_create_global_step()

      (images, one_hot_labels, num_samples_per_epoch,
       num_of_classes) = cifar_data_provider.provide_resnet_data(
           FLAGS.dataset_name,
           'train',
           FLAGS.batch_size,
           dataset_dir=FLAGS.data_dir)

      hps = resnet_model.HParams(
          batch_size=FLAGS.batch_size,
          num_classes=num_of_classes,
          min_lrn_rate=0.0001,
          lrn_rate=FLAGS.learning_rate,
          num_residual_units=5,
          use_bottleneck=False,
          weight_decay_rate=0.0002,
          relu_leakiness=0.1,
          optimizer='mom')

      images.set_shape([FLAGS.batch_size, 32, 32, 3])

      # Define the model:
      resnet = resnet_model.ResNet(hps, images, one_hot_labels, mode='train')
      with tf.variable_scope('ResNet32'):
        logits = resnet.build_model()

      # Specify the loss function:
      loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=one_hot_labels, logits=logits)

      dropout_rates = utils.parse_dropout_rate_list(FLAGS.example_dropout_rates)
      example_dropout_rates = tf.convert_to_tensor(
          dropout_rates, np.float32, name='example_dropout_rates')

      loss_p_percentile = tf.convert_to_tensor(
          np.array([FLAGS.loss_p_percentile] * 100),
          np.float32,
          name='loss_p_percentile')

      loss = tf.reshape(loss, [-1, 1])

      epoch_step = tf.to_int32(
          tf.floor(tf.divide(tf_global_step, max_step_run) * 100))

      zero_labels = tf.zeros([tf.shape(loss)[0], 1], tf.float32)

      mentornet_net_hparams = utils.get_mentornet_network_hyperparameter(
          FLAGS.trained_mentornet_dir)

      # In the simplest case, this function can be replaced with a thresholding
      # function. See loss_thresholding_function in utils.py.
      v = utils.mentornet(
          epoch_step,
          loss,
          zero_labels,
          loss_p_percentile,
          example_dropout_rates,
          burn_in_epoch=FLAGS.burn_in_epoch,
          mentornet_net_hparams=mentornet_net_hparams,
          avg_name='individual')

      v = tf.stop_gradient(v)
      loss = tf.stop_gradient(tf.identity(loss))
      logits = tf.stop_gradient(tf.identity(logits))

      # Perform MentorMix
      images_mix, labels_mix = utils.mentor_mix_up(
          images, one_hot_labels, v, FLAGS.mixup_alpha)
      resnet = resnet_model.ResNet(hps, images_mix, labels_mix, mode='train')
      with tf.variable_scope('ResNet32', reuse=True):
        logits_mix = resnet.build_model()

      loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=labels_mix, logits=logits_mix)
      decay_loss = resnet.decay()

      # second weighting
      if FLAGS.second_reweight:
        loss = tf.reshape(loss, [-1, 1])
        v = utils.mentornet(
            epoch_step,
            loss,
            zero_labels,
            loss_p_percentile,
            example_dropout_rates,
            burn_in_epoch=FLAGS.burn_in_epoch,
            mentornet_net_hparams=mentornet_net_hparams,
            avg_name='mixed')
        v = tf.stop_gradient(v)
        weighted_loss_vector = tf.multiply(loss, v)
        loss = tf.reduce_mean(weighted_loss_vector)
        # reproduced with the following decay loss which should be 0.
        decay_loss = tf.losses.get_regularization_loss()
        decay_loss = decay_loss * (tf.reduce_sum(v) / FLAGS.batch_size)

      # Log data utilization
      data_util = utils.summarize_data_utilization(v, tf_global_step,
                                                   FLAGS.batch_size)

      loss = tf.reduce_mean(loss)
      slim.summaries.add_scalar_summary(
          tf.reduce_mean(loss), 'mentormix/mix_loss')

      weighted_total_loss = loss + decay_loss

      slim.summaries.add_scalar_summary(weighted_total_loss, 'total_loss')
      tf.add_to_collection('total_loss', weighted_total_loss)

      # Set up the moving averages:
      moving_average_variables = tf.trainable_variables()
      moving_average_variables = tf.contrib.framework.filter_variables(
          moving_average_variables, exclude_patterns=['mentornet'])

      variable_averages = tf.train.ExponentialMovingAverage(
          0.9999, tf_global_step)
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                           variable_averages.apply(moving_average_variables))

      decay_steps = FLAGS.num_epochs_per_decay * num_samples_per_epoch / FLAGS.batch_size
      lr = tf.train.exponential_decay(
          FLAGS.learning_rate,
          tf_global_step,
          decay_steps,
          FLAGS.learning_rate_decay_factor,
          staircase=True)
      lr = tf.squeeze(lr)
      slim.summaries.add_scalar_summary(lr, 'learning_rate')

      # Specify the optimization scheme:
      with tf.control_dependencies([weighted_total_loss, data_util]):
        # Set up training.
        trainable_variables = tf.trainable_variables()
        trainable_variables = tf.contrib.framework.filter_variables(
            trainable_variables, exclude_patterns=['mentornet'])

        grads = tf.gradients(weighted_total_loss, trainable_variables)
        optimizer = tf.train.MomentumOptimizer(lr, momentum=0.9)

        apply_op = optimizer.apply_gradients(
            zip(grads, trainable_variables),
            global_step=tf_global_step,
            name='train_step')

        train_ops = [apply_op] + resnet.extra_train_ops + tf.get_collection(
            tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(*train_ops)

      # Parameter restore setup
      if FLAGS.trained_mentornet_dir is not None:
        ckpt_model = FLAGS.trained_mentornet_dir
        if os.path.isdir(FLAGS.trained_mentornet_dir):
          ckpt_model = tf.train.latest_checkpoint(ckpt_model)

        # Fix the mentornet parameters
        variables_to_restore = slim.get_variables_to_restore(
            include=['mentornet', 'mentornet_inputs'])
        iassign_op1, ifeed_dict1 = tf.contrib.framework.assign_from_checkpoint(
            ckpt_model, variables_to_restore)

        # Create an initial assignment function.
        def init_assign_fn(sess):
          tf.logging.info('Restore using customer initializer %s', '.' * 10)
          sess.run(iassign_op1, ifeed_dict1)
      else:
        init_assign_fn = None

      tf.logging.info('-' * 20 + 'MentorMix' + '-' * 20)
      tf.logging.info('loss_p_percentile=%3f', FLAGS.loss_p_percentile)
      tf.logging.info('mixup_alpha=%d', FLAGS.mixup_alpha)
      tf.logging.info('-' * 20)

      saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=24)

      # Run training.
      slim.learning.train(
          train_op=train_op,
          train_step_fn=resnet_train_step,
          logdir=FLAGS.train_log_dir,
          master=FLAGS.master,
          is_chief=FLAGS.task == 0,
          saver=saver,
          number_of_steps=max_step_run,
          init_fn=init_assign_fn,
          save_summaries_secs=FLAGS.save_summaries_secs,
          save_interval_secs=FLAGS.save_interval_secs)


def main(_):
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device_id

  if FLAGS.studentnet == 'resnet32':
    train_resnet_mentormix(FLAGS.max_number_of_steps)
  else:
    tf.logging.error('unknown backbone student network %s', FLAGS.studentnet)


if __name__ == '__main__':
  tf.app.run()
