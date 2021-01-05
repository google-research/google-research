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

r"""Training code based on TF Eager."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v2 as tf

from tcc.algorithms import get_algo
from tcc.config import CONFIG
from tcc.datasets import create_dataset
from tcc.utils import get_lr_fn
from tcc.utils import get_lr_opt_global_step
from tcc.utils import restore_ckpt
from tcc.utils import setup_train_dir
from tcc.utils import Stopwatch


flags.DEFINE_string('logdir', '/tmp/alignment_logs', 'Path to logs.')
flags.DEFINE_boolean('defun', True, 'Defun functions in algo for faster '
                     'training.')
flags.DEFINE_boolean('debug', False, 'Plots detailed summaries on Tensorboard.')
flags.DEFINE_boolean(
    'force_train', False, 'Continue with training even when '
    'train_logs exist. Useful if one has to resume training. '
    'By default switched off to prevent overwriting existing '
    'experiments.')
flags.DEFINE_boolean('visualize', False, 'Visualize images, gradients etc. '
                     'Switched off by for default to speed training up and '
                     'takes less memory.')

FLAGS = flags.FLAGS
layers = tf.keras.layers


def train():
  """Trains model and evaluates on relevant downstream tasks."""
  CONFIG.LOGDIR = FLAGS.logdir
  logdir = CONFIG.LOGDIR
  setup_train_dir(logdir)

  # Common code for multigpu and single gpu. Set devices here if you don't
  # want to use all the GPUs on the machine. Default is to use all GPUs.
  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
    algo = get_algo(CONFIG.TRAINING_ALGO)

    # Setup summary writer.
    summary_writer = tf.summary.create_file_writer(
        os.path.join(logdir, 'train_logs'), flush_millis=10000)

    learning_rate, optimizer, global_step = get_lr_opt_global_step()
    ckpt_manager, _, _ = restore_ckpt(
        logdir=logdir, optimizer=optimizer, **algo.model)

    global_step_value = global_step.numpy()

    # Remember in Eager mode learning rate variable needs to be updated
    # manually. Calling lr_fn each iteration to get current learning rate.
    lr_fn = get_lr_fn(CONFIG.OPTIMIZER)

    # Setup Dataset Iterators from train and val datasets.
    batch_size_per_replica = CONFIG.TRAIN.BATCH_SIZE
    total_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
    train_ds = create_dataset('train', mode='train',
                              batch_size=total_batch_size,
                              return_iterator=False)
    train_iterator = strategy.make_dataset_iterator(train_ds)

    def train_step(data):
      steps = data['chosen_steps']
      seq_lens = data['seq_lens']
      loss = algo.train_one_iter(data, steps, seq_lens, global_step, optimizer)
      return loss

    # This reduction only affects reporting, not the gradients.
    # pylint: disable=g-long-lambda
    dist_train = lambda it: strategy.reduce(
        tf.distribute.ReduceOp.SUM, strategy.experimental_run(train_step, it),
        axis=None)
    # pylint: enable=g-long-lambda
    if FLAGS.defun:
      dist_train = tf.function(dist_train)

    stopwatch = Stopwatch()

    try:
      while global_step_value < CONFIG.TRAIN.MAX_ITERS:
        with summary_writer.as_default():
          with tf.summary.record_if(
              global_step_value % CONFIG.LOGGING.REPORT_INTERVAL == 0):

            loss = dist_train(train_iterator)

            # Update learning rate based in lr_fn.
            learning_rate.assign(lr_fn(learning_rate, global_step))

            tf.summary.scalar('loss', loss, step=global_step)
            tf.summary.scalar('learning_rate', learning_rate, step=global_step)

            # Save checkpoint.
            if global_step_value % CONFIG.CHECKPOINT.SAVE_INTERVAL == 0:
              ckpt_manager.save()
              logging.info('Checkpoint saved at iter %d.', global_step_value)

            # Update global step.
            global_step_value = global_step.numpy()

            time_per_iter = stopwatch.elapsed()

            tf.summary.scalar(
                'timing/time_per_iter', time_per_iter, step=global_step)

            logging.info('Iter[{}/{}], {:.1f}s/iter, Loss: {:.3f}'.format(
                global_step_value, CONFIG.TRAIN.MAX_ITERS, time_per_iter,
                loss.numpy()))
            # Reset stopwatch after iter is complete.
            stopwatch.reset()

    except KeyboardInterrupt:
      logging.info('Caught keyboard interrupt. Saving model before quitting.')

    finally:
      # Save the final checkpoint.
      ckpt_manager.save()
      logging.info('Checkpoint saved at iter %d', global_step_value)


def main(_):
  tf.enable_v2_behavior()
  tf.keras.backend.set_learning_phase(1)

  train()

if __name__ == '__main__':
  app.run(main)
