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

"""A training loop for the various models in this directory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import math
import os
import random
import time
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from depth_from_video_in_the_wild import model

gfile = tf.gfile
MAX_TO_KEEP = 1000000  # Maximum number of checkpoints to keep.

flags.DEFINE_string('data_dir', None, 'Preprocessed data.')

flags.DEFINE_string('file_extension', 'png', 'Image data file extension.')

flags.DEFINE_float('learning_rate', 1e-4, 'Adam learning rate.')

flags.DEFINE_float('reconstr_weight', 0.85, 'Frame reconstruction loss weight.')

flags.DEFINE_float('ssim_weight', 3.0, 'SSIM loss weight.')

flags.DEFINE_float('smooth_weight', 1e-2, 'Smoothness loss weight.')

flags.DEFINE_float('depth_consistency_loss_weight', 0.01,
                   'Depth consistency loss weight')

flags.DEFINE_integer('batch_size', 4, 'The size of a sample batch')

flags.DEFINE_integer('img_height', 128, 'Input frame height.')

flags.DEFINE_integer('img_width', 416, 'Input frame width.')

flags.DEFINE_integer('queue_size', 2000,
                     'Items in queue. Use smaller number for local debugging.')

flags.DEFINE_integer('seed', 8964, 'Seed for random number generators.')

flags.DEFINE_float('weight_reg', 1e-2, 'The amount of weight regularization to '
                   'apply. This has no effect on the ResNet-based encoder '
                   'architecture.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory to save model '
                    'checkpoints.')

flags.DEFINE_integer('train_steps', int(1e6), 'Number of training steps.')

flags.DEFINE_integer('summary_freq', 100, 'Save summaries every N steps.')

flags.DEFINE_bool('debug', False, 'If true, one training step is performed and '
                  'the results are dumped to a folder for debugging.')

flags.DEFINE_string('input_file', 'train', 'Input file name')

flags.DEFINE_float('rotation_consistency_weight', 1e-3, 'Weight of rotation '
                   'cycle consistency loss.')

flags.DEFINE_float('translation_consistency_weight', 1e-2, 'Weight of '
                   'thanslation consistency loss.')

flags.DEFINE_integer('foreground_dilation', 8, 'Dilation of the foreground '
                     'mask (in pixels).')

flags.DEFINE_boolean('learn_intrinsics', True, 'Whether to learn camera '
                     'intrinsics.')

flags.DEFINE_boolean('boxify', True, 'Whether to convert segmentation masks to '
                     'bounding boxes.')

flags.DEFINE_string('imagenet_ckpt', None, 'Path to an imagenet checkpoint to '
                    'intialize from.')


FLAGS = flags.FLAGS
flags.mark_flag_as_required('data_dir')
flags.mark_flag_as_required('checkpoint_dir')


def load(filename):
  with gfile.Open(filename) as f:
    return np.load(io.BytesIO(f.read()))


def _print_losses(dir1):
  for f in gfile.ListDirectory(dir1):
    if 'loss' in f:
      print ('----------', f, end=' ')
      f1 = os.path.join(dir1, f)
      t1 = load(f1).astype(float)
      print (t1)


def main(_):
  # Fixed seed for repeatability
  seed = FLAGS.seed
  tf.set_random_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  if not gfile.Exists(FLAGS.checkpoint_dir):
    gfile.MakeDirs(FLAGS.checkpoint_dir)

  train_model = model.Model(
      boxify=FLAGS.boxify,
      data_dir=FLAGS.data_dir,
      file_extension=FLAGS.file_extension,
      is_training=True,
      foreground_dilation=FLAGS.foreground_dilation,
      learn_intrinsics=FLAGS.learn_intrinsics,
      learning_rate=FLAGS.learning_rate,
      reconstr_weight=FLAGS.reconstr_weight,
      smooth_weight=FLAGS.smooth_weight,
      ssim_weight=FLAGS.ssim_weight,
      translation_consistency_weight=FLAGS.translation_consistency_weight,
      rotation_consistency_weight=FLAGS.rotation_consistency_weight,
      batch_size=FLAGS.batch_size,
      img_height=FLAGS.img_height,
      img_width=FLAGS.img_width,
      weight_reg=FLAGS.weight_reg,
      depth_consistency_loss_weight=FLAGS.depth_consistency_loss_weight,
      queue_size=FLAGS.queue_size,
      input_file=FLAGS.input_file)

  _train(train_model, FLAGS.checkpoint_dir, FLAGS.train_steps,
         FLAGS.summary_freq)

  if FLAGS.debug:
    _print_losses(os.path.join(FLAGS.checkpoint_dir, 'debug'))


def _train(train_model, checkpoint_dir, train_steps, summary_freq):
  """Runs a trainig loop."""
  saver = train_model.saver
  sv = tf.train.Supervisor(logdir=checkpoint_dir, save_summaries_secs=0,
                           saver=None)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with sv.managed_session(config=config) as sess:
    logging.info('Attempting to resume training from %s...', checkpoint_dir)
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    logging.info('Last checkpoint found: %s', checkpoint)
    if checkpoint:
      saver.restore(sess, checkpoint)
    elif FLAGS.imagenet_ckpt:
      logging.info('Restoring pretrained weights from %s', FLAGS.imagenet_ckpt)
      train_model.imagenet_init_restorer.restore(sess, FLAGS.imagenet_ckpt)

    logging.info('Training...')
    start_time = time.time()
    last_summary_time = time.time()
    steps_per_epoch = train_model.reader.steps_per_epoch
    step = 1
    while step <= train_steps:
      fetches = {
          'train': train_model.train_op,
          'global_step': train_model.global_step,
      }
      if step % summary_freq == 0:
        fetches['loss'] = train_model.total_loss
        fetches['summary'] = sv.summary_op

      if FLAGS.debug:
        fetches.update(train_model.exports)

      results = sess.run(fetches)
      global_step = results['global_step']

      if step % summary_freq == 0:
        sv.summary_writer.add_summary(results['summary'], global_step)
        train_epoch = math.ceil(global_step / steps_per_epoch)
        train_step = global_step - (train_epoch - 1) * steps_per_epoch
        this_cycle = time.time() - last_summary_time
        last_summary_time += this_cycle
        logging.info(
            'Epoch: [%2d] [%5d/%5d] time: %4.2fs (%ds total) loss: %.3f',
            train_epoch, train_step, steps_per_epoch, this_cycle,
            time.time() - start_time, results['loss'])

      if FLAGS.debug:
        debug_dir = os.path.join(checkpoint_dir, 'debug')
        if not gfile.Exists(debug_dir):
          gfile.MkDir(debug_dir)
        for name, tensor in results.iteritems():
          if name == 'summary':
            continue
          s = io.BytesIO()
          filename = os.path.join(debug_dir, name)
          np.save(s, tensor)
          with gfile.Open(filename, 'w') as f:
            f.write(s.getvalue())
        return

      # steps_per_epoch == 0 is intended for debugging, when we run with a
      # single image for sanity check
      if steps_per_epoch == 0 or step % steps_per_epoch == 0:
        logging.info('[*] Saving checkpoint to %s...', checkpoint_dir)
        saver.save(sess, os.path.join(checkpoint_dir, 'model'),
                   global_step=global_step)

      # Setting step to global_step allows for training for a total of
      # train_steps even if the program is restarted during training.
      step = global_step + 1


if __name__ == '__main__':
  app.run(main)
