# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Entry point for AWD ENAS with a fixed architecture."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
import time

import numpy as np
import tensorflow as tf

from enas_lm.src import fixed_lib
from enas_lm.src import utils
from tensorflow.contrib import training as contrib_training


flags = tf.app.flags
gfile = tf.gfile
FLAGS = flags.FLAGS

flags.DEFINE_boolean('reset_output_dir', True, '')
flags.DEFINE_string('output_dir', None, '')
flags.DEFINE_string('data_path', None, '')

flags.DEFINE_integer('log_every', 200, '')


def get_ops(params, x_train, x_valid, x_test):
  """Build [train, valid, test] graphs."""
  lm = fixed_lib.LM(params, x_train, x_valid, x_test)
  params.add_hparam('num_train_batches', lm.num_train_batches)
  ops = {
      'train_op': lm.train_op,
      'learning_rate': lm.learning_rate,
      'grad_norm': lm.grad_norm,
      'train_loss': lm.train_loss,
      'global_step': tf.train.get_or_create_global_step(),
      'reset_batch_states': lm.batch_init_states['reset'],
      'eval_valid': lm.eval_valid,
      'eval_test': lm.eval_test,

      'reset_start_idx': lm.reset_start_idx,
      'should_reset': lm.should_reset,
      'moving_avg_started': lm.moving_avg_started,
      'update_moving_avg': lm.update_moving_avg_ops,
      'start_moving_avg': lm.start_moving_avg_op,
  }
  print('-' * 80)
  print('HParams:\n{0}'.format(params.to_json(indent=2, sort_keys=True)))

  return ops


def train(params):
  """Entry point for training."""
  with gfile.GFile(params.data_path, 'rb') as finp:
    x_train, x_valid, x_test, _, _ = pickle.load(finp)
    print('-' * 80)
    print('train_size: {0}'.format(np.size(x_train)))
    print('valid_size: {0}'.format(np.size(x_valid)))
    print(' test_size: {0}'.format(np.size(x_test)))

  g = tf.Graph()
  with g.as_default():
    tf.random.set_random_seed(2126)
    ops = get_ops(params, x_train, x_valid, x_test)
    run_ops = [
        ops['train_loss'],
        ops['grad_norm'],
        ops['learning_rate'],
        ops['should_reset'],
        ops['moving_avg_started'],
        ops['train_op'],
    ]

    saver = tf.train.Saver(max_to_keep=2)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook(
        params.output_dir, save_steps=params.num_train_batches, saver=saver)
    hooks = [checkpoint_saver_hook]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.train.SingularMonitoredSession(config=config, hooks=hooks,
                                             checkpoint_dir=params.output_dir)
    accum_loss = 0.
    accum_step = 0
    best_valid_ppl = []
    start_time = time.time()
    while True:
      try:
        loss, gn, lr, should_reset, moving_avg_started, _ = sess.run(run_ops)
        accum_loss += loss
        accum_step += 1
        step = sess.run(ops['global_step'])
        if step % params.log_every == 0:
          epoch = step // params.num_train_batches
          train_ppl = np.exp(accum_loss / accum_step)
          mins_so_far = (time.time() - start_time) / 60.
          log_string = 'epoch={0:<5d}'.format(epoch)
          log_string += ' step={0:<7d}'.format(step)
          log_string += ' ppl={0:<10.2f}'.format(train_ppl)
          log_string += ' lr={0:<6.3f}'.format(lr)
          log_string += ' |g|={0:<6.3f}'.format(gn)
          log_string += ' avg={0:<2d}'.format(moving_avg_started)
          log_string += ' mins={0:<.2f}'.format(mins_so_far)
          print(log_string)

        if moving_avg_started:
          sess.run(ops['update_moving_avg'])

        # if step % params.num_train_batches == 0:
        if should_reset:
          sess.run(ops['reset_batch_states'])
          accum_loss = 0
          accum_step = 0
          valid_ppl = ops['eval_valid'](sess, use_moving_avg=moving_avg_started)
          sess.run([ops['reset_batch_states'], ops['reset_start_idx']])
          if (not moving_avg_started and
              len(best_valid_ppl) > params.best_valid_ppl_threshold and
              valid_ppl > min(best_valid_ppl[:-params.best_valid_ppl_threshold])
             ):
            print('Starting moving_avg')
            sess.run(ops['start_moving_avg'])
          best_valid_ppl.append(valid_ppl)

        if step >= params.num_train_steps:
          ops['eval_test'](sess, use_moving_avg=moving_avg_started)
          break
      except tf.errors.InvalidArgumentError:
        last_checkpoint = tf.train.latest_checkpoint(params.output_dir)
        print('rolling back to previous checkpoint {0}'.format(last_checkpoint))
        saver.restore(sess, last_checkpoint)
        accum_loss, accum_step = 0., 0
    sess.close()


def main(unused_args):
  print('-' * 80)
  output_dir = FLAGS.output_dir

  print('-' * 80)
  if not gfile.IsDirectory(output_dir):
    print('Path {} does not exist. Creating'.format(output_dir))
    gfile.MakeDirs(output_dir)
  elif FLAGS.reset_output_dir:
    print('Path {} exists. Reseting'.format(output_dir))
    gfile.DeleteRecursively(output_dir)
    gfile.MakeDirs(output_dir)

  print('-' * 80)
  log_file = os.path.join(output_dir, 'stdout')
  print('Logging to {}'.format(log_file))
  sys.stdout = utils.Logger(log_file)

  params = contrib_training.HParams(
      data_path=FLAGS.data_path,
      log_every=FLAGS.log_every,
      output_dir=output_dir,
  )

  train(params)

if __name__ == '__main__':
  tf.app.run()
