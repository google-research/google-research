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

"""Entry point for AWD ENAS search process."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
import time

import numpy as np
import tensorflow.compat.v1 as tf

from enas_lm.src import child
from enas_lm.src import controller
from enas_lm.src import utils
from tensorflow.contrib import training as contrib_training

flags = tf.app.flags
gfile = tf.gfile
FLAGS = flags.FLAGS

flags.DEFINE_boolean('reset_output_dir', False, '')
flags.DEFINE_string('output_dir', None, '')
flags.DEFINE_string('data_path', None, '')

flags.DEFINE_integer('log_every', 200, '')


def get_ops(params, x_train, x_valid):
  """Build [train, valid, test] graphs."""

  ct = controller.Controller(params=params)
  lm = child.LM(params, ct, x_train, x_valid)
  ct.build_trainer(lm)
  params.add_hparam('num_train_batches', lm.num_train_batches)
  ops = {
      'train_op': lm.train_op,
      'learning_rate': lm.learning_rate,
      'grad_norm': lm.grad_norm,
      'train_loss': lm.train_loss,
      'l2_reg_loss': lm.l2_reg_loss,
      'global_step': tf.train.get_or_create_global_step(),
      'reset_batch_states': lm.batch_init_states['reset'],
      'eval_valid': lm.eval_valid,

      'reset_start_idx': lm.reset_start_idx,
      'should_reset': lm.should_reset,

      'controller_train_op': ct.train_op,
      'controller_grad_norm': ct.train_op,
      'controller_sample_arc': ct.sample_arc,
      'controller_entropy': ct.sample_entropy,
      'controller_reward': ct.reward,
      'controller_baseline': ct.baseline,
      'controller_optimizer': ct.optimizer,
      'controller_train_fn': ct.train,
  }
  print('-' * 80)
  print('HParams:\n{0}'.format(params.to_json(indent=2, sort_keys=True)))

  return ops


def train(params):
  """Entry train function."""
  with gfile.GFile(params.data_path, 'rb') as finp:
    x_train, x_valid, _, _, _ = pickle.load(finp)
    print('-' * 80)
    print('train_size: {0}'.format(np.size(x_train)))
    print('valid_size: {0}'.format(np.size(x_valid)))

  g = tf.Graph()
  with g.as_default():
    ops = get_ops(params, x_train, x_valid)
    run_ops = [
        ops['train_loss'],
        ops['l2_reg_loss'],
        ops['grad_norm'],
        ops['learning_rate'],
        ops['should_reset'],
        ops['train_op'],
    ]

    saver = tf.train.Saver(max_to_keep=5)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook(
        params.output_dir, save_steps=params.num_train_batches, saver=saver)
    hooks = [checkpoint_saver_hook]
    hooks.append(ops['controller_optimizer'].make_session_run_hook(True))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.train.SingularMonitoredSession(config=config, hooks=hooks,
                                             checkpoint_dir=params.output_dir)
    accum_loss = 0
    accum_step = 0
    epoch = 0
    best_valid_ppl = []
    start_time = time.time()
    while True:
      try:
        loss, l2_reg, gn, lr, should_reset, _ = sess.run(run_ops)

        accum_loss += loss
        accum_step += 1
        step = sess.run(ops['global_step'])
        if step % params.log_every == 0:
          train_ppl = np.exp(accum_loss / accum_step)
          mins_so_far = (time.time() - start_time) / 60.
          log_string = 'epoch={0:<5d}'.format(epoch)
          log_string += ' step={0:<7d}'.format(step)
          log_string += ' ppl={0:<9.2f}'.format(train_ppl)
          log_string += ' lr={0:<7.2f}'.format(lr)
          log_string += ' |w|={0:<6.2f}'.format(l2_reg)
          log_string += ' |g|={0:<6.2f}'.format(gn)
          log_string += ' mins={0:<.2f}'.format(mins_so_far)
          print(log_string)

        if should_reset:
          ops['controller_train_fn'](sess, ops['reset_batch_states'])
          epoch += 1
          accum_loss = 0
          accum_step = 0
          valid_ppl = ops['eval_valid'](sess)
          sess.run([ops['reset_batch_states'], ops['reset_start_idx']])
          best_valid_ppl.append(valid_ppl)

        if step >= params.num_train_steps:
          break
      except tf.errors.InvalidArgumentError:
        last_checkpoint = tf.train.latest_checkpoint(params.output_dir)
        print('rolling back to previous checkpoint {0}'.format(last_checkpoint))
        saver.restore(sess, last_checkpoint)

    sess.close()


def main(unused_args):
  np.set_printoptions(precision=3, suppress=True, threshold=int(1e9),
                      linewidth=80)

  print('-' * 80)
  if not gfile.IsDirectory(FLAGS.output_dir):
    print('Path {} does not exist. Creating'.format(FLAGS.output_dir))
    gfile.MakeDirs(FLAGS.output_dir)
  elif FLAGS.reset_output_dir:
    print('Path {} exists. Reseting'.format(FLAGS.output_dir))
    gfile.DeleteRecursively(FLAGS.output_dir)
    gfile.MakeDirs(FLAGS.output_dir)

  print('-' * 80)
  log_file = os.path.join(FLAGS.output_dir, 'stdout')
  print('Logging to {}'.format(log_file))
  sys.stdout = utils.Logger(log_file)

  params = contrib_training.HParams(
      data_path=FLAGS.data_path,
      log_every=FLAGS.log_every,
      output_dir=FLAGS.output_dir,
  )

  train(params)

if __name__ == '__main__':
  tf.app.run()
