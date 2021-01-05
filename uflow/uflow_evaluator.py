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

"""Continually polls and evaluates new checkpoints."""

import time

from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow as tf

from uflow import uflow_data
# pylint:disable=unused-import
from uflow import uflow_flags
from uflow import uflow_main
from uflow import uflow_plotting


FLAGS = flags.FLAGS


def evaluate():
  """Eval happens on GPU or CPU, and evals each checkpoint as it appears."""
  tf.compat.v1.enable_eager_execution()

  candidate_checkpoint = None
  uflow = uflow_main.create_uflow()
  evaluate_fn, _ = uflow_data.make_eval_function(
      FLAGS.eval_on,
      FLAGS.height,
      FLAGS.width,
      progress_bar=True,
      plot_dir=FLAGS.plot_dir,
      num_plots=50)

  latest_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
  while 1:
    # Wait for a new checkpoint
    while candidate_checkpoint == latest_checkpoint:
      logging.log_every_n(logging.INFO,
                          'Waiting for a new checkpoint, at %s, latest is %s',
                          20, FLAGS.checkpoint_dir, latest_checkpoint)
      time.sleep(0.5)
      candidate_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    candidate_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    latest_checkpoint = candidate_checkpoint
    logging.info('New checkpoint found: %s', candidate_checkpoint)
    # This forces the checkpoint manager to reexamine the checkpoint directory
    # and become aware of the new checkpoint.
    uflow.update_checkpoint_dir(FLAGS.checkpoint_dir)
    uflow.restore()
    eval_results = evaluate_fn(uflow)
    uflow_plotting.print_eval(eval_results)
    step = tf.compat.v1.train.get_global_step().numpy()
    if step >= FLAGS.num_train_steps:
      logging.info('Evaluator terminating - completed evaluation of checkpoint '
                   'from step %d', step)
      return


def main(unused_argv):

  gin.parse_config_files_and_bindings(FLAGS.config_file, FLAGS.gin_bindings)

  # Make directories if they do not exist yet.
  if FLAGS.checkpoint_dir and not tf.io.gfile.exists(FLAGS.checkpoint_dir):
    logging.info('Making new checkpoint directory %s', FLAGS.checkpoint_dir)
    tf.io.gfile.makedirs(FLAGS.checkpoint_dir)
  if FLAGS.plot_dir and not tf.io.gfile.exists(FLAGS.plot_dir):
    logging.info('Making new plot directory %s', FLAGS.plot_dir)
    tf.io.gfile.makedirs(FLAGS.plot_dir)

  if FLAGS.no_tf_function:
    tf.config.experimental_run_functions_eagerly(True)
    logging.info('TFFUNCTION DISABLED')

  if FLAGS.eval_on:
    evaluate()
  else:
    raise ValueError('evaluation needs --eval_on <dataset>.')


if __name__ == '__main__':
  app.run(main)
