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

# pylint:skip-file
from absl import app
from absl import flags
from absl import logging
import os
import gin
import tensorflow as tf


candidate_checkpoint = None
smurf =build_network(batch_size=1)

weights = {
    'census': FLAGS.weight_census,
    'smooth1': FLAGS.weight_smooth1,
    'smooth2': FLAGS.weight_smooth2,
}
evaluate_fn, _ = smurf_data.make_eval_function(
    FLAGS.eval_on,
    FLAGS.height,
    FLAGS.width,
    progress_bar=True,
    plot_dir=FLAGS.plot_dir,
    num_plots=50,
    weights=weights)

latest_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
while 1:
  # Wait for a new checkpoint
  while candidate_checkpoint == latest_checkpoint:
    logging.log_every_n(logging.INFO,
                        'Waiting for a new checkpoint, at %s, latest is %s',
                        3, FLAGS.checkpoint_dir, latest_checkpoint)
    time.sleep(45.)
    candidate_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
  candidate_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
  latest_checkpoint = candidate_checkpoint
  logging.info('New checkpoint found: %s', candidate_checkpoint)
  # This forces the checkpoint manager to reexamine the checkpoint directory
  # and become aware of the new checkpoint.
  smurf.update_checkpoint_dir(FLAGS.checkpoint_dir)
  smurf.restore()

  step = tf.compat.v1.train.get_global_step().numpy()
  terminate = False
  if step >= FLAGS.num_train_steps:
    # If initializing from another checkpoint directory, the first checkpoint
    # will be the init checkpoint and might have steps > num_train_steps.
    # Don't quit in this case.
    terminate = True
    if FLAGS.init_checkpoint_dir:
      with gfile.Open(os.path.join(FLAGS.checkpoint_dir, 'checkpoint'),
                      'r') as f:
        if len(f.readlines()) == 2:
          logging.info('Continuing evaluation after evaluating '
                        'init_checkpoint.')
          terminate = False

  eval_results = evaluate_fn(smurf)
  smurf_plotting.print_eval(eval_results)

  if terminate or FLAGS.run_eval_once:
    pass
    # return


def main(unused_argv):
  if FLAGS.no_tf_function:
    tf.config.experimental_run_functions_eagerly(True)
    print('TFFUNCTION DISABLED')

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
