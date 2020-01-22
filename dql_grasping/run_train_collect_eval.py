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

"""Main binary for Q-grasping experiments. See README.md for how to run this.

Runs train-collect-eval loop, or just collect-eval, potentially on-policy if
offpolicy training dirs includes where we are writing collect data to.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import gin
import numpy as np
import tensorflow.compat.v1 as tf
from dql_grasping import gin_imports  # pylint: disable=unused-import
from dql_grasping import train_collect_eval

FLAGS = flags.FLAGS

flags.DEFINE_string('gin_config', None,
                    'A string of Gin parameter bindings.')
flags.DEFINE_enum('run_mode', 'both', ['train_only',
                                       'collect_eval_once',
                                       'collect_eval_loop',
                                       'both'],
                  'What mode to run the train-collect-eval loop.')
flags.DEFINE_string('master', '', 'An address of TensorFlow runtime to use.')
flags.DEFINE_bool('distributed', False,
                  'If False, tasks specify independent random trials instead of'
                  ' distributed worker.')
flags.DEFINE_integer('task', 0, 'replica task id. Also used for random seed.')
flags.DEFINE_integer('ps_tasks', 0, 'Number of parameter server tasks.')
flags.DEFINE_string('root_dir', '', 'Root directory for this experiment.')


def main(_):
  np.random.seed(FLAGS.task)
  tf.set_random_seed(FLAGS.task)

  if FLAGS.distributed:
    task = FLAGS.task
  else:
    task = 0

  if FLAGS.gin_config:
    if tf.gfile.Exists(FLAGS.gin_config):
      # Parse as a file.
      with tf.gfile.Open(FLAGS.gin_config) as f:
        gin.parse_config(f)
    else:
      gin.parse_config(FLAGS.gin_config)

  gin.finalize()

  if FLAGS.run_mode == 'collect_eval_once':
    train_collect_eval.train_collect_eval(root_dir=FLAGS.root_dir,
                                          train_fn=None,
                                          task=FLAGS.task)
  elif FLAGS.run_mode == 'train_only':
    train_collect_eval.train_collect_eval(root_dir=FLAGS.root_dir,
                                          do_collect_eval=False,
                                          task=task,
                                          master=FLAGS.master,
                                          ps_tasks=FLAGS.ps_tasks)
  elif FLAGS.run_mode == 'collect_eval_loop':
    raise NotImplementedError('collect_eval_loops')
  else:
    # Synchronous train-collect-eval.
    train_collect_eval.train_collect_eval(root_dir=FLAGS.root_dir,
                                          task=task)

if __name__ == '__main__':
  app.run(main)
