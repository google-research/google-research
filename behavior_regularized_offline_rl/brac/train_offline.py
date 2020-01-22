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

"""Offline training binary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging


import gin
import tensorflow.compat.v1 as tf

from behavior_regularized_offline_rl.brac import agents
from behavior_regularized_offline_rl.brac import train_eval_offline
from behavior_regularized_offline_rl.brac import utils

tf.compat.v1.enable_v2_behavior()


# Flags for which data to load.
flags.DEFINE_string('data_root_dir',
                    os.path.join(os.getenv('HOME', '/'),
                                 'tmp/offlinerl/data'),
                    'Root directory for data.')
flags.DEFINE_string('data_sub_dir', '0', '')
flags.DEFINE_string('data_name', 'eps1', 'data name.')
flags.DEFINE_string('data_file_name', 'data', 'data checkpoint file name.')

# Flags for offline training.
flags.DEFINE_string('root_dir',
                    os.path.join(os.getenv('HOME', '/'), 'tmp/offlinerl/learn'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('sub_dir', '0', '')
flags.DEFINE_string('agent_name', 'sac', 'agent name.')
flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'env name.')
flags.DEFINE_integer('seed', 0, 'random seed, mainly for training samples.')
flags.DEFINE_integer('total_train_steps', int(5e5), '')
flags.DEFINE_integer('n_eval_episodes', 20, '')
flags.DEFINE_integer('n_train', int(1e6), '')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS


def main(_):
  logging.set_verbosity(logging.INFO)
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)
  # Setup data file path.
  data_dir = os.path.join(
      FLAGS.data_root_dir,
      FLAGS.env_name,
      FLAGS.data_name,
      FLAGS.data_sub_dir,
      )
  data_file = os.path.join(
      data_dir, FLAGS.data_file_name)

  # Setup log dir.
  if FLAGS.sub_dir == 'auto':
    sub_dir = utils.get_datetime()
  else:
    sub_dir = FLAGS.sub_dir
  log_dir = os.path.join(
      FLAGS.root_dir,
      FLAGS.env_name,
      FLAGS.data_name,
      'n'+str(FLAGS.n_train),
      FLAGS.agent_name,
      sub_dir,
      str(FLAGS.seed),
      )
  utils.maybe_makedirs(log_dir)
  train_eval_offline.train_eval_offline(
      log_dir=log_dir,
      data_file=data_file,
      agent_module=agents.AGENT_MODULES_DICT[FLAGS.agent_name],
      env_name=FLAGS.env_name,
      n_train=FLAGS.n_train,
      total_train_steps=FLAGS.total_train_steps,
      n_eval_episodes=FLAGS.n_eval_episodes,
      )


if __name__ == '__main__':
  app.run(main)
