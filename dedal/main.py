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

r"""Trains a keras model on a tf.Dataset.

Instructions.

python3 -m dedal.main -- \
--base_dir /tmp/example/1/ \
--gin_config dedal.gin \
--task train \
--alsologtostderr
"""

import os.path

from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow as tf

from dedal.train import training_loop

flags.DEFINE_string('base_dir', None, 'Directory to save trained model in.')
flags.DEFINE_string(
    'reference_dir',
    None,
    'Directory where to read the reference model from (if exists).')
flags.DEFINE_enum(
    'task', 'train', ['train', 'eval', 'downstream'],
    'Whether this is a train, eval or downstream task.')
flags.DEFINE_multi_string(
    'gin_config', [], 'List of paths to the config files.')
flags.DEFINE_multi_string(
    'gin_bindings', [], 'Newline separated list of Gin parameter bindings.')
flags.DEFINE_string(
    'config_path', 'dedal/configs', 'Where to find the gin configurations.')
FLAGS = flags.FLAGS


def main(unused_argv):
  filenames = [os.path.join(FLAGS.config_path, p) for p in FLAGS.gin_config]
  gin.parse_config_files_and_bindings(filenames, FLAGS.gin_bindings)

  strategy = training_loop.get_strategy()
  logging.info('Distribution strategy: %s', strategy)
  logging.info('Devices: %s', tf.config.list_physical_devices())

  reference = FLAGS.reference_dir
  kwargs = {} if reference is None else {'reference_workdir': reference}
  loop = training_loop.TrainingLoop(FLAGS.base_dir, strategy, **kwargs)
  loop.run(FLAGS.task)


if __name__ == '__main__':
  app.run(main)
