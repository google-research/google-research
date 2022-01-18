# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Trains a keras model on a tf.Dataset."""

import os.path
import time

from absl import app
from absl import flags

import gin
import tensorflow.compat.v2 as tf

from soft_sort.matrix_factorization import training


flags.DEFINE_string('base_dir', None, 'Directory to save trained model in.')
flags.DEFINE_integer('seed', None, 'A random seed.')
flags.DEFINE_multi_string('gin_config', [], 'List of config files paths.')
flags.DEFINE_multi_string(
    'gin_bindings', [], 'Multi string of gin parameter bindings.')
flags.DEFINE_string(
    'config_path', 'matrix_factorization/configs/',
    'Where to find the gin config files.')
FLAGS = flags.FLAGS


def main(unused_argv):
  tf.enable_v2_behavior()
  filenames = [os.path.join(FLAGS.config_path, p) for p in FLAGS.gin_config]
  gin.parse_config_files_and_bindings(filenames, FLAGS.gin_bindings)

  seed = FLAGS.seed if FLAGS.seed is not None else int(time.time())
  tf.random.set_seed(seed)
  training.TrainingLoop(workdir=FLAGS.base_dir).run()


if __name__ == '__main__':
  app.run(main)
