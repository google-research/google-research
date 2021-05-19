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

# Lint as: python3
"""Trains a keras model on a tf.Dataset."""

import os

from absl import app
from absl import flags
import gin
import tensorflow.compat.v2 as tf

from perturbations.experiments import training_loop


flags.DEFINE_string('base_dir', None, 'Directory to save trained model in.')
flags.DEFINE_integer('run_number', None, 'Id of the run.')
flags.DEFINE_multi_string(
    'gin_config', [], 'List of paths to the config files.')
flags.DEFINE_multi_string(
    'gin_bindings', [], 'Newline separated list of Gin parameter bindings.')

FLAGS = flags.FLAGS
ROOT = 'third_party/google_research/google_research/perturbations/'
CONFIGS_PATH = os.path.join(ROOT, 'experiments/configs/')


def main(unused_argv):
  tf.enable_v2_behavior()
  filenames = [os.path.join(CONFIGS_PATH, name) for name in FLAGS.gin_config]
  gin.parse_config_files_and_bindings(filenames, FLAGS.gin_bindings)
  tf.random.set_seed(FLAGS.run_number)
  training_loop.TrainingLoop(FLAGS.base_dir).run()


if __name__ == '__main__':
  app.run(main)

