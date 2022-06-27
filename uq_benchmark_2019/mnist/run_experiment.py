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

"""Runs distributional-skew UQ experiments on MNIST.

Lightweight wrapper around experiment.py -- mostly useful to keep flags and
py_binary boilerplate out of colab during development.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from uq_benchmark_2019.mnist import experiment
from uq_benchmark_2019.mnist import models_lib

FLAGS = flags.FLAGS
flags.DEFINE_enum('arch', None, models_lib.ARCHITECTURES,
                  'Name of NN architecture.')
flags.DEFINE_enum('method', None, models_lib.METHODS,
                  'Name of modeling method.')
flags.DEFINE_string('output_dir', None, 'Output directory.')

flags.DEFINE_integer('test_level', 0,
                     '(0) no testing, (1) quick training (2) fake data.')
flags.DEFINE_integer('task', 0, 'Task number.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  experiment.run(FLAGS.method, FLAGS.arch,
                 FLAGS.output_dir.replace('%task%', str(FLAGS.task)),
                 test_level=FLAGS.test_level)

if __name__ == '__main__':
  app.run(main)
