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

r"""General TF-Agents trainer executable.

Runs training on a TFAgent in a specified environment. It is recommended that
the agent be configured using Gin-config and the --gin_file flag, but you
can also import the train function and pass an agent class that you have
configured manually.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow.compat.v1 as tf



flags.DEFINE_string('root_dir', None,
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the study config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding to pass through.')
FLAGS = flags.FLAGS


def main(_):
  logging.set_verbosity(logging.INFO)
  tf.enable_v2_behavior()
  tf.enable_resource_variables()
  tf.enable_control_flow_v2()
  logging.info('Executing eagerly: %s', tf.executing_eagerly())
  logging.info('parsing config files: %s', FLAGS.gin_file)
  gin.parse_config_files_and_bindings(
      FLAGS.gin_file, FLAGS.gin_bindings, skip_unknown=True)

  trainer.train(root_dir, eval_metrics_callback=metrics_callback)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
