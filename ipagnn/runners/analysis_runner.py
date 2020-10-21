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

# Lint as: python3
"""Runner binary for the analysis workflows."""

import os

from absl import app
from absl import flags
from absl import logging  # pylint: disable=unused-import

from ml_collections.config_flags import config_flags

from ipagnn.lib import path_utils
from ipagnn.lib import setup
from ipagnn.workflows import analysis_workflows



DEFAULT_DATA_DIR = os.path.expanduser(os.path.join('~', 'tensorflow_datasets'))
DEFAULT_CONFIG = 'ipagnn/config/config.py'

flags.DEFINE_string('data_dir', DEFAULT_DATA_DIR, 'Where to place the data.')
flags.DEFINE_string('run_dir',
                    '/tmp/learned_interpreters/default/',
                    'The directory to use for this run of the experiment.')
config_flags.DEFINE_config_file(
    name='config',
    default=DEFAULT_CONFIG,
    help_string='config file')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # Unused.

  data_dir = FLAGS.data_dir
  xm_parameters = {}
  run_dir = path_utils.expand_run_dir(FLAGS.run_dir, xm_parameters)
  config = FLAGS.config
  override_values = FLAGS['config'].override_values

  run_configuration = setup.configure(
      data_dir, run_dir, config, override_values, xm_parameters)
  analysis_workflows.run(run_configuration)


if __name__ == '__main__':
  app.run(main)
