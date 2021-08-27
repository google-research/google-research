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

"""Utility for checking config_dict as hparams dictionary."""

import dataclasses
import json
import os
import sys

from absl import app
from absl import flags
from ml_collections import config_flags

from aqt.jax.imagenet import hparams_config
from aqt.jax.imagenet import models
from aqt.utils import hparams_utils

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('hparams_config_dict', None,
                                'Path to file defining a config dict.')
flags.DEFINE_string(
    'output_dir',
    default=None,
    help='directory to dump hparams file to, if not dumps to stdout')


def main(_):
  if not FLAGS['hparams_config_dict'].present:
    raise ValueError(
        'Flag hparams_config_dict is required, but was not provided.')

  config = FLAGS.hparams_config_dict
  if 'configs' in config:
    config = config.configs[0]
    print('getting only first config dict')

  hparams = hparams_utils.load_hparams_from_config_dict(
      hparams_config.TrainingHParams, models.ResNet.HParams, config)

  data_dict = dataclasses.asdict(hparams)

  if FLAGS.output_dir is not None:
    output_path = os.path.join(FLAGS.output_dir, 'hparams_config.json')
    with open(output_path, 'w') as file:
      json.dump(data_dict, file, indent=2)
  else:
    json.dump(data_dict, sys.stdout, indent=2)


if __name__ == '__main__':
  app.run(main)
