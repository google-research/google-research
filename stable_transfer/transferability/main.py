# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Main file for Stable Transferability Experiments.

This file runs an stable transferability experiment, for example it computes
the LEEP score for a target dataset, provided a source model or architecture.
These results are saved on disk and subsequent calls to the same experiment will
simply load the results from disk.

Experiments are defined in config files (given using the --my_config flag).
See the examples (config_*.py) in this directory.
"""

from typing import Sequence

from absl import app
from absl import logging
from ml_collections.config_flags import config_flags


from stable_transfer.transferability import accuracy
from stable_transfer.transferability import gbc
from stable_transfer.transferability import hscore
from stable_transfer.transferability import leep
from stable_transfer.transferability import logme
from stable_transfer.transferability import nleep
from stable_transfer.transferability import transfer_experiment


_CONFIG_DIR = './'

_CONFIG = config_flags.DEFINE_config_file(
    'my_config',
    f'{_CONFIG_DIR}/stable_transfer/transferability/config_transfer_experiment.py',
    )


def run_experiment(experiment):
  """Run the experiment defined in the config file."""
  if experiment.config.experiment.metric == 'accuracy':
    return accuracy.get_test_accuracy(experiment)
  if experiment.config.experiment.metric == 'leep':
    return leep.get_train_leep(experiment)
  if experiment.config.experiment.metric == 'logme':
    return logme.get_train_logme(experiment)
  if experiment.config.experiment.metric == 'hscore':
    return hscore.get_train_hscore(experiment)
  if experiment.config.experiment.metric == 'nleep':
    return nleep.get_train_nleep(experiment)
  if experiment.config.experiment.metric == 'gbc':
    return gbc.get_train_gbc(experiment)
  return ValueError('Metric (%s) unknown', experiment.config.experiment.metric)


def main(_):
  r = run_experiment(transfer_experiment.TransferExperiment(_CONFIG.value))
  logging.info('EXPERIMENT DONE %s', str(r))

if __name__ == '__main__':
  app.run(main)
