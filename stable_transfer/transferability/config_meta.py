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

"""Config utils to run different experimental setups for evaluating transferability metrics."""

from stable_transfer.transferability import config_training_source_selection
from stable_transfer.transferability import config_training_target_selection
from stable_transfer.transferability import config_transferability_source_selection
from stable_transfer.transferability import config_transferability_source_selection_segmentation as config_source_segmentation
from stable_transfer.transferability import config_transferability_target_selection


def get_config(config_string='training_source.oxford_flowers102'):
  """Get config."""
  experiment_config, dataset_name = config_string.split('.')
  assert (experiment_config in [
      'training_source', 'training_target', 'transferability_source',
      'transferability_source_segmentation', 'transferability_target'
  ])

  if experiment_config == 'training_source':
    config = config_training_source_selection.TransferExperimentConfig()
  elif experiment_config == 'training_target':
    config = config_training_target_selection.TransferExperimentConfig()
  elif experiment_config == 'transferability_source':
    config = config_transferability_source_selection.TransferExperimentConfig()
  elif experiment_config == 'transferability_source_segmentation':
    config = config_source_segmentation.TransferExperimentConfig()
  else:
    config = config_transferability_target_selection.TransferExperimentConfig()

  config.build(experiment_config)
  config.config_name = config_string
  config.results.basedir = './official_results/'

  config.target.dataset.name = dataset_name

  return config
