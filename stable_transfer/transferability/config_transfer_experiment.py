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

"""Test config describing small transfer experiment."""

import ml_collections


def get_config(config_string='source_selection'):
  config = TransferExperimentConfig()
  config.build(config_string)
  return config


class TransferExperimentConfig(ml_collections.ConfigDict):
  """Config for a Small Transfer Experiments."""

  def build(self, config_string):
    """Build Config."""
    self.config_name = config_string
    self.only_load = False

    self.results = ml_collections.ConfigDict()
    self.results.basedir = './results/'
    self.results.extension = 'pickle'

    # Source model configuration:
    self.source = ml_collections.ConfigDict()
    self.source.network_architecture = 'resnet50'
    self.source.dataset = ml_collections.ConfigDict()
    self.source.dataset.name = 'imagenet'
    self.source.dataset.num_classes = 1000

    # Target dataset configuration:
    self.target = ml_collections.ConfigDict()
    self.target.dataset = ml_collections.ConfigDict()
    self.target.dataset.name = 'cifar10'
    self.target.dataset.num_classes = 10

    # Class selection methods - implemented: all, random, fixed
    self.target.class_selection = ml_collections.ConfigDict()
    self.target.class_selection.method = 'all'
    self.target.class_selection.seed = 123
    self.target.class_selection.experiment_number = 0
    self.target.class_selection.fixed = ml_collections.ConfigDict()

    # Experiment
    self.experiment = ml_collections.ConfigDict()
    self.experiment.dataset_as_supervised = True
    self.experiment.metric = 'leep'

    self.experiment.leep = ml_collections.ConfigDict()
