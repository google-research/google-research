# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Config to run transferability metrics for source selection in semantic segmentation."""

import ml_collections


def get_config(config_string='source_selection_segmentation'):
  config = TransferExperimentConfig()
  config.build(config_string)
  return config


class TransferExperimentConfig(ml_collections.ConfigDict):
  """Config for Transfer Experiments."""

  def build(self, config_string):
    """Build Config."""
    self.config_name = config_string
    self.only_load = False

    self.results = ml_collections.ConfigDict()
    self.results.basedir = './official_results/'
    self.results.extension = 'pickle'

    # Source model configuration:
    self.source = ml_collections.ConfigDict()
    self.source.network_architecture = 'HRNet'
    self.source.dataset = ml_collections.ConfigDict()
    self.source.dataset.name = 'coco'
    self.source.dataset.name_range = [
        'pvoc', 'isprs', 'vkitti2', 'vgallery', 'sunrgbd', 'suim', 'scannet',
        'msegpcontext', 'mapillary', 'kitti', 'isaid', 'idd', 'coco', 'city',
        'camvid', 'bdd', 'ade20k']

    # Target dataset configuration:
    self.target = ml_collections.ConfigDict()
    self.target.dataset = ml_collections.ConfigDict()
    self.target.dataset.name = 'pvoc'
    self.target.dataset.num_examples = 150  # Number of training examples to use
    # Semantic Segmentation specific parameters for sampling pixels from images
    self.target.dataset.class_balanced_sampling = True
    self.target.dataset.pixels_per_image = 1000  # None to use the whole image

    # Class selection methods - implemented: all, random, fixed
    self.target.class_selection = ml_collections.ConfigDict()

    self.target.class_selection.method = 'all'
    self.target.class_selection.seed = 123
    self.target.class_selection.experiment_number = 0
    self.target.class_selection.fixed = ml_collections.ConfigDict()

    # Experiment
    self.experiment = ml_collections.ConfigDict()
    self.experiment.dataset_as_supervised = True
    self.experiment.metric = 'logme'
    self.experiment.metric_range = ['gbc', 'hscore', 'leep', 'nleep', 'logme']
    for metric in self.experiment.metric_range:
      self.experiment[metric] = ml_collections.ConfigDict()
      self.experiment[metric]['batch_size'] = 1
    # Per-metric specific options
    self.experiment.gbc.gaussian_type = 'diagonal'
    self.experiment.gbc.pca_reduction = 64
    self.experiment.nleep.pca_reduction = 0.8
