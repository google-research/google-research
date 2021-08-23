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

"""Functionality common to pretraining and evaluation."""

from random import shuffle
from typing import Dict
from ml_collections import ConfigDict

import torch
from xirl import factory
from xirl.models import SelfSupervisedModel

DataLoadersDict = Dict[str, torch.utils.data.DataLoader]
ModelType = SelfSupervisedModel


def get_pretraining_dataloaders(
    config: ConfigDict,
    debug: bool = False,
) -> DataLoadersDict:
  """Construct a train/valid pair of pretraining dataloaders.

  Args:
    config: ConfigDict object with config parameters.
    debug: When set to True, the following happens: 1. Data augmentation is
      disabled regardless of config values. 2. Sequential sampling of videos is
      turned on. 3. The number of dataloader workers is set to 0.

  Returns:
    A dict of train/valid pretraining dataloaders.
  """

  def _loader(split):
    dataset = factory.dataset_from_config(config, False, split, debug)
    batch_sampler = factory.video_sampler_from_config(
        config, dataset.dir_tree, downstream=False, sequential=debug)
    return torch.utils.data.DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        batch_sampler=batch_sampler,
        num_workers=4 if torch.cuda.is_available() and not debug else 0,
        pin_memory=torch.cuda.is_available() and not debug,
    )

  return {
      "train": _loader("train"),
      "valid": _loader("valid"),
  }


def get_downstream_dataloaders(
    config: ConfigDict,
    debug: bool = False,
) -> DataLoadersDict:
  """Construct a train/valid pair of downstream dataloaders.

  Args:
    config: ConfigDict object with config parameters.
    debug: When set to True, the following happens: 1. Data augmentation is
      disabled regardless of config values. 2. Sequential sampling of videos is
      turned on. 3. The number of dataloader workers is set to 0.

  Returns:
    A dict of train/valid downstream dataloaders
  """

  def _loader(split):
    datasets = factory.dataset_from_config(config, True, split, debug)
    loaders = {}
    for action_class, dataset in datasets.items():
      batch_sampler = factory.video_sampler_from_config(
          config, dataset.dir_tree, downstream=True, sequential=debug)
      loaders[action_class] = torch.utils.data.DataLoader(
          dataset,
          collate_fn=dataset.collate_fn,
          batch_sampler=batch_sampler,
          num_workers=4 if torch.cuda.is_available() and not debug else 0,
          pin_memory=torch.cuda.is_available() and not debug,
      )
    return loaders

  return {
      "train": _loader("train"),
      "valid": _loader("valid"),
  }


def get_factories(
    config: ConfigDict,
    device: torch.device,
    debug: bool = False,
):
  """Feed config to factories and return objects."""
  pretrain_loaders = get_pretraining_dataloaders(config, debug)
  downstream_loaders = get_downstream_dataloaders(config, debug)
  model = factory.model_from_config(config)
  optimizer = factory.optim_from_config(config, model)
  trainer = factory.trainer_from_config(config, model, optimizer, device)
  eval_manager = factory.evaluator_from_config(config)
  return (
      model,
      optimizer,
      pretrain_loaders,
      downstream_loaders,
      trainer,
      eval_manager,
  )


def get_model(config: ConfigDict) -> ModelType:
  """Construct a model from a config."""
  return factory.model_from_config(config)
