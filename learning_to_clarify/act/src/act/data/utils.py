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

from collections import defaultdict
from typing import Any, Optional, Union

from act.config.base_config import BaseConfig, BaseInitializationConfig
from act.data.base_dataset import ACTDataset
from act.models.preference_model import RejectedSampleModel
from act.utils.storage_utils import read_jsonl
from datasets import Dataset


def prepare_sample(example):
  return [str(example['input_text']) + ' ' + str(example['output_text'])]


def get_data_from_path(path):
  """Gets the data from a path."""
  return read_jsonl(path)


def get_datasets_from_config(
    config,
    preference_model = None,
):
  """Get the datasets from the config."""
  if not preference_model:
    preference_model = RejectedSampleModel(config,
                                           config.preference_model_config)

  train_path = config.data_config.train_path
  dev_path = config.data_config.validation_path
  train, validation = get_data_from_path(train_path), get_data_from_path(
      dev_path
  )
  train_dataset = ACTDataset(
      train, config.training_config.target_label,
      config.training_config.icl_examples, preference_model,
      class_balance=config.training_config.class_balance,
      is_preference=config.training_config.is_preference,
      has_context_metadata=config.data_config.has_context_metadata,
      preference_batch_generation=config.data_config.preference_batch_generation,
  ).prepare_datasets()
  val_dataset = ACTDataset(
      validation, config.training_config.target_label,
      config.training_config.icl_examples, preference_model,
      class_balance=config.training_config.class_balance,
      is_preference=config.training_config.is_preference,
      has_context_metadata=config.data_config.has_context_metadata,
      preference_batch_generation=config.data_config.preference_batch_generation,
  ).prepare_datasets()
  return train_dataset, val_dataset
