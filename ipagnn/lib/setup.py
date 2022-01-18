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

# Lint as: python3
"""Performs setup tasks for Learned Interpreters binaries."""

import dataclasses
import json
import os
import random

from typing import Any, Text

from absl import logging  # pylint: disable=unused-import
import numpy as np
import tensorflow as tf

from ipagnn.adapters import adapters_lib
from ipagnn.config import overrides_lib
from ipagnn.lib import checkpoint_utils
from ipagnn.lib import config_utils
from ipagnn.lib import dataset_utils
from ipagnn.models import models_lib


gfile = tf.io.gfile


@dataclasses.dataclass
class RunConfiguration:
  """The configuration for a single experimental run."""
  mode: Text
  method: Text
  run_dir: Text
  data_dir: Text
  original_checkpoint_path: Text
  model: Any
  info: Any  # Info
  config: Any  # Config
  adapter: Any  # Adapter
  dataset_info: Any  # Tuple


def seed():
  random.seed(0)
  np.random.seed(0)




def configure(data_dir, run_dir, config, override_values, xm_parameters=None):
  """Sets up the Learned Interpreter code with the specified configuration."""
  seed()

  # Apply any overrides set at the command line or in the launcher.
  if config.overrides != config.default_overrides:
    logging.info('Applying overrides set at command line: %s', config.overrides)
    overrides_lib.apply_overrides(
        config, override_names=config.overrides.split(','))
  config.update_from_flattened_dict(override_values)

  # If a checkpoint is specified, it determines the "original run."
  # Otherwise the run_dir, if already present, determines the "original run."
  config_filepath = os.path.join(run_dir, 'config.json')
  if checkpoint_utils.is_checkpoint_specified(config.checkpoint):
    original_checkpoint_path = checkpoint_utils.get_specified_checkpoint_path(
        run_dir, config.checkpoint)
    original_run_dir = checkpoint_utils.get_run_dir(original_checkpoint_path)
    original_config_filepath = os.path.join(original_run_dir, 'config.json')
  else:
    checkpoint_dir = checkpoint_utils.build_checkpoint_dir(run_dir)
    original_checkpoint_path = checkpoint_utils.latest_checkpoint(
        checkpoint_dir)
    original_config_filepath = config_filepath
  original_config_exists = gfile.exists(original_config_filepath)

  # Handle any existing configs.
  if original_config_exists:
    original_config = config_utils.load_config(original_config_filepath)

    # Handle the model config.
    if config.runner.model_config == 'load':
      logging.info('Loading the model config from %s', original_config_filepath)
      config.model.update(original_config.model)
      config.dataset.representation = original_config.dataset.representation
    elif config.runner.model_config == 'assert':
      same_config = config_utils.equals(config.model, original_config.model)
      # Resolution:
      # Either use a new run_dir, or set model_config to 'load' or 'keep'.
      assert same_config, 'Model config has changed.'
    else:
      assert config.runner.model_config == 'keep'

    # Handle the dataset config.
    if config.runner.dataset_config == 'load':
      logging.info('Loading the data config from %s', original_config_filepath)
      config.dataset.update(original_config.dataset)
    elif config.runner.dataset_config == 'assert':
      same_config = config_utils.equals(config.dataset, original_config.dataset)
      assert same_config, 'Dataset config has changed.'
    else:
      assert config.runner.dataset_config == 'keep'

  elif (config.runner.model_config == 'load'
        or config.runner.dataset_config == 'load'):
    raise ValueError('Original model config not found.')

  # In interactive mode, force batch size 1.
  if config.runner.mode == 'interact':
    config.dataset.batch_size = 1

  config_exists = gfile.exists(config_filepath)
  if not config_exists and config.runner.mode in 'train':
    gfile.makedirs(run_dir)
    config_utils.save_config(config, config_filepath)

  # Load dataset.
  if config.setup.setup_dataset:
    dataset_info = dataset_utils.get_dataset(data_dir, config)
    info = dataset_info.info
  else:
    dataset_info = None
    info = None

  # Create model.
  if config.setup.setup_model:
    model = models_lib.get_model(info, config)
  else:
    model = None

  adapter = adapters_lib.get_default_adapter(info, config)

  return RunConfiguration(
      mode=config.runner.mode,
      method=config.runner.method,
      run_dir=run_dir,
      data_dir=data_dir,
      original_checkpoint_path=original_checkpoint_path,
      model=model,
      info=info,
      config=config,
      adapter=adapter,
      dataset_info=dataset_info)
