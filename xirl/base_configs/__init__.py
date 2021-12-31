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

"""Ensure user-defined pretraining & rl configs inherit from base configs."""

from ml_collections import ConfigDict

from .pretrain import get_config as get_pretrain_config
from .rl import get_config as get_rl_config


# Ref: https://github.com/deepmind/jaxline/blob/master/jaxline/base_config.py
def __validate_keys(  # pylint: disable=invalid-name
    base_config,
    config,
    base_filename,
):
  """Validate keys."""
  for key in base_config.keys():
    if key not in config:
      raise ValueError(
          f"Key {key} missing from config. This config is required to have "
          f"keys: {list(base_config.keys())}. See base_configs/{base_filename} "
          "for more details.")
    if (isinstance(base_config[key], ConfigDict) and config[key] is not None):
      __validate_keys(base_config[key], config[key], base_filename)


def validate_config(config, mode):
  """Ensures a config inherits from a base config.

  Args:
    config: The child config to validate.
    mode: Can be one of 'pretraining' or 'rl'.

  Raises:
    ValueError: if the base config contains keys that are not present in config.
  """
  assert mode in ["pretrain", "rl"]
  base_config = get_rl_config() if mode == "rl" else get_pretrain_config()
  base_filename = "rl.py" if mode == "rl" else "pretrain.py"
  __validate_keys(base_config, config, base_filename)
