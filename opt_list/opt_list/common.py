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
"""Shared code for loading configs for learned optimizer lists."""
import json
import os
from typing import Text, Dict, Any
from absl import logging


def _get_opt_name_content():
  """Returns the contents of the file containing optimizer names."""
  path = os.path.join(os.path.dirname(__file__), "nadamw_opt_list.txt")
  return open(path).read()


def _get_config_map():
  """Returns the optimizer name to configuration dict."""
  path = os.path.join(os.path.dirname(__file__), "nadamw_configs.json")
  configs = json.loads(open(path).read())
  return configs


def get_optimizer_config(idx):
  """Get the optimizer config from the list of hparams at the given index."""
  names = [x.strip() for x in _get_opt_name_content().split("\n") if x.strip()]
  name_to_use = names[idx]
  config, _ = _get_config_map()[name_to_use]
  logging.info("Using config:: %s", str(config))
  return config
